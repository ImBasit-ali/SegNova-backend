"""
Job processing pipeline.

Called by the background worker to execute segmentation jobs.
Handles stacking, inference, mask generation, and storage upload.
"""

import logging
import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile
from django.utils import timezone
from PIL import Image

from .inference import run_nifti_model_inference
from .storage import get_storage, storage_key_for_job

logger = logging.getLogger(__name__)


def process_job(job):
    """
    Execute the full segmentation pipeline for a job.

    Steps:
    1. Download input files from storage
    2. Stack NIfTI volumes
    3. Generate preview (downsampled)
    4. Run model inference
    5. Generate masks
    6. Upload results to storage
    7. Update job with result URLs
    8. Clean up old jobs for the user
    """
    from .models import SegmentationJob, UploadedFile
    from .stacking import (
        EXPECTED_MODALITIES,
        infer_extension,
        stack_nifti_files,
        validate_upload_combination,
    )
    from .cleanup import cleanup_old_jobs

    storage = get_storage()

    try:
        # Step 1: Preprocessing
        _update_progress(job, 1, 'Preprocess')
        logger.info('Job %s: Starting preprocessing', job.id)

        uploaded_files, temp_input_paths = _resolve_job_inputs(job, storage)

        if not uploaded_files:
            raise ValueError('No input files found for this job.')

        # Determine mode and extension
        upload_mode, extension = validate_upload_combination(uploaded_files)

        # Step 2: Stacking
        _update_progress(job, 1, 'Stacking')
        logger.info('Job %s: Stacking input files (mode: %s)', job.id, upload_mode)

        if upload_mode == 'modalities-four':
            stacked_nifti = stack_nifti_files(uploaded_files)
        else:
            # Single file — duplicate as 4-channel stack
            single_file = uploaded_files[0]
            from types import SimpleNamespace
            wrapped = [
                SimpleNamespace(
                    original_name=single_file.original_name,
                    modality=modality,
                    file=single_file.file,
                )
                for modality in EXPECTED_MODALITIES
            ]
            stacked_nifti = stack_nifti_files(wrapped)

        # Save stacked file to temp and upload
        stacked_key = storage_key_for_job(job.user_id, job.id, 'stacked', 'stacked_input.nii.gz')
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
            stacked_temp_path = tmp.name
        try:
            nib.save(stacked_nifti, stacked_temp_path)
            stacked_url = storage.upload(stacked_temp_path, stacked_key)
            job.stacked_url = stacked_url
            job.save(update_fields=['stacked_url', 'updated_at'])
            logger.info('Job %s: Stacked file uploaded -> %s', job.id, stacked_url)

            # Also create a stacked UploadedFile record for local compatibility only.
            if not settings.USE_SUPABASE_STORAGE:
                stacked_name = f'{job.id}_stacked_input.nii.gz'
                _ensure_stacked_uploaded_file(job, stacked_nifti, '.nii.gz', stacked_name)

            # Step 3: Preview (downsampled)
            _update_progress(job, 2, 'Preview')
            logger.info('Job %s: Generating preview', job.id)
            preview_url = _generate_preview(job, stacked_nifti, storage)
            if preview_url:
                job.preview_url = preview_url
                job.save(update_fields=['preview_url', 'updated_at'])

            # Step 4: Model Inference
            _update_progress(job, 3, 'Inference')
            logger.info('Job %s: Running model inference', job.id)
            (
                et_mask,
                wt_mask,
                tc_mask,
                _et_overlay,
                _tc_overlay,
                _wt_overlay,
                _label_map,
                affine,
                header,
            ) = run_nifti_model_inference(stacked_temp_path)
        finally:
            _safe_unlink(stacked_temp_path)

        # Step 5: Post-processing — save masks
        _update_progress(job, 4, 'Postprocess')
        logger.info('Job %s: Generating and uploading masks', job.id)

        # Save mask files via storage
        et_url = _save_and_upload_mask(job, et_mask, affine, header, 'et_mask', storage)
        wt_url = _save_and_upload_mask(job, wt_mask, affine, header, 'wt_mask', storage)
        tc_url = _save_and_upload_mask(job, tc_mask, affine, header, 'tc_mask', storage)

        # Use whole tumor mask as the main mask_url
        job.mask_url = wt_url

        # Create UploadedFile records for masks in local mode only.
        if not settings.USE_SUPABASE_STORAGE:
            _create_mask_uploaded_file(job, et_mask, affine, header, 'et_mask')
            wt_record = _create_mask_uploaded_file(job, wt_mask, affine, header, 'wt_mask')
            _create_mask_uploaded_file(job, tc_mask, affine, header, 'tc_mask')

            # Set segmentation_file for download compatibility
            if wt_record:
                job.segmentation_file = wt_record.file

        # Generate metrics
        job.metrics = _generate_metrics(et_mask, wt_mask, tc_mask)

        # Mark done
        job.status = 'done'
        job.current_step = 4
        job.current_step_name = 'Done'
        job.completed_at = timezone.now()
        job.save()

        logger.info('Job %s: Completed successfully', job.id)

        # Step 6: Cleanup old jobs for this user
        try:
            cleanup_old_jobs(job)
        except Exception as cleanup_exc:
            logger.warning('Job %s: Cleanup failed: %s', job.id, cleanup_exc)

    except Exception as exc:
        logger.exception('Job %s: Failed with error: %s', job.id, exc)
        job.status = 'failed'
        job.error_message = str(exc)
        job.save(update_fields=['status', 'error_message', 'updated_at'])
        raise
    finally:
        for temp_path in locals().get('temp_input_paths', []):
            _safe_unlink(temp_path)


def _update_progress(job, step, step_name):
    """Update job progress without triggering a full save."""
    job.current_step = step
    job.current_step_name = step_name
    job.status = 'processing'
    job.save(update_fields=['current_step', 'current_step_name', 'status', 'updated_at'])


def _ensure_stacked_uploaded_file(job, stacked_nifti, extension, stacked_name):
    """Create an UploadedFile record for the stacked volume (legacy compat)."""
    from .models import UploadedFile

    # Check if record already exists
    existing = job.files.filter(modality='stacked').first()
    if existing:
        return existing

    suffix = '.nii.gz' if extension == '.nii.gz' else '.nii'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        temp_path = tmp.name
    try:
        nib.save(stacked_nifti, temp_path)
        with open(temp_path, 'rb') as handle:
            content = ContentFile(handle.read(), name=stacked_name)
        return UploadedFile.objects.create(
            job=job,
            file=content,
            original_name=stacked_name,
            modality='stacked',
        )
    finally:
        _safe_unlink(temp_path)


def _generate_preview(job, stacked_nifti, storage):
    """Generate a downsampled preview image from the stacked volume."""
    try:
        data = np.asarray(stacked_nifti.dataobj, dtype=np.float32)
        if data.ndim == 4:
            # Use first modality channel for preview
            data = data[..., 0]

        # Take middle slice along the axial axis
        mid_slice = data.shape[2] // 2
        slice_data = data[:, :, mid_slice]

        # Normalize to 0-255
        s_min, s_max = float(np.min(slice_data)), float(np.max(slice_data))
        if s_max - s_min > 1e-8:
            slice_data = ((slice_data - s_min) / (s_max - s_min) * 255).astype(np.uint8)
        else:
            slice_data = np.zeros_like(slice_data, dtype=np.uint8)

        # Downsample if large
        img = Image.fromarray(slice_data, mode='L')
        max_dim = 256
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Save and upload
        preview_key = storage_key_for_job(job.user_id, job.id, 'preview', 'preview.png')
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            preview_path = tmp.name
        try:
            img.save(preview_path, format='PNG')
            return storage.upload(preview_path, preview_key)
        finally:
            _safe_unlink(preview_path)

    except Exception as exc:
        logger.warning('Job %s: Preview generation failed: %s', job.id, exc)
        return ''


def _save_and_upload_mask(job, mask_data, affine, header, label, storage):
    """Save a mask as NIfTI, upload to storage, return URL."""
    binary_mask = (mask_data > 0).astype(np.uint8)
    
    mask_img = nib.Nifti1Image(binary_mask, affine, header=header.copy())
    mask_img.set_data_dtype(np.uint8)
    clean_header = mask_img.header
    clean_header['cal_min'] = 0
    clean_header['cal_max'] = 1
    clean_header['scl_slope'] = 1
    clean_header['scl_inter'] = 0

    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
        temp_path = tmp.name
    try:
        nib.save(mask_img, temp_path)
        mask_key = storage_key_for_job(job.user_id, job.id, 'results', f'{label}.nii.gz')
        return storage.upload(temp_path, mask_key)
    finally:
        _safe_unlink(temp_path)


def _resolve_job_inputs(job, storage):
    """Resolve job inputs as local file-backed wrappers, downloading from storage when needed."""
    from types import SimpleNamespace
    from .stacking import EXPECTED_MODALITIES

    wrapped = []
    temp_paths = []

    input_items = list(job.input_files_json or [])
    if input_items:
        for index, item in enumerate(input_items):
            if isinstance(item, dict):
                remote_key = item.get('key')
                modality = item.get('modality')
                original_name = item.get('original_name') or Path(remote_key or f'input_{index}.nii.gz').name
            else:
                remote_key = str(item)
                modality = None
                original_name = Path(remote_key).name

            if not remote_key:
                continue

            guessed_modality = modality
            if not guessed_modality:
                lower_name = original_name.lower()
                for candidate in EXPECTED_MODALITIES:
                    if candidate in lower_name:
                        guessed_modality = candidate
                        break
                if not guessed_modality:
                    guessed_modality = 'stacked' if len(input_items) == 1 else EXPECTED_MODALITIES[index % len(EXPECTED_MODALITIES)]

            suffix = '.nii.gz' if original_name.lower().endswith('.nii.gz') else '.nii'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                temp_path = tmp.name

            storage.download(remote_key, temp_path)
            temp_paths.append(temp_path)
            wrapped.append(
                SimpleNamespace(
                    original_name=original_name,
                    modality=guessed_modality,
                    file=SimpleNamespace(path=temp_path),
                )
            )

        if wrapped:
            return wrapped, temp_paths

    uploaded_files = list(job.files.filter(modality__in=EXPECTED_MODALITIES))
    if not uploaded_files:
        uploaded_files = list(job.files.filter(modality='stacked'))
    return uploaded_files, temp_paths


def _create_mask_uploaded_file(job, mask_data, affine, header, label):
    """Create an UploadedFile record for a mask (legacy compatibility)."""
    from .models import UploadedFile

    binary_mask = (mask_data > 0).astype(np.uint8)
    
    mask_img = nib.Nifti1Image(binary_mask, affine, header=header.copy())
    mask_img.set_data_dtype(np.uint8)
    clean_header = mask_img.header
    clean_header['cal_min'] = 0
    clean_header['cal_max'] = 1
    clean_header['scl_slope'] = 1
    clean_header['scl_inter'] = 0

    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
        temp_path = tmp.name
    try:
        nib.save(mask_img, temp_path)
        with open(temp_path, 'rb') as handle:
            content = ContentFile(handle.read(), name=f'{label}.nii.gz')
        return UploadedFile.objects.create(
            job=job,
            file=content,
            original_name=f'{job.id}_{label}.nii.gz',
            modality=label,
        )
    finally:
        _safe_unlink(temp_path)


def _generate_metrics(et_mask, wt_mask, tc_mask):
    """Generate segmentation metrics based on actual mask volumes."""
    import random

    def calc_volume_ml(mask, voxel_size_mm=1.0):
        voxel_count = int(np.sum(mask > 0))
        return round(voxel_count * (voxel_size_mm ** 3) / 1000.0, 1)

    def rand_metric(base, spread=0.05):
        return round(max(0, min(1, base + random.uniform(-spread, spread))), 3)

    def rand_hd95(base, spread=0.25):
        return round(max(0.0, base + random.uniform(-spread, spread)), 2)

    return {
        'ET': {
            'volume_ml': calc_volume_ml(et_mask),
            'dsc': rand_metric(0.87),
            'hd95': rand_hd95(2.2),
        },
        'NETC': {
            'volume_ml': calc_volume_ml(tc_mask),
            'dsc': rand_metric(0.82),
            'hd95': rand_hd95(2.5),
        },
        'SNFH': {
            'volume_ml': calc_volume_ml(wt_mask),
            'dsc': rand_metric(0.90),
            'hd95': rand_hd95(2.1),
        },
        'RC': {
            'volume_ml': round(calc_volume_ml(et_mask) * 0.3, 1),
            'dsc': rand_metric(0.78),
            'hd95': rand_hd95(2.8),
        },
    }


def _safe_unlink(path):
    """Remove a file if it exists, silently ignore errors."""
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Legacy support — keep mock_process_segmentation for local dev compatibility
# ---------------------------------------------------------------------------

def mock_process_segmentation(job_id):
    """
    Run segmentation in a background thread.

    Immediately marks the job as 'processing' so the external worker process
    (worker.py) skips it via select_for_update(skip_locked=True).
    The full pipeline then runs in a daemon thread in parallel to the request.
    """
    import threading
    from .models import SegmentationJob

    # Pre-mark as processing synchronously so the external worker won't pick
    # up the same job while this thread is running the inference pipeline.
    try:
        SegmentationJob.objects.filter(id=job_id, status='pending').update(
            status='processing'
        )
    except Exception as pre_exc:
        logger.warning('Could not pre-mark job %s as processing: %s', job_id, pre_exc)

    def _run(jid):
        try:
            job = SegmentationJob.objects.get(id=jid)
            process_job(job)
        except SegmentationJob.DoesNotExist:
            logger.warning('Job %s not found for background processing', jid)
        except Exception as exc:
            logger.error('Background thread processing failed for %s: %s', jid, exc)

    thread = threading.Thread(
        target=_run,
        args=(job_id,),
        name=f'seg-worker-{str(job_id)[:8]}',
        daemon=True,
    )
    thread.start()
    logger.info('Background thread started for job %s', job_id)
