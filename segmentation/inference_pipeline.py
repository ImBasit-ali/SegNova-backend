"""Stack validation and model-input preparation for segmentation."""

import logging
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
from django.conf import settings

from .model_loader import get_model_path
from .stacking import EXPECTED_MODALITIES, infer_extension, stack_nifti_files, validate_upload_combination

logger = logging.getLogger(__name__)


def require_nifti_extension(extension):
    if extension not in ('.nii', '.nii.gz'):
        raise ValueError(
            'Model inference requires NIfTI files (.nii or .nii.gz). '
            'Upload four modalities as NIfTI, stack, then run segmentation.'
        )


def ensure_modality_wrappers(file_wrappers):
    """Require four distinct BraTS modalities unless a single stacked file is provided."""
    if len(file_wrappers) == 1:
        return file_wrappers

    modalities = {item.modality for item in file_wrappers}
    missing = [m for m in EXPECTED_MODALITIES if m not in modalities]
    if missing:
        raise ValueError(
            'All four modalities are required for model inference: '
            f'{", ".join(m.upper() for m in EXPECTED_MODALITIES)}. '
            f'Missing: {", ".join(m.upper() for m in missing)}.'
        )
    return file_wrappers


def build_stacked_nifti(file_wrappers, upload_mode):
    """Stack uploads into a 4-channel NIfTI volume for model.keras."""
    require_nifti_extension(infer_extension(file_wrappers[0].original_name))

    if upload_mode == 'modalities-four':
        stacked = stack_nifti_files(file_wrappers)
    elif upload_mode == 'stacked-single':
        source = file_wrappers[0]
        wrapped = [
            SimpleNamespace(
                original_name=source.original_name,
                modality=modality,
                file=source.file,
            )
            for modality in EXPECTED_MODALITIES
        ]
        stacked = stack_nifti_files(wrapped)
    else:
        raise ValueError(
            f'Unsupported upload mode for inference: {upload_mode}. '
            'Upload exactly four modalities or one file to duplicate.'
        )

    data = np.asarray(stacked.dataobj, dtype=np.float32)
    if data.ndim == 3:
        raise ValueError(
            f'Stacked volume must be 4D (X, Y, Z, 4); got 3D shape {data.shape}.'
        )
    if data.ndim != 4 or data.shape[-1] != 4:
        raise ValueError(
            f'Model expects shape (X, Y, Z, 4) after stacking; got {data.shape}.'
        )

    logger.info(
        'Built stacked volume shape=%s dtype=%s model=%s',
        data.shape,
        data.dtype,
        get_model_path(),
    )
    return stacked


def middle_axial_preview_slice(volume_data):
    """Return a normalized uint8 axial slice for PNG preview (3D or 4-channel stack)."""
    arr = np.asarray(volume_data, dtype=np.float32)
    if arr.ndim == 4:
        mid_z = arr.shape[2] // 2
        channel = 0 if arr.shape[-1] >= 1 else 0
        slab = arr[:, :, mid_z, channel]
    elif arr.ndim == 3:
        mid_z = arr.shape[2] // 2
        slab = arr[:, :, mid_z]
    else:
        raise ValueError(f'Cannot build preview from volume shape {arr.shape}')

    vmin, vmax = float(np.min(slab)), float(np.max(slab))
    if vmax > vmin:
        slab = (slab - vmin) / (vmax - vmin) * 255.0
    else:
        slab = np.zeros_like(slab, dtype=np.float32)
    return np.clip(slab, 0, 255).astype(np.uint8)


def write_viewer_base_nifti(stacked_path, job_id, channel_index=2, previews_dir=None):
    """Extract T2 (channel 2) as 3D volume for aligned mask overlay in Niivue."""
    previews_dir = previews_dir or (Path(settings.MEDIA_ROOT) / 'previews')
    previews_dir.mkdir(parents=True, exist_ok=True)
    ref = nib.load(stacked_path)
    data = np.asarray(ref.dataobj, dtype=np.float32)
    if data.ndim == 4:
        vol3d = data[..., min(channel_index, data.shape[-1] - 1)]
    elif data.ndim == 3:
        vol3d = data
    else:
        raise ValueError(f'Cannot build viewer base from shape {data.shape}')

    base_name = f'{job_id}_viewer_t2.nii.gz'
    base_path = previews_dir / base_name
    base_img = nib.Nifti1Image(vol3d.astype(np.float32), ref.affine, ref.header.copy())
    nib.save(base_img, str(base_path))
    relative = f"{settings.MEDIA_URL.rstrip('/')}/previews/{base_name}"
    return str(base_path), relative


def write_stacked_preview(stacked_nifti, job_id, previews_dir=None):
    """Persist {job_id}_stacked.nii.gz under media/previews/."""
    previews_dir = previews_dir or (Path(settings.MEDIA_ROOT) / 'previews')
    previews_dir.mkdir(parents=True, exist_ok=True)
    stacked_name = f'{job_id}_stacked.nii.gz'
    stacked_path = previews_dir / stacked_name
    nib.save(stacked_nifti, str(stacked_path))
    relative = f"{settings.MEDIA_URL.rstrip('/')}/previews/{stacked_name}"
    return str(stacked_path), relative


def prepare_model_input(file_wrappers, job_id, force_restack=True, previews_dir=None):
    """
    Validate uploads, stack to 4-channel NIfTI, save preview, return paths.

    Always rebuilds the stacked file when force_restack=True (default) so inference
    uses the same modalities that were just uploaded.
    """
    file_wrappers = ensure_modality_wrappers(file_wrappers)
    upload_mode, extension = validate_upload_combination(file_wrappers)
    require_nifti_extension(extension)

    stacked_nifti = build_stacked_nifti(file_wrappers, upload_mode)
    stacked_path, stacked_url = write_stacked_preview(
        stacked_nifti,
        job_id,
        previews_dir=previews_dir,
    )

    if not force_restack:
        logger.warning('prepare_model_input called with force_restack=False; still rewrote stack.')

    return {
        'stacked_path': stacked_path,
        'stacked_url': stacked_url,
        'stacked_nifti': stacked_nifti,
        'upload_mode': upload_mode,
        'extension': extension,
    }
