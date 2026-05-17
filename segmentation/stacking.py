import os
import tempfile
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import nibabel as nib
import numpy as np
from PIL import Image

from .models import UploadedFile

ALLOWED_EXTENSIONS = ('.nii', '.nii.gz', '.png')
EXPECTED_MODALITIES = ('t1', 't1ce', 't2', 'flair')


def infer_extension(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith('.nii.gz'):
        return '.nii.gz'
    if lower.endswith('.nii'):
        return '.nii'
    if lower.endswith('.png'):
        return '.png'
    return ''


def _missing_modalities(modalities):
    return [modality for modality in EXPECTED_MODALITIES if modality not in modalities]


def _duplicate_modalities(modalities):
    duplicates = []
    for modality in EXPECTED_MODALITIES:
        if modalities.count(modality) > 1:
            duplicates.append(modality)
    return duplicates


def validate_upload_combination(uploaded_files: List[UploadedFile]) -> Tuple[str, str]:
    """
    Validate user input shape and return (mode, extension).

    mode:
    - stacked-single: one pre-stacked file
    - modalities-four: four modality-specific files to be stacked here
    """
    extensions = [infer_extension(item.original_name) for item in uploaded_files]
    if any(ext not in ALLOWED_EXTENSIONS for ext in extensions):
        raise ValueError('Only .nii, .nii.gz, and .png files are supported.')

    if len(set(extensions)) != 1:
        raise ValueError('All uploaded files must use the same format (.nii/.nii.gz or .png).')

    extension = extensions[0]
    count = len(uploaded_files)
    if count == 1:
        return 'stacked-single', extension

    modalities = [item.modality for item in uploaded_files]

    if count in (2, 3):
        duplicates = _duplicate_modalities(modalities)
        if duplicates:
            raise ValueError(f'File already uploaded for modality {duplicates[0].upper()}. Each modality can only be uploaded once.')

        missing = _missing_modalities(modalities)
        raise ValueError(
            f'Four modalities are required to stack. Upload {", ".join(m.upper() for m in missing)} or duplicate one of the existing files to fill the missing slot.'
        )

    if count != 4:
        raise ValueError('Upload either exactly 4 modality files or exactly 1 file to duplicate into a stacked input.')

    duplicates = _duplicate_modalities(modalities)
    if duplicates:
        raise ValueError(f'File already uploaded for modality {duplicates[0].upper()}. Each modality can only be uploaded once.')

    if set(modalities) != set(EXPECTED_MODALITIES):
        missing = _missing_modalities(modalities)
        raise ValueError(
            f'When uploading 4 files, modalities must include t1, t1ce, t2, and flair exactly once. Missing: {", ".join(m.upper() for m in missing)}.'
        )

    return 'modalities-four', extension


def _load_nifti_file(item):
    """Load a single NIfTI file (used for parallel loading)."""
    temp_path = None
    try:
        if hasattr(item.file, 'path') and item.file.path:
            nii = nib.load(item.file.path)
        elif hasattr(item.file, 'temporary_file_path'):
            nii = nib.load(item.file.temporary_file_path())
        else:
            suffix = '.nii.gz' if str(item.original_name).lower().endswith('.nii.gz') else Path(item.original_name).suffix or '.nii'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                temp_path = tmp.name
                if hasattr(item.file, 'seek'):
                    item.file.seek(0)
                if hasattr(item.file, 'chunks'):
                    for chunk in item.file.chunks():
                        tmp.write(chunk)
                else:
                    tmp.write(item.file.read())
            nii = nib.load(temp_path)

        data = np.asarray(nii.dataobj, dtype=np.float32)

        if data.ndim == 4:
            if data.shape[-1] == 1:
                data = data[..., 0]
            else:
                raise ValueError(f'{item.original_name} is already multi-channel; upload it as a single stacked file instead.')

        if data.ndim != 3:
            raise ValueError(f'{item.original_name} must be a 3D NIfTI volume.')

        return item.modality, data, nii.affine, nii.header.copy()
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def stack_nifti_files(uploaded_files: List[UploadedFile]) -> nib.Nifti1Image:
    """Stack NIfTI files with PARALLEL loading for speed."""
    ordered = sorted(uploaded_files, key=lambda f: EXPECTED_MODALITIES.index(f.modality) if f.modality in EXPECTED_MODALITIES else 999)

    volumes_by_modality = {}
    reference_shape = None
    reference_affine = None
    reference_header = None

    # **PARALLEL LOADING** - Load all files simultaneously
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_load_nifti_file, item): item for item in ordered}
        
        for future in as_completed(futures):
            modality, data, affine, header = future.result()
            
            if reference_shape is None:
                reference_shape = data.shape
                reference_affine = affine
                reference_header = header
            elif data.shape != reference_shape:
                raise ValueError('All modality NIfTI files must have identical dimensions.')
            
            volumes_by_modality[modality] = data

    # If only 1 file, duplicate into 4 channels
    if len(volumes_by_modality) == 1:
        volumes_by_modality = {modality: next(iter(volumes_by_modality.values())) for modality in EXPECTED_MODALITIES}

    if len(volumes_by_modality) != 4:
        raise ValueError('NIfTI stacking requires either 1 file to duplicate or exactly 4 modality files.')

    volumes = [volumes_by_modality[modality] for modality in EXPECTED_MODALITIES]
    stacked = np.stack(volumes, axis=-1)
    return nib.Nifti1Image(stacked.astype(np.float32), reference_affine, reference_header)


def _load_png_file(item):
    """Load a single PNG file (used for parallel loading)."""
    temp_path = None
    try:
        if hasattr(item.file, 'path') and item.file.path:
            img = Image.open(item.file.path).convert('L')
        elif hasattr(item.file, 'temporary_file_path'):
            img = Image.open(item.file.temporary_file_path()).convert('L')
        else:
            suffix = Path(item.original_name).suffix or '.png'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                temp_path = tmp.name
                if hasattr(item.file, 'seek'):
                    item.file.seek(0)
                if hasattr(item.file, 'chunks'):
                    for chunk in item.file.chunks():
                        tmp.write(chunk)
                else:
                    tmp.write(item.file.read())
            img = Image.open(temp_path).convert('L')

        return item.modality, img
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def stack_png_files(uploaded_files: List[UploadedFile]) -> Image.Image:
    """Stack PNG files with PARALLEL loading for speed."""
    ordered = sorted(uploaded_files, key=lambda f: EXPECTED_MODALITIES.index(f.modality) if f.modality in EXPECTED_MODALITIES else 999)

    channels_by_modality = {}
    width = height = None

    # **PARALLEL LOADING** - Load all files simultaneously
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_load_png_file, item): item for item in ordered}
        
        for future in as_completed(futures):
            modality, img = future.result()
            
            if width is None:
                width, height = img.size
            elif img.size != (width, height):
                raise ValueError('All PNG modality files must have identical image dimensions.')
            channels_by_modality[modality] = img

    # If only 1 file, duplicate into 4 channels
    if len(channels_by_modality) == 1:
        single_channel = next(iter(channels_by_modality.values()))
        channels_by_modality = {modality: single_channel for modality in EXPECTED_MODALITIES}

    if len(channels_by_modality) != 4:
        raise ValueError('PNG stacking requires either 1 file to duplicate or exactly 4 modality files.')

    channels = [channels_by_modality[modality] for modality in EXPECTED_MODALITIES]
    return Image.merge('RGBA', channels)


def load_individual_nifti_files(uploaded_files: List[UploadedFile]) -> List[dict]:
    """
    Load individual NIfTI files WITHOUT stacking for preview/inspection.
    
    Returns a list of dicts with:
    - modality: modality name (t1, t1ce, t2, flair)
    - data: 3D numpy array of voxels
    - affine: NIfTI affine matrix
    - header: NIfTI header
    - shape: volume shape (x, y, z)
    - original_name: original filename
    """
    # Sort by expected modality order
    ordered = sorted(
        uploaded_files,
        key=lambda f: EXPECTED_MODALITIES.index(f.modality) if f.modality in EXPECTED_MODALITIES else 999
    )

    volumes = []

    # **PARALLEL LOADING** - Load all files simultaneously
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_load_nifti_file, item): item for item in ordered}
        
        for future in as_completed(futures):
            item = futures[future]
            modality, data, affine, header = future.result()
            volumes.append({
                'modality': modality,
                'data': data,
                'affine': affine,
                'header': header,
                'shape': tuple(data.shape),
                'original_name': item.original_name,
            })

    return volumes


def load_individual_png_files(uploaded_files: List[UploadedFile]) -> List[dict]:
    """
    Load individual PNG files WITHOUT stacking for preview/inspection.
    
    Returns a list of dicts with:
    - modality: modality name (t1, t1ce, t2, flair)
    - image: PIL Image object (grayscale)
    - size: image size (width, height)
    - original_name: original filename
    """
    # Sort by expected modality order
    ordered = sorted(
        uploaded_files,
        key=lambda f: EXPECTED_MODALITIES.index(f.modality) if f.modality in EXPECTED_MODALITIES else 999
    )

    images = []

    # **PARALLEL LOADING** - Load all files simultaneously
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_load_png_file, item): item for item in ordered}
        
        for future in as_completed(futures):
            item = futures[future]
            modality, img = future.result()
            images.append({
                'modality': modality,
                'image': img,
                'size': img.size,
                'original_name': item.original_name,
            })

    return images
