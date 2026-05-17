"""
Session upload workflow — local media or Supabase (brain-mri bucket).

Stacking and segmentation still run on local temp paths; only persistence
and URLs returned to the frontend use remote storage when enabled.
"""

import shutil
from pathlib import Path
from types import SimpleNamespace

from django.conf import settings

from .stacking import EXPECTED_MODALITIES, infer_extension
from .storage import (
    get_storage,
    session_preview_key,
    session_result_key,
    session_upload_key,
)


def uses_remote_storage():
    return bool(settings.USE_SUPABASE_STORAGE)


def _disk_id_prefixes(disk_id):
    normalized = str(disk_id)
    prefixes = [normalized]
    compact = normalized.replace('-', '')
    if compact and compact not in prefixes:
        prefixes.append(compact)
    return prefixes


def clear_session_artifacts(session_id):
    """Delete previous uploads, previews, and results for a session."""
    storage = get_storage()
    session_id = str(session_id)

    if uses_remote_storage():
        for category in ('uploads', 'previews', 'results'):
            storage.delete_prefix(f'{category}/{session_id}')
        return

    media_root = Path(settings.MEDIA_ROOT)
    for category in ('uploads', 'previews', 'results'):
        folder = media_root / category
        if not folder.exists():
            continue
        for prefix in _disk_id_prefixes(session_id):
            for path in folder.glob(f'{prefix}_*'):
                if path.is_file():
                    path.unlink(missing_ok=True)
            session_dir = folder / session_id
            if session_dir.exists():
                shutil.rmtree(session_dir, ignore_errors=True)


def get_session_work_dir(session_id):
    """Scratch directory for model inference (always local)."""
    work_root = Path(settings.TEMP_MEDIA_ROOT)
    work_dir = work_root / str(session_id)
    for sub in ('uploads', 'previews', 'results'):
        (work_dir / sub).mkdir(parents=True, exist_ok=True)
    return work_dir


def local_upload_path(session_id, modality, extension):
    """Path under work dir or media/uploads for a modality file."""
    filename = f'{session_id}_{modality}{extension}'
    if uses_remote_storage():
        return get_session_work_dir(session_id) / 'uploads' / filename
    return Path(settings.MEDIA_ROOT) / 'uploads' / filename


def local_previews_dir(session_id):
    if uses_remote_storage():
        return get_session_work_dir(session_id) / 'previews'
    previews = Path(settings.MEDIA_ROOT) / 'previews'
    previews.mkdir(parents=True, exist_ok=True)
    return previews


def local_results_dir(session_id):
    if uses_remote_storage():
        return get_session_work_dir(session_id) / 'results'
    results = Path(settings.MEDIA_ROOT) / 'results'
    results.mkdir(parents=True, exist_ok=True)
    return results


def persist_upload_stream(session_id, modality, extension, file_obj):
    """Write upload to local scratch, then push to storage when remote."""
    dest_path = local_upload_path(session_id, modality, extension)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    chunk = 1024 * 1024

    if hasattr(file_obj, 'seek'):
        file_obj.seek(0)
    with open(dest_path, 'wb') as fh:
        if hasattr(file_obj, 'temporary_file_path'):
            with open(file_obj.temporary_file_path(), 'rb') as tmp:
                shutil.copyfileobj(tmp, fh, length=chunk)
        elif hasattr(file_obj, 'chunks'):
            for block in file_obj.chunks(chunk):
                fh.write(block)
        elif hasattr(file_obj, 'read'):
            while True:
                block = file_obj.read(chunk)
                if not block:
                    break
                fh.write(block)
        else:
            with open(file_obj.path, 'rb') as disk:
                shutil.copyfileobj(disk, fh, length=chunk)

    if not dest_path.is_file() or dest_path.stat().st_size == 0:
        raise OSError(f'Failed to write upload for {modality}')

    if uses_remote_storage():
        key = session_upload_key(session_id, modality, extension)
        url = get_storage().upload(str(dest_path), key)
    else:
        url = get_storage().get_public_url(f'uploads/{dest_path.name}')

    return {
        'modality': modality,
        'url': url,
        'original_name': dest_path.name,
        'path': str(dest_path),
        'storage_key': session_upload_key(session_id, modality, extension)
        if uses_remote_storage()
        else f'uploads/{dest_path.name}',
    }


def public_url_for_local(session_id, category, filename, local_path=None):
    if uses_remote_storage():
        if category == 'uploads':
            ext = infer_extension(filename) or ''
            modality = filename.replace(f'{session_id}_', '').replace(ext, '')
            key = session_upload_key(session_id, modality, ext)
        elif category == 'previews':
            key = session_preview_key(session_id, filename)
        else:
            key = session_result_key(session_id, filename)
        if local_path and Path(local_path).is_file():
            return get_storage().upload(str(local_path), key)
        return get_storage().get_public_url(key)

    return get_storage().get_public_url(f'{category}/{filename}')


def publish_artifact(session_id, category, filename, local_path):
    """Upload a generated file and return its public URL."""
    if uses_remote_storage():
        if category == 'previews':
            key = session_preview_key(session_id, filename)
        elif category == 'results':
            key = session_result_key(session_id, filename)
        else:
            key = f'{category}/{session_id}/{filename}'
        return get_storage().upload(str(local_path), key)

    dest = Path(settings.MEDIA_ROOT) / category / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = Path(local_path).resolve()
    dst = dest.resolve()
    if src != dst:
        shutil.copy2(str(src), str(dst))
    return get_storage().get_public_url(f'{category}/{filename}')


def list_session_upload_entries(session_id):
    """Return viewer entries for all modalities saved for this session."""
    session_id = str(session_id)
    entries_by_modality = {}

    if uses_remote_storage():
        storage = get_storage()
        prefix = f'uploads/{session_id}/'
        keys = storage.list_keys(prefix)
        work_uploads = get_session_work_dir(session_id) / 'uploads'
        work_uploads.mkdir(parents=True, exist_ok=True)
        for key in keys:
            name = Path(key).name
            modality = _modality_from_session_filename(session_id, name)
            if not modality:
                continue
            local_path = work_uploads / name
            if not local_path.exists():
                storage.download(key, str(local_path))
            entries_by_modality[modality] = {
                'modality': modality,
                'url': storage.get_public_url(key),
                'original_name': name,
                'path': str(local_path),
            }
    else:
        uploads_dir = Path(settings.MEDIA_ROOT) / 'uploads'
        if not uploads_dir.exists():
            return []
        paths = []
        for prefix in _disk_id_prefixes(session_id):
            paths = sorted(uploads_dir.glob(f'{prefix}_*'))
            if paths:
                break
        for path in paths:
            modality = _modality_from_session_filename(session_id, path.name)
            if not modality:
                continue
            relative = f"{settings.MEDIA_URL.rstrip('/')}/uploads/{path.name}"
            entries_by_modality[modality] = {
                'modality': modality,
                'url': get_storage().get_public_url(f'uploads/{path.name}'),
                'original_name': path.name,
                'path': str(path),
            }

    def sort_key(item):
        try:
            return EXPECTED_MODALITIES.index(item['modality'])
        except ValueError:
            return 99

    return sorted(entries_by_modality.values(), key=sort_key)


def _modality_from_session_filename(session_id, filename):
    for prefix in _disk_id_prefixes(session_id):
        head = f'{prefix}_'
        if not filename.startswith(head):
            continue
        rest = filename[len(head):]
        ext = infer_extension(filename)
        if ext and rest.endswith(ext):
            return rest[:-len(ext)]
        return rest.rsplit('.', 1)[0] if '.' in rest else rest
    return None


def load_wrappers_from_session(session_id):
    """Load session uploads as file wrappers (local paths for stacking/inference)."""
    from .inference_pipeline import ensure_modality_wrappers

    entries = list_session_upload_entries(session_id)
    if not entries:
        raise ValueError(f'No uploaded files found for session {session_id}.')

    file_wrappers = []
    extension = None
    for entry in entries:
        path = Path(entry['path'])
        ext = infer_extension(path.name)
        if extension is None:
            extension = ext
        elif ext != extension:
            raise ValueError(
                'All uploaded files must use the same format (.nii/.nii.gz or .png).'
            )
        file_wrappers.append(SimpleNamespace(
            file=SimpleNamespace(path=str(path)),
            original_name=entry['original_name'],
            modality=entry['modality'],
        ))

    ensure_modality_wrappers(file_wrappers)
    if extension is None:
        raise ValueError('Could not determine file extension for session uploads.')
    return file_wrappers, extension


def sync_stacked_preview_png(session_id, png_path):
    """Upload stacked_preview.png to previews/{session_id}/."""
    return publish_artifact(session_id, 'previews', 'stacked_preview.png', png_path)
