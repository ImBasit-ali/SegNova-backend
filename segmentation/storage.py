"""
Storage backends for BraTS uploads and results.

- LocalStorage: media/ on disk (development)
- SupabaseStorage: brain-mri bucket (production on Render)
"""

import logging
import shutil
from pathlib import Path

from django.conf import settings

logger = logging.getLogger(__name__)


def session_upload_key(session_id, modality, extension):
    """Object key for one modality upload."""
    filename = f'{session_id}_{modality}{extension}'
    return f'uploads/{session_id}/{filename}'


def session_preview_key(session_id, filename):
    return f'previews/{session_id}/{filename}'


def session_result_key(session_id, filename):
    return f'results/{session_id}/{filename}'


def storage_key_for_job(user_id, job_id, category, filename):
    """Legacy key layout used by background tasks."""
    safe_user_id = str(user_id or 'anonymous')
    return f'user_{safe_user_id}/job_{job_id}/{category}/{filename}'


class LocalStorage:
    """Stores files under MEDIA_ROOT for local development."""

    def __init__(self):
        self.media_root = Path(settings.MEDIA_ROOT)

    def upload(self, local_path, remote_key):
        dest = self.media_root / remote_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(local_path), str(dest))
        return self.get_public_url(remote_key)

    def upload_content(self, content_bytes, remote_key, content_type='application/octet-stream'):
        dest = self.media_root / remote_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content_bytes)
        return self.get_public_url(remote_key)

    def download(self, remote_key, local_path):
        src = self.media_root / remote_key
        if not src.exists():
            raise FileNotFoundError(f'Local file not found: {src}')
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dest))

    def delete(self, remote_key):
        target = self.media_root / remote_key
        if target.exists():
            target.unlink()

    def delete_prefix(self, prefix):
        target_dir = self.media_root / prefix
        if target_dir.exists() and target_dir.is_dir():
            shutil.rmtree(str(target_dir), ignore_errors=True)
        else:
            parent = target_dir.parent
            pattern = f'{target_dir.name}*'
            if parent.exists():
                for path in parent.glob(pattern):
                    if path.is_file():
                        path.unlink(missing_ok=True)

    def list_keys(self, prefix):
        folder = self.media_root / prefix.rstrip('/')
        if not folder.exists():
            return []
        if folder.is_file():
            return [prefix]
        keys = []
        for path in folder.rglob('*'):
            if path.is_file():
                keys.append(str(path.relative_to(self.media_root)).replace('\\', '/'))
        return keys

    def get_public_url(self, remote_key):
        base = (settings.PUBLIC_MEDIA_BASE_URL or '').rstrip('/')
        path = f'/media/{remote_key.lstrip("/")}'
        return f'{base}{path}' if base else path


class SupabaseStorage:
    """Supabase Storage (brain-mri bucket) for production."""

    def __init__(self):
        from supabase import create_client

        url = settings.SUPABASE_URL
        key = settings.SUPABASE_SERVICE_ROLE_KEY
        if not url or not key:
            raise RuntimeError(
                'SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required when '
                'USE_SUPABASE_STORAGE=true'
            )
        self.client = create_client(url, key)
        self.bucket = settings.SUPABASE_BUCKET

    def _bucket_api(self):
        return self.client.storage.from_(self.bucket)

    def upload(self, local_path, remote_key):
        remote_key = remote_key.lstrip('/')
        content_type = 'application/octet-stream'
        lower = remote_key.lower()
        if lower.endswith('.png'):
            content_type = 'image/png'
        elif lower.endswith('.nii.gz'):
            content_type = 'application/gzip'
        elif lower.endswith('.nii'):
            content_type = 'application/octet-stream'
        elif lower.endswith('.stl'):
            content_type = 'model/stl'
        elif lower.endswith('.ply'):
            content_type = 'application/octet-stream'
        elif lower.endswith('.obj'):
            content_type = 'model/obj'

        with open(local_path, 'rb') as handle:
            self._bucket_api().upload(
                remote_key,
                handle.read(),
                file_options={'content-type': content_type, 'upsert': 'true'},
            )
        return self.get_public_url(remote_key)

    def upload_content(self, content_bytes, remote_key, content_type='application/octet-stream'):
        remote_key = remote_key.lstrip('/')
        self._bucket_api().upload(
            remote_key,
            content_bytes,
            file_options={'content-type': content_type, 'upsert': 'true'},
        )
        return self.get_public_url(remote_key)

    def download(self, remote_key, local_path):
        remote_key = remote_key.lstrip('/')
        data = self._bucket_api().download(remote_key)
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    def delete(self, remote_key):
        remote_key = remote_key.lstrip('/')
        try:
            self._bucket_api().remove([remote_key])
        except Exception as exc:
            logger.warning('Supabase delete failed for %s: %s', remote_key, exc)

    def delete_prefix(self, prefix):
        prefix = prefix.rstrip('/')
        keys = self.list_keys(prefix)
        if not keys:
            return
        try:
            self._bucket_api().remove(keys)
        except Exception as exc:
            logger.warning('Supabase delete_prefix failed for %s: %s', prefix, exc)

    def list_keys(self, prefix):
        prefix = prefix.rstrip('/')
        folder = prefix
        try:
            items = self._bucket_api().list(folder) or []
        except Exception:
            return []

        keys = []
        for item in items:
            name = item.get('name')
            if not name:
                continue
            key = f'{folder}/{name}'
            if item.get('id') is None and not name.endswith(('.nii', '.gz', '.png', '.nii.gz')):
                keys.extend(self.list_keys(key))
            else:
                keys.append(key)
        return keys

    def get_public_url(self, remote_key):
        remote_key = remote_key.lstrip('/')
        public_base = (settings.SUPABASE_PUBLIC_URL or '').rstrip('/')
        if public_base:
            return f'{public_base}/{remote_key}'
        return self._bucket_api().get_public_url(remote_key)


_storage_instance = None


def get_storage():
    global _storage_instance
    if _storage_instance is None:
        if settings.USE_SUPABASE_STORAGE:
            _storage_instance = SupabaseStorage()
        else:
            _storage_instance = LocalStorage()
    return _storage_instance


def reset_storage():
    """Reset singleton (tests / settings reload)."""
    global _storage_instance
    _storage_instance = None
