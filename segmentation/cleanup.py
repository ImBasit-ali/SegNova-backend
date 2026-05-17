"""
File lifecycle management.

Keeps only the latest completed job per user. Deletes older jobs
(and their storage files) after a new job finishes successfully.
Also cleans up temporary preview files that accumulate in media/previews/.
"""

import logging
import os
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)


def cleanup_old_jobs(current_job):
    """
    Delete all older jobs for the same user after the current job completes.

    Safety rules:
    - Never deletes the current job
    - Never deletes a job that is still processing
    - Only runs when the current job is 'done'
    """
    from .models import SegmentationJob
    from .storage import get_storage, storage_key_for_job

    if current_job.status != 'done':
        return

    user_id = current_job.user_id
    if not user_id:
        # No user scoping — skip cleanup
        return

    # Find all other jobs for this user
    other_jobs = SegmentationJob.objects.filter(
        user_id=user_id,
    ).exclude(
        id=current_job.id,
    )

    storage = get_storage()

    for job in other_jobs:
        if job.status == 'processing':
            logger.info(
                'Skipping cleanup of job %s — still processing', job.id
            )
            continue

        # Delete storage files
        job_prefix = f'user_{user_id}/job_{job.id}'
        try:
            storage.delete_prefix(f'{job_prefix}/uploads')
            storage.delete_prefix(f'{job_prefix}/stacked')
            storage.delete_prefix(f'{job_prefix}/results')
            storage.delete_prefix(f'{job_prefix}/preview')
        except Exception as exc:
            logger.warning(
                'Failed to delete storage files for job %s: %s', job.id, exc
            )

        # Delete database records (cascades to UploadedFile)
        logger.info('Deleting old job %s for user %s', job.id, user_id)
        job.delete()


def cleanup_old_preview_files(hours=24):
    """
    Delete temporary stacked preview files older than the specified hours.

    Removes .nii.gz and .png files from media/previews/ that haven't been
    accessed in the given time period. This prevents disk accumulation from
    temporary preview files created during stacking operations.

    Args:
        hours (int): Age threshold in hours. Files older than this are deleted.
                    Default is 24 hours.

    Returns:
        dict: Summary of cleanup operation with 'deleted' count and 'failed' count.
    """
    preview_dir = Path(settings.MEDIA_ROOT) / 'previews'

    if not preview_dir.exists():
        logger.info('Preview directory does not exist: %s', preview_dir)
        return {'deleted': 0, 'failed': 0}

    cutoff_time = timezone.now() - timedelta(hours=hours)
    cutoff_timestamp = cutoff_time.timestamp()

    deleted_count = 0
    failed_count = 0

    try:
        for file_path in preview_dir.glob('*'):
            if not file_path.is_file():
                continue

            # Check file modification time
            file_mtime = file_path.stat().st_mtime
            if file_mtime < cutoff_timestamp:
                try:
                    file_path.unlink()
                    logger.debug('Deleted preview file: %s', file_path.name)
                    deleted_count += 1
                except Exception as exc:
                    logger.warning(
                        'Failed to delete preview file %s: %s', file_path.name, exc
                    )
                    failed_count += 1

        logger.info(
            'Preview cleanup complete: deleted %d files, %d failures',
            deleted_count,
            failed_count,
        )
    except Exception as exc:
        logger.error('Preview cleanup failed: %s', exc)
        return {'deleted': 0, 'failed': 1}

    return {'deleted': deleted_count, 'failed': failed_count}
