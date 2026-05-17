#!/usr/bin/env python
"""
Background worker for processing segmentation jobs.
"""

import os
import sys
import time
import logging
import signal

# 🔥 Logging config (ADDED)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)

# Ensure Django settings are loaded
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

# Add the backend directory to the Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import django
django.setup()

from segmentation.models import SegmentationJob
from segmentation.tasks import process_job
from segmentation.model_loader import get_model

logger = logging.getLogger('segmentation.worker')

# Configuration
POLL_INTERVAL = int(os.environ.get('WORKER_POLL_INTERVAL', '3'))
MAX_RETRIES = int(os.environ.get('WORKER_MAX_RETRIES', '2'))

# Graceful shutdown
_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    logger.info('Received signal %s — shutting down gracefully...', signum)
    _shutdown = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# 🔧 UPDATED preload_model (with debug)
def preload_model():
    """Load the ML model once at worker startup (singleton pattern)."""
    try:
        logger.info('🔄 Pre-loading segmentation model...')

        # 🔍 Check model directory
        model_dir = os.path.join(os.getcwd(), "model")
        logger.info('📂 Checking model directory: %s', model_dir)

        if os.path.exists(model_dir):
            logger.info('📁 Model directory exists')
            logger.info('📄 Files inside model dir: %s', os.listdir(model_dir))
        else:
            logger.error('❌ Model directory NOT found!')

        model = get_model()

        logger.info('✅ Model loaded successfully: %s', type(model).__name__)

    except Exception as exc:
        logger.exception('❌ Model pre-load failed: %s', exc)


def pick_next_job():
    from django.db import transaction

    with transaction.atomic():
        job = (
            SegmentationJob.objects
            .select_for_update(skip_locked=True)
            .filter(status='pending')
            .order_by('created_at')
            .first()
        )
        if job:
            job.status = 'processing'
            job.save(update_fields=['status', 'updated_at'])
        return job


def run_worker():
    logger.info('=== BraTS Segmentation Worker Started ===')
    logger.info('Poll interval: %ds | Max retries: %d', POLL_INTERVAL, MAX_RETRIES)

    # 🔥 Preload model
    preload_model()

    while not _shutdown:
        try:
            logger.info('🔍 Checking for pending jobs...')

            job = pick_next_job()

            if job is None:
                time.sleep(POLL_INTERVAL)
                continue

            logger.info('📦 Picked job %s (created: %s)', job.id, job.created_at)

            retries = 0
            while retries <= MAX_RETRIES:
                try:
                    # 🔥 Ensure model is ready
                    try:
                        model = get_model()
                        logger.info('🧠 Model ready for job %s', job.id)
                    except Exception as e:
                        logger.exception('❌ Model not available for job %s: %s', job.id, e)
                        raise e

                    logger.info('🚀 Starting processing for job %s', job.id)

                    # 🔥 Run actual task
                    process_job(job)

                    logger.info('✅ Job %s completed successfully', job.id)
                    break

                except Exception as exc:
                    retries += 1

                    logger.exception(
                        '❌ ERROR in job %s (attempt %d/%d): %s',
                        job.id, retries, MAX_RETRIES + 1, exc
                    )

                    if retries > MAX_RETRIES:
                        logger.error(
                            '💥 Job %s failed after %d retries',
                            job.id, MAX_RETRIES
                        )
                        break
                    else:
                        logger.warning(
                            '🔁 Retrying job %s...',
                            job.id
                        )
                        job.status = 'processing'
                        job.error_message = ''
                        job.save(update_fields=['status', 'error_message', 'updated_at'])
                        time.sleep(1)

        except KeyboardInterrupt:
            break
        except Exception as exc:
            logger.exception('💥 Worker loop error: %s', exc)
            time.sleep(POLL_INTERVAL)

    logger.info('=== Worker shut down ===')


if __name__ == '__main__':
    run_worker()