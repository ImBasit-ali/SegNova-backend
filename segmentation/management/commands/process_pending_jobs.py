"""
Django management command to manually process pending segmentation jobs.

Usage:
    python manage.py process_pending_jobs              # Process all pending jobs
    python manage.py process_pending_jobs --job-id UUID  # Process specific job
    python manage.py process_pending_jobs --check        # Just show pending count

Useful when the background worker is not running (e.g., local development).
"""

from django.core.management.base import BaseCommand
from segmentation.models import SegmentationJob
from segmentation.tasks import process_job


class Command(BaseCommand):
    help = 'Manually process pending segmentation jobs (for when worker is not running)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--job-id',
            type=str,
            help='Process a specific job by ID',
        )
        parser.add_argument(
            '--check',
            action='store_true',
            help='Just check pending job count without processing',
        )

    def handle(self, *args, **options):
        job_id = options.get('job_id')
        check_only = options.get('check', False)

        if check_only:
            pending_count = SegmentationJob.objects.filter(status='pending').count()
            self.stdout.write(
                self.style.SUCCESS(
                    f'✓ Pending jobs: {pending_count}'
                )
            )
            return

        if job_id:
            try:
                job = SegmentationJob.objects.get(id=job_id)
                if job.status == 'done':
                    self.stdout.write(
                        self.style.WARNING(f'⚠ Job {job_id} is already done')
                    )
                    return
                
                self.stdout.write(
                    self.style.WARNING(f'Processing job {job_id}...')
                )
                process_job(job)
                job.refresh_from_db()
                self.stdout.write(
                    self.style.SUCCESS(
                        f'✓ Job {job_id} completed with status: {job.status}'
                    )
                )
            except SegmentationJob.DoesNotExist:
                self.stdout.write(
                    self.style.ERROR(f'✗ Job {job_id} not found')
                )
        else:
            pending_jobs = SegmentationJob.objects.filter(status='pending').order_by('created_at')
            
            if not pending_jobs.exists():
                self.stdout.write(
                    self.style.SUCCESS('✓ No pending jobs to process')
                )
                return

            count = pending_jobs.count()
            self.stdout.write(
                self.style.WARNING(f'Processing {count} pending job(s)...')
            )

            for job in pending_jobs:
                try:
                    self.stdout.write(f'  Processing job {job.id}...', ending=' ')
                    process_job(job)
                    job.refresh_from_db()
                    self.stdout.write(
                        self.style.SUCCESS(f'✓ {job.status}')
                    )
                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(f'✗ Failed: {str(e)}')
                    )

            self.stdout.write(
                self.style.SUCCESS(f'\n✓ Processed {count} job(s)')
            )
