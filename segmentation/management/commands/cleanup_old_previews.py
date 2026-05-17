"""
Django management command to clean up old temporary preview files.

Usage:
    python manage.py cleanup_old_previews            # Default: 24 hours
    python manage.py cleanup_old_previews --hours 48  # Custom: 48 hours

Can be scheduled with cron:
    0 2 * * * cd /path/to/backend && python manage.py cleanup_old_previews
    (runs daily at 2 AM)
"""

from django.core.management.base import BaseCommand

from segmentation.cleanup import cleanup_old_preview_files


class Command(BaseCommand):
    help = 'Clean up old temporary stacked preview files from media/previews/'

    def add_arguments(self, parser):
        parser.add_argument(
            '--hours',
            type=int,
            default=24,
            help='Delete files older than this many hours (default: 24)',
        )

    def handle(self, *args, **options):
        hours = options['hours']
        self.stdout.write(
            self.style.WARNING(
                f'Cleaning up preview files older than {hours} hours...'
            )
        )

        result = cleanup_old_preview_files(hours=hours)

        self.stdout.write(
            self.style.SUCCESS(
                f'✓ Deleted {result["deleted"]} files, {result["failed"]} failures'
            )
        )
