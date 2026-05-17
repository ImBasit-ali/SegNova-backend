import uuid
from django.db import models


class SegmentationJob(models.Model):
    """Represents a brain tumor segmentation job."""

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('done', 'Done'),
        ('failed', 'Failed'),
        ('error', 'Error'),  # kept for backward compatibility
    ]

    GRADE_CHOICES = [
        ('LGG', 'Low-Grade Glioma'),
        ('HGG', 'High-Grade Glioma'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    grade = models.CharField(max_length=5, choices=GRADE_CHOICES, default='HGG')
    regions = models.JSONField(default=dict, blank=True)
    opacity = models.IntegerField(default=70)

    # User scoping (anonymous session-based for now)
    user_id = models.CharField(max_length=64, blank=True, default='')

    # Processing progress
    current_step = models.IntegerField(default=0)
    current_step_name = models.CharField(max_length=50, default='Preprocessing')
    total_steps = models.IntegerField(default=4)

    # File paths / URLs (populated by worker after processing)
    input_files_json = models.JSONField(default=list, blank=True,
                                         help_text='List of input file paths or storage keys')
    stacked_url = models.URLField(max_length=1024, blank=True, default='',
                                   help_text='URL to the stacked NIfTI volume')
    mask_url = models.URLField(max_length=1024, blank=True, default='',
                                help_text='URL to the combined segmentation mask')
    preview_url = models.URLField(max_length=1024, blank=True, default='',
                                   help_text='URL to the preview image')

    # Results
    metrics = models.JSONField(null=True, blank=True)
    segmentation_file = models.FileField(upload_to='results/', null=True, blank=True)
    error_message = models.TextField(blank=True, default='')

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Job {self.id} - {self.status}"


class UploadedFile(models.Model):
    """Represents an uploaded NIfTI file associated with a segmentation job."""

    MODALITY_CHOICES = [
        ('t1', 'T1-weighted'),
        ('t1ce', 'T1-CE'),
        ('t2', 'T2-weighted'),
        ('flair', 'FLAIR'),
        ('stacked', 'Stacked Input'),
        ('et_mask', 'Enhancing Tumor Mask'),
        ('wt_mask', 'Whole Tumor Mask'),
        ('tc_mask', 'Tumor Core Mask'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.ForeignKey(SegmentationJob, on_delete=models.CASCADE, related_name='files')
    file = models.FileField(upload_to='uploads/')
    original_name = models.CharField(max_length=255)
    modality = models.CharField(max_length=10, choices=MODALITY_CHOICES, default='t1')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.original_name} ({self.modality})"
