from django.contrib import admin
from .models import SegmentationJob, UploadedFile


class UploadedFileInline(admin.TabularInline):
    model = UploadedFile
    extra = 0
    readonly_fields = ['id', 'original_name', 'modality', 'uploaded_at']


@admin.register(SegmentationJob)
class SegmentationJobAdmin(admin.ModelAdmin):
    list_display = ['id', 'status', 'grade', 'current_step_name', 'created_at', 'completed_at']
    list_filter = ['status', 'grade']
    search_fields = ['id']
    readonly_fields = ['id', 'created_at', 'updated_at', 'completed_at']
    inlines = [UploadedFileInline]


@admin.register(UploadedFile)
class UploadedFileAdmin(admin.ModelAdmin):
    list_display = ['id', 'job', 'original_name', 'modality', 'uploaded_at']
    list_filter = ['modality']
