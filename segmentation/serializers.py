import os

from rest_framework import serializers
from .models import SegmentationJob, UploadedFile


class UploadedFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedFile
        fields = ['id', 'original_name', 'modality', 'uploaded_at']


class SegmentationJobStatusSerializer(serializers.ModelSerializer):
    progress = serializers.SerializerMethodField()
    stacked_url = serializers.SerializerMethodField()
    mask_url = serializers.SerializerMethodField()
    preview_url = serializers.SerializerMethodField()

    class Meta:
        model = SegmentationJob
        fields = [
            'id', 'status', 'progress',
            'stacked_url', 'mask_url', 'preview_url',
            'created_at', 'updated_at',
        ]

    def get_progress(self, obj):
        steps = ['Preprocess', 'Inference', 'Postprocess', 'Done']
        return {
            'step': obj.current_step,
            'step_name': obj.current_step_name,
            'total_steps': obj.total_steps,
            'steps': steps,
        }

    def get_stacked_url(self, obj):
        if obj.stacked_url:
            return self._resolve(obj.stacked_url)
        stacked = obj.files.filter(modality='stacked').order_by('-uploaded_at').first()
        if stacked:
            return self._build_abs(stacked.file.url)
        return None

    def get_mask_url(self, obj):
        if obj.mask_url:
            return self._resolve(obj.mask_url)
        wt = obj.files.filter(modality='wt_mask').order_by('-uploaded_at').first()
        if wt:
            return self._build_abs(wt.file.url)
        return None

    def get_preview_url(self, obj):
        if obj.preview_url:
            return self._resolve(obj.preview_url)
        return None

    def _resolve(self, url):
        if url.startswith('http://') or url.startswith('https://'):
            return url
        return self._build_abs(url)

    def _build_abs(self, path):
        request = self.context.get('request')
        if not request:
            return path
        return request.build_absolute_uri(path)


class SegmentationJobResultSerializer(serializers.ModelSerializer):
    files = UploadedFileSerializer(many=True, read_only=True)
    download_url = serializers.SerializerMethodField()
    model_input_url = serializers.SerializerMethodField()
    stacked_url = serializers.SerializerMethodField()
    mask_url = serializers.SerializerMethodField()
    preview_url = serializers.SerializerMethodField()
    overlays = serializers.SerializerMethodField()

    class Meta:
        model = SegmentationJob
        fields = [
            'id', 'status', 'grade', 'regions', 'metrics',
            'segmentation_file', 'files', 'download_url',
            'model_input_url', 'stacked_url', 'mask_url', 'preview_url',
            'overlays',
            'created_at', 'completed_at',
        ]

    def get_download_url(self, obj):
        if obj.segmentation_file:
            return self._build_abs(obj.segmentation_file.url)
        return None

    def _build_abs(self, path):
        request = self.context.get('request')
        if not request:
            return path
        return request.build_absolute_uri(path)

    def _resolve(self, url):
        if not url:
            return None
        if url.startswith('http://') or url.startswith('https://'):
            return url
        return self._build_abs(url)

    def _file_url(self, file_obj):
        if file_obj:
            return self._build_abs(file_obj.file.url)
        return None

    def get_model_input_url(self, obj):
        if obj.stacked_url:
            return self._resolve(obj.stacked_url)
        f = obj.files.filter(modality='stacked').order_by('-uploaded_at').first()
        return self._file_url(f)

    def get_stacked_url(self, obj):
        if obj.stacked_url:
            return self._resolve(obj.stacked_url)
        f = obj.files.filter(modality='stacked').order_by('-uploaded_at').first()
        return self._file_url(f)

    def get_mask_url(self, obj):
        if obj.mask_url:
            return self._resolve(obj.mask_url)
        wt = obj.files.filter(modality='wt_mask').order_by('-uploaded_at').first()
        return self._file_url(wt)

    def get_preview_url(self, obj):
        if obj.preview_url:
            return self._resolve(obj.preview_url)
        return None

    def get_overlays(self, obj):
        et = obj.files.filter(modality='et_mask').order_by('-uploaded_at').first()
        wt = obj.files.filter(modality='wt_mask').order_by('-uploaded_at').first()
        tc = obj.files.filter(modality='tc_mask').order_by('-uploaded_at').first()
        return {
            'enhancing_tumor': self._file_url(et),
            'whole_tumor': self._file_url(wt),
            'tumor_core': self._file_url(tc),
        }
