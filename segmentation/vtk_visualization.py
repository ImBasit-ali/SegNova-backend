"""
VTK brain + tumor mesh visualization for BraTS segmentation jobs.

Exports PNG preview and STL meshes, then publishes to storage (local or Supabase).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

BRAIN_OPACITY = 0.5
BRAIN_COLOR = (234 / 255, 234 / 255, 234 / 255)

MASK_OPACITY = 0.85
# ET, NETC/edema component, SNFH, RC — aligned with BraTS label map 1–4
MASK_COLORS = [
    (221 / 255, 110 / 255, 66 / 255),
    (235 / 255, 193 / 255, 43 / 255),
    (66 / 255, 177 / 255, 221 / 255),
    (180 / 255, 90 / 255, 200 / 255),
]

VTK_AVAILABLE = False
_VTK_IMPORT_ERROR = None

try:
    from vtkmodules.vtkIOImage import vtkNIFTIImageReader
    from vtkmodules.vtkIOGeometry import vtkSTLWriter
    from vtkmodules.vtkIOExport import vtkPNGWriter
    from vtkmodules.vtkFiltersCore import (
        vtkFlyingEdges3D,
        vtkDecimatePro,
        vtkSmoothPolyDataFilter,
        vtkPolyDataNormals,
    )
    from vtkmodules.vtkFiltersGeneral import vtkDiscreteMarchingCubes
    from vtkmodules.vtkRenderingCore import (
        vtkActor,
        vtkPolyDataMapper,
        vtkRenderWindow,
        vtkRenderer,
    )
    from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter
    import vtkmodules.vtkInteractionStyle  # noqa: F401
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401

    VTK_AVAILABLE = True
except ImportError as exc:
    _VTK_IMPORT_ERROR = str(exc)


class VtkBrainViz:
    """Build brain shell and label-wise tumor meshes from NIfTI volumes."""

    def __init__(self, nifti_file: str, mask_file: str):
        self.nifti_file = str(nifti_file)
        self.mask_file = str(mask_file)

    @staticmethod
    def read_nifti(file_name: str):
        reader = vtkNIFTIImageReader()
        reader.SetFileName(str(file_name))
        reader.Update()
        return reader

    @staticmethod
    def decimate(input_connection, target_reduction=0.5):
        decimate = vtkDecimatePro()
        decimate.SetInputConnection(input_connection)
        decimate.SetTargetReduction(target_reduction)
        decimate.PreserveTopologyOn()
        decimate.Update()
        return decimate

    @staticmethod
    def smooth(input_connection, number_of_iterations=120):
        smooth = vtkSmoothPolyDataFilter()
        smooth.SetInputConnection(input_connection)
        smooth.SetNumberOfIterations(number_of_iterations)
        smooth.Update()
        return smooth

    @staticmethod
    def normals(input_connection, feature_angle=60.0):
        normals = vtkPolyDataNormals()
        normals.SetInputConnection(input_connection)
        normals.SetFeatureAngle(feature_angle)
        normals.Update()
        return normals

    @staticmethod
    def _write_stl(polydata, path: Path):
        writer = vtkSTLWriter()
        writer.SetFileName(str(path))
        writer.SetInputData(polydata)
        writer.Write()

    def _surface_pipeline(self, extract_filter):
        decimate = self.decimate(extract_filter.GetOutputPort(), target_reduction=0.45)
        smooth = self.smooth(decimate.GetOutputPort(), number_of_iterations=80)
        normals = self.normals(smooth.GetOutputPort(), feature_angle=60)
        normals.Update()
        return normals.GetOutput()

    def brain_polydata(self, brain_threshold=20):
        brain_reader = self.read_nifti(self.nifti_file)
        brain_extract = vtkFlyingEdges3D()
        brain_extract.SetInputConnection(brain_reader.GetOutputPort())
        brain_extract.SetValue(0, brain_threshold)
        brain_extract.Update()
        return self._surface_pipeline(brain_extract)

    def tumor_polydata_list(self):
        tumor_reader = self.read_nifti(self.mask_file)
        max_label = int(tumor_reader.GetOutput().GetScalarRange()[1])
        meshes = []
        for label in range(1, max_label + 1):
            tumor_extract = vtkDiscreteMarchingCubes()
            tumor_extract.SetInputConnection(tumor_reader.GetOutputPort())
            tumor_extract.SetValue(0, label)
            tumor_extract.Update()
            if tumor_extract.GetOutput().GetNumberOfCells() == 0:
                continue
            poly = self._surface_pipeline(tumor_extract)
            color = MASK_COLORS[(label - 1) % len(MASK_COLORS)]
            meshes.append({'label': label, 'polydata': poly, 'color': color})
        return meshes

    def build_scene(self, brain_threshold=20):
        renderer = vtkRenderer()
        renderer.SetBackground(0.02, 0.05, 0.09)

        brain_poly = self.brain_polydata(brain_threshold=brain_threshold)
        brain_mapper = vtkPolyDataMapper()
        brain_mapper.SetInputData(brain_poly)
        brain_mapper.ScalarVisibilityOff()
        brain_actor = vtkActor()
        brain_actor.SetMapper(brain_mapper)
        brain_actor.GetProperty().SetColor(*BRAIN_COLOR)
        brain_actor.GetProperty().SetOpacity(BRAIN_OPACITY)
        renderer.AddActor(brain_actor)

        tumor_meshes = []
        for item in self.tumor_polydata_list():
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(item['polydata'])
            mapper.ScalarVisibilityOff()
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*item['color'])
            actor.GetProperty().SetOpacity(MASK_OPACITY)
            renderer.AddActor(actor)
            tumor_meshes.append(item)

        renderer.ResetCamera()
        return renderer, brain_poly, tumor_meshes


def _resolve_brain_and_mask_paths(session_id: str, job_id: str) -> tuple[Path, Path]:
    from . import upload_workflow

    results_dir = upload_workflow.local_results_dir(session_id)
    seg_path = results_dir / f'{job_id}_seg_labels.nii.gz'
    if not seg_path.is_file():
        raise FileNotFoundError(f'Segmentation label map not found: {seg_path}')

    brain_path = None
    for entry in upload_workflow.list_session_upload_entries(session_id):
        if entry.get('modality') == 't1ce':
            brain_path = Path(entry['path'])
            break

    if brain_path is None or not brain_path.is_file():
        previews_dir = upload_workflow.local_previews_dir(session_id)
        viewer_t2 = previews_dir / f'{job_id}_viewer_t2.nii.gz'
        if viewer_t2.is_file():
            brain_path = viewer_t2
        else:
            stacked = previews_dir / f'{session_id}_stacked.nii.gz'
            if not stacked.is_file():
                stacked = previews_dir / f'{job_id}_stacked.nii.gz'
            if stacked.is_file():
                brain_path = stacked

    if brain_path is None or not brain_path.is_file():
        raise FileNotFoundError(
            'Brain volume not found (expected t1ce upload or viewer T2 base).'
        )

    return brain_path, seg_path


def generate_3d_visualization_assets(
    session_id: str,
    job_id: str,
    *,
    brain_threshold: int = 20,
) -> dict[str, Any]:
    """
    Run VTK pipeline, write PNG + STL files under previews work dir, publish to storage.
    Returns public URLs for preview and meshes.
    """
    if not VTK_AVAILABLE:
        raise RuntimeError(
            f'VTK is not installed ({_VTK_IMPORT_ERROR}). '
            'Install with: pip install vtk'
        )

    from . import upload_workflow

    brain_path, mask_path = _resolve_brain_and_mask_paths(session_id, job_id)
    previews_dir = upload_workflow.local_previews_dir(session_id)
    previews_dir.mkdir(parents=True, exist_ok=True)

    viz = VtkBrainViz(str(brain_path), str(mask_path))
    renderer, brain_poly, tumor_meshes = viz.build_scene(brain_threshold=brain_threshold)

    png_name = f'{job_id}_visualization.png'
    png_path = previews_dir / png_name

    render_window = vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.AddRenderer(renderer)
    render_window.SetSize(1280, 960)
    render_window.SetWindowName('Brain')
    render_window.Render()

    window_filter = vtkWindowToImageFilter()
    window_filter.SetInput(render_window)
    window_filter.SetScale(1)
    window_filter.SetInputBufferTypeToRGB()
    window_filter.ReadFrontBufferOff()
    window_filter.Update()

    png_writer = vtkPNGWriter()
    png_writer.SetFileName(str(png_path))
    png_writer.SetInputConnection(window_filter.GetOutputPort())
    png_writer.Write()

    meshes: dict[str, str] = {}
    brain_stl = previews_dir / f'{job_id}_brain_mesh.stl'
    viz._write_stl(brain_poly, brain_stl)
    meshes['brain'] = str(brain_stl)

    label_names = {1: 'et', 2: 'netc', 3: 'edema', 4: 'rc'}
    for item in tumor_meshes:
        label = item['label']
        suffix = label_names.get(label, f'label{label}')
        tumor_stl = previews_dir / f'{job_id}_tumor_{suffix}.stl'
        viz._write_stl(item['polydata'], tumor_stl)
        meshes[suffix] = str(tumor_stl)

    preview_url = upload_workflow.publish_artifact(session_id, 'previews', png_name, png_path)
    mesh_urls = {}
    for key, local_path in meshes.items():
        fname = Path(local_path).name
        mesh_urls[key] = upload_workflow.publish_artifact(
            session_id, 'previews', fname, Path(local_path),
        )

    return {
        'preview_url': preview_url,
        'preview_png': preview_url,
        'meshes': mesh_urls,
        'status': 'ready',
    }


def get_existing_visualization_urls(session_id: str, job_id: str) -> dict[str, Any] | None:
    """Return URLs if visualization artifacts were already generated."""
    from django.conf import settings
    from . import upload_workflow
    from .storage import get_storage, session_preview_key

    names = [f'{job_id}_visualization.png', f'{job_id}_brain_mesh.stl']
    label_suffixes = ('et', 'netc', 'edema', 'rc')
    for suffix in label_suffixes:
        names.append(f'{job_id}_tumor_{suffix}.stl')

    if upload_workflow.uses_remote_storage():
        storage = get_storage()
        preview_key = session_preview_key(session_id, names[0])
        remote_keys = set(storage.list_keys(f'previews/{session_id}/'))
        if preview_key not in remote_keys:
            return None
        preview_url = storage.get_public_url(preview_key)
        meshes = {}
        for suffix in label_suffixes:
            fname = f'{job_id}_tumor_{suffix}.stl'
            key = session_preview_key(session_id, fname)
            try:
                meshes[suffix] = storage.get_public_url(key)
            except Exception:
                pass
        brain_key = session_preview_key(session_id, f'{job_id}_brain_mesh.stl')
        try:
            meshes['brain'] = storage.get_public_url(brain_key)
        except Exception:
            pass
        return {
            'preview_url': preview_url,
            'preview_png': preview_url,
            'meshes': meshes,
            'status': 'ready',
        }

    previews_dir = Path(settings.MEDIA_ROOT) / 'previews'
    png_path = previews_dir / names[0]
    if not png_path.is_file():
        session_png = previews_dir / session_id / names[0]
        if session_png.is_file():
            png_path = session_png
        else:
            return None

    base = upload_workflow.public_url_for_local(session_id, 'previews', names[0], png_path)
    meshes = {}
    brain_stl = previews_dir / session_id / f'{job_id}_brain_mesh.stl'
    if not brain_stl.is_file():
        brain_stl = previews_dir / f'{job_id}_brain_mesh.stl'
    if brain_stl.is_file():
        meshes['brain'] = upload_workflow.public_url_for_local(
            session_id, 'previews', brain_stl.name, brain_stl,
        )
    for suffix in label_suffixes:
        fname = f'{job_id}_tumor_{suffix}.stl'
        for candidate in (previews_dir / session_id / fname, previews_dir / fname):
            if candidate.is_file():
                meshes[suffix] = upload_workflow.public_url_for_local(
                    session_id, 'previews', fname, candidate,
                )
                break

    return {
        'preview_url': base,
        'preview_png': base,
        'meshes': meshes,
        'status': 'ready',
    }
