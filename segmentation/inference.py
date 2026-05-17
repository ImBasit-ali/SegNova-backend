import logging

import nibabel as nib
import numpy as np
from scipy import ndimage

from .model_loader import get_model, get_model_path

logger = logging.getLogger(__name__)

# Model output channels: 0=BG, 1=ET, 2=NETC, 3=SNFH (edema), 4=RC (necrosis)
CLASS_BG = 0
CLASS_ET = 1
CLASS_NETC = 2
CLASS_SNFH = 3
CLASS_RC = 4

FOREGROUND_CONFIDENCE = 0.35
MIN_COMPONENT_VOXELS = 64


def run_nifti_model_inference(stacked_path):
    """
    Run model.keras on a 4-channel stacked NIfTI.

    Returns ET / WT / TC binary masks, a multi-class label map (0–4), affine, and header.
    BraTS nested regions (TC is always inside WT when edema exists):
      - ET  = enhancing tumor (class 1)
      - TC  = ET + NETC + RC (tumor core, no edema)
      - WT  = ET + NETC + SNFH + RC (whole tumor including edema)
    """
    logger.info('Starting model inference on %s (model=%s)', stacked_path, get_model_path())

    nii = nib.load(stacked_path)
    volume = np.asarray(nii.dataobj, dtype=np.float32)

    if volume.ndim == 3:
        volume = volume[..., np.newaxis]
    if volume.ndim != 4:
        raise ValueError(f'Stacked NIfTI must be 4D (x, y, z, channels); got {volume.shape}.')
    if volume.shape[-1] != 4:
        raise ValueError(
            f'Model expects 4 channels (T1, T1ce, T2, FLAIR); stacked volume has '
            f'{volume.shape[-1]} channels.'
        )

    logger.info('Stacked volume shape=%s — loading model', volume.shape)
    model = get_model()
    logger.info('Model input=%s output=%s', model.input_shape, model.output_shape)

    prepared, mapping, channels_first = _prepare_for_model(volume, model)
    logger.info('Prepared batch shape=%s channels_first=%s', prepared.shape, channels_first)

    prediction = model.predict(prepared, verbose=0)
    prediction = _unwrap_prediction(prediction)
    prediction = _to_channels_last(prediction, channels_first)
    logger.info('Raw prediction shape=%s', prediction.shape)

    labels_model = _labels_from_prediction(prediction)
    original_shape = volume.shape[:3]
    label_map = _restore_label_map(labels_model, mapping, original_shape)

    brain_mask = _build_brain_mask(volume)
    label_map = _clean_label_map(label_map, brain_mask)

    regions = _brats_regions_from_labels(label_map)

    et_n = int(np.sum(regions['et'] > 0))
    wt_n = int(np.sum(regions['wt'] > 0))
    tc_n = int(np.sum(regions['tc'] > 0))
    ed_n = int(np.sum(regions['wt_overlay'] > 0))
    ncr_net_n = int(np.sum(regions['tc_overlay'] > 0))
    logger.info(
        'Inference complete — ET=%s TC=%s WT=%s voxels (ED=%s, NCR/NET=%s)',
        et_n,
        tc_n,
        wt_n,
        ed_n,
        ncr_net_n,
    )

    if wt_n == 0:
        logger.warning('Whole-tumor mask is empty after post-processing.')

    return (
        regions['et'].astype(np.uint8),
        regions['wt'].astype(np.uint8),
        regions['tc'].astype(np.uint8),
        regions['et_overlay'].astype(np.uint8),
        regions['tc_overlay'].astype(np.uint8),
        regions['wt_overlay'].astype(np.uint8),
        label_map.astype(np.uint8),
        nii.affine,
        nii.header.copy(),
    )


def _brats_regions_from_labels(label_map):
    """
    BraTS nested regions (model: 1=ET, 2=NETC, 3=SNFH/ED, 4=RC/NCR):

      ET = ET
      TC = ET + NCR/NET
      WT = ED + ET + NCR/NET

    Exclusive overlay layers (no voxel painted twice when all are shown):
      ET layer  → ET
      TC layer  → NCR/NET (NETC + RC)
      WT layer  → ED (SNFH)
    """
    et = label_map == CLASS_ET
    netc = label_map == CLASS_NETC
    ncr_net = netc | (label_map == CLASS_RC)
    ed = label_map == CLASS_SNFH

    tc_full = et | ncr_net
    wt_full = tc_full | ed

    return {
        'et': et,
        'tc': tc_full,
        'wt': wt_full,
        'et_overlay': et,
        'tc_overlay': ncr_net,
        'wt_overlay': ed,
    }


def _labels_from_prediction(prediction):
    """Softmax argmax with background confidence gating."""
    logits = prediction.astype(np.float64)
    logits -= logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-8)
    labels = np.argmax(probs, axis=-1).astype(np.uint8)
    fg_prob = 1.0 - probs[..., CLASS_BG]
    labels[fg_prob < FOREGROUND_CONFIDENCE] = CLASS_BG
    return labels


def _clean_label_map(label_map, brain_mask):
    """Keep the largest tumor cluster and drop small noisy islands."""
    label_map = np.where(brain_mask, label_map, CLASS_BG).astype(np.uint8)
    tumor = label_map > CLASS_BG
    if not np.any(tumor):
        return label_map

    labeled, n_comp = ndimage.label(tumor)
    if n_comp == 0:
        return label_map

    sizes = ndimage.sum(tumor, labeled, index=range(1, n_comp + 1))
    main_id = int(np.argmax(sizes)) + 1
    main_mask = labeled == main_id

    cleaned = np.zeros_like(label_map, dtype=np.uint8)
    cleaned[main_mask] = label_map[main_mask]

    for cls in (CLASS_ET, CLASS_NETC, CLASS_SNFH, CLASS_RC):
        cls_mask = cleaned == cls
        if not np.any(cls_mask):
            continue
        sub_labeled, sub_n = ndimage.label(cls_mask)
        for sub_id in range(1, sub_n + 1):
            island = sub_labeled == sub_id
            if int(np.sum(island)) < MIN_COMPONENT_VOXELS:
                cleaned[island] = CLASS_BG

    return cleaned


def _build_brain_mask(volume):
    """Union of brain tissue across all modalities (tighter than a single channel)."""
    combined = np.max(volume.astype(np.float32), axis=-1)
    positive = combined[combined > 0]
    if positive.size == 0:
        return np.zeros(combined.shape, dtype=bool)
    threshold = float(np.percentile(positive, 5.0))
    return combined > threshold


def _prepare_for_model(volume, model):
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) != 5:
        raise ValueError(f'Expected a 3D model input shape of rank 5, got: {input_shape}')

    channels_first = _is_channels_first(input_shape)

    if channels_first:
        target_spatial = _resolve_target_shape(input_shape[2:5], volume.shape[:3])
        target_channels = input_shape[1] if input_shape[1] is not None else volume.shape[-1]
    else:
        target_spatial = _resolve_target_shape(input_shape[1:4], volume.shape[:3])
        target_channels = input_shape[4] if input_shape[4] is not None else volume.shape[-1]

    prepared = _zscore_per_channel(volume)
    prepared, mapping = _center_crop_or_pad(prepared, target_spatial)
    prepared = _align_channels(prepared, int(target_channels))

    if channels_first:
        prepared = np.transpose(prepared, (3, 0, 1, 2))

    prepared = np.expand_dims(prepared, axis=0)
    return prepared.astype(np.float32), mapping, channels_first


def _unwrap_prediction(prediction):
    if isinstance(prediction, list):
        if not prediction:
            raise ValueError('Model returned an empty prediction list.')
        prediction = prediction[0]

    prediction = np.asarray(prediction)
    if prediction.ndim == 5 and prediction.shape[0] == 1:
        prediction = prediction[0]

    if prediction.ndim not in (3, 4):
        raise ValueError(f'Unexpected prediction shape: {prediction.shape}')

    if prediction.ndim == 3:
        prediction = prediction[..., np.newaxis]

    return prediction


def _to_channels_last(prediction, channels_first):
    if channels_first:
        return np.transpose(prediction, (1, 2, 3, 0))
    return prediction


def _resolve_target_shape(target_shape, fallback_shape):
    return tuple(int(ts if ts is not None else fallback_shape[i]) for i, ts in enumerate(target_shape))


def _is_channels_first(input_shape):
    channels_dim = input_shape[1]
    if channels_dim in (1, 2, 3, 4):
        return True

    last_dim = input_shape[-1]
    if last_dim in (1, 2, 3, 4):
        return False

    return False


def _zscore_per_channel(volume, clip_std=4.0):
    """BraTS-style per-modality z-score inside brain (matches typical training prep)."""
    vol = volume.astype(np.float32, copy=False)
    out = np.zeros_like(vol)
    for c in range(vol.shape[-1]):
        ch = vol[..., c]
        tissue = ch[ch > 0]
        if tissue.size == 0:
            continue
        mean = float(tissue.mean())
        std = float(tissue.std())
        if std < 1e-8:
            std = 1.0
        z = (ch - mean) / std
        out[..., c] = np.clip(z, -clip_std, clip_std)
    return out


def _center_crop_or_pad(volume, target_spatial):
    src_shape = volume.shape[:3]
    channels = volume.shape[-1]
    output = np.zeros((*target_spatial, channels), dtype=volume.dtype)

    src_slices = []
    dst_slices = []

    for src, dst in zip(src_shape, target_spatial):
        if src >= dst:
            src_start = (src - dst) // 2
            src_end = src_start + dst
            dst_start = 0
            dst_end = dst
        else:
            src_start = 0
            src_end = src
            dst_start = (dst - src) // 2
            dst_end = dst_start + src

        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))

    output[
        dst_slices[0], dst_slices[1], dst_slices[2], :
    ] = volume[
        src_slices[0], src_slices[1], src_slices[2], :
    ]

    return output, {'src': tuple(src_slices), 'dst': tuple(dst_slices), 'target_shape': target_spatial}


def _align_channels(volume, target_channels):
    current_channels = volume.shape[-1]
    if target_channels == current_channels:
        return volume

    if target_channels < current_channels:
        return volume[..., :target_channels]

    pad = np.zeros((*volume.shape[:3], target_channels - current_channels), dtype=volume.dtype)
    return np.concatenate([volume, pad], axis=-1)


def _restore_to_original_shape(mask, mapping, original_shape):
    restored = np.zeros(original_shape, dtype=np.uint8)
    src = mapping['src']
    dst = mapping['dst']
    restored[src[0], src[1], src[2]] = mask[dst[0], dst[1], dst[2]].astype(np.uint8)
    return restored


def _restore_label_map(labels, mapping, original_shape):
    return _restore_to_original_shape(labels, mapping, original_shape)
