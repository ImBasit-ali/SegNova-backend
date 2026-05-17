import importlib
import logging
import os

from django.conf import settings

logger = logging.getLogger(__name__)

MODEL = None


def _resolve_load_model():
    try:
        tf_models = importlib.import_module('tensorflow.keras.models')
        return tf_models.load_model
    except ImportError:
        raise ImportError(
            'TensorFlow/Keras is not installed. Install tensorflow to run segmentation.'
        )


def instance_normalization(x):
    try:
        tf = importlib.import_module('tensorflow')
        axes = tuple(range(1, len(x.shape) - 1))
        mean = tf.reduce_mean(x, axis=axes, keepdims=True)
        variance = tf.math.reduce_variance(x, axis=axes, keepdims=True)
        return (x - mean) / tf.sqrt(variance + 1e-5)
    except Exception:
        keras_ops = importlib.import_module('keras.ops')
        axes = tuple(range(1, len(x.shape) - 1))
        mean = keras_ops.mean(x, axis=axes, keepdims=True)
        variance = keras_ops.var(x, axis=axes, keepdims=True)
        return (x - mean) / keras_ops.sqrt(variance + 1e-5)


def get_model_path():
    configured = getattr(settings, 'MODEL_KERAS_PATH', None)
    if configured:
        return str(configured)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, 'model', 'model.keras')


def get_model():
    global MODEL

    if MODEL is not None:
        return MODEL

    model_path = get_model_path()
    logger.info('Loading Keras model from %s', model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f'Model file not found at {model_path}. '
            'Place model.keras under backend/model/.'
        )

    load_model = _resolve_load_model()
    MODEL = load_model(
        model_path,
        custom_objects={'instance_normalization': instance_normalization},
        compile=False,
    )

    logger.info('Model loaded (input=%s, output=%s)', MODEL.input_shape, MODEL.output_shape)
    return MODEL
