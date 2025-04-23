from logging import getLogger
import tensorflow as tf
from pathlib import Path
import os
import threading
import logging
import keras


logger = getLogger(__name__)
logger.setLevel(logging.INFO)

_model_file_name = "model.keras"
_device = "/GPU:0" if len(tf.config.list_physical_devices("GPU")) > 0 else "/CPU:0"

content_types = [
    "application/json",
    "text/csv",
    "text/plain",
    "application/jsonlines",
]

logger.info("[INFO] load-thread id: {}".format(threading.current_thread().name))
logger.info("[INFO] load-process id: {}".format(os.getpid()))
logger.info(
    "[INFO] load-SAGEMAKER_MODEL_SERVER_WORKERS: {}".format(
        os.environ.get("SAGEMAKER_MODEL_SERVER_WORKERS", "nonon")
    )
)


logger.info(
    f"[INFO] Container started, model {_model_file_name} will load on device {_device}"
)


def model_fn(model_dir, context) -> dict:
    """
    Load the model for inference
    """

    logger.info("[INFO] model_fn-thread id: {}".format(threading.current_thread().name))
    logger.info("[INFO] loading model on device: {}".format(_device))
    logger.info("[INFO] loading model from dir: {}".format(model_dir))

    model_path = Path(model_dir) / _model_file_name
    logger.info(
        "[INFO] model_path: {}, exists: {}".format(model_path, model_path.exists())
    )

    model: keras.Model = keras.models.load_model(model_path)
    # set inference mode
    model.trainable = False
    model.compile()

    logger.info("[INFO] model loaded successfully")

    return model


def input_fn(input_data: bytes, content_type):
    """
    Parse input data payload
    """
    logger.info("[INFO] input_fn-thread id: {}".format(threading.current_thread().name))
    logger.info("[INFO] input_fn-process id: {}".format(os.getpid()))
    logger.info("[INFO] input_fn-content_type: {}".format(content_type))

    # TODO
    ...


def predict_fn(image_tensor, model):
    logger.info("predict_fn: Predicting for an image...")

    # TODO

    with tf.device(_device):
        ...
    ...


def output_fn(prediction, accept) -> bytes:
    logger.info(
        "[INFO] output_fn-thread id: {}".format(threading.current_thread().name)
    )
    logger.info("[INFO] output_fn-process id: {}".format(os.getpid()))

    # TODO
    ...
