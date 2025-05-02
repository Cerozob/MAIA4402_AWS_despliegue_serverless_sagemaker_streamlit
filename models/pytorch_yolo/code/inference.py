import io
from logging import getLogger
import numpy as np
import torch
import torchvision
from torchvision.io import decode_image, encode_jpeg, encode_png
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms import v2
import torchvision
from pathlib import Path
import torch
import os
import threading
import logging
import json

# Constants
COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 0),  # Maroon
    (0, 128, 0),  # Green (dark)
    (0, 0, 128),  # Navy
    (128, 128, 0),  # Olive
]


logger = getLogger(__name__)
logger.setLevel(logging.INFO)


_model_file_name = "model.pt"

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_img_content_types = [
    "image/jpeg",
    "image/png",
    "application/x-image",
]

_python_content_types = [
    "application/x-npy",
]

logger.info("[INFO] load-thread id: {}".format(threading.currentThread().getName()))
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

    logger.info(
        "[INFO] model_fn-thread id: {}".format(threading.currentThread().getName())
    )
    logger.info("[INFO] loading model on device: {}".format(_device))
    logger.info("[INFO] loading model from dir: {}".format(model_dir))

    model_path = Path(model_dir) / _model_file_name
    logger.info(
        "[INFO] model_path: {}, exists: {}".format(model_path, model_path.exists())
    )

    model = torch.jit.load(model_path, map_location=_device)
    model.eval()
    logger.info("[INFO] model loaded successfully")

    return model


def input_fn(input_data: bytes, content_type):
    """
    Parse input data payload
    """
    logger.info(
        "[INFO] input_fn-thread id: {}".format(threading.currentThread().getName())
    )
    logger.info("[INFO] input_fn-process id: {}".format(os.getpid()))
    logger.info("[INFO] input_fn-content_type: {}".format(content_type))

    image_transforms = v2.Compose(
        [v2.ToDtype(torch.float, scale=True), v2.ToPureTensor()]
    )
    array = None
    if content_type in _img_content_types:
        array = decode_image(
            torch.frombuffer(input_data, dtype=torch.uint8),
            mode=torchvision.io.ImageReadMode.RGB,
        )
    elif content_type in _python_content_types:
        array = torch.load(input_data, map_location=_device)
    else:
        raise Exception("Unsupported content type: {}".format(content_type))

    # remove alpha channel if present
    return image_transforms(array)[:3, ...].to(_device)


def predict_fn(image_tensor, model):
    logger.info("predict_fn: Predicting for an image...")
    with torch.no_grad():
        predictions = model([image_tensor])
        pred = predictions[0] if len(predictions[0]) > 0 else predictions[1][0]
        preds = {
            "labels": pred["labels"],
            "scores": pred["scores"],
            "boxes": pred["boxes"],
            "masks": pred["masks"],
        }
        logger.info("predict_fn: Predicting for an image done")

    return {"predictions": preds, "image_tensor": image_tensor}


def _build_image(prediction_dict):

    _segmentation_threshold = 0.8
    _prediction_threshold = 0.8
    logger.info(
        f"[INFO] Image requested - masking with threshold {_segmentation_threshold}. To get all predictions, request a JSON response"
    )
    image_tensor = prediction_dict["image_tensor"]
    preds: dict = prediction_dict["predictions"]

    score_mask = preds["scores"] > _prediction_threshold

    new_preds = {
        "labels": preds["labels"][score_mask],
        "scores": preds["scores"][score_mask],
        "boxes": preds["boxes"][score_mask],
        "masks": preds["masks"][score_mask],
    }

    image = (
        255.0
        * (image_tensor - image_tensor.min())
        / (image_tensor.max() - image_tensor.min())
    ).to(torch.uint8)
    image = image[:3, ...]

    pred_labels = [
        f"person: {score:.3f}"
        for _, score in zip(new_preds["labels"], new_preds["scores"])
    ]
    pred_boxes = new_preds["boxes"].long()
    box_colors = [COLORS[i % len(COLORS)] for i in range(len(pred_boxes))]
    output_image = draw_bounding_boxes(
        image, pred_boxes, pred_labels, colors=box_colors, width=3
    )

    masks = (new_preds["masks"] > _segmentation_threshold).squeeze(1)
    output_image = draw_segmentation_masks(
        output_image, masks, alpha=0.5, colors="blue"
    )
    return output_image


def output_fn(prediction, accept):
    # returns a tuple of the serialized prediction and the content type
    logger.info(
        "[INFO] output_fn-thread id: {}".format(threading.currentThread().getName())
    )
    logger.info("[INFO] output_fn-process id: {}".format(os.getpid()))

    # return the response
    if accept == "application/json":
        pred_dict = prediction["predictions"]
        # Get the prediction dictionary
        # masks = (pred_dict["masks"] > 0.2).detach().cpu().numpy().astype(int)

        json_output = {
            "predictions": {
                "boxes": pred_dict["boxes"].detach().cpu().numpy().tolist(),
                "labels": pred_dict["labels"].detach().cpu().numpy().tolist(),
                "scores": pred_dict["scores"].detach().cpu().numpy().tolist(),
            }
        }
        return json.dumps(json_output).encode("utf-8"), accept
    else:
        output_image = _build_image(prediction).cpu()
        if accept in _python_content_types:
            output_array = output_image.numpy()
            buffer = io.BytesIO()
            np.save(buffer, output_array)
            return buffer.getvalue(), accept
        elif accept in _img_content_types:
            if accept == "image/jpeg":

                return encode_jpeg(output_image).numpy().tobytes(), accept
            elif accept == "image/png" or accept == "application/x-image":

                return encode_png(output_image).numpy().tobytes(), accept
        else:
            raise Exception("Unsupported content type: {}".format(accept))
