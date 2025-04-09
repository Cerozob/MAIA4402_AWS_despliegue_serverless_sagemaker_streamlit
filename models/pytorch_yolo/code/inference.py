import torch
import torchvision
from torchvision.io import decode_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms import v2
import torchvision
from pathlib import Path
import torch
import os
import threading
from model_class import inference_model


_prediction_threshold = 0.5
_segmentation_threshold = 0.6

_model_file_name = "inference_model.pt"

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_img_content_types = [
    "image/jpeg",
    "image/png",
    "application/x-image",
]

_python_content_types = [
    "application/x-npy",
]

print("[INFO] load-thread id: {}".format(threading.currentThread().getName()))
print("[INFO] load-process id: {}".format(os.getpid()))
print(
    "[INFO] load-SAGEMAKER_MODEL_SERVER_WORKERS: {}".format(
        os.environ.get("SAGEMAKER_MODEL_SERVER_WORKERS", "nonon")
    )
)


def model_fn(model_dir, context) -> dict:
    """
    Load the model for inference
    """

    print("[INFO] model_fn-thread id: {}".format(threading.currentThread().getName()))
    print("[INFO] loading model on device: {}".format(_device))

    model_path = Path(model_dir) / _model_file_name

    inference_model.load_state_dict(torch.load(model_path, map_location=_device))

    print("[INFO] model loaded successfully")

    return inference_model


def input_fn(input_data: bytes, content_type):
    """
    Parse input data payload
    """
    print("[INFO] input_fn-thread id: {}".format(threading.currentThread().getName()))
    print("[INFO] input_fn-process id: {}".format(os.getpid()))
    print("[INFO] input_fn-content_type: {}".format(content_type))

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
    return {
        "raw_image": input_data,
        "input_image": image_transforms(array)[:3, ...].to(_device),
    }


def predict_fn(input_dict, model):
    print("predict_fn: Predicting for an image...")
    model.eval()
    input_image = input_dict["input_image"]
    raw_image = input_dict["raw_image"]
    with torch.no_grad():
        predictions = model([input_image])
        pred = predictions[0]
        score_mask = pred["scores"] > _prediction_threshold
        pred = {
            "labels": pred["labels"][score_mask],
            "scores": pred["scores"][score_mask],
            "boxes": pred["boxes"][score_mask],
            "masks": pred["masks"][score_mask],
        }
        print("predict_fn: Predicting for an image done")
        print("predict_fn: pred: {}".format(pred))
        print("drawing bounding boxes and masks")

    image = (
        255.0
        * (input_image - input_image.min())
        / (input_image.max() - input_image.min())
    ).to(torch.uint8)
    image = image[:3, ...]

    pred_labels = [
        f"person: {score:.3f}" for _, score in zip(pred["labels"], pred["scores"])
    ]
    pred_boxes = pred["boxes"].long()
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    masks = (pred["masks"] > _segmentation_threshold).squeeze(1)
    output_image = draw_segmentation_masks(
        output_image, masks, alpha=0.5, colors="blue"
    )

    return output_image


def output_fn(prediction, content_type):
    print("[INFO] output_fn-thread id: {}".format(threading.currentThread().getName()))
    print("[INFO] output_fn-process id: {}".format(os.getpid()))

    # return the response
    if content_type in _python_content_types + _img_content_types:
        return prediction.permute(1, 2, 0)
    else:
        raise Exception("Unsupported content type: {}".format(content_type))
