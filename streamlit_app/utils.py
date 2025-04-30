import boto3
import json
import logging
import io
import base64
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def get_endpoint_details_from_sagemaker(endpoint_name: str) -> Optional[Dict]:
    """
    Get model details from SageMaker

    Args:
        model_name (str): Name of the model

    Returns:
        Optional[Dict]: Model details if found, else None
    """
    try:
        sagemaker_client = boto3.client("sagemaker")
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        # return simple dictionary with the arn of the endpoint, the name again, and if it serverless or not, from the response
        prod_variant = response["ProductionVariants"][0]
        resp_dict = {
            "endpoint_name": response["EndpointName"],
            "endpoint_arn": response["EndpointArn"],
            "serverless": prod_variant["CurrentServerlessConfig"] is not None,
        }
        return resp_dict
    except Exception as e:
        logger.error(f"Error fetching model details from SageMaker: {e}")
        return None


def get_models_from_ssm() -> List[Dict]:
    """
    Get model information from SSM Parameter Store

    Returns:
        List[Dict]: List of model configurations
    """
    try:
        ssm_client = boto3.client("ssm")
        response = ssm_client.get_parameter(Name="ModelsParameter")
        models_json = response["Parameter"]["Value"]
        # The parameter contains a comma-separated list of JSON objects
        models_list = f"[{models_json}]"
        return json.loads(models_list)
    except Exception as e:
        logger.error(f"Error fetching models from SSM: {e}")
        return []


def get_content_types_for_problem_type(problem_type: str) -> List[str]:
    """
    Get appropriate content types based on the model's problem type

    Args:
        problem_type (str): The problem type (e.g., 'object detection', 'text classification')

    Returns:
        List[str]: List of appropriate content types
    """
    if problem_type.lower() == "object detection":
        return [
            "image/jpeg",
            "image/png",
            "application/x-image",
            "application/x-npy",
            "application/json",
        ]
    elif problem_type.lower() == "text classification":
        return ["text/plain", "application/json", "text/csv", "application/jsonlines"]
    else:
        # Default content types
        return ["application/json", "text/plain"]


def invoke_endpoint(
    endpoint_name: str,
    data: bytes,
    content_type: str = "application/x-image",
    accept: str = "application/json",
) -> Dict:
    """
    Invoke a SageMaker endpoint with data

    Args:
        endpoint_name (str): Name of the SageMaker endpoint
        data (bytes): Binary data to send to the endpoint
        content_type (str, optional): Content type of the data. Defaults to "application/json".

    Returns:
        Dict: Response from the endpoint
    """
    try:
        runtime_client = boto3.client("sagemaker-runtime")
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType=content_type,
            Body=data,
            Accept=accept,
        )

        # Read the response body once
        response_body = response["Body"].read()

        # Process based on content type
        if accept == "application/json" or accept == "text/plain":
            try:
                result = json.loads(response_body.decode())
            except json.JSONDecodeError:
                # If not valid JSON, return as text
                result = {"predictions": response_body.decode()}
        elif accept == "application/x-image":
            # For image responses, return the raw bytes
            img = Image.open(io.BytesIO(response_body))
            result = {"image": img}
        elif accept == "application/x-npy":
            # For numpy array responses
            np_array = np.load(io.BytesIO(response_body))
            result = {"predictions": np_array.tolist()}
        elif accept == "image/jpeg":
            img = Image.open(io.BytesIO(response_body))
            result = {"image": img}
        elif accept == "image/png":
            img = Image.open(io.BytesIO(response_body))
            result = {"image": img}
        else:
            # For binary responses or other formats
            result = {"predictions": response_body}

        return result
    except Exception as e:
        logger.error(f"Error invoking endpoint {endpoint_name}: {e}")
        return {}


def draw_bounding_boxes(
    image: Image.Image, predictions: List[Dict], confidence_threshold: float = 0.5
) -> Image.Image:
    """
    Draw bounding boxes on an image based on model predictions

    Args:
        image (Image.Image): PIL Image object
        predictions (List[Dict]): List of prediction dictionaries
        confidence_threshold (float, optional): Minimum confidence threshold. Defaults to 0.5.

    Returns:
        Image.Image: Image with bounding boxes drawn
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()

    for i, pred in enumerate(predictions):
        if pred.get("confidence", 0) < confidence_threshold:
            continue

        # Extract bounding box coordinates
        # Format may vary based on model output, adjust as needed
        if "bbox" in pred:
            # Format: [x_min, y_min, x_max, y_max] normalized
            bbox = pred["bbox"]
            x_min, y_min, x_max, y_max = bbox

            # Convert normalized coordinates to pixel values
            x_min = int(x_min * width)
            y_min = int(y_min * height)
            x_max = int(x_max * width)
            y_max = int(y_max * height)
        elif all(k in pred for k in ["x_min", "y_min", "x_max", "y_max"]):
            # Format with explicit keys
            x_min = int(pred["x_min"] * width)
            y_min = int(pred["y_min"] * height)
            x_max = int(pred["x_max"] * width)
            y_max = int(pred["y_max"] * height)
        else:
            logger.warning(f"Unsupported bounding box format: {pred}")
            continue

        # Get color for this detection
        color = COLORS[i % len(COLORS)]

        # Draw rectangle
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)

        # Draw label
        label = f"{pred.get('class', 'Object')}: {pred.get('confidence', 0):.2f}"
        text_width, text_height = (
            draw.textsize(label, font=font) if hasattr(draw, "textsize") else (100, 15)
        )
        draw.rectangle(
            [x_min, y_min - text_height - 2, x_min + text_width, y_min], fill=color
        )
        draw.text(
            (x_min, y_min - text_height - 2), label, fill=(255, 255, 255), font=font
        )

    return image


def get_image_download_link(img: Image.Image, filename: str, text: str) -> str:
    """
    Generate a link to download an image

    Args:
        img (Image.Image): PIL Image object
        filename (str): Name of the file to download
        text (str): Text to display for the download link

    Returns:
        str: HTML link for downloading the image
    """
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href


def prepare_image_for_model(image: Image.Image) -> Tuple[bytes, str]:
    """
    Convert a PIL Image to bytes for sending to a model based on content type

    Args:
        image (Image.Image): PIL Image object

    Returns:
        Tuple[bytes, str]: Image as bytes and corresponding mime type
    """
    img_byte_arr = io.BytesIO()

    # Get image format from the image itself
    img_format = image.format or "PNG"  # Default to PNG if format is None

    mime_type = "image/png"  # Default mime type
    if img_format.upper() == "JPEG":
        image.save(img_byte_arr, format="JPEG")
        mime_type = "image/jpeg"
    elif img_format.upper() == "PNG":
        image.save(img_byte_arr, format="PNG")
        mime_type = "image/png"
    elif img_format.upper() == "NPY":
        # Convert to numpy array and save
        import numpy as np

        img_array = np.array(image)
        np.save(img_byte_arr, img_array)
        mime_type = "application/x-npy"
    else:
        # For other formats, use PNG as default
        image.save(img_byte_arr, format="PNG")
        mime_type = "image/png"

    return img_byte_arr.getvalue(), mime_type
