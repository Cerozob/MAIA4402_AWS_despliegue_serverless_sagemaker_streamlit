import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
import os
import time
import random
import logging
from typing import Dict, List, Tuple, Optional

# Import helper functions from utils.py
from utils import (
    get_endpoint_details_from_sagemaker,
    get_models_from_ssm,
    invoke_endpoint,
    draw_bounding_boxes,
    get_image_download_link,
    prepare_image_for_model,
    get_content_types_for_problem_type,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="SageMaker Endpoint Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Main application
def main():
    st.title("SageMaker Endpoint Demo")

    # Sidebar
    st.sidebar.title("Configuration")

    # Get models from SSM Parameter Store
    models = get_models_from_ssm()

    if not models:
        st.warning("No models found in SSM Parameter Store.")
        # Create a mock model for demo purposes
        models = []

    # Model selection
    model_names = [model["name"] for model in models]
    selected_model_name = st.sidebar.selectbox("Select Model", model_names)

    # Find the selected model details
    selected_model = next(
        (model for model in models if model["name"] == selected_model_name), None
    )

    if selected_model:
        st.sidebar.write("Model Details:")
        st.sidebar.json(selected_model)

        # Get appropriate content types for the selected model
        content_types = get_content_types_for_problem_type(
            selected_model["problem_type"]
        )
        selected_content_type = st.sidebar.selectbox(
            "Content Type",
            content_types,
            # this content type is to set the OUTPUT of the model
            help="Select the content type that you want the model to output",
            index=0,
        )

        # Confidence threshold for object detection
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05
        )

        # Main content
        st.write(f"## Testing {selected_model['name']}")

        if selected_model["problem_type"] == "object detection":
            st.write("Upload an image to detect objects:")

            # File uploader
            uploaded_file = st.file_uploader(
                "Choose an image...", type=["jpg", "jpeg", "png"]
            )

            if uploaded_file is not None:
                # Display the uploaded image
                image = Image.open(uploaded_file)

                st.image(image, caption="Uploaded Image", use_container_width=True)

                # Process the image when the user clicks the button
                if st.button("Detect Objects"):
                    with st.spinner("Processing..."):
                        # Prepare the image for the model with the selected content type
                        img_byte_arr = prepare_image_for_model(image)

                        # Invoke the real SageMaker endpoint
                        model_details = get_endpoint_details_from_sagemaker(
                            selected_model["endpoint"]
                        )
                        # write in screen the model data, show as a json

                        st.json(model_details)

                        result = invoke_endpoint(
                            selected_model["endpoint"],
                            img_byte_arr,
                            content_type=selected_content_type,
                        )

                        # Display the results
                        if result and "predictions" in result:
                            st.write("### Detection Results:")

                            # Convert predictions to DataFrame for display
                            predictions = result["predictions"]
                            if predictions:
                                df = pd.DataFrame(predictions)
                                st.dataframe(df)

                                # Draw bounding boxes on the image
                                annotated_image = draw_bounding_boxes(
                                    image.copy(), predictions, confidence_threshold
                                )
                                st.image(
                                    annotated_image,
                                    caption="Detection Results",
                                    use_container_width=True,
                                )

                                # Provide download link for the annotated image
                                st.markdown(
                                    get_image_download_link(
                                        annotated_image,
                                        "detection_result.jpg",
                                        "Download Annotated Image",
                                    ),
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.info(
                                    "No objects detected with the current confidence threshold."
                                )
                        else:
                            st.error("Failed to get valid predictions from the model.")

        elif selected_model["problem_type"] == "text classification":
            st.write("Enter text for classification:")

            # Text input
            text_input = st.text_area("Input Text", height=150)

            if st.button("Classify Text") and text_input:
                with st.spinner("Processing..."):

                    # Invoke the real SageMaker endpoint
                    result = invoke_endpoint(
                        selected_model["endpoint"],
                        text_input.encode("utf-8"),
                        content_type=selected_content_type,
                    )

                    # Display the results
                    if result and "predictions" in result:
                        st.write("### Classification Results:")

                        # Convert predictions to DataFrame for display
                        predictions = result["predictions"]
                        if predictions:
                            df = pd.DataFrame(predictions)
                            df = df.sort_values(by="probability", ascending=False)

                            # Display as dataframe
                            st.dataframe(df)

                            # Display as chart
                            st.bar_chart(df.set_index("class")["probability"])
                        else:
                            st.info("No classification results returned.")
                    else:
                        st.error("Failed to get valid predictions from the model.")
        else:
            st.write(
                f"Support for {selected_model['problem_type']} is not implemented yet."
            )
    else:
        st.error("No model selected or model information not available.")


if __name__ == "__main__":
    main()
