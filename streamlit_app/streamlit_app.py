import streamlit as st
from PIL import Image
import logging

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

        # Main content
        st.write(f"## Testing {selected_model['name']}")

        if selected_model["problem_type"] == "object detection":
            # Confidence threshold for object detection
            confidence_threshold = st.sidebar.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
            )
            st.write("Upload an image to detect objects:")

            # File uploader
            uploaded_file = st.file_uploader(
                "Choose an image...", type=["jpg", "jpeg", "png"]
            )

            if uploaded_file is not None:
                # Display the uploaded image
                image = Image.open(uploaded_file)

                button = st.button("Detect People", use_container_width=True)
                placeholder = st.empty()
                spinner_obj = st.spinner("Inference in progress...", show_time=True)

                col1, col2 = st.columns(
                    [0.5, 0.5],
                )

                with col1:
                    col1.image(
                        image, caption="Uploaded Image", use_container_width=True
                    )
                with col2:

                    # display the same image, then change it later to the overlaid predictions
                    col2image = col2.image(
                        image,
                        caption="Click 'Detect People' to highlight where are people in the picture",
                        use_container_width=True,
                    )

                # Prepare the image for the model with the selected content type
                img_byte_arr, mimetype = prepare_image_for_model(image)

                # Invoke the real SageMaker endpoint
                model_details = get_endpoint_details_from_sagemaker(
                    selected_model["endpoint"]
                )
                # write in screen the model data, show as a json
                with col1:
                    st.write("Model Details:")
                    st.json(model_details)

                # Process the image when the user clicks the button
                if button:
                    with placeholder, spinner_obj:

                        result = invoke_endpoint(
                            selected_model["endpoint"],
                            img_byte_arr,
                            content_type=mimetype,
                            accept=selected_content_type,
                        )

                        # result is a dict that contains a key with its content-type, and the corresponding object in that type
                        # if accept was an image, there will be an Image object ready to display, if its a json, use the draw_bboxes function

                        if selected_content_type in [
                            "application/x-image",
                            "image/jpeg",
                            "image/png",
                        ]:

                            with col2:

                                col2image.image(
                                    result[selected_content_type],
                                    caption="Detection Results",
                                    use_container_width=True,
                                )

                                # Provide download link for the image
                                st.markdown(
                                    get_image_download_link(
                                        result[selected_content_type],
                                        "detection_result.jpg",
                                        "Download Detection Result",
                                    ),
                                    unsafe_allow_html=True,
                                )
                        elif selected_content_type == "application/json":
                            # Draw bounding boxes on the image
                            annotated_image = draw_bounding_boxes(
                                image.copy(),
                                result[selected_content_type],
                                confidence_threshold,
                            )
                            with col2:

                                col2image.image(
                                    annotated_image,
                                    caption="Detection Results with Bounding Boxes",
                                    use_container_width=True,
                                )
                                # Provide download link for the annotated image
                                col2.markdown(
                                    get_image_download_link(
                                        annotated_image,
                                        "detection_result.jpg",
                                        "Download Annotated Image",
                                    ),
                                    unsafe_allow_html=True,
                                )
                                col2.json(result[selected_content_type])
                        elif selected_content_type == "application/x-npy":
                            # draw the raw numpy array as an image

                            # transpose 120

                            result[selected_content_type] = result[
                                selected_content_type
                            ].transpose(1, 2, 0)

                            with col2:

                                col2image.image(
                                    result[selected_content_type],
                                    caption="Detection Results with Bounding Boxes",
                                    use_container_width=True,
                                )
                                # Provide download link for the annotated image
                                col2.markdown(
                                    "To download the image, select another content type"
                                )

                                col2.json(
                                    result[selected_content_type].tolist(), expanded=1
                                )

                        else:
                            with col2:
                                col2.error(
                                    f"Unsupported content type for detection: {selected_content_type}, model answered with: {
                                    result
                                }"
                                )
        else:
            st.write(
                f"Support for {selected_model['problem_type']} is not implemented yet."
            )
    else:
        st.error("No model selected or model information not available.")


if __name__ == "__main__":
    main()
