# SageMaker Endpoint Demo App

This Streamlit application provides a user-friendly interface to test SageMaker endpoints deployed using the CDK infrastructure in this repository.

## Features

- **Model Selection**: Choose from available models deployed as SageMaker endpoints
- **Object Detection**: Upload images and visualize object detection results with bounding boxes
- **Text Classification**: Input text and view classification results
- **Demo Mode**: Fallback to demo mode with mock predictions if no endpoints are available
- **Result Visualization**: View detection results as annotated images and download them
- **Confidence Threshold**: Adjust confidence threshold for object detection

## How It Works

1. The app fetches model information from AWS SSM Parameter Store
2. Users can select a model and upload appropriate input (image or text)
3. The app invokes the SageMaker endpoint with the input data
4. Results are displayed in a user-friendly format

## Requirements

The application requires the following Python packages:
- streamlit
- boto3
- pandas
- numpy
- Pillow (PIL)
- requests

## Local Development

To run the app locally for development:

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deployment

The app is deployed as a Docker container on AWS ECS Fargate using the CDK infrastructure in this repository. The deployment process is handled by the `StreamlitStack` class in `stacks/streamlit_stack.py`.

## Architecture

- `streamlit_app.py`: Main application file
- `utils.py`: Helper functions for AWS interactions and image processing
- `Dockerfile`: Container definition for deployment
- `requirements.txt`: Python dependencies

## AWS Permissions

The app requires the following AWS permissions:
- `sagemaker:InvokeEndpoint`: To call the SageMaker endpoints
- `sagemaker:ListEndpoints`: To list available endpoints
- `sagemaker:DescribeEndpoint`: To get endpoint details
- `ssm:GetParameter`: To fetch model information from SSM Parameter Store

These permissions are configured in the CDK stack.