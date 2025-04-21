#!/usr/bin/env python3
from os import environ

import aws_cdk as cdk
from aws_cdk import Tags

from stacks.inference_endpoint_stack import InferenceEndpointStack
from stacks.model_deployment_stack import ModelDeploymentStack
from stacks.model_deployment_base_stack import ModelDeploymentBaseStack
import json
from pathlib import Path
from BaseModel import BaseModel

file_path = Path(__file__).parent / Path("config.json")
config = {}

with open(file_path) as f:
    config = json.load(f)

if config == {}:
    raise Exception("No config.json file found")

models = [BaseModel(**model) for model in config.get("models")]

app = cdk.App()

env = cdk.Environment(
    account=environ.get("CDK_DEFAULT_ACCOUNT"),
    region=environ.get("CDK_DEFAULT_REGION"),
)

base_stack = ModelDeploymentBaseStack(
    app, "MAIA4402-DemoStack", stack_name="maia4402-demo-app", env=env
)

for base_model in models:
    deployed_model_stack = ModelDeploymentStack(
        base_stack,
        f"ModelDeploymentStack-{base_model.name}",
        model_obj=base_model,
        data_bucket=base_stack.data_bucket,
        model_bucket=base_stack.models_bucket,
        # env=env
        # stack_name=f"model-deployment-stack-{base_model.name}",
    )

    model_inference_endpoint = InferenceEndpointStack(
        base_stack,
        f"ModelEndpointStack-{base_model.name}",
        metadata=deployed_model_stack.model_obj,
        model=deployed_model_stack.sagemaker_model,
        # env=env
        # stack_name=f"model-endpoint-stack-{base_model.name}",
    )

Tags.of(app).add("Project", "MAIA4402-Demo")

app.synth()
