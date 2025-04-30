#!/usr/bin/env python3
from os import environ

import aws_cdk as cdk
from aws_cdk import Tags

from stacks.model_deployment_stack import ModelDeploymentStack
from stacks.model_deployment_base_stack import ModelDeploymentBaseStack
import json
from pathlib import Path
from BaseModel import BaseModel
from stacks.streamlit_stack import StreamlitStack

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
    app, "MAIA4402-BaseStack", stack_name="maia4402-base-stack", env=env
)


for base_model in models:
    deployed_model_stack = ModelDeploymentStack(
        base_stack,
        f"ModelDeploymentStack-{base_model.name}",
        model_obj=base_model,
        data_bucket=base_stack.data_bucket,
        model_bucket=base_stack.models_bucket,
        # env=env,
        # stack_name=f"maia4402-model-deployment-stack-{base_model.name}",
    )

    base_model.endpoint = deployed_model_stack.endpoint

frontend_stack = StreamlitStack(
    base_stack,
    "StreamlitStack",
    models=models,
    # env=env,
    # stack_name="maia4402-streamlit-stack",
)

Tags.of(app).add("Project", "MAIA4402-Demo")

app.synth()
