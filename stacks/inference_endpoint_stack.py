from typing import TypedDict
from aws_cdk import (
    NestedStack,
    aws_sagemaker as sagemaker,
    CfnOutput,
)

from BaseModel import BaseModel
from constructs import Construct


class InferenceEndpointStack(NestedStack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        metadata: BaseModel,
        model: sagemaker.CfnModel,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        model_name = model.attr_model_name

        # We define buckets to upload the data to test them/train them and the models themselves

        # this Stack intends to deploy a sagemaker model

        # TODO: Make this a parameter in a config file
        # this can be used for non-serverless configuration
        ec2_instance_type = "ml.inf1.xlarge"

        if metadata.serverless:
            serverless_prod_variant = sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                initial_variant_weight=1,
                model_name=model_name,
                variant_name=f"{model_name}-ServerlessProductionVariant",
                serverless_config=sagemaker.CfnEndpointConfig.ServerlessConfigProperty(
                    max_concurrency=1,
                    memory_size_in_mb=4096,
                ),
            )
            production_variant = serverless_prod_variant
        else:
            normal_prod_variant = sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                instance_type=ec2_instance_type,
                initial_instance_count=1,
                initial_variant_weight=1,
                model_name=model_name,
                variant_name=f"{model_name}-WithServers",
            )
            production_variant = normal_prod_variant

        # create an endpoint configuration
        endpoint_config = sagemaker.CfnEndpointConfig(
            self,
            f"EndpointConfig-{metadata.name}",
            production_variants=[production_variant],
        )

        endpoint_config.node.add_dependency(model)

        # create an endpoint

        if metadata.serverless:
            serverless_endpoint = sagemaker.CfnEndpoint(
                self,
                f"ServerlessEndpoint-{metadata.name}",
                endpoint_config_name=endpoint_config.attr_endpoint_config_name,
                endpoint_name=f"{model_name}-ServerlessEndpoint",
            )
            endpoint = serverless_endpoint
        else:
            normal_endpoint = sagemaker.CfnEndpoint(
                self,
                f"Endpoint-{metadata.name}",
                deployment_config=sagemaker.CfnEndpoint.DeploymentConfigProperty(
                    # rolling update to ensure the endpoints keeps the latest model in use
                    rolling_update_policy=sagemaker.CfnEndpoint.RollingUpdatePolicyProperty(
                        maximum_batch_size=sagemaker.CfnEndpoint.CapacitySizeProperty(
                            type="CAPACITY_PERCENT", value=50
                        ),
                        wait_interval_in_seconds=60,
                        maximum_execution_timeout_in_seconds=1800,
                    )
                ),
                endpoint_config_name=endpoint_config.attr_endpoint_config_name,
                endpoint_name=f"{model_name}-Endpoint",
            )
            endpoint = normal_endpoint

        CfnOutput(
            self,
            f"{metadata.name}-sagemakerEndpoint",
            value=endpoint.attr_endpoint_name,
            description="The sagemaker endpoint where the model is deployed",
        )
