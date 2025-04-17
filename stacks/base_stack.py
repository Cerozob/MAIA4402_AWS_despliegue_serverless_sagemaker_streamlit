from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_iam as iam,
    aws_sagemaker as sagemaker,
    aws_s3_deployment as s3deploy,
    RemovalPolicy,
    CfnOutput,
)

from os import environ
from pathlib import Path
from constructs import Construct
from tarfile import TarFile


class BaseStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # We define buckets to upload the data to test them/train them and the models themselves

        # this Stack intends to package a model directory in a file
        # and set it up as an Amazon SageMaker model

        # a bucket
        acc_id = environ.get("CDK_DEFAULT_ACCOUNT")
        acc_region = environ.get("CDK_DEFAULT_REGION")

        data_bucket = s3.Bucket(
            self,
            "data_bucket",
            bucket_name=f"cdk-sagemaker-data-{acc_id}-{acc_region}",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            enforce_ssl=True,
        )

        models_bucket = s3.Bucket(
            self,
            "models_bucket",
            bucket_name=f"cdk-sagemaker-models-{acc_id}-{acc_region}",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            enforce_ssl=True,
        )

        self.data_bucket = data_bucket
        self.models_bucket = models_bucket

        CfnOutput(
            self,
            "data_bucket_url",
            value=data_bucket.bucket_arn,
            description="The bucket where the data is stored",
        )

        CfnOutput(
            self,
            "models_bucket_url",
            value=models_bucket.bucket_arn,
            description="The bucket where the models are stored",
        )
