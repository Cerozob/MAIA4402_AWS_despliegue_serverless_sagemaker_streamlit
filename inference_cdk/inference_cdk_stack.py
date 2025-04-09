from aws_cdk import (
    # Duration,
    Stack,
    # aws_sqs as sqs,
    aws_s3 as s3,
    aws_iam as iam,
    aws_sagemaker as sagemaker,
    aws_s3_deployment as s3deploy,
    RemovalPolicy,
    CfnOutput,
    aws_ec2 as ec2,
)
from sagemaker import image_uris

from os import environ
from pathlib import Path
from constructs import Construct
from tarfile import TarFile


def package_sagemaker_model(model_dir: Path) -> Path:

    # packages a model directory into a tar.gz file
    # returns the path to the tar.gz file
    # creates the file in a subdirectory called /asset/

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory {model_dir} does not exist")

    if not model_dir.is_dir():
        raise NotADirectoryError(f"Model directory {model_dir} is not a directory")

    target_path = model_dir / "asset"
    target_path.mkdir(exist_ok=True)

    tar_path = target_path / Path(f"{model_dir.name}.tar.gz")
    print(f"Packaging model {model_dir.name} into {tar_path}")
    with TarFile.open(tar_path, "w:gz") as tar:
        tar.add(model_dir, arcname=model_dir.name)
    print(f"Model packaged into {tar_path} successfully")
    return tar_path


class Inference_CDK_Stack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # We define buckets to upload the data to test them/train them and the models themselves

        # this Stack intends to deploy a sagemaker endpoint with an already trained model

        # a bucket
        acc_id = environ.get("CDK_DEFAULT_ACCOUNT")
        acc_region = environ.get("CDK_DEFAULT_REGION")

        data_bucket = s3.Bucket(
            self,
            "data_bucket",
            bucket_name=f"cdk-sagemaker-data-{acc_id}-{acc_region}",
            removal_policy=RemovalPolicy.DESTROY,
        )

        models_bucket = s3.Bucket(
            self,
            "models_bucket",
            bucket_name=f"cdk-sagemaker-models-{acc_id}-{acc_region}",
            removal_policy=RemovalPolicy.DESTROY,
        )

        data_path = Path(__file__).parent / Path("../data")

        # deploy the data
        dataBucketDeployment = s3deploy.BucketDeployment(
            self,
            "deploy_data",
            sources=[s3deploy.Source.asset(str(data_path))],
            destination_bucket=data_bucket,
            retain_on_delete=False,
            memory_limit=1024,  # a gb in mb
        )

        dataBucketDeployment.node.add_dependency(data_bucket)

        model_path = Path(__file__).parent / Path("../models/pytorch_yolo")
        tarfile = package_sagemaker_model(model_path)
        print(f"model file saved at {tarfile.resolve()}")
        # upload the models
        modelBucketDeployment = s3deploy.BucketDeployment(
            self,
            "deploy_models",
            sources=[s3deploy.Source.asset(str(tarfile.parent))],
            destination_bucket=models_bucket,
            retain_on_delete=True,
            memory_limit=512,  # a gb in mb, model weights around 60mb tho
        )

        modelBucketDeployment.node.add_dependency(models_bucket)

        # create a role for the sagemaker endpoint
        sagemaker_role = iam.Role(
            self,
            "sagemaker_role",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess"
                )
            ],
        )

        sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:*"],
                resources=[
                    data_bucket.bucket_arn,
                    f"{data_bucket.bucket_arn}/*",
                    models_bucket.bucket_arn,
                    f"{models_bucket.bucket_arn}/*",
                ],
            )
        )

        modelurl = f"s3://{models_bucket.bucket_name}/{tarfile.name}"

        img_uri = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-cpu-py312-ubuntu22.04-sagemaker"

        container = sagemaker.CfnModel.ContainerDefinitionProperty(
            image=img_uri,
            model_data_url=modelurl,
            image_config=sagemaker.CfnModel.ImageConfigProperty(
                repository_access_mode="Platform",
            ),
        )

        # create a model
        model = sagemaker.CfnModel(
            self,
            "PytorchPersonDetectionModel",
            execution_role_arn=sagemaker_role.role_arn,
            primary_container=container,
        )

        model.node.add_dependency(sagemaker_role)
        model.node.add_dependency(modelBucketDeployment)

        ec2_instance_type = "ml.inf1.xlarge"

        # create an endpoint configuration
        endpoint_config = sagemaker.CfnEndpointConfig(
            self,
            "PytorchPersonDetectionEndpointConfig",
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    instance_type=ec2_instance_type,
                    initial_instance_count=1,
                    initial_variant_weight=1,
                    model_name=model.attr_model_name,
                    variant_name="AllTraffic",
                )
            ],
        )

        endpoint_config.node.add_dependency(model)

        # create an endpoint
        endpoint = sagemaker.CfnEndpoint(
            self,
            "PytorchPersonDetectionEndpoint",
            endpoint_config_name=endpoint_config.attr_endpoint_config_name,
        )

        # As outputs, we want the bucket urls, and the model endpoint

        self.data_bucket_url = data_bucket.bucket_name
        self.models_bucket_url = models_bucket.bucket_name
        self.sagemaker_endpoint = endpoint.attr_endpoint_name

        CfnOutput(
            self,
            "data_bucket_url",
            value=data_bucket.bucket_name,
            description="The bucket where the data is stored",
        )

        CfnOutput(
            self,
            "models_bucket_url",
            value=models_bucket.bucket_name,
            description="The bucket where the models are stored",
        )

        CfnOutput(
            self,
            "sagemaker_endpoint",
            value=endpoint.attr_endpoint_name,
            description="The sagemaker endpoint where the model is deployed",
        )
