from aws_cdk import (
    NestedStack,
    aws_s3 as s3,
    aws_iam as iam,
    aws_sagemaker as sagemaker,
    aws_s3_deployment as s3deploy,
    CfnOutput,
)

from pathlib import Path
from constructs import Construct
from tarfile import TarFile

from BaseModel import BaseModel


def package_sagemaker_model(model_dir: Path, model_file_name: str) -> Path:

    # packages a model directory into a tar.gz file
    # returns the path to the tar.gz file
    # creates the file in a subdirectory called /asset/

    target_path = model_dir
    target_path.mkdir(exist_ok=True)

    tar_path = target_path / Path(f"{model_dir.name}.tar.gz")

    if tar_path.exists():
        print(f"Model already packaged at {tar_path}, delete the file to repackage")
    else:
        print(f"Packaging model {model_dir.name} into {tar_path}")
        try:
            with TarFile.open(tar_path, "w:gz") as tar:
                tar.add(model_dir / model_file_name, arcname=model_file_name)
                tar.add(model_dir / "code", arcname="code")
            print(f"Model packaged into {tar_path} successfully")
        except Exception as e:
            print(f"Error packaging model: {e}")
            raise e
    print(f"model file saved at {tar_path.resolve()}")
    return tar_path


class ModelDeploymentStack(NestedStack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        model_obj: BaseModel,
        data_bucket: s3.Bucket,
        model_bucket: s3.Bucket,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # We define buckets to upload the data to test them/train them and the models themselves

        # this Stack intends to package a model directory in a file
        # and set it up as an Amazon SageMaker model

        go_serverless = model_obj.serverless

        data_path: Path = model_obj.data_path

        # deploy the data

        dataBucketDeployment = s3deploy.BucketDeployment(
            self,
            "deploy_data",
            sources=[s3deploy.Source.asset(str(data_path))],
            destination_bucket=data_bucket,
            retain_on_delete=False,
            memory_limit=512,
        )

        dataBucketDeployment.node.add_dependency(data_bucket)

        model_path: Path = model_obj.path
        model_file_name = model_obj.file_name

        tarfile = package_sagemaker_model(model_path, model_file_name)

        modelBucketDeployment = s3deploy.BucketDeployment(
            self,
            "deploy_models",
            sources=[
                s3deploy.Source.asset(
                    str(tarfile.parent.resolve()), exclude=["*.py", "*.txt", "*.pt"]
                ),
            ],
            destination_bucket=model_bucket,
            retain_on_delete=False,
            memory_limit=512,
        )

        modelBucketDeployment.node.add_dependency(model_bucket)

        modelurl = f"s3://{model_bucket.bucket_name}/{tarfile.name}"

        img_uri = model_obj.image_uri

        if go_serverless:
            serverless_container_definition = (
                sagemaker.CfnModel.ContainerDefinitionProperty(
                    image=img_uri,
                    model_data_url=modelurl,
                    mode="SingleModel",
                )
            )
            container_definition = serverless_container_definition
        else:
            normal_container_definition = (
                sagemaker.CfnModel.ContainerDefinitionProperty(
                    image=img_uri,
                    model_data_url=modelurl,
                    image_config=sagemaker.CfnModel.ImageConfigProperty(
                        repository_access_mode="Platform",
                    ),
                    mode="SingleModel",
                )
            )
            container_definition = normal_container_definition

        # create a role for the sagemaker model
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
                    model_bucket.bucket_arn,
                    f"{model_bucket.bucket_arn}/*",
                ],
            )
        )

        timestamp = tarfile.stat().st_mtime

        model = sagemaker.CfnModel(
            self,
            f"Model-{model_obj.name}-{model_obj.framework}-{timestamp}",
            execution_role_arn=sagemaker_role.role_arn,
            primary_container=container_definition,
            model_name=model_obj.name,
        )

        model.node.add_dependency(sagemaker_role)
        model.node.add_dependency(modelBucketDeployment)

        self.model_obj = model_obj
        self.sagemaker_model = model

        CfnOutput(
            self,
            "sagemaker_model",
            value=model.attr_model_name,
            description="The sagemaker model to deploy",
        )
