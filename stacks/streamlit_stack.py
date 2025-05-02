import json
from aws_cdk import (
    CfnOutput,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ssm as ssm,
    aws_iam as iam,
    aws_ecs_patterns as ecs_patterns,
    aws_ecr_assets as ecr_assets,
    aws_sagemaker as sagemaker,
    aws_elasticloadbalancingv2 as elb,
    aws_autoscaling as autoscaling,
    NestedStack,
    Stack,
    Duration,
)

from pathlib import Path

from constructs import Construct

from BaseModel import BaseModel


class StreamlitStack(Stack):

    def __init__(
        self, scope: Construct, id: str, models: list[BaseModel], **kwargs
    ) -> None:
        super().__init__(scope, id, **kwargs)

        # Create a VPC
        vpc = ec2.Vpc(
            self,
            "StreamlitDemoVPC",
            max_azs=2,
        )

        # Create ECS cluster
        cluster = ecs.Cluster(
            self, "WebDemoCluster", vpc=vpc, enable_fargate_capacity_providers=True
        )

        sagemaker_endpoint_policy = iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "sagemaker:InvokeEndpoint",
                "sagemaker:ListEndpoints",
                "sagemaker:DescribeEndpoint",
            ],
            # over the endpoints
            resources=["*"],
        )

        ssm_endpoint_policy = iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "ssm:GetParameter",
                "ssm:GetParameters",
                "ssm:GetParametersByPath",
                "ssm:ListTagsForResource",
            ],
            resources=["*"],
        )

        ecs_task_policy = iam.ManagedPolicy.from_aws_managed_policy_name(
            "service-role/AmazonECSTaskExecutionRolePolicy"
        )

        task_role = iam.Role(
            self,
            "TaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            inline_policies={
                "ECSTaskpolicy": iam.PolicyDocument(
                    statements=[
                        sagemaker_endpoint_policy,
                        ssm_endpoint_policy,
                    ]
                )
            },
        )

        execution_role = iam.Role(
            self,
            "ExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[ecs_task_policy],
        )

        fargate_task_definition = ecs.FargateTaskDefinition(
            self,
            "StreamlitTaskDefinition",
            memory_limit_mib=4096,
            cpu=2048,
            execution_role=execution_role,
            task_role=task_role,
        )

        streamlit_source_folder = str(Path(__file__).parent.parent / "streamlit_app")

        image = ecs.ContainerImage.from_asset(
            str(streamlit_source_folder), platform=ecr_assets.Platform.LINUX_AMD64
        )

        fargate_task_definition.add_container(
            "StreamlitContainer",
            image=image,
            port_mappings=[ecs.PortMapping(container_port=8501)],
            logging=ecs.LogDriver.aws_logs(stream_prefix="StreamlitContainer"),
        )

        fargate_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "StreamlitService",
            cluster=cluster,
            desired_count=1,
            public_load_balancer=True,
            task_definition=fargate_task_definition,
            load_balancer_name="streamlit-demo",
            protocol=elb.ApplicationProtocol.HTTP,
            health_check_grace_period=Duration.seconds(60),
            min_healthy_percent=100,
        )

        # Setup task "auto-scaling"
        fargate_service.service.auto_scale_task_count(max_capacity=1)

        CfnOutput(
            self,
            "LoadBalancerDNS",
            value=fargate_service.load_balancer.load_balancer_dns_name,
            description="The DNS name of the load balancer",
        )
