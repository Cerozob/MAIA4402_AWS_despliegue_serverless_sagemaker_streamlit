import aws_cdk as core
import aws_cdk.assertions as assertions

from maia_sagemaker.maia_sagemaker_stack import MaiaSagemakerStack

# example tests. To run these tests, uncomment this file along with the example
# resource in maia_sagemaker/maia_sagemaker_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = MaiaSagemakerStack(app, "maia-sagemaker")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
