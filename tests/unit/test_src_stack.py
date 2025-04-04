import aws_cdk as core
import aws_cdk.assertions as assertions

from src.src_stack import SrcStack

# example tests. To run these tests, uncomment this file along with the example
# resource in src/src_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = SrcStack(app, "src")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
