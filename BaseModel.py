from pathlib import Path
from typing import Optional, Union, Any
from dataclasses import dataclass, field
from aws_cdk.aws_sagemaker import CfnEndpoint, CfnModelCard


@dataclass
class BaseModel:
    name: str
    framework: str
    data_path: Union[str, Path]
    path: Union[str, Path]
    file_name: str  # with extension!
    image_uri: str
    problem_type: str
    serverless: bool = True
    endpoint: Optional[CfnEndpoint] = None
    model_card: Optional[
        Union[
            CfnModelCard,
            dict[str, Any],
        ]
    ] = field(default_factory=dict)

    def __post_init__(self):
        self.data_path = Path(self.data_path)
        self.path = Path(self.path)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path {self.data_path} does not exist")
        if not self.path.exists():
            raise FileNotFoundError(f"Model directory {self.path} does not exist")

        if not self.path.is_dir():
            raise NotADirectoryError(f"Model directory {self.path} is not a directory")

        if not (self.path / self.file_name).exists():
            raise FileNotFoundError(f"Model file {self.file_name} does not exist")

    def __str__(self):
        base_str = f"Model {self.name} | {"serverless" if self.serverless else ""} {self.problem_type} on {self.framework}."
        if self.endpoint is not None:
            base_str += f" Endpoint: {self.endpoint.attr_endpoint_name}"
        return base_str

    def build_model_card(self, scope) -> CfnModelCard:
        """
        Build a model card for the model

        Returns:
            CfnModelCard: Model card for the model
        """
        if isinstance(self.model_card, CfnModelCard):
            return self.model_card
        elif isinstance(self.model_card, dict):

            return CfnModelCard(
                scope,
                id=f"{self.name}-model-card",
                model_card_name=f"{self.name}-model-card",
                model_card_status="Approved",
                content=self.model_card,
            )
        else:
            raise ValueError(
                f"Model card must be a CfnModelCard or a dict, got {type(self.model_card)}"
            )
