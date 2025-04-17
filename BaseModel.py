from pathlib import Path
from dataclasses import dataclass


@dataclass
class BaseModel:
    name: str
    framework: str
    data_path: str | Path
    path: str | Path
    file_name: str  # with extension!
    image_uri: str
    problem_type: str
    serverless: bool = True

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
        return f"Model {self.name} | {"serverless" if self.serverless else ""} {self.problem_type} on {self.framework}."
