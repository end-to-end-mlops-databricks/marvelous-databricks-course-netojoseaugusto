from pathlib import Path
from typing import Any, Union

import yaml


def open_yaml_file(file_path: Union[str, Path]) -> Any:
    """
    Open a YAML file and return its contents.

    Parameters:
    file_path (str or Path): The path to the YAML file.

    Returns:
    Any: The contents of the YAML file.

    Raises:
    FileNotFoundError: If the file does not exist.
    PermissionError: If the file cannot be accessed.
    yaml.YAMLError: If the YAML is malformed.
    ValueError: If the file_path is empty or not a string or Path object.
    """
    if not file_path:
        raise ValueError("file_path must be a non-empty string or Path object")

    if not isinstance(file_path, (str, Path)):
        raise ValueError("file_path must be a string or Path object")

    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"The file does not exist: {file_path}")

    try:
        with file_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {file_path}") from e
