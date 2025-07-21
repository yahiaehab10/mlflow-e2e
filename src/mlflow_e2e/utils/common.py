import os
import yaml
from mlflow_e2e import logger
import json
import joblib
from ensure import ensure_annotations
from box.box import Box
from box.exceptions import BoxValueError
from pathlib import Path
from typing import Any
import base64


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> Box:
    """Reads YAML file and returns it as a Box object.

    Args:
        path_to_yaml (Path): path-like input

    Raises:
        ValueError: if YAML file is empty
        e: unexpected errors

    Returns:
        Box: data accessible via dot notation
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file loaded: {path_to_yaml}")
            return Box(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Create a list of directories.

    Args:
        path_to_directories (list): list of paths
        verbose (bool, optional): whether to log creation messages
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """Save data as a JSON file.

    Args:
        path (Path): destination path
        data (dict): dictionary to save
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> Box:
    """Load JSON file and return as Box.

    Args:
        path (Path): path to JSON file

    Returns:
        Box: dot-accessible config
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded from: {path}")
    return Box(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """Save data as binary using joblib.

    Args:
        data (Any): object to save
        path (Path): destination path
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """Load binary file using joblib.

    Args:
        path (Path): path to binary file

    Returns:
        Any: deserialized object
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """Get file size in KB.

    Args:
        path (Path): file path

    Returns:
        str: size as string
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, "wb") as f:
        f.write(imgdata)


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
