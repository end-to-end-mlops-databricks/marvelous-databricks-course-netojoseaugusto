import yaml

def open_yaml_file(file_path: str) -> dict:
    """
    Open a YAML file and return its contents as a dictionary.

    Parameters:
    file_path (str): The path to the YAML file.

    Returns:
    dict: The contents of the YAML file.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)