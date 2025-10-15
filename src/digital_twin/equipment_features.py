import json
from typing import List, Dict, Any, Optional


def get_equipment_features(
    equipment_name: str,
    config_file_path: str = "dataset/carboptim/DHC_unit_config.json",
) -> List[Dict[str, Any]]:
    """
    Extracts all features for a given equipment from the DHC_unit_config.json file.

    Args:
        equipment_name (str): The name of the equipment (e.g., "F101", "F102")
        config_file_path (str): Path to the DHC_unit_config.json file

    Returns:
        List[Dict[str, Any]]: List of all features (tags with classification="feature") for the equipment

    Raises:
        FileNotFoundError: If the config file is not found
        ValueError: If the equipment is not found in the config file
        json.JSONDecodeError: If the JSON file is malformed
    """
    try:
        # Load the JSON configuration file
        with open(config_file_path, "r", encoding="utf-8") as file:
            config = json.load(file)

        # Find the equipment in the configuration
        equipment_data = None
        for equipment in config.get("equipment", []):
            if equipment.get("name") == equipment_name:
                equipment_data = equipment
                break

        if equipment_data is None:
            available_equipment = [eq.get("name") for eq in config.get("equipment", [])]
            raise ValueError(
                f"Equipment '{equipment_name}' not found. Available equipment: {available_equipment}"
            )

        # Extract all features (tags with classification="feature")
        features = []
        for tag in equipment_data.get("tags", []):
            if (
                tag.get("classification") == "feature"
                or tag.get("classification") == "target"
                or tag.get("classification") == "variable"
            ):
                features.append(tag)

        return features

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in configuration file: {e}")


def get_equipment_features_only(
    equipment_name: str,
    config_file_path: str = "dataset/carboptim/DHC_unit_config.json",
) -> List[Dict[str, Any]]:
    """
    Extracts all features for a given equipment from the DHC_unit_config.json file.

    Args:
        equipment_name (str): The name of the equipment (e.g., "F101", "F102")
        config_file_path (str): Path to the DHC_unit_config.json file

    Returns:
        List[Dict[str, Any]]: List of all features (tags with classification="feature") for the equipment

    Raises:
        FileNotFoundError: If the config file is not found
        ValueError: If the equipment is not found in the config file
        json.JSONDecodeError: If the JSON file is malformed
    """
    try:
        # Load the JSON configuration file
        with open(config_file_path, "r", encoding="utf-8") as file:
            config = json.load(file)

        # Find the equipment in the configuration
        equipment_data = None
        for equipment in config.get("equipment", []):
            if equipment.get("name") == equipment_name:
                equipment_data = equipment
                break

        if equipment_data is None:
            available_equipment = [eq.get("name") for eq in config.get("equipment", [])]
            raise ValueError(
                f"Equipment '{equipment_name}' not found. Available equipment: {available_equipment}"
            )

        # Extract all features (tags with classification="feature")
        features = []
        for tag in equipment_data.get("tags", []):
            if (
                tag.get("classification") == "feature"
                or tag.get("classification") == "variable"
            ):
                features.append(tag)

        return features

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in configuration file: {e}")


def get_equipment_feature_only_names(
    equipment_name: str,
    config_file_path: str = "dataset/carboptim/DHC_unit_config.json",
) -> List[str]:
    """
    Extracts only the feature names for a given equipment.

    Args:
        equipment_name (str): The name of the equipment (e.g., "F101", "F102")
        config_file_path (str): Path to the DHC_unit_config.json file

    Returns:
        List[str]: List of feature names for the equipment
    """
    features = get_equipment_features_only(equipment_name, config_file_path)
    return [feature.get("name") for feature in features if feature.get("name")]


def get_equipment_feature_names(
    equipment_name: str,
    config_file_path: str = "dataset/carboptim/DHC_unit_config.json",
) -> List[str]:
    """
    Extracts only the feature names for a given equipment.

    Args:
        equipment_name (str): The name of the equipment (e.g., "F101", "F102")
        config_file_path (str): Path to the DHC_unit_config.json file

    Returns:
        List[str]: List of feature names for the equipment
    """
    features = get_equipment_features(equipment_name, config_file_path)
    return [feature.get("name") for feature in features if feature.get("name")]


def get_all_equipment_names(
    config_file_path: str = "dataset/carboptim/DHC_unit_config.json",
) -> List[str]:
    """
    Gets all available equipment names from the configuration file.

    Args:
        config_file_path (str): Path to the DHC_unit_config.json file

    Returns:
        List[str]: List of all equipment names
    """
    try:
        with open(config_file_path, "r", encoding="utf-8") as file:
            config = json.load(file)

        return [
            equipment.get("name")
            for equipment in config.get("equipment", [])
            if equipment.get("name")
        ]

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in configuration file: {e}")


def output_equipment(
    equipment_name: str,
    config_file_path: str = "dataset/carboptim/DHC_unit_config.json",
):
    """
    Outputs the names of all output features for a given equipment.
    Args:
        equipment_name (str): The name of the equipment (e.g., "F101", "F102")
        config_file_path (str): Path to the DHC_unit_config.json file
    """
    try:
        l = []
        features = get_equipment_features(equipment_name, config_file_path)

        for _, feature in enumerate(features, 1):
            if feature.get("type") == "output":
                l.append(feature.get("name", "Unknown"))

        return l

    except Exception as e:
        print(f"Error: {e}")


def get_shared_features_across_equipment(
    config_file_path: str = "dataset/carboptim/DHC_unit_config.json",
) -> List[str]:
    """
    Returns the names of features that appear in more than 1 equipment in the DHC unit.

    Args:
        config_file_path (str): Path to the DHC_unit_config.json file

    Returns:
        List[str]: List of feature names that are shared across multiple equipment
    """
    try:
        # Load the JSON configuration file
        with open(config_file_path, "r", encoding="utf-8") as file:
            config = json.load(file)

        # Dictionary to count occurrences of each feature across equipment
        feature_count = {}

        # Iterate through all equipment
        for equipment in config.get("equipment", []):
            equipment_name = equipment.get("name")

            # Get all feature names for this equipment
            for tag in equipment.get("tags", []):
                if (
                    tag.get("classification") == "feature"
                    or tag.get("classification") == "target"
                    or tag.get("classification") == "variable"
                ):
                    feature_name = tag.get("name")
                    if feature_name:
                        if feature_name not in feature_count:
                            feature_count[feature_name] = set()
                        feature_count[feature_name].add(equipment_name)

        # Return only features that appear in more than 1 equipment
        shared_features = [
            feature_name
            for feature_name, equipment_set in feature_count.items()
            if len(equipment_set) > 1
        ]

        return sorted(shared_features)  # Sort for consistent output

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in configuration file: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Test the functions
    try:
        # Get all available equipment
        equipment_list = get_all_equipment_names()
        print("Available equipment:", equipment_list)

        # Test with F101
        print("\n" + "=" * 50)
        features_f101 = get_equipment_features("F101")
        feature_names_f101 = get_equipment_feature_names("F101")
        print(feature_names_f101)
    except Exception as e:
        print(f"Error during testing: {e}")
