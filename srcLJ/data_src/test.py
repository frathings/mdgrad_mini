import json
import os
from ase.visualize import view
from data import load_config, get_unit_len, visualize_system_with_ase_3d

def test_visualization():
    """
    Test visualization of a system using ASE.
    """
    # Ensure the config.json file is present
    config_file = r"C:\Users\franc\mdml_project\data_src\config.json"
    if not os.path.exists(config_file):
        print(f"Configuration file '{config_file}' not found.")
        return

    # Load the configuration
    config = load_config(config_file)

    # Select a data tag to test visualization
    data_tag = "Si_2.293_100K"  # Change this to a valid data tag from your config
    size = 2  # System size for visualization

    # Check if the data tag exists in the configuration
    if data_tag not in config["exp_rdf_data_dict"]:
        print(f"Data tag '{data_tag}' not found in config.")
        return

    # Run visualization
    print(f"Visualizing system for data tag '{data_tag}'...")
    visualize_system_with_ase_3d(data_tag, size)
    print("Visualization completed.")

# Run the visualization test
if __name__ == "__main__":
    test_visualization()
