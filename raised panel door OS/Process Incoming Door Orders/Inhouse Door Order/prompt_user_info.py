"""
Interactive Prompt for Missing Order Information
Used by Inhouse Door Order processing to gather specification details
"""
import json
import os


def load_defaults():
    """Load defaults from config file"""
    config_path = os.path.join(os.path.dirname(__file__), 'inhouse_defaults.json')

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Return empty defaults if config doesn't exist
        return {
            "last_used": {},
            "overlay_conversions": {},
            "presets": {}
        }


def save_defaults(config):
    """Save updated defaults to config file"""
    config_path = os.path.join(os.path.dirname(__file__), 'inhouse_defaults.json')

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def prompt_with_default(prompt_text, default_value=""):
    """
    Prompt user with a default value shown

    Args:
        prompt_text: Text to display
        default_value: Default value to use if user presses Enter

    Returns:
        str: User input or default value
    """
    if default_value:
        user_input = input(f"{prompt_text} [{default_value}]: ").strip()
        return user_input if user_input else default_value
    else:
        return input(f"{prompt_text}: ").strip()


def prompt_for_order_info(image_path=None):
    """
    Interactively prompt user for missing order information
    First checks for order_config.json in the same directory as the image

    Args:
        image_path: Path to the image being processed (optional)

    Returns:
        tuple: (order_info dict, specifications dict, should_save_defaults bool)
    """
    # First, try to load order_config.json from image directory
    if image_path:
        image_dir = os.path.dirname(os.path.abspath(image_path))
        order_config_path = os.path.join(image_dir, 'order_config.json')

        if os.path.exists(order_config_path):
            print(f"\n[OK] Found order_config.json in {image_dir}")
            with open(order_config_path, 'r') as f:
                order_config = json.load(f)

            return (
                order_config.get('order_info', {}),
                order_config.get('specifications', {}),
                False
            )

    print("\n" + "="*60)
    print("INHOUSE ORDER - Missing Information Required")
    print("="*60)

    # Load defaults
    config = load_defaults()
    last_used = config.get("last_used", {})

    # Show presets if available
    presets = config.get("presets", {})
    if presets:
        print("\nAvailable Presets:")
        for key, preset in presets.items():
            print(f"  {key}: {preset['name']}")
        use_preset = input("\nUse a preset? (preset_1/preset_2/preset_3 or Enter to skip): ").strip()

        if use_preset in presets:
            preset_data = presets[use_preset]
            print(f"\nLoaded preset: {preset_data['name']}")
            last_used.update(preset_data)

    print("\nPlease provide the following information:")
    print("(Press Enter to use default value shown in brackets)\n")

    # Order Information
    customer_company = prompt_with_default(
        "Customer Company",
        last_used.get("customer_company", "")
    )

    jobsite = prompt_with_default(
        "Jobsite/Customer Name",
        last_used.get("jobsite", "")
    )

    submitted_by = prompt_with_default(
        "Submitted By",
        last_used.get("submitted_by", "")
    )

    # Specifications
    wood_type = prompt_with_default(
        "Wood Type (e.g., 'Beech Paint Grade', 'Cherry')",
        last_used.get("wood_type", "Beech Paint Grade")
    )

    door_style = prompt_with_default(
        "Door Style Number (e.g., '101', '102')",
        last_used.get("door_style", "101")
    )

    edge_profile = prompt_with_default(
        "Edge Profile (e.g., 'square', 'eased')",
        last_used.get("edge_profile", "square")
    )

    panel_cut = prompt_with_default(
        "Panel Cut (e.g., '3/8 MDF', '1/4 plywood')",
        last_used.get("panel_cut", "3/8 MDF")
    )

    sticking_cut = prompt_with_default(
        "Sticking Cut",
        last_used.get("sticking_cut", "square")
    )

    # Ask if user wants to save as defaults
    print()
    save_response = input("Save these values as defaults for next time? (y/n): ").strip().lower()
    should_save = save_response == 'y'

    if should_save:
        config["last_used"] = {
            "customer_company": customer_company,
            "jobsite": jobsite,
            "submitted_by": submitted_by,
            "wood_type": wood_type,
            "door_style": door_style,
            "edge_profile": edge_profile,
            "panel_cut": panel_cut,
            "sticking_cut": sticking_cut
        }
        save_defaults(config)
        print("[OK] Defaults saved")

    # Build return dictionaries
    order_info = {
        "customer_company": customer_company,
        "jobsite": jobsite,
        "submitted_by": submitted_by
    }

    specifications = {
        "wood_type": wood_type,
        "door_style": door_style,
        "edge_profile": edge_profile,
        "panel_cut": panel_cut,
        "sticking_cut": sticking_cut
    }

    print("\n" + "="*60)
    return order_info, specifications, should_save


if __name__ == "__main__":
    # Test the prompt
    order_info, specs, saved = prompt_for_order_info()

    print("\nCollected Information:")
    print("\nOrder Info:")
    for key, value in order_info.items():
        print(f"  {key}: {value}")

    print("\nSpecifications:")
    for key, value in specs.items():
        print(f"  {key}: {value}")

    print(f"\nDefaults saved: {saved}")
