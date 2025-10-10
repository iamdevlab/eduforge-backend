import json
import os

REGION_PATH = "app/regions"
_cached_regions = {}

def load_region(region_name: str):
    """
    Load curriculum/region configuration from JSON files.

    - Accepts the `region_name` (e.g., "Nigeria").
    - Caches results in `_cached_regions` to avoid repeated file reads.
    - Expects the JSON file to contain:
        - region (string)
        - curriculum (string)
        - levels (dict of class levels, subjects, exam styles, syllabus anchors, etc.)
    - Adds default values for optional keys like difficulty_levels and command_words
      if they are missing, so downstream code never breaks.
    """
    key = region_name.lower()
    if key in _cached_regions:
        return _cached_regions[key]

    file_path = os.path.join(REGION_PATH, f"{key}.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Region '{region_name}' not found")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate required top-level keys
    for k in ["region", "curriculum", "levels"]:
        if k not in data:
            raise ValueError(f"Region JSON missing key: {k}")

    # Provide safe defaults for optional fields
    if "difficulty_levels" not in data:
        data["difficulty_levels"] = {"easy": "", "medium": "", "hard": ""}
    if "command_words" not in data:
        data["command_words"] = {"easy": [], "medium": [], "hard": []}

    # Cache for reuse
    _cached_regions[key] = data
    return data
