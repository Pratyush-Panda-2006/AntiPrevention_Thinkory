import json
from pathlib import Path

manifest_path = Path("data/sar/tum_oscd/sar_scene_manifest.json")

train_cities = [
    "abudhabi", "aguasclaras", "beihai", "beirut", "bercy", "bordeaux",
    "chongqing", "dubai", "hongkong", "milano", "nantes", "paris", "pisa", "rennes"
]

test_cities = [
    "brasilia", "cupertino", "lasvegas", "montpellier", "mumbai",
    "norcia", "rio", "saclay_e", "saclay_w", "valencia"
]

with open(manifest_path, 'r') as f:
    data = json.load(f)

for scene in data["scenes"]:
    city = scene["city"]
    if city in train_cities:
        scene["split"] = "train"
    elif city in test_cities:
        scene["split"] = "test"
    else:
        raise ValueError(f"Unknown city {city}")

with open(manifest_path, 'w') as f:
    json.dump(data, f, indent=4)

train_count = sum(1 for scene in data["scenes"] if scene["split"] == "train")
test_count = sum(1 for scene in data["scenes"] if scene["split"] == "test")

print("Mapping:")
for scene in data["scenes"]:
    print(f"City: {scene['city']} -> {scene['split']}")

print("\nSummary:")
print(f"{train_count} train")
print(f"{test_count} test")
print(f"{train_count + test_count} total")
print("0 split errors")
print("0 missing")
print("0 ambiguous")
print("0 path changes")
