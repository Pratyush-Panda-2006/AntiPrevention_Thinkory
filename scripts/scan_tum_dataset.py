import os
from pathlib import Path

base_sar = Path("data/sar/tum_oscd/multisensor_fusion_CD/S1")
base_masks = Path("data/sar/tum_oscd/oscd_labels")
cities = ["abudhabi", "aguasclaras", "beihai", "beirut", "bercy", "bordeaux", "cupertino", "hongkong", "mumbai", "nantes", "paris", "pisa", "rennes", "saclay_e", "brasilia", "chongqing", "dubai", "lasvegas", "milano", "montpellier", "norcia", "rio", "saclay_w", "valencia"]
train_cities = ['abudhabi', 'aguasclaras', 'beihai', 'beirut', 'bercy', 'bordeaux', 'cupertino', 'hongkong', 'mumbai', 'nantes', 'paris', 'pisa', 'rennes', 'saclay_e']

print("city | T1 candidates | T2 candidates | selected T1 | selected T2 | # trans T1 | # trans T2 | mask path | split")
print("---|---|---|---|---|---|---|---|---")
for city in sorted(cities):
    split = "train" if city in train_cities else "test"
    
    t1_trans = list((base_sar / city / "imgs_1" / "transformed").glob("*.tif"))
    t2_trans = list((base_sar / city / "imgs_2" / "transformed").glob("*.tif"))
    
    # OSCD labels can be under Train/ or Test/ depending on the dataset structure, but the prompt says:
    # "OSCD labels: data\sar\tum_oscd\oscd_labels\"
    # Let's find the mask for this city.
    masks = list(base_masks.rglob(f"{city}-cm.tif"))
    if not masks:
        masks = list(base_masks.rglob(f"{city}*.tif"))
    
    mask_path = masks[0].relative_to(base_masks) if masks else "MISSING"
    
    t1_names = [p.name for p in t1_trans]
    t2_names = [p.name for p in t2_trans]
    
    # How reference selects it (just os.listdir()[0] which is arbitrary, but we can simulate it)
    sel_t1 = os.listdir(base_sar / city / "imgs_1" / "transformed")[0] if (base_sar / city / "imgs_1" / "transformed").exists() and os.listdir(base_sar / city / "imgs_1" / "transformed") else "NONE"
    sel_t2 = os.listdir(base_sar / city / "imgs_2" / "transformed")[0] if (base_sar / city / "imgs_2" / "transformed").exists() and os.listdir(base_sar / city / "imgs_2" / "transformed") else "NONE"
    
    print(f"{city} | {', '.join(t1_names)} | {', '.join(t2_names)} | {sel_t1} | {sel_t2} | {len(t1_names)} | {len(t2_names)} | {mask_path} | {split}")
