import json
import numpy as np
import rasterio
from pathlib import Path

def test_patch_index():
    root_dir = Path(r"d:\Projects\border surv")
    index_path = root_dir / "data" / "sar" / "tum_oscd" / "sar_patch_index.json"
    manifest_path = root_dir / "data" / "sar" / "tum_oscd" / "sar_scene_manifest.json"
    
    with open(index_path, 'r') as f:
        patch_index = json.load(f)
        
    patches = patch_index['patches']
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    # Build dictionary of scenes from manifest for easy lookup
    scenes = {s['city']: s for s in manifest['scenes']}
    
    # 4A. Pick one tiny known scene (bercy) and print all generated windows
    print("--- 4A: Tiny Scene (bercy) Windows ---")
    bercy_patches = [p for p in patches if p['city'] == 'bercy']
    for p in bercy_patches:
        print(f"y={p['y']:3d}, x={p['x']:3d} | valid_h={p['valid_height']:3d}, valid_w={p['valid_width']:3d} | pad_h={p['pad_bottom']:3d}, pad_w={p['pad_right']:3d}")
        
    # Group patches by scene
    from collections import defaultdict
    scene_patches = defaultdict(list)
    for p in patches:
        scene_patches[p['city']].append(p)
        
    # 4D. Verify test cities are not included
    test_cities = {'brasilia', 'cupertino', 'lasvegas', 'montpellier', 'mumbai', 'norcia', 'rio', 'saclay_e', 'saclay_w', 'valencia'}
    found_test_cities = set(scene_patches.keys()).intersection(test_cities)
    assert len(found_test_cities) == 0, f"Found test cities in index: {found_test_cities}"
    print("\n--- 4D: Test cities exclusion verified ---")

    print("\n--- 4B, 4C, 5: Coverage and Bounds Verification ---")
    print(f"{'City':12s} | {'Size':10s} | {'Patches':7s} | {'Uncovered':9s}")
    print("-" * 50)
    for city, city_patches in scene_patches.items():
        scene_meta = scenes[city]
        h = scene_meta['image_height']
        w = scene_meta['image_width']
        
        # Coverage map
        coverage = np.zeros((h, w), dtype=bool)
        
        # Check duplicates
        window_set = set()
        
        for p in city_patches:
            y, x = p['y'], p['x']
            v_h, v_w = p['valid_height'], p['valid_width']
            
            # 4B Check bounds
            assert y >= 0 and x >= 0, f"Negative coordinates in {city}"
            assert y + v_h <= h, f"Window exceeds height in {city}: {y}+{v_h} > {h}"
            assert x + v_w <= w, f"Window exceeds width in {city}: {x}+{v_w} > {w}"
            
            # For padding validation: 
            # If valid_h < patch_size, we MUST be at the bottom edge. (y + v_h == h)
            if v_h < p['patch_size']:
                assert y + v_h == h, f"Padded window not at bottom edge in {city}"
            if v_w < p['patch_size']:
                assert x + v_w == w, f"Padded window not at right edge in {city}"
            
            assert (y, x) not in window_set, f"Duplicated window {(y,x)} in {city}"
            window_set.add((y, x))
            
            coverage[y:y+v_h, x:x+v_w] = True
            
        # 5. Border coverage test
        uncovered = np.sum(~coverage)
        print(f"{city:12s} | {h}x{w:4d} | {len(city_patches):7d} | {uncovered:9d}")
        assert uncovered == 0, f"Scene {city} has {uncovered} uncovered pixels!"
        
    print("\nAll boundary and coverage tests passed.")
    
    # 6. Mask alignment test
    print("\n--- 6: Mask Alignment Test ---")
    # Picking 3 scenes of different sizes
    test_scenes = ['bercy', 'pisa', 'beirut'] 
    for city in test_scenes:
        if city not in scene_patches:
            print(f"Skipping {city} as it is not in the index.")
            continue
            
        scene_meta = scenes[city]
        t1_path = root_dir / "data" / "sar" / "tum_oscd" / scene_meta['t1_image']
        t2_path = root_dir / "data" / "sar" / "tum_oscd" / scene_meta['t2_image']
        mask_path = root_dir / "data" / "sar" / "tum_oscd" / scene_meta['mask']
        
        with rasterio.open(t1_path) as src_t1, rasterio.open(t2_path) as src_t2, rasterio.open(mask_path) as src_mask:
            h_t1, w_t1 = src_t1.shape
            h_t2, w_t2 = src_t2.shape
            h_m, w_m = src_mask.shape
            
            assert h_t1 == h_t2 == h_m, f"Height mismatch in {city}"
            assert w_t1 == w_t2 == w_m, f"Width mismatch in {city}"
            
            city_patches = scene_patches[city]
            # Verify coordinates are valid for all 3 datasets
            for p in city_patches:
                y, x = p['y'], p['x']
                v_h, v_w = p['valid_height'], p['valid_width']
                window = rasterio.windows.Window(x, y, v_w, v_h)
                
                # Reading just to assert no out-of-bounds error is raised by rasterio
                _ = src_t1.read(1, window=window)
                _ = src_t2.read(1, window=window)
                _ = src_mask.read(1, window=window)
                
        print(f"Alignment verified for {city} (H={h_t1}, W={w_t1}).")
        
    print("\nAll validation tests successfully passed!")

if __name__ == "__main__":
    test_patch_index()
