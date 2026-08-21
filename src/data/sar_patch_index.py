import json
import os
from pathlib import Path

def generate_1d_windows(length, patch_size, stride):
    """Generates 1D window coordinates with padding information."""
    windows = []
    for start in range(0, length, stride):
        valid_len = min(patch_size, length - start)
        pad_len = patch_size - valid_len
        windows.append((start, valid_len, pad_len))
        if start + patch_size >= length:
            break
    return windows

def main():
    root_dir = Path(r"d:\Projects\border surv")
    manifest_path = root_dir / "data" / "sar" / "tum_oscd" / "sar_scene_manifest.json"
    split_path = root_dir / "data" / "sar" / "tum_oscd" / "sar_split.json"
    output_index_path = root_dir / "data" / "sar" / "tum_oscd" / "sar_patch_index.json"
    output_report_path = root_dir / "data" / "sar" / "tum_oscd" / "sar_patch_index_report.md"

    # Load splits
    with open(split_path, 'r') as f:
        splits = json.load(f)
    
    train_cities = set(splits['train'])
    val_cities = set(splits['validation'])
    test_cities = set(splits['test'])

    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    patch_size = 256
    stride = 128

    records = []
    
    for scene in manifest['scenes']:
        city = scene['city']
        
        # Only process train and validation
        if city in test_cities:
            continue
        
        if city in train_cities:
            split_label = 'train'
        elif city in val_cities:
            split_label = 'validation'
        else:
            continue
            
        h = scene['image_height']
        w = scene['image_width']
        
        y_windows = generate_1d_windows(h, patch_size, stride)
        x_windows = generate_1d_windows(w, patch_size, stride)
        
        for y, valid_h, pad_h in y_windows:
            for x, valid_w, pad_w in x_windows:
                record = {
                    "city": city,
                    "split": split_label,
                    "t1_image": scene['t1_image'],
                    "t2_image": scene['t2_image'],
                    "mask": scene['mask'],
                    "y": y,
                    "x": x,
                    "valid_height": valid_h,
                    "valid_width": valid_w,
                    "pad_bottom": pad_h,
                    "pad_right": pad_w,
                    "patch_size": patch_size,
                    "original_scene_height": h,
                    "original_scene_width": w
                }
                records.append(record)

    # Sort deterministic: city -> y -> x
    records.sort(key=lambda r: (r['city'], r['y'], r['x']))

    # Metadata
    patch_index = {
        "metadata": {
            "version": "1.0",
            "patch_size": patch_size,
            "stride": stride,
            "padding": "zero_bottom_right",
            "description": "SAR deterministic patch index for train/val splits."
        },
        "counts": {
            "train": sum(1 for r in records if r['split'] == 'train'),
            "validation": sum(1 for r in records if r['split'] == 'validation'),
            "total": len(records)
        },
        "patches": records
    }

    with open(output_index_path, 'w') as f:
        json.dump(patch_index, f, indent=4)
        
    print(f"Saved {len(records)} patches to {output_index_path}")

    # Generate Report
    train_count = patch_index['counts']['train']
    val_count = patch_index['counts']['validation']
    
    scene_counts = {}
    padded_count = 0
    for r in records:
        city = r['city']
        scene_counts[city] = scene_counts.get(city, 0) + 1
        if r['pad_bottom'] > 0 or r['pad_right'] > 0:
            padded_count += 1
            
    report_lines = [
        "# SAR Patch Index Report",
        "",
        f"- **Train Patches:** {train_count}",
        f"- **Validation Patches:** {val_count}",
        f"- **Total Patches:** {len(records)}",
        f"- **Padded Patches (Partial Windows):** {padded_count}",
        "",
        "## Per-Scene Breakdown",
        "| City | Split | Patches | Scene Size |",
        "|---|---|---|---|"
    ]
    
    for scene in manifest['scenes']:
        city = scene['city']
        if city in test_cities: continue
        split_label = 'train' if city in train_cities else 'validation'
        cnt = scene_counts.get(city, 0)
        h = scene['image_height']
        w = scene['image_width']
        report_lines.append(f"| {city} | {split_label} | {cnt} | {h}x{w} |")
        
    with open(output_report_path, 'w') as f:
        f.write("\n".join(report_lines) + "\n")
        
    print(f"Saved report to {output_report_path}")

if __name__ == "__main__":
    main()
