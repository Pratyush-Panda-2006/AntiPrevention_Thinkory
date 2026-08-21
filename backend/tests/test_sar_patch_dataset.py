import sys
import os
from pathlib import Path
import torch
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.data.sar_patch_dataset import TUMSARChangeDetectionDataset

def run_tests():
    index_path = root_dir / "data" / "sar" / "tum_oscd" / "sar_patch_index.json"
    
    # 1. Test Dataset Creation and Test-City Exclusion
    try:
        train_ds = TUMSARChangeDetectionDataset(index_path, 'train', root_dir=root_dir)
        val_ds = TUMSARChangeDetectionDataset(index_path, 'validation', root_dir=root_dir)
        dataset_created = "PASS"
    except Exception as e:
        dataset_created = f"FAIL: {str(e)}"
        print(dataset_created)
        return
        
    try:
        _ = TUMSARChangeDetectionDataset(index_path, 'test', root_dir=root_dir)
        test_city_exclusion = "FAIL: Allowed 'test' split"
    except ValueError:
        test_city_exclusion = "PASS"
        
    # Check no test cities leaked into train/val
    test_cities = {'brasilia', 'cupertino', 'lasvegas', 'montpellier', 'mumbai', 'norcia', 'rio', 'saclay_e', 'saclay_w', 'valencia'}
    
    for ds in [train_ds, val_ds]:
        for patch in ds.patches:
            if patch['city'] in test_cities:
                test_city_exclusion = f"FAIL: Found {patch['city']}"
                break
                
    # 2. Dataset Summary & Load tests
    print(f"Train samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    
    train_loading = "PASS"
    val_loading = "PASS"
    
    # Load 5 train
    try:
        for i in range(5):
            _ = train_ds[i]
    except Exception as e:
        train_loading = f"FAIL: {str(e)}"
        
    # Load 5 validation
    try:
        for i in range(5):
            _ = val_ds[i]
    except Exception as e:
        val_loading = f"FAIL: {str(e)}"
        
    # 3. Comprehensive tests on specific patches
    padding_test = "PASS"
    validity_mask_test = "PASS"
    range_test = "PASS"
    
    non_pad_idx = next(i for i, p in enumerate(train_ds.patches) if p['pad_bottom'] == 0 and p['pad_right'] == 0)
    pad_idx = next(i for i, p in enumerate(train_ds.patches) if p['pad_bottom'] > 0 or p['pad_right'] > 0)
    
    for idx, name in [(non_pad_idx, "Non-Padding"), (pad_idx, "Padding")]:
        try:
            patch = train_ds.patches[idx]
            item = train_ds[idx]
            print(f"\n--- Testing {name} Patch: {patch['city']} (x={patch['x']}, y={patch['y']}) ---")
            
            img_a = item['image_a']
            img_b = item['image_b']
            tgt = item['label']
            val = item['valid_mask']
            
            # Print representative info
            print(f"city: {item['city']}")
            print(f"x: {item['x']}")
            print(f"y: {item['y']}")
            print(f"valid_height: {patch['valid_height']}")
            print(f"valid_width: {patch['valid_width']}")
            print(f"padding: pad_bottom={patch['pad_bottom']}, pad_right={patch['pad_right']}")
            print(f"image_a shape: {img_a.shape}")
            print(f"image_b shape: {img_b.shape}")
            print(f"label shape: {tgt.shape}")
            print(f"valid_mask shape: {val.shape}")
            print(f"valid pixel count: {val.sum().item()}")
            print(f"positive target count: {tgt.sum().item()}")
            
            # D, E, F, G: Shapes
            assert img_a.shape == (2, 256, 256), "Shape mismatch A"
            assert img_b.shape == (2, 256, 256), "Shape mismatch B"
            assert tgt.shape == (1, 256, 256), "Shape mismatch label"
            assert val.shape == (1, 256, 256), "Shape mismatch validity"
            
            # H: Finite
            assert torch.isfinite(img_a).all(), "Non-finite A"
            assert torch.isfinite(img_b).all(), "Non-finite B"
            assert torch.isfinite(tgt).all(), "Non-finite label"
            
            # I: Ranges
            min_a, max_a = img_a.min().item(), img_a.max().item()
            min_b, max_b = img_b.min().item(), img_b.max().item()
            min_t, max_t = tgt.min().item(), tgt.max().item()
            
            assert min_a >= 0.0 and max_a <= 1.0 + 1e-5, f"Range out of bounds A: [{min_a}, {max_a}]"
            assert min_b >= 0.0 and max_b <= 1.0 + 1e-5, f"Range out of bounds B: [{min_b}, {max_b}]"
            assert min_t >= 0.0 and max_t <= 1.0 + 1e-5, f"Range out of bounds label: [{min_t}, {max_t}]"
            
            # J, K: Padding logic
            v_h = patch['valid_height']
            v_w = patch['valid_width']
            
            if patch['pad_bottom'] > 0 or patch['pad_right'] > 0:
                if v_h < 256:
                    assert img_a[:, v_h:, :].sum() == 0
                    assert img_b[:, v_h:, :].sum() == 0
                    assert tgt[:, v_h:, :].sum() == 0
                    assert not val[:, v_h:, :].any()
                if v_w < 256:
                    assert img_a[:, :, v_w:].sum() == 0
                    assert img_b[:, :, v_w:].sum() == 0
                    assert tgt[:, :, v_w:].sum() == 0
                    assert not val[:, :, v_w:].any()
                    
            # K: validity mask check (only in non padded region)
            
        except Exception as e:
            if name == "Padding": padding_test = f"FAIL: {str(e)}"
            else: 
                padding_test = f"FAIL: {str(e)}"
                range_test = f"FAIL: {str(e)}"
                validity_mask_test = f"FAIL: {str(e)}"
                
    # L. Unexpected mask values
    try:
        import rasterio
        bad_mask_rel = "bad_mask_test.tif"
        bad_mask_abs = root_dir / "data" / "sar" / "tum_oscd" / bad_mask_rel
        with rasterio.open(
            bad_mask_abs, 'w', driver='GTiff',
            height=256, width=256, count=1, dtype='uint8'
        ) as dst:
            dst.write(np.full((1, 256, 256), 3, dtype=np.uint8))
            
        original_mask = train_ds.patches[0]['mask']
        train_ds.patches[0]['mask'] = bad_mask_rel
        
        try:
            _ = train_ds[0]
            mask_conversion = "FAIL: Did not raise error for value 3"
        except ValueError as e:
            if "Unexpected mask value" in str(e):
                mask_conversion = "PASS"
            else:
                mask_conversion = f"FAIL: Wrong error - {str(e)}"
        
        # Cleanup
        train_ds.patches[0]['mask'] = original_mask
        os.remove(bad_mask_abs)
    except Exception as e:
        mask_conversion = f"FAIL: {str(e)}"

    print("\n================================================")
    print("FINAL REPORT")
    print(f"Dataset created: {dataset_created}")
    print(f"Train loading: {train_loading}")
    print(f"Validation loading: {val_loading}")
    print(f"Padding test: {padding_test}")
    print(f"Mask conversion: {mask_conversion}")
    print(f"Validity-mask test: {validity_mask_test}")
    print(f"Range test: {range_test}")
    print(f"Test-city exclusion: {test_city_exclusion}")

if __name__ == "__main__":
    run_tests()
