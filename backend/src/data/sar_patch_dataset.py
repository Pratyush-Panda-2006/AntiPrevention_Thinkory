import json
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset
from pathlib import Path
import sys

# Ensure src is accessible
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.preprocessing.sar_loader import normalize_sar_tensor

class TUMSARChangeDetectionDataset(Dataset):
    """
    PyTorch Dataset for SAR Change Detection on the TUM OSCD dataset.
    Consumes the deterministically generated patch index.
    """
    def __init__(self, patch_index_path, split, root_dir=None):
        if split not in ('train', 'validation'):
            raise ValueError(f"Split must be 'train' or 'validation', got {split}. 'test' is not permitted.")
            
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        
        with open(patch_index_path, 'r') as f:
            index_data = json.load(f)
            
        self.patches = [p for p in index_data['patches'] if p['split'] == split]
        
    def __len__(self):
        return len(self.patches)
        
    def __getitem__(self, idx):
        patch = self.patches[idx]
        
        x = patch['x']
        y = patch['y']
        v_h = patch['valid_height']
        v_w = patch['valid_width']
        patch_size = patch['patch_size']
        
        t1_path = self.root_dir / "data" / "sar" / "tum_oscd" / patch['t1_image']
        t2_path = self.root_dir / "data" / "sar" / "tum_oscd" / patch['t2_image']
        mask_path = self.root_dir / "data" / "sar" / "tum_oscd" / patch['mask']
        
        window = Window(x, y, v_w, v_h)
        
        # Open per item to be worker-safe
        with rasterio.open(t1_path) as src_t1:
            t1_arr = src_t1.read(window=window)
            
        with rasterio.open(t2_path) as src_t2:
            t2_arr = src_t2.read(window=window)
            
        with rasterio.open(mask_path) as src_mask:
            mask_arr = src_mask.read(1, window=window)
            
        # 3. SAR Preprocessing
        # The TUM data is already in dB. We use the locked constants and finite checks.
        t1_norm, t1_valid = normalize_sar_tensor(t1_arr, is_linear=False, return_validity=True)
        t2_norm, t2_valid = normalize_sar_tensor(t2_arr, is_linear=False, return_validity=True)
        
        # 5. Mask Handling
        unique_vals = np.unique(mask_arr)
        for v in unique_vals:
            if v not in (1, 2):
                raise ValueError(f"Unexpected mask value {v} in city {patch['city']} at x={x}, y={y}")
                
        target_arr = np.zeros_like(mask_arr, dtype=np.float32)
        target_arr[mask_arr == 2] = 1.0
        
        target_arr = np.expand_dims(target_arr, axis=0)
        
        # 4. Validity Mask
        validity_mask = t1_valid & t2_valid
        validity_mask = np.expand_dims(validity_mask, axis=0)
        
        # 6. Padding
        if v_h < patch_size or v_w < patch_size:
            t1_padded = np.zeros((2, patch_size, patch_size), dtype=np.float32)
            t1_padded[:, :v_h, :v_w] = t1_norm
            
            t2_padded = np.zeros((2, patch_size, patch_size), dtype=np.float32)
            t2_padded[:, :v_h, :v_w] = t2_norm
            
            target_padded = np.zeros((1, patch_size, patch_size), dtype=np.float32)
            target_padded[:, :v_h, :v_w] = target_arr
            
            validity_padded = np.zeros((1, patch_size, patch_size), dtype=bool)
            validity_padded[:, :v_h, :v_w] = validity_mask
            
            t1_norm = t1_padded
            t2_norm = t2_padded
            target_arr = target_padded
            validity_mask = validity_padded
            
        # 7. Tensor Contract
        return {
            "image_a": torch.from_numpy(t1_norm).float(),
            "image_b": torch.from_numpy(t2_norm).float(),
            "label": torch.from_numpy(target_arr).float(),
            "valid_mask": torch.from_numpy(validity_mask).bool(),
            "city": patch['city'],
            "x": x,
            "y": y
        }
