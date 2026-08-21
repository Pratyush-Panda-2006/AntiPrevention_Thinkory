import ftplib
import os
import zipfile
import rasterio
from pathlib import Path
import json
import time

def download_with_resume(ftp, filename, local_path):
    max_retries = 10
    for attempt in range(max_retries):
        try:
            local_size = 0
            if local_path.exists():
                local_size = os.path.getsize(local_path)
            
            ftp.voidcmd('TYPE I')
            remote_size = ftp.size(filename)
            print(f"[{filename}] Remote size: {remote_size}, Local size: {local_size}")
            
            if local_size >= remote_size:
                print(f"[{filename}] Download complete.")
                return True
                
            print(f"[{filename}] Resuming from byte {local_size}...")
            with open(local_path, "ab") as f:
                ftp.retrbinary(f"RETR {filename}", f.write, rest=local_size)
                
            # Double check size
            if os.path.getsize(local_path) >= remote_size:
                print(f"[{filename}] Download complete.")
                return True
                
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)
            try:
                # Reconnect if connection dropped
                ftp.quit()
            except:
                pass
            print("Reconnecting...")
            ftp.connect("dataserv.ub.tum.de")
            ftp.login("m1619966", "m1619966")
            
    print(f"Failed to download {filename} after {max_retries} attempts.")
    return False

def download_and_inspect():
    host = "dataserv.ub.tum.de"
    user = "m1619966"
    passwd = "m1619966"
    dest_dir = Path("data/sar/tum_oscd")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print("Connecting to FTP...")
    ftp = ftplib.FTP(host)
    ftp.login(user, passwd)
    
    files = ftp.nlst()
    
    downloaded_files = []
    
    for f in files:
        if f in ['.', '..'] or not f.endswith('.zip'):
            continue
            
        local_path = dest_dir / f
        success = download_with_resume(ftp, f, local_path)
        if success:
            downloaded_files.append(local_path)
    
    try:
        ftp.quit()
    except:
        pass
    
    print("\n--- Inspecting Archives ---")
    
    report = {
        "files": [],
        "sample_info": {}
    }
    
    for archive_path in downloaded_files:
        print(f"\nInspecting {archive_path.name}")
        with zipfile.ZipFile(archive_path, 'r') as zf:
            namelist = zf.namelist()
            print(f"Total files in archive: {len(namelist)}")
            
            dirs = set([p.split('/')[1] if p.startswith('OSCD/') else p.split('/')[0] for p in namelist if '/' in p])
            print("Top level directories/cities:", list(dirs)[:10])
            
            report["files"].extend(namelist[:10]) 
            
            tif_files = [f for f in namelist if f.endswith('.tif')]
            if tif_files:
                sample_tif = tif_files[0]
                print(f"Extracting sample: {sample_tif}")
                zf.extract(sample_tif, path=dest_dir / "temp_extract")
                
                sample_path = dest_dir / "temp_extract" / sample_tif
                with rasterio.open(sample_path) as ds:
                    print(f"Sample CRS: {ds.crs}")
                    print(f"Sample Dimensions: {ds.width}x{ds.height}")
                    print(f"Sample Bands: {ds.count}")
                    print(f"Sample Dtype: {ds.dtypes[0]}")
                    
                    arr = ds.read()
                    print(f"Stats: min={arr.min()}, max={arr.max()}, mean={arr.mean():.4f}")
                    
                    report["sample_info"] = {
                        "name": sample_tif,
                        "crs": str(ds.crs),
                        "width": ds.width,
                        "height": ds.height,
                        "bands": ds.count,
                        "dtype": ds.dtypes[0],
                        "min": float(arr.min()),
                        "max": float(arr.max()),
                        "mean": float(arr.mean())
                    }
                
                import shutil
                shutil.rmtree(dest_dir / "temp_extract")
                
    with open(dest_dir / "inspection_summary.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print("Inspection complete.")

if __name__ == "__main__":
    download_and_inspect()
