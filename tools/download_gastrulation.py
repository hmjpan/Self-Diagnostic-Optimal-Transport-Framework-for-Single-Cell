"""Download mouse gastrulation atlas data (Pijuan-Sala et al. 2019).
Real developmental time points: E6.5-E8.5.
This is a non-hematopoietic system - embryonic development."""
import requests, gzip, io, os, time, sys
from pathlib import Path

# Data directory relative to project root
data_dir = Path(__file__).parent.parent / 'data'
data_dir.mkdir(parents=True, exist_ok=True)
os.chdir(str(data_dir))

base = 'https://content.cruk.cam.ac.uk/jmlab/atlas_data/'

def download_with_retry(url, fname, max_retries=3, chunk_size=8192):
    """Download with retry logic and progress reporting."""
    for attempt in range(max_retries):
        try:
            print(f"  Connecting to {url[:80]}...")
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, 
                           stream=True, timeout=120)
            if r.status_code == 200:
                total = int(r.headers.get('Content-Length', 0))
                downloaded = 0
                chunks = []
                start = time.time()
                for chunk in r.iter_content(chunk_size=chunk_size):
                    chunks.append(chunk)
                    downloaded += len(chunk)
                    if total > 0 and downloaded % (10*1024*1024) < chunk_size:
                        pct = downloaded / total * 100
                        elapsed = time.time() - start
                        speed = downloaded / (elapsed + 1) / 1024 / 1024
                        print(f"    {pct:.0f}% ({downloaded/1e6:.1f}MB, {speed:.1f} MB/s)")
                
                data = b''.join(chunks)
                Path(fname).write_bytes(data)
                print(f"  Downloaded: {len(data):,} bytes -> {fname}")
                return data
            else:
                print(f"  HTTP {r.status_code}")
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {type(e).__name__}: {str(e)[:80]}")
            if attempt < max_retries - 1:
                time.sleep(5)
    return None

# Step 1: Download metadata (smaller file, verify time labels)
print("[1] Downloading metadata...")
meta_data = download_with_retry(base + 'meta.tab.gz', 'gastrulation_meta.tab.gz')
if meta_data:
    meta_text = gzip.decompress(meta_data).decode()
    lines = meta_text.strip().split('\n')
    header = lines[0].split('\t')
    print(f"  Metadata: {len(lines)} lines, {len(header)} columns")
    
    # Find time/stage column
    time_cols = [c for c in header if any(t in c.lower() for t in ['stage','time','day','age','somite'])]
    print(f"  Time columns: {time_cols}")
    
    # Print unique stage values
    for col in time_cols[:3]:
        idx = header.index(col)
        vals = set()
        for line in lines[1:]:
            parts = line.split('\t')
            if idx < len(parts):
                vals.add(parts[idx].strip())
        print(f"    {col}: {sorted(vals)[:20]}")
    
    # Step 2: Try to download the count matrix
    print("\n[2] Attempting count matrix download...")
    print("  File: raw_counts.mtx.gz (1.4GB)")
    print("  This may take a while...")
    
    # Try to download with a very long timeout
    count_data = download_with_retry(base + 'raw_counts.mtx.gz', 
                                      'gastrulation_counts.mtx.gz', 
                                      max_retries=1, chunk_size=1024*1024)
    if count_data:
        print(f"  SUCCESS: {len(count_data):,} bytes")
    else:
        print("  Count matrix download failed (timeout/network)")
        print("  Using metadata alone to document dataset availability")
        # Write metadata summary
        summary = {
            'dataset': 'Pijuan-Sala 2019 mouse gastrulation',
            'access': 'https://content.cruk.cam.ac.uk/jmlab/atlas_data/',
            'cells': len(lines) - 1,
            'time_points': {col: sorted(list(set(
                line.split('\t')[header.index(col)].strip() 
                for line in lines[1:] if header.index(col) < len(line.split('\t'))
            ))) for col in time_cols[:3]},
            'status': 'metadata_downloaded_counts_pending'
        }
        import json
        with open('gastrulation_info.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print("  Saved metadata summary to gastrulation_info.json")
else:
    print("  Metadata download failed - gastrulation data not accessible")
    print("  Documenting as attempted but unavailable")
