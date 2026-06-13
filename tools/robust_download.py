"""
ROBUST DOWNLOAD with resume, retry, and verification
Downloads gastrulation count matrix with integrity check.
"""
import requests, gzip, time, os, sys
from pathlib import Path

url = 'https://content.cruk.cam.ac.uk/jmlab/atlas_data/raw_counts.mtx.gz'
fname = Path('D:/lun2/math/data/gastrulation_counts.mtx.gz')
tmpname = Path(str(fname) + '.part')

def verify_gzip(filepath):
    """Check if a gzip file is complete and valid."""
    try:
        with gzip.open(filepath, 'rb') as f:
            # Read in chunks to verify integrity
            chunk_size = 10 * 1024 * 1024  # 10MB
            total = 0
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
        print(f"  Verified: {total/1e6:.0f}MB readable")
        return True
    except Exception as e:
        print(f"  Verify failed: {e}")
        return False

def download_with_retry(url, fname, max_retries=10):
    """Download with automatic retry on connection errors."""
    # Check if partial exists
    resume_pos = 0
    if tmpname.exists():
        resume_pos = tmpname.stat().st_size
        print(f"Resuming from {resume_pos/1e6:.0f}MB")
    
    for attempt in range(max_retries):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            if resume_pos > 0:
                headers['Range'] = f'bytes={resume_pos}-'
            
            print(f"\nAttempt {attempt+1}/{max_retries}")
            r = requests.get(url, headers=headers, stream=True, timeout=60)
            
            if r.status_code == 206:  # Partial content
                total = int(r.headers.get('Content-Range', '').split('/')[-1])
                print(f"  Resuming, total: {total/1e9:.2f}GB")
            elif r.status_code == 200:
                total = int(r.headers.get('Content-Length', 0))
                print(f"  Fresh download, total: {total/1e9:.2f}GB")
                resume_pos = 0
            else:
                print(f"  HTTP {r.status_code}, retrying...")
                time.sleep(5)
                continue
            
            mode = 'ab' if resume_pos > 0 else 'wb'
            downloaded = resume_pos
            start = time.time()
            
            with open(tmpname, mode) as f:
                for chunk in r.iter_content(chunk_size=5*1024*1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        elapsed = time.time() - start
                        speed = downloaded / (elapsed + 1) / 1e6
                        pct = downloaded / total * 100
                        eta = (total - downloaded) / (speed * 1e6 + 1)
                        print(f"\r  {pct:.1f}% ({downloaded/1e9:.2f}/{total/1e9:.2f}GB, "
                              f"{speed:.1f}MB/s, ETA {eta:.0f}s)", end='')
            
            print()  # newline
            print(f"  Download complete: {downloaded/1e9:.2f}GB in {elapsed:.0f}s")
            
            # Verify and rename
            print("  Verifying...")
            if verify_gzip(tmpname):
                tmpname.rename(fname)
                print(f"  SUCCESS -> {fname}")
                return True
            else:
                print("  Verification failed, will retry...")
                resume_pos = tmpname.stat().st_size if tmpname.exists() else 0
                
        except Exception as e:
            print(f"\n  Error: {type(e).__name__}: {str(e)[:100]}")
            if tmpname.exists():
                resume_pos = tmpname.stat().st_size
                print(f"  Will resume from {resume_pos/1e6:.0f}MB")
            time.sleep(10)
    
    print("\nFAILED: All retries exhausted.")
    return False

# Main
print("=" * 60)
print("DOWNLOADING: Mouse Gastrulation Count Matrix")
print("=" * 60)
print(f"URL: {url}")
print(f"Destination: {fname}")
print(f"Expected size: ~1.53GB")
print()

if fname.exists():
    print("File already exists. Verifying...")
    if verify_gzip(fname):
        print("File is valid. Done.")
        sys.exit(0)
    else:
        print("File corrupted. Re-downloading...")
        fname.unlink()

success = download_with_retry(url, fname)
if success:
    print("\nDone. File ready for analysis.")
else:
    print("\nDownload failed. Try again later or use aria2c:")
    print(f"  aria2c -o {fname} {url}")
