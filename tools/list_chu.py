"""List GSE75748 supplementary files for download."""
import urllib.request, re
url = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE75nnn/GSE75748/suppl/'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
resp = urllib.request.urlopen(req, timeout=30)
html = resp.read().decode('utf-8', errors='ignore')
files = re.findall(r'href="([^"]+)"', html)
data = [f for f in files if not f.startswith('/') and not f.startswith('?')]
print('GSE75748 (Chu 2016) supplementary files:')
for f in data:
    print(f'  {f}')
