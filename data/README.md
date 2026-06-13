# Data Directory

This directory holds input data for the pipeline scripts.
All files are gitignored. Download them before running analyses.

## Auto-downloaded by scanpy/scvelo
These are downloaded automatically when running the pipeline scripts:
- `moignard15/` — Moignard et al. 2015 (sc.datasets.moignard15())
- `paul15/` — Paul et al. 2015 (sc.datasets.paul15())
- `Pancreas/` — Bastidas-Ponce et al. 2019 (scvelo.datasets.pancreas())

## Manual download required
These must be downloaded before running the corresponding scripts:

### Chu et al. 2016 (GSE75748)
URL: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE75nnn/GSE75748/suppl/GSE75748_sc_time_course_ec.csv.gz
Save as: `GSE75748_sc_time_course_ec.csv.gz`
Script: `scripts/run_chu2016.py`

### Pijuan-Sala et al. 2019 Mouse Gastrulation Atlas
URL: https://content.cruk.cam.ac.uk/jmlab/atlas_data/
Files needed:
- `raw_counts.mtx.gz` (~1.5GB)
- `meta.tab.gz`
- `genes.tsv.gz`
- `sizefactors.tab.gz`
Script: `scripts/run_gastrulation.py`
