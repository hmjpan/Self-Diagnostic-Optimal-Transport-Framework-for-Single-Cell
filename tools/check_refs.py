"""Verify LaTeX references."""
import re
with open('paper/main.tex', 'r', encoding='utf-8') as f:
    text = f.read()

labels = re.findall(r'\\label\{([^}]+)\}', text)
refs = re.findall(r'\\ref\{([^}]+)\}', text)
cites_raw = re.findall(r'\\cite\{([^}]+)\}', text)
bibitems = re.findall(r'\\bibitem\{([^}]+)\}', text)

all_cites = set()
for c in cites_raw:
    for ci in c.split(','):
        all_cites.add(ci.strip())

missing_refs = [r for r in refs if r not in labels]
missing_cites = [c for c in all_cites if c not in bibitems]
dupe_labels = [l for l, n in __import__('collections').Counter(labels).items() if n > 1]

print(f"Labels: {len(labels)}, Refs: {len(refs)}, Cites: {len(all_cites)}, Bibitems: {len(bibitems)}")
if dupe_labels: print(f"DUPLICATE LABELS: {dupe_labels}")
else: print("Labels: all unique")
if missing_refs: print(f"MISSING REFS: {missing_refs}")
else: print("Refs: all resolved")
if missing_cites: print(f"MISSING CITES: {missing_cites}")
else: print("Cites: all resolved")
