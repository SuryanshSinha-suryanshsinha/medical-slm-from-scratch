import urllib.request, gzip, os
import xml.etree.ElementTree as ET

os.makedirs("data/raw", exist_ok=True)
abstracts = []

base_url = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"

for i in range(1, 36):
    fname_gz = f"pubmed26n{i:04d}.xml.gz"
    url = base_url + fname_gz
    local = f"data/raw/{fname_gz}"
    print(f"Downloading file {i}/35: {fname_gz}")
    try:
        urllib.request.urlretrieve(url, local)
        print(f"  parsing...")
        with gzip.open(local, 'rb') as f:
            tree = ET.parse(f)
            for article in tree.findall('.//AbstractText'):
                if article.text and len(article.text) > 150:
                    abstracts.append(article.text.strip())
        os.remove(local)
        print(f"  abstracts so far: {len(abstracts):,}")
    except Exception as e:
        print(f"  failed: {e}")

with open("data/raw/pubmed_abstracts.txt", "w", encoding="utf-8") as f:
    for a in abstracts:
        f.write(a + "\n")

print(f"\nDone! {len(abstracts):,} abstracts saved")
print(f"Size: {os.path.getsize('data/raw/pubmed_abstracts.txt') / 1e6:.1f} MB")