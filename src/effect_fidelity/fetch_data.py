"""Download any public raw dataset missing from data/raw (pure Python, works on macOS without curl/unzip).

  python -m effect_fidelity.fetch_data
"""
from __future__ import annotations
import urllib.request, zipfile
from .experiments import RAW, ML1, GSS, HILL

SOURCES = [
    # (file the code needs, url, zip to extract into)
    (ML1, "https://osf.io/download/nqg97/", RAW / "ml1"),                     # Many Labs 1 (CC0), OSF wx7ck
    (RAW / "ml1" / "Data" / "CleanedDataset.sav", "https://osf.io/download/nqg97/", RAW / "ml1"),  # optional SPSS copy (not used by the code)
    (GSS, "https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/2018_stata.zip", RAW),
    (HILL, "https://raw.githubusercontent.com/YashhMahajan/Causal-Decision-Intelligence-using-Uplift-Modeling/HEAD/datasets/"
           "phase%201%20-%20Main%20Development/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv", None),
]


def fetch(required_only: bool = False):
    RAW.mkdir(parents=True, exist_ok=True)
    for target, url, unzip_to in SOURCES:
        if target.exists() or (required_only and target.suffix == ".sav"):
            continue
        dl = RAW / ("_dl.zip" if unzip_to else target.name)
        print(f"downloading {target.relative_to(RAW.parent.parent)} ...", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "effect-fidelity"})
        with urllib.request.urlopen(req) as r, open(dl, "wb") as f:
            f.write(r.read())
        if unzip_to:
            with zipfile.ZipFile(dl) as z:
                z.extractall(unzip_to)
            dl.unlink()
        if not target.exists():
            print(f"  warning: {target.name} still missing after download")


if __name__ == "__main__":
    fetch()
    print("data ready")
