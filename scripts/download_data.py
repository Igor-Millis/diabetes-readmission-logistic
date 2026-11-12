"""Download the UCI 'Diabetes 130-US Hospitals for Years 1999-2008' dataset (CC BY 4.0) into data/raw/."""
from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "diabetic_data.csv").exists():
        print("already downloaded:", OUT / "diabetic_data.csv")
        return
    print("downloading", URL)
    blob = urllib.request.urlopen(URL, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in ("diabetic_data.csv", "IDS_mapping.csv"):
            (OUT / name).write_bytes(z.read(name))
    digest = hashlib.md5((OUT / "diabetic_data.csv").read_bytes()).hexdigest()
    print("saved to", OUT, "| md5(diabetic_data.csv) =", digest)


if __name__ == "__main__":
    main()
