"""Descarga y verifica el ZIP de Mendeley Data, versión 1."""
from hashlib import file_digest
from pathlib import Path
from urllib.request import urlopen
import shutil

URL = ("https://data.mendeley.com/public-files/datasets/mzb4b6dff3/files/"
       "8d63eebb-fdcb-4d8e-b18c-683a9f570c82/file_downloaded")
SHA256 = "427fec9d689a22856c51c7773adc6024158d5b228e36b2c9b58a13ce9af41c8c"
ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "data/m2_conduccion/raw/afridi.zip"


def download():
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    partial = ARCHIVE.with_suffix(".zip.part")
    if not ARCHIVE.exists() and not (partial.exists() and partial.stat().st_size == 2556850881):
        with urlopen(URL, timeout=90) as response, partial.open("wb") as file:
            shutil.copyfileobj(response, file, length=1024 * 1024)
    candidate = ARCHIVE if ARCHIVE.exists() else partial
    with candidate.open("rb") as file:
        if file_digest(file, "sha256").hexdigest() != SHA256:
            raise ValueError(f"El SHA-256 de {candidate} difiere del publicado en Mendeley.")
    if candidate == partial:
        partial.replace(ARCHIVE)
    return ARCHIVE


if __name__ == "__main__":
    print(download())
