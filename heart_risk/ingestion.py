"""
Pulls the Cleveland subset of the UCI Heart Disease dataset and writes a
clean, analysis-ready CSV to disk.

Implemented as a small class (``ClevelandHeartDiseaseIngestor``) rather
than a bag of functions, mainly so the source URL / output path / column
names are grouped together as configuration instead of scattered module
constants, and so a notebook or test can swap in a different URL (e.g. a
cached mirror) by just instantiating with different arguments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]


@dataclass
class ClevelandHeartDiseaseIngestor:
    """Fetches + lightly cleans the Cleveland processed heart-disease file.

    Usage:
        ingestor = ClevelandHeartDiseaseIngestor()
        df = ingestor.run()
    """

    source_url: str = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "heart-disease/processed.cleveland.data"
    )
    output_path: Path = field(default_factory=lambda: Path("data/raw/heart_disease.csv"))
    request_timeout: int = 20

    def fetch_raw_text(self) -> str:
        logger.info("Fetching Cleveland dataset from %s", self.source_url)
        response = requests.get(self.source_url, timeout=self.request_timeout)
        response.raise_for_status()
        return response.text

    def parse(self, raw_text: str) -> pd.DataFrame:
        """UCI encodes missing values as literal '?' characters; the
        original 'num' column is a 0-4 severity score that we collapse
        into a binary 'target' (0 = no disease, 1 = any disease)."""
        frame = pd.read_csv(StringIO(raw_text), header=None, names=RAW_COLUMNS, na_values="?")
        frame["target"] = (frame["num"] > 0).astype(int)
        return frame.drop(columns=["num"])

    def run(self) -> pd.DataFrame:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            raw_text = self.fetch_raw_text()
            frame = self.parse(raw_text)
        except (requests.RequestException, ValueError) as exc:
            logger.error("Dataset fetch/parse failed: %s", exc)
            raise SystemExit(
                f"Could not retrieve the dataset from {self.source_url}. "
                f"Download it manually and save it to {self.output_path}."
            ) from exc

        frame.to_csv(self.output_path, index=False)
        logger.info("Wrote %d rows x %d columns to %s", *frame.shape, self.output_path)
        return frame


def main() -> int:
    ingestor = ClevelandHeartDiseaseIngestor()
    frame = ingestor.run()
    logger.info("Target distribution:\n%s", frame["target"].value_counts(normalize=True).round(3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
