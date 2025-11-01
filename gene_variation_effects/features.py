from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer

import pandas as pd
import numpy as np

from config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()

def main(
    input_path: Path = PROCESSED_DATA_DIR / "variant_summary.txt",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv"
    ):
    pass

if __name__ == "__main__":
    app()
