from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import pandas as pd
import numpy as np

from config import DATA_DIR

app = typer.Typer()


@app.command()
def main(input_path: Path = DATA_DIR / "variant_summary.txt",
         output_path: Path = DATA_DIR / "dataset.csv"
         ):
    logger.info("Generating raw data")

    df = pd.read_csv(input_path, sep = "\t", low_memory = False)

    logger.info("Cleaning raw data")

    useless_columns = ['#AlleleID', 'GeneID', 'Name', 'HGNC_ID', 'RS# (dbSNP)', 'nsv/esv (dbVar)', 'RCVaccession',
                       'PhenotypeIDS', 'OriginSimple', 'ReferenceAllele', 'AlternateAllele', 'Guidelines', 'OtherIDs',
                       'VariationID', 'SomaticClinicalImpactLastEvaluated', 'SomaticClinicalImpact', 'Oncogenicity', 
                       'OncogenicityLastEvaluated', 'ReviewStatusOncogenicity', 'SCVsForAggregateGermlineClassification', 'ReviewStatus',
                       'SCVsForAggregateGermlineClassification', 'SCVsForAggregateSomaticClinicalImpact', 'SCVsForAggregateOncogenicityClassification', 'SubmitterCategories',
                       'ClinicalSignificance', 'LastEvaluated', 'PhenotypeList', 'Assembly', 'NumberSubmitters', 'TestedInGTR', 'ReviewStatusClinicalImpact', 'Origin',
                       'ChromosomeAccession', 'Cytogenetic', 'Chromosome', 'PositionVCF'
                       ]

    df.drop(columns = useless_columns, inplace = True)
    df = df.loc[df['ClinSigSimple'] != -1]
    df = df.replace(['na', '-', -1], np.nan)
    df['VariantLength'] = df['Stop'] - df['Start']
    df['VariantLengthDifference'] = [
        len(alt) - len(ref) if not any(pd.isna([alt, ref])) else np.nan 
        for _, (alt, ref) in df[['AlternateAlleleVCF', 'ReferenceAlleleVCF']].iterrows()
        ]
    df.drop(columns = ['Start', 'Stop'], inplace = True)
    df.drop(columns = ['AlternateAlleleVCF', 'ReferenceAlleleVCF'], inplace = True)
    df.to_csv(output_path)

    logger.success("Features generation complete.")


if __name__ == "__main__":
    app()
    main()