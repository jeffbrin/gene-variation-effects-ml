from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer


import pandas as pd
import numpy as np

from config import DATA_DIR


def generate_bed_format_variant_dataset(input_path: Path, output_path: Path):
    logger.info("Generating evolution API input data")

    df = pd.read_csv(input_path, sep = "\t", low_memory = False)
    df.dropna(axis="index", inplace=True)
    bed_df = df[["Type", "GeneSymbol", "Chromosome", "Start", "Stop", "ReferenceAlleleVCF", "AlternateAlleleVCF"]]
    bed_df.to_csv(output_path, sep='\t', index=False)


def generate_main_dataset(input_path: Path, output_path: Path):
    logger.info("Generating raw data")

    df = pd.read_csv(input_path, sep = "\t", low_memory = False)

    logger.info("Cleaning raw data")

    useless_columns = ['#AlleleID', 'GeneID', 'Name', 'HGNC_ID', 'RS# (dbSNP)', 'nsv/esv (dbVar)', 'RCVaccession',
                       'PhenotypeIDS', 'ReferenceAllele', 'AlternateAllele', 'Guidelines', 'OtherIDs',
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
    df['OriginGermline'] = (df['OriginSimple'] == "germline").astype(int)

    df.drop(columns = ['Start', 'Stop', "OriginSimple"], inplace = True)
    df.drop(columns = ['AlternateAlleleVCF', 'ReferenceAlleleVCF'], inplace = True)

    # nans are destroying out output
    df.dropna(axis="index", inplace=True)
    df.to_csv(output_path, index=False)

    logger.success("Features generation complete.")

def generate_BRCA1_dataset(variants_filepath: Path, phylo_scores_filepath: Path, output_path: Path) -> None:

    logger.info("Generating BRCA1 data")

    df = pd.read_csv(variants_filepath, sep = "\t", low_memory = False)

    logger.info("Cleaning raw data")

    columns = ["Type", "ClinSigSimple", "Chromosome", "Start", "Stop", "ReferenceAlleleVCF", "AlternateAlleleVCF", "OriginSimple"]

    df = df[columns]
    df = df.loc[df['ClinSigSimple'] != -1]
    df = df.replace(['na', '-', -1], np.nan)
    df['VariantLength'] = df['Stop'] - df['Start']
    df['VariantLengthDifference'] = [
        len(alt) - len(ref) if not any(pd.isna([alt, ref])) else np.nan 
        for _, (alt, ref) in df[['AlternateAlleleVCF', 'ReferenceAlleleVCF']].iterrows()
        ]
    phylo_scores_df = pd.read_csv(phylo_scores_filepath)
    phylo_scores_df['Chromosome'] = phylo_scores_df['Chromosome'].astype(str)
    phylo_scores_df = phylo_scores_df[["Chromosome", "Start", "Stop", "PhyloScore", "Type"]]

    df = df.merge(phylo_scores_df, on=['Chromosome', "Start", "Stop", "Type"], how="inner")

    gene_start_position_hg38 = 43044292
    gene_end_position_hg38 = 43170245
    gene_sequence_length = gene_start_position_hg38 - gene_end_position_hg38 + 1

    # Convert start into relative positon
    df['RelativeStart'] = (df['Start'] - gene_start_position_hg38) / gene_sequence_length
    df['ConservationDisruption'] = df['PhyloScore'] * df['VariantLength']
    df['DistanceFromEnd'] = [min(start-gene_start_position_hg38, gene_end_position_hg38-end) for start, end in zip(df['Start'], df['Stop'])]
    df['OriginGermline'] = (df['OriginSimple'] == "germline").astype(int)

    # nans are destroying out output
    df.drop(columns = ['Start', 'Stop', "Chromosome", "OriginSimple", 'AlternateAlleleVCF', 'ReferenceAlleleVCF'], inplace = True)
    df.dropna(axis="index", inplace=True)
    df.to_csv(output_path, index=False)

    logger.success("Features generation complete.")

def main(input_path: Path = DATA_DIR / "variant_summary.txt",
         output_path: Path = DATA_DIR / "dataset.csv"
         ):
    generate_BRCA1_dataset(input_path, DATA_DIR / "BRCA1_phylo_scores.csv", DATA_DIR / "BRCA1_dataset.csv")
    # generate_bed_format_variant_dataset(input_path, DATA_DIR / "variants.csv")
    # generate_main_dataset(input_path, output_path)


if __name__ == "__main__":
    main()