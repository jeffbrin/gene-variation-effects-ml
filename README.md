# Gene Variation Classifier

A machine learning pipeline aiming to identify and classify variations in genes to identify pathogenic or benign variations.

## What does this repository contain?
This repository has many files which aren't laid out here, but the key files for running and understanding the dataset are outlined below.

```markdown
├── gene_variation_effects          <- Contains files and classes used in the notebooks.
|  └── dataset.py                   <- Cleans data from a specified data folder and creates the input for the pipeline
|  └── model                        <- Contains classes used for training
|       └── modelarchitectures.py   <- Our custom neural network class
|       └── pipelines.py            <- Our custom data pipeline
|       └── train.py                <- Contains the training loop
|
├── models                          <- Contains weights for our different models
|  └── BRCA1model.pth               <- The model used to classify variations exclusively found on the BRCA1 gene
|  └── model.pth                    <- The model used to classify variations across genes
|
├── notebooks                       <- Contains the notebooks used for training and analysis
|  └── main.ipynb                   <- Contains the training and analysis for variation classification across genes
|  └── BRCA1.ipynb                  <- Contains the training and analysis for variation classification exclusively found in BRCA1
|
├── references                      <- Contains reference files
|  └── dict.txt                     <- Data dictionary for the ClinVar variations dataset
```

## Data Processing

### Dataset

We trained our neural networks on a relatively small number of features due to the limited number of seemingly meaningful fields in the ClinVar dataset. See [Dataset](#dataset) for the specific datasets used. These fields are 
- Type: Indicates the type of variation (indel, deletion, insertion, ...)
- (Cross-Gene Only) GeneSymbol: Indicates which gene this variation was found in.
- VariantLength: The number of nucleotide bases affected by this variation.
- VariantLengthDifference: The difference between the length of the alternate allele, and the reference allele.
- OriginGermline: Calculated field indicating whether the origin of the variation of germline. The alternative (negative) is that the origin is somatic.
- (BRCA1 Only) PhyloScore: The conservation score for this variation sourced from the PhyloP 100 way bigwig file.
- (BRCA1 Only) ConvervationDisruption: Calculated field capturing the significance of a disruption to a highly conserved gene section. ConvervationDisruption is the product of PhyloScore and VariantLength.
- (BRCA1 Only) DistanceFromEnd: The shortest distance from the start or end of the variation to the start or end of the gene. This is potentially insightful since variations at the start or end of genes tend to be more pathogenic.
- (BRCA1 Only) RelativeStart: The distance from the start of the gene to the start of the variation (normalized to the size of the gene)

## Findings
Despite the limited number of fields, and their relatively low correlation with pathogenicity -- with our most correlated numeric features being OriginGermline for general variations with a -0.17 correlaton, and PhyloScore with a 0.285 correlation for BRCA1 variations.

TODO: What is correlation? What are the values from df.corr()?
### General Variation Classification
Findings

![image](figures/image.png)

### BRCA1 Variation Classification
Findings

![image](figures/image.png)

## Future Work

## References
...

## License
This project is licensed under the MIT License.

## Dataset 
Source: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/ <br>
File: variant_summary.txt.gz

Source: https://hgdownload.cse.ucsc.edu/goldenpath/hg38/phyloP100way/ <br>
File hg38.phyloP10way.bw
