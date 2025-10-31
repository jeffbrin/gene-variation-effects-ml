# Notes:


## For cleaning:
- GeneID and GeneSymbol seem redundant. Remove GeneID and extract Gene with multiple symbols
- GeneName to be removed. Seems like a way to name the gene and summarizes features we already have columns for
- ClinicalSignificance to be removed. Details of the evaluation of the gene. Data leakage
- LastEvaluated to be removed
- HGNC_ID to be removed. Says if there is more than one GeneID. Redundant
- RS# (dbSNP) to be removed. Seems like an ID for another DB. Almost all are missing
- nsv/esv (dbVar) to be removed. Same thing as ^^
- RCVaccession to be removed. Seems like redundant information Gene, ClinicalSignificance etc. 
- PhenotypeIDS and PhenotypeList seems to have a bunch of missing. ID seems to say no missing but List seems to indicate most are missing
- Origin seems like more granular than OriginSimple. Remove OriginSimple?
- Chromosome indicates in which chromosome this variation was found in. 1-22 for the first 22. chromsomes, "X" or "Y" for the sex chromosomes, and "na" for incomplete data. Removed since genes normally occur in only one chromosome, so this is redundant with gene.
- ReferenceAllele and AlternateAllele to be removed. Almost all missing
- Guidelines to be removed
- TestedInGTR idk if it should be removed. 98 % are No and 2 % are yes. Jeff thoughts?
- OtherIDs to be removed
- SubmitterCategories idk what to extract of this but we keep?
- VariationID to be removed
- PositionVCF has a shit load of -1, don't know whats up with that. To be removed, seems like this is mostly useful to derive other features we already have (gene). It represents the distance from the start of the chromosome.
- SomaticClinicalImpact to be removed. All missing
- SomaticClinicalImpactLastEvaluated to be removed. All missing
- Oncogenecity to be removed. All missing
- OncogenicityLastEvaluated to be removed. All missing
- ReviewStatusOncogenicity to be removed. All missing
- remove all the ones left
- Cytogenetic to be removed. Basically just a human-readable, less granular "start".


Saw that there were missing (but didnt check extremely in depth):
- GeneSymbol ('-')
- Assembly('na')
- Cytogenetic ('-')
- ReferenceAlleleVCF ('na')
- AlternativeAlleleVCF ('na')
- ChromosomeAccession ('na')
- Chromosome ('na')
- ReferenceAlleleVCF ('na')
- Start, Stop, PositionVCF ('-')
- AlternateAlleleVCF ('na')

Ideas of transformations:
- Type (cat): one hot encoding
- GeneSymbol (cat): LabelEncoder (assigning a number per category) and Add an embedding layer in the neural network
- ChromosomeAccession (cat): one hot encoding (might be too memory intensive idk tho)
- Chromosome (cat): one hot encoding
- Cytogenetic (cat): LabelEncoder and Add an embedding layer in the neural network
*** Start, Stop and PositionVCF have to be transformed I think. Length of the sequence depends on the chromosome and Assembly. 
The start and stop are relative and so maybe we use the length of the variant instead of start and stop. For positionVCF, not
exactly sure what should be done. For now, I'm excluding PositionVCF and taking the length. ***
- VariantLength (Int): min max normalization
- ReferenceAlleleVCF (cat): LabelEncoder and Add an embedding layer in the neural network
- AlternateAlleleVCF (cat): LabelEncoder and Add an embedding layer in the neural network

## Potentially useful future features
- Normalize start with the gene's length to get a start position between 0-1. This might be able to teach the model to identify "hotspots" within the gene if a particular region is more pathogen-inducing. Should be able to get length from here: https://www.ensembl.org/biomart/martview/f472cfd32e9956359032f89e96854f40
- Look at context around the variation, apparently this can make the reference -> alternate alleles more relevant.
