# Introduction

The advance of high-throughput whole-genome mapping methods for the three-dimensional (3D) genome organization such as Hi-C has revealed multi-scale structures of chromatin folding within the cell nucleus, including A/B compartments, subcompartments, topologically associating domains (TADs), and chromatin loops.
These structures play critical roles in gene regulation, cellular development, and disease progression.
However, the cell-to-cell variation of 3D genome structures and their functional significance remain poorly understood.
Recent developments in single-cell Hi-C (scHi-C) technologies allow us to explore chromatin interactions with unparalleled detail, ranging from a few cells of specific types to thousands of cells from complex tissues.
Emerging co-assayed technologies enabled the profiling and joint analysis of multiple modalities of complex tissues at the same time.
These emerging technologies and datasets hold the potential to reveal how genome structure relates to function in single cells across various biological settings, both in health and disease.
These new technologies and datasets hold the potential to unveil the structure-function connections of the genome for a wide range of biological contexts in health and disease.

However, computational methods that can effectively utilize Hi-C data to reveal the roles of DNA sequence and 3D genome structure in transcriptional regulations are significantly lacking.
Recently, predictive neural networks have been developed to understand the effect of coding and non-coding DNA sequences on transcriptomes, such as DeepSEA, Basenji2, ExPecto, and Enformer.
Methodologically, these neural networks take a sequence as input and cannot directly incorporate a 2D Hi-C contact map as part of the input.
Conceptually, the DNA sequence, which is the sole input to these algorithms, is shared for all cells from the same biological context, and these algorithms generate cell-specific predictions by including cell-specific model parameters.
As a result, these algorithms' generalizability to unseen cells is far from ideal.
Besides, the feature extraction modules (e.g., convolutional layers and transformer layers) are largely shared for all cells, obscuring the interpretation of the cell-to-cell variability in the model reasoning.
Therefore, new algorithms are urgently needed to fill these important gaps.

Here, we present Hi-CFormer, a new computational method for understanding the intricate interplay of DNA sequence, 3D genome structure, and transcriptome using large language models.
We formulate this goal as predicting mRNA signals from DNA sequence and 3D genome structure.
On a mouse brain dataset, we show that the superior predictive performance of Hi-CFormer over sequence-only baselines.
The interpretation of the trained Hi-CFormer model demonstrates its ability to capture cell-type-specific interaction among DNA sequence, 3D genome structure, and transcriptome.
Hi-CFormer has the potential to shed new light on the functions of DNA sequence and 3D genome structure on transcriptional regulations.

# Overview of Hi-CFormer

![figures/fig1.png](https://github.com/tianmingzhou/SeqHiC2RNA/blob/main/figures/fig1.png)

Hi-CFormer predicts mRNA signals from DNA sequence and 3D genome structure.
Hi-CFormer requires two types of input for each sample: a 409,600 bp-long DNA sequence and a Hi-C contact map for that genomic region at 1,024-bp resolution.
Hi-CFormer predicts the mRNA signals, i.e., the normalized number of transcripts for each 1,024-bp genomic locus.
Although the DNA sequence is shared across all cells from the same biological context, Hi-CFormer learns the variability among cell types from the Hi-C information.
As the entire Hi-CFormer model is shared across all cell types, i.e., there are no cell-type-specific model parameters, Hi-CFormer is able to generalize to unseen cell types.

As a proof of principle, we apply Hi-CFormer to pseudo-bulk data at the cell type level on a GAGE-seq dataset from mouse brains containing 28 cell types and 3740 highly variable genes [[link to paper]]([https://doi.org/10.1038/s41588-022-01256-z](https://www.nature.com/articles/s41588-024-01745-3)).
We construct one sample centered at the transcription start site (TSS) of each highly variable for every cell type, resulting in 104,720 samples in total.
