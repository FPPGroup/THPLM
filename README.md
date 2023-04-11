# THPLM
A sequence-based deep learning framework to predict protein stability changes upon point mutations using pretrained protein language model

## Table of Contents

- [THPLM](#thplm)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Dependencies](#dependencies)
    - [ESM pretrain model download](#ESM-pretrain-model-download)
    - [Installation](#Installation)
  - [Usage](#usage)
    - [For prediction](#for-prediction)
    - [Running](#Running)
  - [Citation](#citation)

## Installation

### Dependencies

```bash
conda env create -f environment.yml
```
### ESM pretrain model download

The pretain model used in THPLM is [**esm2_t36_3B_UR50D**](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50), it automatically downloaded to `~/.cache/torch/hub/checkpoints` based on the tutorial of [ESM](https://github.com/facebookresearch/esm).

### Installation

If the dependencies are satified, you can install the THPLM using following commands:

```bash
conda activate THPLM
git clone https://github.com/FPPGroup/THPLM.git
cd THPLM
```

## Usage

### For prediction

```bash
python THPLM_predict.py -h ## show the parameters
```
paramaters:

```bash
usage: THPLM_predict.py [-h] [--model_location MODEL_LOCATION]
                        [--gpunumber GPUNUMBER]
                        [--toks_per_batch TOKS_PER_BATCH] [--THPLM THPLM]
                        [--extractfile EXTRACTFILE] [--pythonbin PYTHONBIN]
                        variants_file fasta_file output_dir variants_fasta_dir

Extract mean representations and model outputs for sequences in a FASTA file
and to predict DDGs

positional arguments:
  variants_file         files inclusing variants, format is
                        <wildtype><position><mutation> (see README for models)
  fasta_file            FASTA file on which to extract representations
  output_dir            output directory for extracted representations
  variants_fasta_dir    FASTA file was used to store variant and wildtype protein

optional arguments:
  -h, --help            show this help message and exit
  --gpunumber GPUNUMBER
                        GPU number for use
  --toks_per_batch TOKS_PER_BATCH
                        maximum batch size
  --THPLM THPLM         maximum batch size
  --extractfile EXTRACTFILE
                        the path of extract.py file from esm-2
  --pythonbin PYTHONBIN
                        the path of python bin
```

THPLM input format:
```bash
  variants_file:
      A file like "./examples/var.txt", in every line contain one variant formated by <wildtype><position><mutation>. 
      <wildtype><position><mutation> defines the mutations which are to be predicted for the protein sequence, which stands for wild-type amino acid, mutation position in sequence, and mutated amino acid.
      for example:
    
        D3H
        E17A
  
  fasta_file:
      fasta_file defines the target protein sequence in a one-letter code, containing sequence ID and sequence. It is better to use Uniprot ID or PDB ID added chain ID for the ID.
      for example:
      
        >1A7VA
        QTDVIAQRKAILKQMGEATKPIAAMLKGEAKFDQAVVQKSLAAIADDSKKLPALFPADSKTGGDTAALPKIWEDKAKFDDLFAKLAAAATAAQGTIKDEASLKANIGGVLGNCKSCHDDFRAKKS
  
  variants_fasta_dir:
      FASTA file was used to store variant and wildtype protein.
      for example:
      
        >1A7VA
        QTDVIAQRKAILKQMGEATKPIAAMLKG....
        >1A7VA_D3H
        QTHVIAQRKAILKQMGEATKPIAAMLKG...
        >1A7VA_E17A
        QTDVIAQRKAILKQMGAATKPIAAMLKGEAKF...
            
  output_dir:
      output directory for extracted representations from ESM-2
```
THPLM output format:

```bash
Output a dict contained variants' ID and DDGs. The variants ID was constructed by sequence ID from "fasta_file" and variants <wildtype><position><mutation>.
  for example:
    {'1A7VA_D3H': -0.28042653, '1A7VA_E17A': -0.2851434}
```
### Running

This is the example

```bash

python THPLM_predict.py ./examples/var.txt ./examples/wild.fasta ./examples/esm3Bout/ ./examples/varlist.fasta --gpunumber 0 --extractfile ./esmscripts/extract.py

```
You can to build your fasta file firstly and change to your directory.

## Citation


