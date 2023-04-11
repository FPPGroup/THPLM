# THPLM
A sequence-based deep learning framework for predicting protein stability changes upon point mutations using pretrained protein language model



## Table of Contents

- [THPLM](#thplm)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Dependencies](#dependencies)
    - [ESM pretrain model download](#ESM pretrain model download)
    - [Installation](#Installation)
  - [Usage](#usage)
    - [For prediction](#for-prediction)
    - [Training your own model](#training-your-own-model)
  - [Docker](#Docker)
  - [Webserver and tutorial](#Webserver)
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
  variants_fasta_dir    FASTA file stored to get representations

optional arguments:
  -h, --help            show this help message and exit
  --model_location MODEL_LOCATION
                        PyTorch model file OR name of pretrained model to
                        download (see README for models)
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

### Training your own model

To train your own model, you need to build your fasta file firstly. And then you can use the 

```bash

python THPLM_predict.py ./examples/var.txt ./examples/wild.fasta ./examples/esm3Bout/ ./examples/varlist.fasta --gpunumber 0 --extractfile ./esmscripts/extract.py

```
## Docker

## Webserver and tutorial

## Citation


