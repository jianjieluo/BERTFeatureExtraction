# An Example to use BERT to extract Features of ANY Text

This repo aims at providing an easy to use and efficient code for extracting text features using BERT pre-trained model.

It has been originally designed to extract features of text instructions in [the R2R dataset for Visual and Language Navigation Task](https://bringmeaspoon.org/) in an efficient manner.

This repo provides a simple python script for the BERT Feature Extraction: Just imitate the `instr_loader.py` to design another PyTorch dataset class for your text data (mainly your text data reading method) if necessary and import your dataset class in `extract.py`, and the script will take care of the BERT text data preprocessing (e.g. BERT tokenization, adding special keys to each sentence, padding, etc) and feature extraction using state-of-the-art models.

# Requirements
- Python 3
- PyTorch (>= 1.0)
- [transformers](https://github.com/huggingface/transformers)

# Quick Start

Take `R2R_test.json` annotation file as an example:

```sh
python extract.py \
    --input data/raw_data/R2R_test.json \
    --num_workers 2 \
    --bert_model bert-base-uncased
```

Please note that the script is intended to be run on ONE single GPU only. If multiple GPUs are available, please make sure that only one free GPU is set visible by the script with the CUDA_VISIBLE_DEVICES variable environment for example.

# Downloading pre-trained models (Optional)

Since the BERT pre-trained model's download speed of the `transformers` package is not fast enough in some areas of the world, we also create a mirror on Baidu Drive (i.e., Baidu PAN). Some BERT pre-trained models cache listed below can be downloaded with the shared link https://pan.baidu.com/s/1CFVzy5we8JM2PP-TPFq_Sg and access code `xl9v`.

- Model List
    - bert-base-uncased
    - bert-large-uncased

Put the pre-trained model cache file in `~/.cache/torch/transformers/` and you can load a model directly via transformers API without modifying any package code.