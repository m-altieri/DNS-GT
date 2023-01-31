_This README is a draft and placeholder._

# GNN-NIDS (CDP No. 35452 -- Bari)

GNN-NIDS is a project for malicious domain detection based on DNS queries using graph neural networks.

---

![Python 3.9](https://img.shields.io/badge/python-3.9-green)
![TensorFlow 2.10](https://img.shields.io/badge/tensorflow-2.10-green)

## Installation

### Clone the repository:
```
git clone https://gitlab.jrc.ec.europa.eu/jrc-projects/createg/cdp-bari/dns.git ~/gnn-nids
```

### Get the data:
```
mkdir -p ~/gnn-nids/datasets/TI-2016-Partial && cd gnn-nids/datasets/TI-2016-Partial
wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zh3wnddzxy-2.zip
unzip zh3wnddzxy-2.zip && rm -rf zh3wnddzxy-2.zip
```

### Install the required libraries:

For example,
```
python3 -m pip install <requirement>
```

## Usage

### Quick Start

Launch the script with the smallest `.pcap` file:
```
cd ~/gnn-nids/src
cp ../datasets/TI-2016-Partial/Day0_24_04_2016/20160424_055409.pcap .
python3 preprocess.py -v .
```

_TODO: now that `.pcap` file is included in `src/` for convenience but I have to remove it._

### Docker
```
docker run -it --gpus all -p 6006:6006 delm "python3 train.py --gpu 0 --quick-tb & { sleep 30; tensorboard --host=0.0.0.0 --logdir=tmp; };"
```

## Visuals

## Roadmap
*Current Paper:*
- [x] General dataset analysis and overview
- [x] PCAP preprocessing and query information extraction into csv
- [x] Vocabulary and query dataset creation from csv
- [x] Design and implement core model components
- [x] Configure the model with external conf files
- [x] Training script with tf.data API
- [x] Add `<START>` token
    - [ ] Show sequence contextualized representation examples 
- [x] Demo for model playesting and debugging 
- [x] Calculate gradients only for the masked tokens
    - [x] Zero-ed out loss for non-masked tokens
    - [ ] Check that gradients are correctly calculated
- [x] Use the BERT token masking technique (80% mask, 10% random, 10% stays)
- [x] Improve propagation towards `<START>` and `<MASK>` tokens
    - [x] `<START>` and `<MASK>` are always connected with all tokens
- [x] Add an add & norm as well as a skip-connection to the MH-GAT block
- [x] Add a `blocks` hyperparameter that tells how many times to repeat the MH-GAT block
- [ ] (Possibly.) Only connect token embeddings with the same host
- [ ] (Possibly.) Visualize clusters of domain embeddings

*Future Extensions:*
- [ ] Implement at least two domain adjacency matrices (hierarchical and behavioral)
    - [x] Implemented hierarchical similarity
    - [ ] Implement behavioral similarity
- [ ] Dynamic vocabulary and embedding extension
    - [ ] Add `<UNK>` token
- [ ] Improve the sequencing module with custom algorithm
    - [ ] Add `<PAD>` token

## Support
If you need any direct help, contact me: massimiliano.altieri@ec.europa.eu.
