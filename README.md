
_This README is a draft._

# GNN-NIDS (CDP No. 35452 -- Bari)

GNN-NIDS is a project for malicious domain detection based on DNS queries using graph neural networks.

---

![Python 3.9](doc/shields/python-3.9-green.png)
![TensorFlow 2.12](doc/shields/tensorflow-2.12-green.png)

![Flower](doc/images/flower.png)
![Rocs](doc/images/rocs.png)

## Requirements

### Install the required libraries:

For example,
```
python3 -m pip install <requirement>
```

## Usage

### 1. Get the Data

- Download the raw DNS traffic dataset (TI-2016) from https://ieee-dataport.org/documents/ti-2016-dns-dataset or //DATASET_URL// and save it in `<DATA_PATH>`.
It should contain 10 folders: Day0, Day1, Day2, Day3, Day4, Day5, Day6, Day7, Day8 and Day9, for a total of 240 pcap files.
The total size of the dataset should be around 113 GiB.

- The blacklists are included in the code repository in the `data` folder


### 2. Get the Code

Clone the repository containing the necessary code and scripts: // update link with github:
```
git clone https://github.com/m-altieri/DNS-GT.git
cd DNS-GT
```


### 3. Data Preprocessing

- Convert the pcap files into csv files using the tshark script:

```
src/preprocessing/pcap2csv.sh <DATA_PATH>
```

This step performs an initial filtering, removing packets that are not DNS queries or that are malformed, and extracts only the relevant fields from each packet.
It will create a new `tcsv` folder in `<DATA_PATH>` with 240 csv files, one for each pcap file.
The process can take up to a few hours and the resulting folder is around 72 GiB in size.


- Preprocess csv files:

```
src/preprocessing/process_tshark_csvs.py <DATA_PATH>
```

This will create a new `pcsv` folder in `<DATA_PATH>` with again 240 csv files, but a total size of just around 1.8 GiB, cleaning the DNS traffic and extracting only the useful information.


- Create the vocabulary from the processed queries and convert the csv files into the final npy files:

```
src/preprocessing/csv2npy.py <DATA_PATH>
```

This will create, at the same time, the vocabulary, and save it to `<VOCAB_PATH>` as a pair of two files, one for hosts and one for domains, and also the final query files in npy format. The vocabulary will be created in `<DATA_PATH>/vocab/`. The npy files will be created in `<DATA_PATH>/npy/`, and will be split automatically into a `train` folder containing the first 70% files, and a `test` folder containing the remaining 30% files.

- Process the blacklists into a single csv
Move the blacklists to the same path as the other data, to have all data-related files in a single place:

```
mv </path/to/>DNS-GT/data/blacklists <DATA_PATH>
```

Merge blacklists in a single file, for each category and quality:

```
src/scripts/merge_blacklists.py <DATA_PATH>
```

This script will traverse the `blacklists` folder and merge the blacklists for each category and quality in a single filed called `blacklist.txt`, save in the respective directory.


- Create labels for domains in the dictionary:

```
src/scripts/label_domains.py <DATA_PATH>
```

This script will create a `labels.csv` file in `<DATA_PATH>` containing domains names, and booleans (0 or 1) for their blacklisted status, for all blacklist categories (advertising, malicious, suspicious, tracking, other, and any) and quality (good and ok), for a total of 13 significative columns.


### 4. Pretrain the Models

- Configure the default training parameters in `</path/to/>DNS-GT/runs/default.yaml` with the correct paths.

- Run the training:
First move to the src folder:
```
cd </path/to/>DNS-GT/src
```
There are several parameters that can be customized. To start, run:
```
python3 train.py --help
```
Then start training the desired model with the desired parameters.
The parameters selected will be saved and will be reused if you wish to stop training and resume in the future for the same model and the same run name.
Each model has a default parameter configuration, that can be seen in `</path/to/>DNS-GT/runs/<MODEL>/default.yaml`.

Start training with, for instance:
```
python3 train.py DELM -r first_run --seq-strategy cluster
```
The model weights, embeddings and configuration will be saved in the `runs` folder.
 

### 5. Evaluate the Models

- Finetune the Models End-to-end
After a model is pretrained (and its weights are available in the `runs` folder), it can be finetuned in an end-to-end way with the `--ft` or `--finetune` flag. For instance:
```
python3 train.py DELM -r first_run --ft
```
- Afterwards, get predictions on the downstream task:
```
python3 train.py DELM -r first_run --evaluate
```

- Use the pretrained embeddings to train external classifiers
Instead of tinetuning, you can use the saved embeddings (`embeddings.npy` file) to train and evalute external classifiers using:
```
python3 train_classifier.py <MODEL> <RUN_NAME>
```
Optionally, it's possible to balance the training classes automatically with the `-b` flag. By default, the used blacklist is the one for category 'any' and quality 'good'. This can be changed with flags `--category` and `--q`, respectively.


### Docker (unstable)
```
docker run -it --gpus all -p 6006:6006 delm "python3 train.py --gpu 0 --quick-tb & { sleep 30; tensorboard --host=0.0.0.0 --logdir=tmp; };"
```


## Roadmap (out of date)
*Current Paper:*
- [x] General dataset analysis and overview
- [x] PCAP preprocessing and query information extraction into csv
- [x] Vocabulary and query dataset creation from csv
- [x] Design and implement core model components
- [x] Configure the model with external conf files
- [x] Training script with tf.data API
- [x] Add `<START>` token
    - [x] Show sequence contextualized representation examples 
- [x] Demo for model playesting and debugging 
- [x] Calculate gradients only for the masked tokens
    - [x] Zero-ed out loss for non-masked tokens
    - [x] Check that gradients are correctly calculated
- [x] Use the BERT token masking technique (80% mask, 10% random, 10% stays)
- [x] Improve propagation towards `<START>` and `<MASK>` tokens
    - [x] `<START>` and `<MASK>` are always connected with all tokens
- [x] Add an add & norm as well as a skip-connection to the MH-GAT block
- [x] Add a `blocks` hyperparameter that tells how many times to repeat the MH-GAT block
- [x] (Possibly.) Only connect token embeddings with the same host
    - [x] It's not an adjacency, it's done at data pipeline level. Use --group-hosts to sort queries by host and make each sequence contain queries from the same host. The order of queries for the same host stays unchanged.
- [x] Visualize clusters of domain embeddings

*Future Extensions:*
- [ ] Implement at least two domain adjacency matrices (hierarchical and behavioral)
    - [x] Implemented hierarchical similarity
    - [ ] Implement behavioral similarity
- [ ] Dynamic vocabulary and embedding extension
    - [x] Add `<UNK>` token
- [x] Improve the sequencing module with custom algorithm
    - [x] Add `<PAD>` token
    
---

**Contacts:**
Massimiliano ALTIERI
massimiliano.altieri@ec.europa.eu
