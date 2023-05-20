# Patton<img src="figure/patton.svg" width="30" height="30" />: Language Model Pretraining on Text-rich Networks

## Data Preparation
### Download Data
Download data from MAG and [Amazon](http://jmcauley.ucsd.edu/data/amazon/links.html).


### Data Processing
1. Run the cells in data_process/process_amazon.ipynb and data_process/process_mag.ipynb for amazon domain network and MAG domain network respectively.
2. Tokenize the text in train/val/test.
```
cd src/scripts
bash build_train.sh
```

## Pretraining Patton
Pretraining start from bert-base-uncased.
```
bash run_pretrain.sh
```
Pretraining start from scibert-base-uncased.
```
bash run_pretrain_sci.sh
```

We support both single GPU training and multi-GPU training.

## Finetuning Patton

### Classification
Run classification train.
```
bash nc_class_train.sh
```

Run classification test.
```
bash nc_class_test.sh
```


### Retrieval
Run bm25 to prepare hard negatives.
```
cd bm25/
bash bm25.sh
```

Prepare data for retrieval.
```
cd src/
bash nc_retrieve_gen_bm25neg.sh
bash build_train.sh
```

Run retrieval train.
```
bash nc_retrieve_train.sh
```

Run retrieval test.
```
bash nc_infer.sh
bash nc_retrieval.sh
```

### Reranking
Prepare data for reranking.
```
bash scripts/match.sh
```

Run reranking train.
```
bash nc_rerank_train.sh
```

Run reranking test.
```
bash nc_rerank_test.sh
```

### Link Prediction
Run link prediction train.
```
bash lp_train.sh
```

Run link prediction test.
```
bash lp_test.sh
```



