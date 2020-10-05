# Semantic Matching And Aggregation Network for Few-shot Intent Detection
This repository provides PyTorch implementation for the paper *Semantic Matching and Aggregation Network for Few-shot Intent Detection*

## Requirements
Python 3.6.2 <br />
Numpy <br />
Pandas <br />
Pytorch 1.0.1 <br />
Scikit-learn 0.21.1 <br />

## Dataset
We conduct the split on NLUE and SNIPS dataset in dataset directory. Please take a look at our paper for details of the split.

## Usage
Please obtain and put the pre-trained FasTText embedding in our fasttext directory of the directory.</br>
 
## Configuration
* ```--ckpt_dir```: Saved directory for checkpoint
* ```--eps```: Evaluate as nonepisodic or episodic procedure (i.e. eps or noneps)
* ```--num_eps```: Number of episodes used for episodic training and/or evaluation
* ```--dataset```: Choose dataset to train/evaluate (i.e. SNIPS/ NLUE)
* ```--num_run```: number of runs (only for SNIPS)
* ```--num_fold```: number of KFold counting from 1 to 10 (only for NLUE)
* ```--src```: Source data used for training (i.e. seen)
* ```--tgt```: Data used for evaluation (i.e. novel or joint)
* ```--num_samples_per_class```: K in C-way K-shot
* ```--num_class```: C class in C-way K-shot
* ```--num_query_per_class```: num query per class (Q)
* ```--num_test_class```: Number of classes used for evaluation (i.e. C for episodic, #total classes in joint/novel space)
* ```--fasttext_path```: FastText pretrained embedding file location



## Running

```
python main.py
```


## Citation
If you use our code or dataset, please cite our paper

## Acknowledgement
https://github.com/ZhixiuYe/MLMAN </br>
https://github.com/galsang/BIMPM-pytorch



