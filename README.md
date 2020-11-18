# Dynamic Semantic Matching And Aggregation Network for Few-shot Intent Detection
This repository provides PyTorch implementation for the paper [*Dynamic Semantic Matching and Aggregation Network for Few-shot Intent Detection*](https://www.aclweb.org/anthology/2020.findings-emnlp.108/) **(Findings of EMNLP'2020)**

## Requirements
Python 3.6.2 <br />
Numpy <br />
Pandas <br />
Pytorch 1.0.1 <br />
Scikit-learn 0.21.1 <br />

## Dataset
We conduct the split on NLUE and SNIPS dataset in dataset directory. Please take a look at our paper for details of the split.

## Usage
Please obtain and put the pre-trained FastText embedding in our fasttext directory of the repository (named as **vectors-en.txt**). Otherwise, create your own FastText embedding directory and update the argument in Configuration below. </br>
 
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

**Regularization hyperparameters**
* ```--self_attn_loss```: coefficient for Self-attention Regularization 
* ```--uniform_loss```: coefficient for Head Uniform Regularization
* ```--same_intent_loss```: coefficient for Head Distribution Regularization



## Running
### SNIPS_Dataset
**FSL (episode-nonepisode)**
```
python main.py --dataset='SNIPS' --tgt='novel' --eps='eps' --num_class=2 --num_test_class=2
```
```
python main.py --dataset='SNIPS' --tgt='novel' --eps ='noneps' --num_class=2 --num_test_class=2
```

**GFSL (episode-nonepisode)**
```
python main.py --dataset='SNIPS' --tgt='joint' --eps='eps' --num_class=2 --num_test_class=2
```
```
python main.py --dataset='SNIPS' --tgt='joint' --eps ='noneps' --num_class=2 --num_test_class=7
```

### NLUE Dataset
**FSL (episode-nonepisode)**
```
python main.py --dataset='NLUE' --tgt='novel' --eps='eps' --num_class=5 --num_test_class=5
```
```
python main.py --dataset='NLUE' --tgt='novel' --eps ='noneps' --num_class=5 --num_test_class=16
```

**GFSL (episode-nonepisode)**
```
python main.py --dataset='NLUE' --tgt='joint' --eps='eps' --num_class=5 --num_test_class=5
```
```
python main.py --dataset='NLUE' --tgt='joint' --eps ='noneps' --num_class=5 --num_test_class=64
```

## Citation
If you use our ideas, code or dataset, please cite the following paper:
<pre>
@inproceedings{nguyen-etal-2020-dynamic,
    title = "Dynamic Semantic Matching and Aggregation Network for Few-shot Intent Detection",
    author = "Nguyen, Hoang  and
      Zhang, Chenwei  and
      Xia, Congying  and
      Yu, Philip",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.108",
    pages = "1209--1218",
}
</pre>

## Acknowledgement
https://github.com/ZhixiuYe/MLMAN </br>
https://github.com/galsang/BIMPM-pytorch



