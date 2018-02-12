# textGAN (Adversarial Feature Matching for Text Generation)
Theano and tensorflow implementation for 

* **Adversarial Feature Matching for Text Generation**,
Yizhe Zhang, Zhe Gan, Kai Fan, Zhi Chen, Ricardo Henao, Lawrence Carin. ICML, 2017.
* **Generating Text via Adversarial Training.**
Yizhe Zhang, Zhe Gan, Lawrence Carin.  Workshop on Adversarial Training, NIPS, 2016.

## Run 

* (Theano) Run: 

`THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python textGAN.py`

* (Tensorflow) Run: 

`python textGAN.py`


## Prerequisite
* Theano: 
	* Theono >= 0.8
	* cPickle
	* foxhound
	
* Tensorflow: 
	* Tensorflow version == 1.2
	* cPickle



# Evaluation
* (Theano only) `python eval_kde.py`

# Citation
Please cite our paper if it helps with your research.

* Arxiv link: [https://arxiv.org/abs/1706.03850](https://arxiv.org/abs/1706.03850)

```latex
@inproceedings{zhang2017adversarial,
  title={Adversarial Feature Matching for Text Generation},
  author={Zhang, Yizhe and Gan, Zhe and Fan, Kai and Chen, Zhi and Henao, Ricardo and Shen, Dinghan and Carin, Lawrence},
  booktitle={ICML},
  year={2017}
}
```

