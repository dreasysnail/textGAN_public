# textGAN
textGAN

Run: `THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python textGAN.py`

# Prerequisite

* Theono version >= 0.8
* cuda version 8.0
* cPickle
* foxhound

# Data link
 https://drive.google.com/open?id=0B52eYWrYWqIpd2o2T1E3aUU0cEk

# Pretrained parameter
http://people.duke.edu/~yz196/zip/param.zip

# Evaluation
*`python eval_kde.py`

# Citation
* Arxiv link: [https://arxiv.org/abs/1706.03850](https://arxiv.org/abs/1706.03850)

```latex
@inproceedings{zhang2017adversarial,
  title={Adversarial Feature Matching for Text Generation},
  author={Zhang, Yizhe and Gan, Zhe and Fan, Kai and Chen, Zhi and Henao, Ricardo and Shen, Dinghan and Carin, Lawrence},
  booktitle={ICML},
  year={2017}
}
```

