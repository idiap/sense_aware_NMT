# Description

Implementation of "Integrating Weakly Supervised Word Sense Disambiguation into Neural
Machine Translation".

This work is based on [OpenNMT](https://github.com/OpenNMT/OpenNMT), an open-source
(MIT) neural machine translation system. We did modification by integrating sense
information.


<center style="padding: 40px"><img width="70%" src="http://opennmt.github.io/simple-attn.png" /></center>

# Quickstart

## 1) Preprocessing.

```bash
python preprocess.py -train_src train.tok.$lsource -train_tgt train.tok.$ltarget -src_vocab src.dict -tgt_vocab tgt.dict -feature_vocab feature.dict -valid_src dev.tok.$lsource -valid_tgt dev.tok.$ltarget -save_data $savePath''demo -feature 1 -lower
```

## 2) Train the model.

```bash
python train.py -data $savePath''demo.train.pt -pre_feature_vecs_enc feature_embed.dict -pre_word_vecs_enc src_embed.dict -save_model model -gpus 0 -epochs  -brnn 
```

## 3) Translate sentences.

```bash
python translate.py -model model.pt -src $testPath''test.tok.$lsource -replace_unk -verbose -output test.$ltarget -gpu 0

