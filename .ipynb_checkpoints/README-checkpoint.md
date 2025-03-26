# Nystorm vs Simple Transformer

Nystrom Attention applies the nystorm attention to the matrix multiplicaiton of Self-Attention.


### Install Requirements
```
python -m pip install -r requirements.txt
```

### English -> German Europarl dataset

Multi-Head Attention Transformer
```
python train_translate.py
```

Nystrom Attention Transformer
```
python train_translate_nystorm.py
```

### Plot Loss for training checkpoint.pkl

`Nystrom-Simple-Transformer/Extra_Scripts/plot_loss.ipynb`


### Credits
Simple Transformer Implementaiton: https://github.com/IpsumDominum/Pytorch-Simple-Transformer

Nystromformer: https://arxiv.org/pdf/2102.03902