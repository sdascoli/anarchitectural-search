# anarchitectural-search

This repository contains a PyTorch implementation of the experiments described in the paper "Finding the Needle in the Haystack withConvolutions: on the benefits of architectural bias" by Stéphane d'Ascoli, Levent Sagun, Joan Bruna and Giulio Biroli.

It allows to convert a convolutional network (CNN) to its equivalent fully-connected network (eFCN) or locally-connected network (eLCN), and perform interpolations in weight and output space between models.

Usage :

To install requirements :
```pip install -r requirements.txt```

To check the mapping is exact :
```python test.py```

To train AlexNet on CIFAR-10 :
```python train.py```

To train the the eFCN of AlexNet on CIFAR-10 :
```python train.py --convert_to fc```

To perform interpolations:
```python interp.py --net1_path $NET1_PATH --net2_path $NET2_PATH --interp_method string```
