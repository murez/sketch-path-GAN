# Sketch-Simplification
My implementation of the sig 16 paper 

[Learning to Simplify: Fully Convolutional Networks for Rough Sketch Cleanup](http://hi.cs.waseda.ac.jp/~esimo/en/research/sketch/)

Edgar Simo-Serra*, Satoshi Iizuka*, Kazuma Sasaki, Hiroshi Ishikawa   (*equal contribution)

using **PyTorch**

## TODO
1. out of memory when predicting
```
batch_x= Variable(batch_x,volatile = True).cuda()
```
2. input color image