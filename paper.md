# <font><center>基于生成式对抗神经网络的素描路径生成系统</center></font>
## <font><center>张厶元  龚敬洋</center></font>

###   关键字 人工神经网络;深度学习;生成式对抗网络;绘画风格转移


##  1.选题背景

  现在网路上风靡的相机滤镜都有素描画的图片滤镜。同时在执法和刑事案件中嫌疑人的素描画任然作为目击者提供线索的重要依据。在娱乐和社会安全两方面都有对素描画像的需要。但人工素描价格昂贵，生产效率低，而普通基于图片简单滤镜变换的素描画真实度不高，通常伴有不合适的线条或低清晰度，同时笔画线条不清楚。

  最近GAN(Generative Adversarial Networks)和CNN(convolutional neural network)等神经网络技术的发展，直接产生矢量素描笔画路径的机器算法成为可能。本文将介绍我们利用GAN技术提出的素描路径生成系统的算法和框架，并进行详细的实验证明和与普通滤镜素描生成器和人工绘画的结果比较和改进方案。

##  2.相关工作

  基于神经网络算法的发展和CUDA(Compute Unified Device Architecture)显卡计算技术的发展，原先许多必须由人工生成图像合成算法现在可以通过神经网络得出我们想要的模型。而2014年Ian Goodfellow等人提出的GAN(Generative Adversarial Networks)
