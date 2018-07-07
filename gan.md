## 我们的方法
想象阿尔布雷 希特·丢勒(Albrecht Dürer ,1471－1528)大师在文艺复兴时期面对铜镜中的自己画出了世界上第一幅自画像，他的铅笔线条柔和流畅明暗中的人栩栩如生，如果丢勒来到现在的城市公园，为富有年轻活力的少女画画像会怎么样？
### cycleGAN(Cycle Generative Adversarial Networks)
cycleGAN是一种不成对的图像到图像转换的神经网络算法，由Berkeley AI Research (BAIR) laboratory, UC Berkeley在2018年提出。 算法主要基于GAN生成式对抗网络算法。

该算法的原理可以概述为：将A风格图片转换成B风格图片。也就是说，现在有两个样本空间(sample space)，<a href="http://www.codecogs.com/eqnedit.php?latex=X" target="_blank"><img src="http://latex.codecogs.com/svg.latex?X" title="X" /></a>和<a href="http://www.codecogs.com/eqnedit.php?latex=Y" target="_blank"><img src="http://latex.codecogs.com/svg.latex?Y" title="Y" /></a>，我们希望把<a href="http://www.codecogs.com/eqnedit.php?latex=X" target="_blank"><img src="http://latex.codecogs.com/svg.latex?X" title="X" /></a>空间中的样本通过cycleGAN转换成<a href="http://www.codecogs.com/eqnedit.php?latex=Y" target="_blank"><img src="http://latex.codecogs.com/svg.latex?Y" title="Y" /></a>空间中的样本。所以我们的目的就是拟合学习一个生成器或映射，设这个映射为<a href="http://www.codecogs.com/eqnedit.php?latex=F" target="_blank"><img src="http://latex.codecogs.com/svg.latex?F" title="F" /></a>，则它就对应着GAN中的生成器(Generator)，F可以将X中的样本空间x映射成为Y中的样本空间<a href="http://www.codecogs.com/eqnedit.php?latex=F(x)" target="_blank"><img src="http://latex.codecogs.com/svg.latex?F(x)" title="F(x)" /></a>。对于生成的图片，我们还需要GAN中的判别器(Discriminator)来判别它是否为理想图片，即为我们所希望要的图片，由此建立GAN(Generative Adversarial Networks)。设这个判别器为<a href="http://www.codecogs.com/eqnedit.php?latex=D_{Y}" target="_blank"><img src="http://latex.codecogs.com/svg.latex?D_{Y}" title="D_{Y}" /></a>这样的话，根据这里的生成器和判别器，我们就可以构造一个GAN损失(loss function)，表达式为：
<a href="http://www.codecogs.com/eqnedit.php?latex=L_{GAN}(F,D_{Y},X,Y)=E_{y\sim&space;Pdata(y)}[T(D_{Y}(y))]&plus;E_{x\sim&space;Pdata(x)}[T(1-D_{Y}(F(x)))]" target="_blank"><img src="http://latex.codecogs.com/svg.latex?L_{GAN}(F,D_{Y},X,Y)=E_{y\sim&space;Pdata(y)}[T(D_{Y}(y))]&plus;E_{x\sim&space;Pdata(x)}[T(1-D_{Y}(F(x)))]" title="L_{GAN}(F,D_{Y},X,Y)=E_{y\sim Pdata(y)}[T(D_{Y}(y))]+E_{x\sim Pdata(x)}[T(1-D_{Y}(F(x)))]" /></a>

但单纯的使用这一个损失训练是无法达到目的的。为了追求最低的生成差异，映射F完全可以将所有<a href="http://www.codecogs.com/eqnedit.php?latex=X" target="_blank"><img src="http://latex.codecogs.com/svg.latex?X" title="X" /></a>都映射为<a href="http://www.codecogs.com/eqnedit.php?latex=Y" target="_blank"><img src="http://latex.codecogs.com/svg.latex?Y" title="Y" /></a>空间中的同一张图片，使损失无效化，即“循环一致性损失”（cycle consistency loss）。

所以为了保证映射结果的差异性，需要再假设映射<a href="http://www.codecogs.com/eqnedit.php?latex=G" target="_blank"><img src="http://latex.codecogs.com/svg.latex?G" title="G" /></a>将<a href="http://www.codecogs.com/eqnedit.php?latex=Y" target="_blank"><img src="http://latex.codecogs.com/svg.latex?Y" title="Y" /></a>空间中的样本空间<a href="http://www.codecogs.com/eqnedit.php?latex=y" target="_blank"><img src="http://latex.codecogs.com/svg.latex?y" title="y" /></a>转换为<a href="http://www.codecogs.com/eqnedit.php?latex=X" target="_blank"><img src="http://latex.codecogs.com/svg.latex?X" title="X" /></a>中的样本空间<a href="http://www.codecogs.com/eqnedit.php?latex=G(y)" target="_blank"><img src="http://latex.codecogs.com/svg.latex?G(y)" title="G(y)" /></a>。CycleGAN同时学习<a href="http://www.codecogs.com/eqnedit.php?latex=F" target="_blank"><img src="http://latex.codecogs.com/svg.latex?F" title="F" /></a>和<a href="http://www.codecogs.com/eqnedit.php?latex=G" target="_blank"><img src="http://latex.codecogs.com/svg.latex?G" title="G" /></a>两个映射，并要求<a href="http://www.codecogs.com/eqnedit.php?latex=F(G(y))\approx&space;y" target="_blank"><img src="http://latex.codecogs.com/svg.latex?F(G(y))\approx&space;y" title="F(G(y))\approx y" /></a>并且<a href="http://www.codecogs.com/eqnedit.php?latex=G(F(x))\approx&space;x" target="_blank"><img src="http://latex.codecogs.com/svg.latex?G(F(x))\approx&space;x" title="G(F(x))\approx x" /></a>。

根据<a href="http://www.codecogs.com/eqnedit.php?latex=F(G(y))\approx&space;y" target="_blank"><img src="http://latex.codecogs.com/svg.latex?F(G(y))\approx&space;y" title="F(G(y))\approx y" /></a>  <a href="http://www.codecogs.com/eqnedit.php?latex=G(F(x))\approx&space;x" target="_blank"><img src="http://latex.codecogs.com/svg.latex?G(F(x))\approx&space;x" title="G(F(x))\approx x" /></a>,cycleGAN的损失函数就可以定义为：
<a href="http://www.codecogs.com/eqnedit.php?latex=L_{cyc}(F,G,X,Y)=E_{x\sim&space;Pdata(x)}[||G(F(x))-x||_{1}]&plus;" target="_blank"><img src="http://latex.codecogs.com/svg.latex?L_{cyc}(F,G,X,Y)=E_{x\sim&space;Pdata(x)}[||G(F(x))-x||_{1}]&plus;" title="L_{cyc}(F,G,X,Y)=E_{x\sim Pdata(x)}[||G(F(x))-x||_{1}]+" /></a><a href="http://www.codecogs.com/eqnedit.php?latex=E_{y\sim&space;Pdata(y)}[||F(G(y))-y||_{1}]" target="_blank"><img src="http://latex.codecogs.com/svg.latex?E_{y\sim&space;Pdata(y)}[||F(G(y))-y||_{1}]" title="E_{y\sim Pdata(y)}[||F(G(y))-y||_{1}]" /></a>
同样G(X)也需要一个D判断器来学习，可以引入其损失函数
<a href="http://www.codecogs.com/eqnedit.php?latex=L_{GAN}(F,D_{X},Y,X)" target="_blank"><img src="http://latex.codecogs.com/svg.latex?L_{GAN}(F,D_{X},Y,X)" title="L_{GAN}(F,D_{X},Y,X)" /></a>

综上，我们就可以完整的创造一个cycleGAN网络，它的损失函数表达式为：
<a href="http://www.codecogs.com/eqnedit.php?latex=L=L_{GAN}(F,D_{Y},X,Y)&plus;L_{GAN}(G,D_{X},Y,X)&plus;\lambda&space;L_{cyc}(F,G,X,Y)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?L=L_{GAN}(F,D_{Y},X,Y)&plus;L_{GAN}(G,D_{X},Y,X)&plus;\lambda&space;L_{cyc}(F,G,X,Y)" title="L=L_{GAN}(F,D_{Y},X,Y)+L_{GAN}(G,D_{X},Y,X)+\lambda L_{cyc}(F,G,X,Y)" /></a>

### 回归素描绘画过程
普通的机器滤镜实现的主要原理都是由基于图片灰度特征或者卷积计算后得到特征，而人在绘画中恰恰不会一开始就从整体的色调或者从整体描画出图像。通过模仿人类艺术创作的过程实现机器进行艺术创作的效果会明显比简单机器滤镜的效果要好很多，Combining Sketch and Tone for Pencil Drawing Production[此处为论文] <http://www.cse.cuhk.edu.hk/~leojia/projects/pencilsketch/pencil_drawing.htm>通过绘画主体的描绘和背景的描绘拆分实现了机器素描绘画的最高效果，所以我们回归到人类人像的素描绘画过程中，实现更好的结果。

考虑一位素描艺术家看到了一张人脸，(感谢陈同学的出镜)[![PEftPJ.md.jpg](https://s1.ax1x.com/2018/07/04/PEftPJ.md.jpg)](https://imgchr.com/i/PEftPJ)
他会考虑人物的脸型轮廓，五官位置和形状，头发式样，光线强弱......所以我们就需要根据图片提取这些特征。

### 人脸特征信息提取
#### 1、人脸识别API的选择
将肖像照片转化为素描照片，首先需要提取照片中人脸中相关特征点的位置，以方便后续的GAN网络生成素描画。目前互联网上提供了大量已训练成熟的基于CNN(convolutional neural network)的面部识别API可供调用，但它们在响应时间，面部关键点数量和调用流量计费方式上均有差异。通过对国内三款主流面部识别服务供应商提供的API进行大量测试，基本可以得出三款API的相关差异。

| API名称        | 平均响应时间   |  关键点数量  | 免费调用流量限制 |
| --------   | -----:  | :----:  | :----:  |
| 百度云 | 315ms |    72    | 2QPS |
| 旷世FACE++ | 206ms |   106   | 不限量，与其他用户共享QPS池 |
| 腾讯 |   294ms     |  88  | 1万张/月 |

由上表可知，对于小规模面部识别调用而言，矿世FACE++在关键点数量及响应时间上相比其他两款API均有明显优势。因此本文中我们将选择该API进行人脸特征信息的提取。
#### 2、图片预处理
矿石FACE++对于上传图片有最大4096*4096像素，2MB文件大小的限制要求，而目前绝大多数拍摄设备拍出的图片文件参数均高于该值，同时为了减少因进行人脸识别而产生的流量，需要首先对图片进行适当压缩和s缩放。我们首先将图片进行适当锐化以保证其不因缩放导致锐度下降，进而影响识别成功率。随后通过OpenCV图像压缩算法对文件体积进行适当压缩，并对图片进行等比例缩放以保证其大小和体积被控制在合理范围内。为了防止图片缩小时出现波纹，我们使用了像素关系重采样的方式(CV_INTER_AREA)对图片进行缩放。具体函数用法如下：
```python
#通过CV_INTER_AREA方法缩放图片至目标大小
cv2.resize(SourceImage, (DXsize,DYsize), interpolation = cv2.INTER_AREA)
#适当降低照片质量以减小图片质量
cv2.imwrite(TargetFileName, SourceImage, [int(cv2.IMWRITE_JPEG_QUALITY), QualityKeepValue])
```
经过测试，处理后图像的文件体积已基本被控制在可接受范围内。测试数据如下：
| 原图片大小 | 原图片体积 | 压缩后图片大小 | 压缩后图片体积 |
| --------   | -----:  | :----:  | :----:  |
| 4288 * 2848 | 3.01MB | 1500 * 995 | 554kb |
| 6000 * 4000 | 6.15MB | 1500 * 1000 | 682kb |
| 2048 * 2048 | 2.18MB | 2048 * 2048 | 325kb |

#### 3、获得人脸特征信息
将照片进行预处理后，我们通过POST方式调用旷世FACE++的面部识别接口，并获得包含人脸特征信息的JSON数据。获得的人脸特征信息包含面部的矩形位置(face_rectangle)，面部器官的轮廓位置(landmarks)以及人脸的特征信息(attributes)。通过对JSON数据进行解析和分离，便可得到精确的人脸特征点位置信息。测试结果如下：
[![PEftPJ.md.jpg](https://s1.ax1x.com/2018/07/04/PEftPJ.md.jpg)](https://imgchr.com/i/PEftPJ)
(图3-1 原图片)
[![PEHKBR.md.jpg](https://s1.ax1x.com/2018/07/04/PEHKBR.md.jpg)](https://imgchr.com/i/PEHKBR)
(图3-2 识别结果(face_rectangle))
[![PEfJ54.jpg](https://s1.ax1x.com/2018/07/04/PEfJ54.jpg)](https://imgchr.com/i/PEfJ54)
(图3-2 识别结果(landmarks))

输出的数据是人物轮廓的内容和点阵位置，这就是先把握人物的整体形象。

