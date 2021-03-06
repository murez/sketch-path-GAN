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
