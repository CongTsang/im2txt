# 欢迎

> 本案例使用的环境：win10  + python3.5.6 + tensorflow1.10.0
>
> 建议使用Anaconda或者Miniconda安装python等集成环境，方便自我管理以及自我自闭

&nbsp;&nbsp;本实例使用model-3000000，欢迎小伙伴们下载安装，git一下你就知道。

## 安装

&nbsp;&nbsp;在电脑端安装好[git](https://git-scm.com/download/win)，然后建立[ssh](https://git-scm.com/book/zh/v2/%E6%9C%8D%E5%8A%A1%E5%99%A8%E4%B8%8A%E7%9A%84-Git-%E7%94%9F%E6%88%90-SSH-%E5%85%AC%E9%92%A5)，之后在你本地自己想要的目录右键打开**git bash**.

&nbsp;&nbsp;在上面的一切配置好后，输入`git clone git://github.com/CongTsang/im2txt.git`就可以把我这个库clone到你的电脑了，之后可以愉快和我创造世界了

## 运行

>  本案例是在[Tensorflow项目](https://github.com/tensorflow/models)的基础上跑的，checkpoint为3000000，训练效果已经达到预期

使用*PyCharm*打开此项目

```
.
├─.idea
├─pic
└─research
    └─im2txt
        ├─data
        │  └─__pycache__
        ├─inference_utils
        │  └─__pycache__
        ├─ops
        │  └─__pycache__
        └─__pycache__
```

**默认一切环境配置好后**

&nbsp;&nbsp;目录结构为亚子，我们只需要进入`research/im2txt`目录运行`run_inference.py`该文件即可

## 配置

&nbsp;&nbsp;默认识别图片放置在`./pic/cat.jpg`，你可以根据需要修改`run_inference.py`

```python
tf.flags.DEFINE_string("checkpoint_path", "data", ## checkpoint
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "data/word_counts.txt", ## 该项为词集
                       "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", r"../../pic/cat.jpg", ## 该项为识别图片
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
```

