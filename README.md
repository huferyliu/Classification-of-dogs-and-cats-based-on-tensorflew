# Classification-of-dogs-and-cats-based-on-tensorflew
This is a project about dog and cat identification based on tensorflew.

在main.py中存放的是我们项目的基础结构,可以参照.

由于本实验代码是基于GPU版的 如果没有GPU,就需要使用cpu版本的.


猫狗分类数据集：本实验使用实验数据基于kaggle Dogs vs. Cats 竞赛提供的官方数据集，数据集可在百度网盘中进行下载：
链接：https://pan.baidu.com/s/13hw4LK8ihR6-6-8mpjLKDA 密码：dmp4。


我们在获得数据后,需要先对数据进行照片的分类,利用classify.py  先修改好路径,这样方便分类.
![Snipaste_2025-05-30_23-12-26](https://github.com/user-attachments/assets/140308b0-b3c4-43d8-91f7-5101cd6dc41f)



环境配置

要配置好环境,关于本实验要用到的python环境,下面给出了配置的视频,跟着视频操作即可
https://www.bilibili.com/video/BV1rd4y187nM/?spm_id_from=333.1391.0.0&vd_source=bffb1b050d9ccb5c74ac3b797aa33a24



实验环节
在上述都做好后,达到了main.py文件的结构后,就可以点击到train.py去运行代码，训练模型了.

在得到训练的模型后,就可以运行predict.py,这是验证模型的测试代码,我们选择一张图片，等待给出结果即可.



![test](https://github.com/user-attachments/assets/cf4cc47c-494e-4fce-ae8b-f32fad1ab4cc)

![test1](https://github.com/user-attachments/assets/e0d94270-2d50-42a5-8f71-717039c78347)
