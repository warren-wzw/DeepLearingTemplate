# 项目介绍
深度学习项目模版

## 有哪些功能？
* 快速建立深度学习项目;
* 多GPU并行训练
* 加入多进程训练,训练使用卡1,验证使用卡2。
* 包含一些基础功能例如，学习率预热衰减、打印模型结构及参数信息、加载缓存数据等;
* 包含基础模块例如DSC、SelfAttention、TransformerEncoder、CBAM、SE;
* 加入save/loadcheckpoint已保存模型及当前训练状态，实现训练中断后也可恢复之前训练状态;
* 增加ModelLib包含EfficientNet/ShuffleNet/Mobilenet/GhostNet/ShuffleNet
* 增加无需缓存数据预处理函数,适用于大批量图像数据情况,训练时get 1batch数据处理1batch数据

## 环境搭建
* bash env_setup.sh

## 关于作者
* warren@伟
* 个人博客：[CSDN-warren@伟](https://blog.csdn.net/warren103098?type=blog)