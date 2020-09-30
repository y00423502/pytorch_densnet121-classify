# 说明
- 简介：本文档是基于densenet121网络结构训练明火二分类及模型检测的工程，后续可拓展至其他分类任务及更换对应网络结构
# 运行环境
- pytorch>1.3,PIL包，最好是在docker hub上下载一个pytorch镜像，创建对应容器，然后映射到jupyter notebook上在线调试
# 文件结构
- data：下面存放所有的数据，train,val,test,注意命名格式按照czx_fire_30k的规范创建自己的数据文件夹；

- lib;需要导入的一些py文件,比如我自己修改的抽取embedding_feature的densenet的extract_EmbedingFeature_densenet121.py，后续添加别的；

- pretrain_models：这里存放了预训练模型的地址，目前是空的，待添加，normal下存放的是原生densenet121的网络训练的模型，extract_EmbedingFeature下存放的是修改的网络训练的模型，注意训练和测试的网络结构必须选择一致

- output_models：目录结构同上，存放路径在train.py中对应行修改

- tools:存放一些工具，比如search_engine.py

- train.ipynb/train.py:训练代码

- test.ipynnb/test.py；测试代码

## 注意：
- 训练和测试的时候建议都用我修改的extract_EmbedingFeature_densenet121.py网络结构，便于训练的模型后续用于特征的抽取
##  train.ipynb/test.ipynb:
- notebook按照代码注释更改训练地址data_dir，模型保存地址save_path，和最后最优模型的地址output_model_SavePath

## train.py: 
- 例子：python train.py  --d ./data/czx_fire_30k --p ./output_models/extract_EmbedingFeature/czx_fire_30k/ --re ./output_models/normal/czx_fire_30k/best_500epoch.pth --epochs 500 --eps 20 --b 32 --num 8
- --d :输入训练图片地址
- --p 中间epoch保存地址
- --epochs 训练迭代次数
- --eps 间隔多少epoch保存中间模型
- --b batach_size数
- --num numworkers数

## test.py
- 例子; python test.py --d ./data/czx_fire_30k --p ../fire_RGB_classify_pro/300.pth --re ./data/czx_fire_30k/result/ --b 32 --num 8
- --d 测试图片路径
- --p 测试用到的模型路径
- --re 图片测试结果保存的路径
- --b batchsize数
- --num numworker数
