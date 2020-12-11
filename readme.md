[TOC]

### 1 任务
本项目任务为训练模型实现文本摘要生成，从京东发现好货栏目中的商品标题及宣传文案中，生成该商品的简介。


### 2 数据
本项目数据源于京东电商发现好货栏目，其中训练集43996条，验证集5000条，测试集1000条。数据中源文本由商品标题、商品参数、商品宣传文案组成，目标文本来自真实写手为商品写的简介。


### 3 目录
+ files: 该目录用于存放数据，受限于数据大小，可与我联系获取数据。
+ data: 该目录用于存放数据集生成工具函数，包括：
    + data_utils.py: 用于读取数据的工具函数
    + dataset.py：用于生成符合pytorch格式的数据集
+ model：该目录用于工程用到的工具函数，包括：
    + config.py：设置各项参数
    + model.py: 设置Seq2Seq+Attention网络结构
    + pgn.py：设置Point Generation Network网络结构
    + predict.py：实现摘要生成
    + utils.py：工具函数
+ main.py：工程主程序


### 4 模型说明

#### 4.1 模型选择
在config.py中，设置```pointer=True```可选择模型为PGN，否则模型为Seq2Seq。
PGN与普通Seq2Seq的区别在于：
（1）拓展了词汇表，使得OOV词汇也可被生成。
（2）coverage机制抑制了attention，减少词汇的重复产生。

#### 4.2 Beam Search与Greedy Search
在config.py中，设置```beam_width```为大于1的值则采用波束方式生成摘要；设置```beam_width=1```则束宽为1，也即采用贪心方式生成摘要。


### 5 实验结果

| 模型名称 | 词汇表大小 | 生成方式 | Rouge-1 | Rouge-2 | Rouge-L |
| :-----: | :----: | :----: | :----: | :----: | :----: |
| Seq2Seq | 30k <br> 50k | greedy <br> beam <br> greedy <br> beam | 14.3 <br> 15.4 <br> 14.5 <br> 16.8 | 0.5 <br> 2.0 <br> 0.4 <br> 1.7 | 10.4 <br> 14.1 <br> 10.0 <br> 14.1 |
| PGN | 30k <br> 50k | greedy <br> beam <br> greedy <br> beam | 17.3 <br> 20.3 <br> 20.1 <br> 22.7 | 0.4 <br> 1.9 <br> 1.1 <br> 2.5 | 11.2 <br> 13.3 <br> 12.9 <br> 15.7 |
