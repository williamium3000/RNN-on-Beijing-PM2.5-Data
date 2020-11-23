# RNN-on-Beijing-PM2.5-Data
RNN on Beijing PM2.5 Data Data prediction
在北京PM2.5数据集上利用RNN（循环神经网络）通过前2个小时的PM2.5污染数据以及天气条件预测第三小时的PM2.5污染。
## 数据集
北京PM2.5数据集（https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data），共43824条数据。
## 数据预处理
本实验参考样板数据预处理代码，首先对缺失值进行去除处理，然后利用滑动窗口（sliding window）对数据进行处理，得到$2\times 8$的输入格式，将2010，11，12年数据作为训练样本，剩余为测试样本进行划分，共得到26280条训练样本以及17518条测试数据。
## 实验思路
按照题目要去，构建RNN（循环神经网络），网络架构如下图，其中输入数据为前两小时的数据，隐藏变量为一个100纬维的向量，通过5个隐藏层后，将最后数据放入一个全连接层，输出回归一个PM2.5值，用这个值与真实值（groud truth）通过MAE进行计算损失（此处MAE使用均值MAE，后续实验结果也是均值MAE的实验结果），反向传播并且更新参数。每一个epoch，我们对全体训练样本以及测试样本进行测试，求的均值MAE并记录绘图。
![](rnn示意图.jpg)
## implementation detail
本实验采用GPU利用cuda10.2对pytorch rnn实现进行，利用默认参数值的Adam优化器，batch size为256，learning rate为0.0001，训练了50个epoch，并取在测试样本上损失最小的那个模型作为最终模型。
## 实验结果
在实验结果部分，将训练损失和测试损失绘制成图
![](train_test_loss.png)

![](rnn_lstm_gru_loss.png)
在实验中，通过利用gru，lstm，rnn进行实验，并对比效果，可以看到gru的收敛速度较快，但最终结果lstm更加好
