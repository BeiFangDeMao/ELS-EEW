# BAZ-Network追加对比实验

## Python启动接口

直接在IDE运行本目录的 `train_additional_baselines.py`。
默认仅训练 **baz_network**；用户已取消DistPT训练。适配代码保留，但不在默认模型列表中，不会运行。

```powershell
& "$env:USERPROFILE/.conda/envs/lightweight/python.exe" discussion/model_comparison/additional_baselines/train_additional_baselines.py
```

仅检查，不训练：在上述命令后增加 `--check`。
默认继续本目录已有实验，跳过已完成模型，未完成模型从最后一个完整epoch恢复。
配置或代码改变后必须使用 `--new-run` 开启新实验；原始SeisT/MagNet实验不受影响。

## 数据与训练

固定使用已完成SeisT/MagNet的同一套100 Hz、三分量Z/N/E、gal、288点因果H5与cohort。
训练/验证/测试记录数分别为605166/84564/51772；不重新划分，不重复滤波或归一化。
唯一新增标签变换是将现有azi（台站指向震中、从北顺时针）编码为cos和sin。
输出第一项cos、第二项sin；atan2(sin,cos)恢复角度，最小圆周角差用于验证早停和测试。
两个分量MSE之和用于训练，不把角度标量直接传给双输出损失。

配置：batch512、AdamW lr=1e-3、weight_decay=1e-5、seed42、梯度L2裁剪0.8、最多300轮、patience20、min_delta1e-6，无学习率调度器。
16个进程读取H5后复用原训练框架；不安装依赖，不自动降低batch，不跳过非有限批次。
只有验证集用于选模型，之后在固定测试集保存逐记录预测。
输出：本目录 `training_results/<时间戳>/baz_network/`，含best.pth、resume.pth、训练日志、逐轮历史、test_predictions.csv、completed.json。

## BAZ适配与检查

复用仓库原始CNN+协方差/特征值/特征向量分支，设置in_samples=288；总参数62602。
上游代码以整个batch的最大值归一化协方差及特征值，会让一条记录依赖同批其他记录。
本地子类改为逐记录最大值归一化，分母用float32 epsilon保护；保留原eig顺序及实部处理，不修改上游源文件。
这属于明确披露的本地数值适配，不应宣称逐项复现原论文实验。
协方差仅使用截至2.88 s的已到达波形，不读取未来波形。
预检包括288点随机输入的前向/反向有限性、零输入有限性、同批其他记录放大100倍时首条预测不变、训练/验证/测试各32条真实短窗口（共96条）的协方差/特征及输出有限性，以及0/90/180/270/359.9度的编码恢复。
这些是启动前抽查，不保证每条数据都没有问题；正式训练遇到非有限输出或梯度会保留已完成epoch并明确报错。

## DistPT未启用

原网络同时输出距离头与走时头，各含均值与log-variance。当前数据缺少走时监督，已编写的distance-only适配冻结并忽略走时头；用户决定本轮不训练它。
原配置kernel6、11层膨胀尺度1..1024、每块两次卷积，对应理论感受野20471点（204.71 s）。288点虽能前向/反向，但大量上下文来自左侧零填充；不能保证短窗能发挥原结构优势。

只编写和执行预检，不由本脚本作者自动启动正式训练。
