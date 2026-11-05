# CompactGRU 对比模型源码

本目录仅收集上游模型与训练代码。未启动训练，未制作统一数据，未修改 CompactGRU 或上游模型。

## 已下载

源码位于 `upstream/SeisT-main/`，来自 https://github.com/senli1073/SeisT ，保留上游 LICENSE、README、依赖和配置。

|模型|模型主体|训练入口与配置|实现来源|
|---|---|---|---|
|SeisT|`upstream/SeisT-main/models/seist.py`|`main.py`、`training/train.py`、`config.py`|SeisT 作者官方 PyTorch 实现|
|MagNet|`upstream/SeisT-main/models/magnet.py`|共享上述训练框架；配置名 `magnet`|SeisT 作者提供的 PyTorch 复现，不是 MagNet 原作者实现|

以上路径除特别注明外均相对于上游目录。`datasets/`、`training/preprocess.py` 和 `utils/` 保留以支持后续审查训练依赖；不代表已经适配本项目数据。下载包中附带的示例、图片、权重等保留上游原状，不作为本项目训练结果。

SeisT 的震级、方位角和震中距配置分别为 `seist_.*?_emg`、`seist_.*?_baz`、`seist_.*?_dis`，不是默认联合三任务输出。MagNet 仅用于震级任务，其双输出应结合上游损失和后处理解释，不能直接视为两个震源参数。

## E3EW / E3WS 状态

按前文语境将用户所写 E3EW 理解为 E3WS；官方来源：https://github.com/PabloELara/E3WS 。

根据用户后续要求，E3WS 也纳入对比，保留原本的特征加集成学习方法，不强制转换为 PyTorch。

官方源码已下载至 `upstream/E3WS/`，包括特征计算函数、实时推理示例和检测器训练脚本。具体入口、数据要求与复现缺口见 [E3WS_REPRODUCTION.md](E3WS_REPRODUCTION.md)。源码已逐文件通过 GitHub blob 哈希核对，记录在 `E3WS_source_manifest.json`。

尚未下载约8GB的全部上游资产，尤其是各窗口的预训练 joblib 模型与示例波形。完整文件清单保留在 `downloads/E3WS-tree.json`。当前是源码收集阶段，不代表已完成 E3WS 训练复现；震源参数训练脚本缺口需要后续依据论文和公开材料补齐。

## 后续数据统一前需核对

- 采样率、分量顺序、单位、震前窗口、归一化和滤波要求。
- 震级类型、震中距/震源距、方位角方向和损失函数。
- 使用项目固定划分，训练选择仅依据验证集；原数据加载器不可直接当作统一协议。
- SeisT 和 MagNet 当前代码不能视为已有 CompactGRU 式缓存增量接口。在线比较需要仅使用已到达波形并重新计算，不能读入未来波形。

## 溯源

`source_manifest.json` 保存下载来源、源码包 SHA256 和本地文件哈希。`downloads/SeisT-tree.json` 保存 GitHub API 源码树快照；该快照与源码包分开获取，具体文件匹配情况见 manifest。

本次仅核对静态源码和下载完整性，未安装依赖，未进行模型前向或训练验证。
