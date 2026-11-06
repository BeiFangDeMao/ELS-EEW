# ELS-EEW

ELS-EEW 是用于单台站地震预警的轻量化因果流式模型，代码中的模型名称为 `CompactGRU`。模型接收 100 Hz 三分量波形，每 32 个采样点更新一次，输出震级、震中距和反方位角的 sin/cos 分量。

## 项目结构

- `els_eew/`：ELS-EEW 模型、因果预处理、训练与测试。
- `discussion/`：可靠性与预警评价、规模实验、消融、模型对比及真实事件回放。PureGRU 仅保留在消融目录。
- `plotting/`：共享绘图样式。

## 使用

```bash
pip install -r requirements.txt
python els_eew/train_CompactGRU.py
```

运行前在 `els_eew/experiment_config.py` 中设置 H5 数据路径、输出目录和训练参数。模型实现位于 `els_eew/models/compact_gru.py`，流式推理使用 `model.eval()` 和 `stream_step()`，将返回的状态传给下一次更新，新记录开始时重置状态。

讨论实验分别使用各目录的 Python 或 `.bat` 入口，配置位于对应目录的 YAML、JSON 或 Python 文件中。数据集、权重和结果需自行放入配置指定的位置。

## 许可与来源

项目许可见 `LICENSE`。第三方对比代码位于 `discussion/model_comparison/upstream/`：SeisT 来自 https://github.com/senli1073/SeisT （MIT），E3WS 来自 https://github.com/PabloELara/E3WS （CC BY 4.0），各自许可文件予以保留。MagNet、BAZ-Network 等实现来源见上游代码与来源清单。
