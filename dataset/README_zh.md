# ELS-EEW：数据索引与日本强震数据获取指南

## 概述

本归档提供用于识别和重建 **ELS-EEW（Edge-oriented Lightweight Streaming Earthquake Early Warning，面向边缘计算的轻量化流式地震预警模型）** 研究所用地震波形数据集的相关材料。其目的在于，在不重新分发第三方强震波形文件的前提下，提高数据筛选过程的透明度，并支持研究实验的复现。

本研究使用的地震观测数据来源于日本防灾科学技术研究所（**National Research Institute for Earth Science and Disaster Resilience，NIED**）运营的 **K-NET和/或KiK-net强震观测网络**。研究人员须遵守NIED的注册程序和数据使用条件，自行从NIED官方渠道获取相应波形记录。

**本归档不包含任何NIED地震波形文件。** 随附的记录索引和数据集构建材料旨在帮助研究人员识别本研究使用的记录，并在已公开材料所支持的范围内重现作者的数据集构建过程。

## 1. 原始数据来源与引用

- **数据提供机构：** 日本防灾科学技术研究所（NIED）
- **观测网络：** K-NET（Kyoshin Network）和KiK-net（Kiban Kyoshin Network）
- **官方数据网站：** https://www.kyoshin.bosai.go.jp/en/
- **原始数据集DOI：** https://doi.org/10.17598/NIED.0004

使用上述记录时，请引用NIED原始数据集：

> National Research Institute for Earth Science and Disaster Resilience. (2019). *NIED K-NET, KiK-net* [Dataset]. National Research Institute for Earth Science and Disaster Resilience. https://doi.org/10.17598/NIED.0004

上述DOI对应的是**NIED原始数据集**，而非作者整理并归档的记录索引。

## 2. 如何获取日本强震波形数据

### 步骤1：注册NIED数据访问账号

下载K-NET和KiK-net波形数据前，需要按照NIED的要求完成用户注册。

1. 打开K-NET/KiK-net官方网站：https://www.kyoshin.bosai.go.jp/en/
2. 找到 **User Registration（用户注册）** 或 **New User Registration（新用户注册）** 入口。
3. 阅读NIED的数据使用说明，并通过官方注册页面完成相关信息填写：https://hinetwww11.bosai.go.jp/nied/registration/
4. 完成注册流程，获取数据下载所需的账号凭据。
5. 根据NIED网站后续提示完成必要操作。注册和访问流程可能发生变化，应以官网实时说明为准。

如遇账号注册或访问问题，可参考NIED官方[常见问题页面](https://www.kyoshin.bosai.go.jp/en/faq/)中的联系信息。

### 步骤2：查找相应地震记录

NIED提供多种官方波形检索及下载方式。

**方式A：按地震事件检索（适用于已知地震事件）**  
https://www.kyoshin.bosai.go.jp/en/eqdownload/

- 根据发震日期及其他已知事件信息检索目标地震。
- 选择对应事件，查看 **Strong-Motion Record List（强震记录列表）**。
- 结合本归档提供的记录索引，核对台站代码和所属观测网络。
- 选择所需波形格式（例如 **K-NET ASCII、K-NET Binary或K-NET CSV**），通过NIED认证下载服务获取记录。

**方式B：按数据条件检索（适用于台站或筛选条件检索）**  
https://www.kyoshin.bosai.go.jp/en/dtdownload/

- 使用日期、观测网络、台站代码、震级、震中距等条件进行查询。
- 将检索结果与本归档的记录索引进行比对。
- 通过NIED官方服务下载相应波形。

**方式C：通过HTTPS下载（适用于批量获取）**  
https://www.kyoshin.bosai.go.jp/en/https_download/

- 使用本人注册的NIED账号进行身份认证。
- 查阅NIED官方说明，了解下载目录结构和经认证的`wget`命令示例。
- 仅下载重建研究数据集所需的事件及台站记录。
- **不要将账号、密码或其他凭据写入公开代码、代码仓库、日志或共享文件中。**

NIED官方HTTPS数据下载目录为：

https://www.kyoshin.bosai.go.jp/kyoshin/download/

例如，NIED文档中介绍了如下**目录命名模式**（仅为目录结构示例，不代表ELS-EEW实际使用的记录清单）：

```text
knet/data/YYYY/MM/YYYYMMDDHHMISS/
kik/data/YYYY/MM/YYYYMMDDHHMISS/
knet/zip/YYYY/MM/YYYYMMDDHHMISS/
kik/zip/YYYY/MM/YYYYMMDDHHMISS/
```

其中，`YYYYMMDDHHMISS`为NIED所使用的事件目录标识。应根据NIED网站的实际事件记录核对，不能仅凭四舍五入后的发震时刻或其他来源的事件编号推断目录名称。

### 步骤3：确认正确的波形分量

对于**K-NET**，三分量强震记录包括：

- `NS`：南北向加速度；
- `EW`：东西向加速度；
- `UD`：竖直向（上下向）加速度。

对于**KiK-net**，加速度计可能分别安装在地表和井下。在K-NET ASCII格式的文件命名中，NIED使用以下分量标识：

- `NS1`、`EW1`、`UD1`：井下观测分量；
- `NS2`、`EW2`、`UD2`：地表观测分量。

依据研究数据索引选择正确的传感器安装位置。除非研究的数据集构建方案明确要求，否则不得以井下记录替代地表记录，反之亦然。

对于K-NET ASCII文件，波形采样值为数字计数值，需要结合文件头中的 **Scale Factor（比例因子）** 转换为实际加速度单位。NIED提供的CSV格式包含物理量数值。请保留原始采样率及元数据，并按照本研究配套代码执行单位转换和数据预处理。

有关文件分量、格式及比例因子约定，请参阅[NIED官方FAQ](https://www.kyoshin.bosai.go.jp/en/faq/)。

## 3. 如何使用归档的记录索引

本归档提供的记录索引用于标识研究中选取的地震观测记录。使用时，应以**实际发布的索引文件中的字段和标识符**为准。以下仅说明常见的记录匹配信息，并不意味着本归档必然采用这些文件名或字段结构：

- 地震事件标识或事件目录时间；
- 观测网络（K-NET或KiK-net）；
- 台站代码，以及适用时的传感器安装位置；
- 记录时间和/或具体波形文件标识；
- 数据集划分（训练集、验证集或测试集），如果索引中包含该信息；
- P波到时及其他研究特定的筛选元数据，如果索引中包含该信息。

**重要说明：** NIED指出，部分事件元数据中的发震时间秒数可能为占位值（例如`00`或`30`）。因此，匹配记录时，应以NIED事件列表及台站/文件标识为依据，而不应将这些秒数直接视为精确的物理发震时刻。NIED的波形下载服务**不提供P波和S波到时拾取结果**；若本归档包含到时信息，这些信息属于研究过程中产生的结果，并非NIED官方拾取结果。

若某条索引记录无法与NIED中的波形唯一对应，请查阅配套的数据集说明文件，不要自行替换为其他地震事件或台站记录。

## 4. 重建研究数据集的基本流程

从NIED官方渠道下载相应波形后，按照以下流程重建数据集：

1. **保留原始文件。** 在本地受控目录中保存从NIED下载的原始波形及其文件头，不修改原始文件。
2. **匹配记录。** 使用归档索引确定对应的地震事件、台站、通道及波形文件。
3. **执行数据集构建。** 使用ELS-EEW项目提供的数据筛选、单位转换、波形截取及预处理代码，并采用与研究对应的配置参数。
4. **复现数据划分。** 根据归档索引中的训练集、验证集和测试集归属进行划分，而不是重新随机分配。
5. **校验结果。** 将记录数量、样本标识、波形长度及其他已公开检查项与相关研究的描述进行核对。

ELS-EEW源代码仓库：

https://github.com/BeiFangDeMao/ELS-EEW

如需复现特定论文中的实验，请使用该论文引用的**已归档软件版本**，不要假定GitHub最新的`main`分支一定能够复现既往实验。具体依赖环境与执行命令应以对应软件版本附带的文档为准；本文档不虚构尚未核实的脚本名称或命令行参数。

## 5. 数据访问、归属及再分发限制

K-NET/KiK-net原始波形属于NIED提供的第三方数据。NIED官方网站及其用户注册协议对数据复制和再分发设有限制。因此：

- 本归档**不镜像或重新分发NIED原始波形文件**。
- 本归档**不提供重新打包或预处理后的NIED波形文件**。
- 希望使用这些地震记录的研究人员，应通过**NIED官方服务**自行获取数据，遵守适用的数据使用条件，并引用NIED原始数据集。
- 作者自行整理的索引、说明文档或代码即使采用开放许可，也**不意味着获得重新分发NIED波形数据的权利**。

本归档中的记录索引与重建材料同样应在其适用的权利及使用条件下使用。如果对第三方内容的重复使用或再分发权限存在疑问，应在再次发布之前咨询NIED。

NIED数据使用声明：https://www.kyoshin.bosai.go.jp/en/  
NIED用户注册协议：https://hinetwww11.bosai.go.jp/nied/registration/

## 6. 如何引用本项目

如使用本项目提供的数据目录、标签、源代码、预处理脚本或模型权重，请引用**本项目 DOI**。这些材料属于同一个 ELS-EEW 联合归档，使用同一个项目 DOI，不分别引用数据目录 DOI 和模型软件 DOI。

使用 NIED 原始波形时，还应独立引用 NIED 原始数据集：

> National Research Institute for Earth Science and Disaster Resilience. (2019). *NIED K-NET, KiK-net* [Dataset]. National Research Institute for Earth Science and Disaster Resilience. https://doi.org/10.17598/NIED.0004

本项目 DOI 与 NIED 原始数据 DOI 对应不同资源，不能相互替代。

本项目提供的四列目录包含记录名称、震级、震中距和反方位角标签，不包含人工 P 波到时标注。因果处理脚本假设输入波形已按 P 波对齐；精确重建论文中的输入窗口仍需要相应的到时标注。

## 7. 官方参考资源

- K-NET/KiK-net官方网站及数据使用条款：https://www.kyoshin.bosai.go.jp/en/
- NIED用户注册页面：https://hinetwww11.bosai.go.jp/nied/registration/
- 按地震事件下载：https://www.kyoshin.bosai.go.jp/en/eqdownload/
- 按数据条件下载：https://www.kyoshin.bosai.go.jp/en/dtdownload/
- HTTPS下载与目录规范：https://www.kyoshin.bosai.go.jp/en/https_download/
- 文件格式、分量定义及常见问题：https://www.kyoshin.bosai.go.jp/en/faq/
- NIED K-NET/KiK-net数据集引用DOI：https://doi.org/10.17598/NIED.0004

---

**适用范围说明：** 本README介绍如何从NIED获取原始观测数据，以及本研究公开的资料如何支持数据集重建。它不表示第三方地震波形文件可以被自由再分发，也不意味着研究人员可以在不遵守NIED访问条件的情况下获得索引中列出的全部记录。
