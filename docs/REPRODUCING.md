# 复现与资产说明

## 归档内容与依赖

源码来自最后验收的联合包，再叠加已经验证的 `visual_fast_v002` 覆盖。只提取最终导入所需的 58 个新增或改动 Python 文件，原文件字节保持一致。官方框架、检测器和原权重不重新发布；旧 3D vendor、未采用支线权重也不加入最终运行依赖。

本仓库没有改造或重新训练模型。源码可阅读、编译和核验；实际视频推理需官方工程、模型资产和兼容的 NVIDIA GPU 环境。

## 获取材料

1. 自备官方 `participant.zip`，SHA256：`93fa695d27596f085131756073669a5c255486a935e2c7b94ef7719270126052`。
2. 从本仓库 [final154-20261110 Release](https://github.com/Livercirrhosis-creator/multiview-table-tennis-bounce-detection-2026/releases/tag/final154-20261110) 获取 `final154_learned_assets.zip`。它只有最终学习资产，不含官方球模型、视频、标签或凭据。
3. 在本地或 Linux 实例安装匹配的官方环境。不要用不受控的全量升级代替环境匹配。

最后验收环境记录为 Python 3.10、PyTorch `2.1.2+cu118`、NumPy `1.26.4`、TensorRT `8.6.1`、onnxruntime-gpu `1.18.1`、PyAV `17.1.0`、OpenCV `4.11.0`、支持 CUDA 解码的 ffmpeg、RTX 4090。还需与 PyTorch 匹配的 TorchVision、Pillow。固定版本来源于实际环境检查，不是“任意版本均可”的兼容性承诺。

## 恢复独立运行目录

在本仓库根目录运行：

```bash
python scripts/verify_archive.py --assets-zip /path/to/final154_learned_assets.zip
python scripts/prepare_runtime.py \
  --official-zip /path/to/participant.zip \
  --assets-zip /path/to/final154_learned_assets.zip \
  --output /path/to/new-runtime
```

准备脚本校验两份 ZIP、解压官方包到**新目录**，将原乒乓球 `solution.py` 保存为 `official_reference.py`，再复制本仓库源码与学习资产。框架、完整性清单、原球检测器、官方球模型和公开数据保持原字节。不会操作已有实验目录，也不会联网安装或启动训练。

Linux GPU 环境下：

```bash
cd /path/to/new-runtime/participant
python pingpang/integrity.py --verify
python pingpang/scripts/check_env.py
cd pingpang
python run.py --input public_data --output /path/to/new-output \
  --solution participant.visual_fast_v002:Solution --decoder gpu
```

`new-output` 应使用新的目录。这里直接调用原运行器，因为官方 `run.sh` 使用固定 `/participant` 路径；只有把恢复工程放到该平台路径时才适合直接用该脚本。GPU 动态库搜索路径须按官方环境配置。首次运行可构建与本机匹配的 TensorRT engine；引擎缓存不随本仓库分发。请按 `run_status.json` 区分准备时间、推理时间、完整视频数和格式检查。复现出的 FPS 依赖硬件、解码与运行负载。

此展示包只恢复乒乓球候选。官方包中的篮球仍是官方基础方案；不能把这个恢复目录当作原篮球 C01 联合交付包。历史联合自测 JSON 证明当时两题共同运行成功，不意味着这里重新分发了队友篮球实现。

## 五类学习模块

| 模块 | 权重路径，相对内层 participant | 拟合用途 |
| --- | --- | --- |
| 密集事件分类 | `weights/channel_adaptation_v001/dense/model.pt` | 154 段最终拟合 |
| 窗口存在与选帧 | `weights/channel_adaptation_v001/window/model.pt` | 154 段最终拟合 |
| 缺测补漏 | `weights/masked_gap_v001/model.json` | 154 段最终拟合；JSON 本身也是模型 |
| 桌框 | `weights/table_guard_v001/model_final.pth` | 154 段最终拟合 |
| 局部视觉 | `weights/visual_calibration_v001/model.pt` | 154 段最终拟合 |

配套归一化、桌框 metadata、视觉 calibration 一并随资产发布。视觉校准标量 `a=0.1785668588891146` 使用 121 段来源留出的预测拟合，不能写成 154 段训练内校准。完整文件名、字节数、SHA256 见 [assets.json](../metadata/assets.json)。

## 训练与评价源码的可复现边界

`research/` 保存当时的原始训练／评分实现，不是另写的一套示意训练代码：

- `20261108_channel_event_adaptation`：原始／增强观测、事件头训练与部署拟合。
- `20261107_masked_gap_decision`：自然缺测／遮挡增强的数据集和补漏分类器。
- `20261108_table_module_refinement`：桌框训练与评价。
- `20261109_final154_refit`：dense/window 各 80 轮、gap 200 轮、桌框 12 轮的最终拟合入口。
- `20261110_visual_calibration`：嵌套来源留出、标量校准、最终视觉拟合与固定配对评分。

这些历史脚本保留原相对路径、缓存身份及哈希约束；还依赖原始工作区中的视频、标签、来源划分、特征缓存、预训练初始化和辅助模块。**克隆本展示仓库不能直接完整重训全部历史实验。** 私有完整归档保留恢复材料，公开源码用于核查训练方法；不要删除脚本校验来伪装复现成功。

原 33 段自最终拟合起转为训练内；外部130与公开15多次参与固定取舍。不同数据池、时间容差和坐标要求见 [RESULTS.md](RESULTS.md)。证据 JSON 已去除本机路径，但数值不变；其原始 SHA 与公开副本 SHA 均在 [provenance.json](../metadata/provenance.json)。
