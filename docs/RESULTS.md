# 结果与评价边界

最终交付链为 `VISUAL_CALIBRATED_SCALE_AWARE_FINAL154_V001`，目标实例采用输出等价的 `visual_fast_v002` 入口。这里汇总已保存的本地评价和完整运行证据；没有官方隐藏测试成绩、名次或获奖回执。

项目面向不同拍摄视角的视频。每次推理处理一个视频，“多视角”表示跨视角适用场景，不表示同步多摄像机标定、三角测量或跨摄像机融合。“三通道”指同一次球检测的不同时间通道，也不是三个摄像机。

## 数据分别承担什么作用

| 数据 | 用途与边界 |
| --- | --- |
| 开发 121 段、10 个来源组 | 用于研发、来源分组留出及部署拟合。反复参与开发，来源留出结果不能称为独立盲测。 |
| 原保留 33 段 | 前期用于冻结方案的预定比较；自 2026-11-10 12:19:12+08 起并入最终 154 段训练，此后的结果属于训练内回归。见[阶段切换及拟合摘要](../evidence/final154_comparison.json)。 |
| 外部 130 段、48 个来源组 | 使用混合 AI 视觉参考与局部人工补标；有未知区域、时间和坐标不确定性，且已多次参与版本取舍。与开发材料的内容独立性未充分证明。 |
| 官方公开 15 段 | 已曝光的运行回归和本地计分参考。总分按公开任务规则复算，部分忽略区实现属于有记录的本地假设，不等于官方隐藏评分。 |

外部 SAFE 主口径先匹配明确事件，再对特殊接触、未决类型、未知区域等按固定协议处理。时间评价以代表帧 ±1 帧为条件，共 813 个可评事件；时空评价再要求估计坐标距离 ≤50 像素，共 788 个可评事件。两者分母不同，不能直接把 TP 差当作纯坐标错误。[固定五版证据](../evidence/external130_five_versions.json)

## 121 段来源留出：视觉校准的开发效果

| 固定方案 | 严格时间 TP / FP / FN | 严格时间 F1 | 40 ms 时间 F1 |
| --- | --- | ---: | ---: |
| 原 CHANNEL_TABLE | 601 / 112 / 79 | 86.2886% | 89.9494% |
| 校准局部视觉＋保持分支分数尺度 | 600 / 93 / 80 | 87.3999% | 90.7096% |

严格时间 F1 提高 1.1113 个百分点，10 个来源组中 7 升、2 降、1 平；主要收益是减少误报，净少 1 个命中。严格时间与 40 ms 是内部时间评价，均未加入正式落点的坐标条件，不能写成比赛总分。[研究证据](../evidence/visual_calibration_research.json)

## 外部 130 段：固定版本参考评价

| 版本 | SAFE 时间 F1 | SAFE 时空 F1 | 证据 |
| --- | ---: | ---: | --- |
| 官方基础版 | 60.0257% | 57.7807% | [五版对照](../evidence/external130_five_versions.json) |
| 历史完整终版 | 77.2757% | 74.3624% | [五版对照](../evidence/external130_five_versions.json) |
| CHANNEL_TABLE，121 段拟合 | 80.0238% | 76.5957% | [五版对照](../evidence/external130_five_versions.json) |
| CHANNEL_TABLE，154 段拟合 | 81.7204% | 78.5321% | [154 固定比较](../evidence/final154_comparison.json) |
| 最终校准视觉，154 段拟合 | 82.1782% | 79.0108% | [视觉固定比较](../evidence/visual_calibration_final_scores.json) |

各行沿用同一外部参考标注和 SAFE 口径，但来自不同时间的固定批次，不能据此比较稳定运行速度。最末两行是同批原版／视觉候选配对：时空 TP / FP / FN 从 642 / 205 / 146 变为 623 / 166 / 165；F1 上升伴随 19 个净命中损失和 39 个参考误报减少。48 个来源组中 14 升、14 降、20 平，不能描述为所有场景都改善。[完整配对证据](../evidence/visual_calibration_final_scores.json)

这些 FP 是参考协议下的未匹配预测，并非逐例人工确认的真实误报。最终结果支持该批候选的取舍，不构成全人工精确真值、独立泛化或隐藏比赛成绩。

## 公开 15 段与完整运行

| 场景与版本 | 本地总分 / 100 | TP / FP / FN | 完整推理 FPS |
| --- | ---: | --- | ---: |
| 研发实例，同批 CHANNEL_TABLE 154 | 87.6768 | 80 / 20 / 12 | 66.3 |
| 研发实例，同批校准视觉 154 | 88.5942 | 78 / 15 / 14 | 61.4 |
| 目标实例，最终等价提速版联合自测 | 88.3180 | 79 / 18 / 13 | 53.0 |

前两行来自[视觉固定比较](../evidence/visual_calibration_final_scores.json)；最后一行来自[目标实例公开评分](../evidence/captain_final_public_score.json)和[联合自测回执](../evidence/captain_final_release_receipt.json)。目标实例处理完 15 段、2,109 帧；同次联合包的篮球分支处理完 2 段、1,500 帧，整体退出码为 0。两题输出均与目标实例提速前的热启动结果逐字节一致。

本地精度分对 0°、45°、90°三个视角分别聚合 F1，再各乘 25 分；速度最多 25 分。TP 同时要求帧差 ≤1、欧氏坐标距离 ≤50 像素。因此总分既不是合并事件 F1，也不能与上面的时间 F1 混用。FPS 使用官方运行框架的推理计时，含解码、方案执行、写出及运行器开销，准备和引擎构建另计。[计分规则与计时字段](../evidence/captain_final_public_score.json)

研发实例与目标实例的引擎和运行条件不同，输出也存在差异，应分别报告。最终目标实例的一次验收达到 53.0 FPS，不能承诺每次运行都保持该速度，也不写成已达到 55 FPS 目标。

## 原始工作区证据索引

以下为原始工作区相对路径，用于追溯；仓库中的公开副本见各指标旁链接。

- `experiments/20261109_external130_fixed_comparison/analysis/results_v001.json`
- `experiments/20261109_external130_fixed_comparison/实验记录.md`
- `experiments/20261109_final154_refit/analysis/final_comparison_v001.json`
- `experiments/20261109_final154_refit/实验记录.md`
- `experiments/20261110_visual_calibration/analysis/research_result_v001.json`
- `experiments/20261110_visual_calibration/runs/final_scores_v001/summary.json`
- `experiments/20261110_visual_calibration/analysis/final_decision_v001.json`
- `experiments/20261110_visual_calibration/实验记录.md`
- `experiments/20261110_captain_frame_overlap/runs/selftest_fast_v003/public_score.json`
- `experiments/20261110_captain_frame_overlap/runs/selftest_fast_v003/release_receipt.json`
- `experiments/20261110_captain_frame_overlap/实验记录.md`
