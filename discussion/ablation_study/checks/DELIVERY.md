# 交付检查

- 默认新增训练6个模型：三个单任务，以及C0/C1/C2三种训练权重消融。
- 现有8K与PureGRU严格加载成功，分别8263/5791参数，原训练跳过。
- PureGRU历史run_config的已核实设置未发现差异；原训练源码快照缺失的限制仍保留。参数匹配PureGRU没有现成权重，按要求不实现或训练。
- 全部新模型前后向、冻结q、单任务删除无关输出/q检查通过。
- 原三任务损失及梯度逐值一致；统一时间权重计算核对通过。
- 合成CPU恢复测试与连续训练的模型、AdamW、CyclicLR和训练历史逐位一致。
- 圆周359°→1°变化为2°、球面零定位误差检查通过。
- D0/D1/D2短序列及一条真实2976点、93更新的一致性与MACs检查通过。完整记录的检查数据在full_record_cache_check；它不是正式多记录CPU性能结论。
- 已用原有真实8K/PureGRU统计和实际短CPU测量检查分析图画布，不绘制虚构实验结果。
- 未启动真实数据训练，正式results仅有占位文件，不存在活动正式实验指针。

预检结果：preflight_report.json、reference_audit.json、synthetic_checks.json、preflight.log。

正式启动：上一级run_ablation.bat或run_ablation.py。首次按配置执行，之后默认跳过已完成项目，完整训练状态可恢复。
