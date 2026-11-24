# Qwen 机制与策略搜索

本入口复用 `find_pareto_frontier` 的镜像、冻结、基线、指标和预算工具，使用独立 Qwen 配置及运行目录。研究 agent 可在隔离工作树中跨模块修改 Rust/Python 实现及公共接口，包括 engine、graph、resources、KV、系统、调度器、运行器、worker/bridge 和 Rooflang 执行模块；可自行新增模块与测试，不受既有优化名称限制。模型事实、专家选择、硬件/基础定价、输入、评分器、控制器、外部可信测试和依赖保持冻结。具体边界见 `candidate.py` 与 [跨模块契约](CROSS_MODULE.md)。

## 启动

```bash
# 静态检查，不启动 Docker、仿真或 API。
./run_qwen_moe_agent.sh --check

# 尚无匹配共用镜像时构建；已有匹配镜像可跳过。
./run_qwen_moe_agent.sh --build-only

# 本地完整基线 → 封存 → Docker agent 搜索。
BUILD_IMAGE=0 ./run_qwen_moe_agent.sh
```

也可以分开执行：

```bash
./run_qwen_moe_baseline.sh
./run_qwen_moe_agent.sh --resume agents/qwen_moe_optimization/runs/运行时间
```

`--prepare-only` 只冻结副本；`--resume` 继续原运行，复用封存基线、候选和剩余预算，不新建研究。新运行会创建 `runs/YYYYMMDD-HHMMSS/`。当前不支持把另一 run 的基线直接导入新 run。

本次扩展需新建 run。`--resume` 维持旧 run 冻结的实现和权限，不会自动升级搜索范围。本次只修改 Qwen 运行文件，没有增加共用镜像构建输入；已有匹配镜像无需因此重建，可用 `BUILD_IMAGE=0 ./run_qwen_moe_agent.sh` 新建研究。

模型、资源、完整负载、SLO、研究 agent 和预算统一在 `experiment.toml`。研究 agent 与被模拟的 Qwen 模型不同。默认凭据复用 `agents/find_pareto_frontier/agent.env`，可用 `RUNTIME_ENV` 覆盖；不复制进冻结源码，不传给候选执行容器。

用户提供的《推理的计算与数据搬运》原 PDF 随 `inputs.assets` 冻结，容器内路径查 manifest；[完整 33 页文本](references/compute-data-movement.txt) 与[页码/源码/消融映射](references/compute-data-movement.md) 随可信源码提供。先读全文，再用当前基线证据检验启发；不得套用 Kimi/Blackwell 预测数值，也不将文档方向作为新白名单。

`BUILD_IMAGE=0` 只跳过构建，仍校验镜像身份。首次共享 launcher 或依赖变化后需重建；一般候选源码变化无需重建环境镜像。构建沿用专用 host 网络 builder 和脚本中的代理，不修改 Docker 全局配置；构建结束停止本次专用 builder，保留缓存。

## 研究循环

agent 读取基线与诊断，记录瓶颈、替代解释和可证伪假设，再跨模块实现并提交固定案例。宿主按实际差异选择可信检查；机制提交须附结构化 plan，给出状态 owner、不变量推导、测试和封存证据。评测在无网络、无 API 凭据的独立容器进行；停止后宿主重算指标和保存图。agent 无权修改可信源码、输入、基线、评分或控制器账本。

默认 `research.workflow="bottleneck-v1"`：所有新提交（含纯策略）须带 plan.research；Pareto 额外比较总输出吞吐/GPU 和平均单用户速度。`submit.py --status` 读取宿主预算、方向覆盖和诊断入口。基线诊断在 `artifacts/baseline-diagnostics.json`，候选诊断随 metrics 保存并哈希；额外原生采样仍默认关闭。

agent 提前退出后，宿主排空已提交任务并在剩余预算内续开会话，继续使用原 worktree、notes、候选和账本。默认连续 3 次无新评测退出标记 STALLED（退出码 3）；预算耗尽标记 BUDGET_EXHAUSTED（124），用户取消为 CANCELLED（130）。这些状态均不代表已验证。到达 SLO 不停止；新模式为最终重放预留全局时间。缺少冷重放样本时按单次超时上界估算，`full_matrix_replay_fits=false` 是预算警示，不伪造完成保证。旧配置不设置 workflow 时保持单会话行为；旧 run 的冻结控制器不变。

plan.research 示例（合并到下方 plan 对象）：

```json
{"research":{"name":"减少末尾 Decode 排队","family":"batch_decode","causes":["长输出请求集中在一个组","替代：KV 依赖阻塞"],"discriminating_check":"比较两组队列及 KV ready 时刻，固定策略做消融","expected_metrics":["output_tokens_per_gpu","tokens_per_user"],"implementation":"联合路由与 Decode 批处理；不足时扩展运行器接口","overhead":"新增观察有界；用相同输入做配对 wall/RSS 检查并报告样本"}}
```

研究结构和覆盖不能证明因果；证据仍须引用封存 baseline/trial。主因及至少一个替代解释须不同，family 开放。优先填满最有希望候选的完整矩阵，再花冷重放预算；`phase=validate` 仅接收 replay。新增热路径需配对宿主性能验证，不能混用编译耗时、模拟服务时间与真实 GPU 实测。

可选压力套件：运行 `python agents/qwen_moe_optimization/workloads.py --help`，按 `experiment.pressure.toml` 生成新目录下的 experiment.toml，再将生成文件传给 `--experiment`。原100请求保持不变，额外三个24请求负载明确标为合成、可见 validation，不能宣称隐藏 holdout；预算继承原配置，运行前检查完整矩阵复验是否可容纳。

在研究容器内：

```bash
CLIENT=/workspace/simthesizer/agents/qwen_moe_optimization/submit.py
python "$CLIENT" --list
python "$CLIENT" --scope
python "$CLIENT" --case CASE --reference \
  --plan /workspace/notes/plan.json \
  --hypothesis "改变哪项决策及预期结果" --bottleneck "压力位置与替代解释"
```

`--scope` 只预览变更和 `impact.required_checks`，不构建、不仿真。机制 plan 中的 checks 必须逐项覆盖该清单；宿主重新推导，不能靠少填减少验证。已有冻结外部测试和宿主资源 fixture 是可信检查；实现文件内的单元测试可修改，属于补充证据。新增 Python 测试须是 `tests/test*.py` 的 unittest 入口，plan 的 Rust 用例须是已有 crate 的 `crates/*/tests/*.rs` 集成测试。

提交通过队列同步等待结果，一次评测一个完整 case。`--reference` 选封存参考策略，**不负责关闭机制**；开关需由候选实现。`--ablation-of ID` 只是关联标签。`--replay CANDIDATE_ID` 使用封存源码及 plan，不接受新 `--plan`；`CANDIDATE_ID` 是 64 位源码身份。重放须保持原策略选择，并比较 `result.jsonl` 与 `service_finish_s`。

## 机制计划示例

以下仅示范字段；将 `CASE` 换成 `--list` 的 case ID，将 tests 换成已新增并封存的实际文件。每个 `--scope` 返回的 required check 都要填写一项，不能只复制下面的一项。

```json
{
  "schema_version": 1,
  "mode": "control",
  "summary": "关闭新资源仲裁机制，验证参考行为",
  "state_ownership": "Resources 持有预约，完成或取消时释放",
  "preserved_semantics": "固定工作量、物理容量和依赖，保持请求全部完成",
  "expected_effect": "关闭版本逐 token 与基线一致",
  "alternatives": ["压力来自 HBM，改变仲裁不能改善"],
  "stop_conditions": ["关闭回归失败或新机制增加延迟尾部"],
  "evidence": [{"kind": "baseline", "id": "CASE", "observation": "填写该案例实际观察及量纲"}],
  "control": {"kind": "baseline", "id": "CASE"},
  "checks": [{"id": "baseline_compatibility", "argument": "关闭时仍执行原有决策路径", "tests": ["tests/test_new_mechanism.py"]}]
}
```

先提交实际关闭版本，`mode=control` 要求结果逐 token 和服务结束时间复现引用参考。成功后提交开启版本：改 `mode` 为 `candidate`，`control` 指向该次 control 的 `{"kind":"trial","id":"32位提交ID"}`，更新假设与检查。也可引用同 case 的其他封存 trial 作为证据；笔记路径和任意目录不能代替证据身份。匹配比较还核对策略种类和源码身份。

逐 case 评测并干净重放后，控制器写 `artifacts/review-candidates.json`，汇总全案例覆盖及待补证据。`ready_for_review` 仅表示可进入审阅；结果始终 `EXPLORATORY`。`comparison_reference_reproduced` 说明引用 control 复现了参考，不证明后来 candidate 的同源关闭路径正确，还需专门审查和区别性检查。

按相同冻结案例保留对照：A 当前基线、B 旧机制配优化策略、C 新机制配参考策略、D 新机制配优化策略、E 关键组件消融。机制必须由瓶颈证据支持；固定物理资源、模型专家选择、请求工作量、基础成本和评分口径，不得截断输出或降低成本常数换取收益。

失败、超时、重试和重放消耗预算。停止后恢复原 run，不清零历史。退出只处理本 run 记录的容器，不执行全局 prune 或停止其他人的 Docker。

## 当前结论边界

控制器记录 `EXPLORATORY` / `EVALUATED` 与干净重放是否一致，待审清单也保留该证据边界。完整请求、固定规则通过、匹配 control 和结果可重复，不足以证明任意新机制的成本与因果正确，不能自动宣称 `VERIFIED` 或真实 GPU 性能已验证。

可信事件 driver / 私有路由 oracle 隔离尚未实现；当前不是强因果隔离，也没有不可见 holdout 保证。新增机制应提供工作量、依赖、容量、KV 生命周期及关闭回归证据；超出独立检查能力的结论保留为探索结果。设计文档中的未来权限与验证能力不能视为本入口已经兑现的保证。
