# 接入与验收计划

状态：核心策略、评测、归档与宿主复验已接入；本文件保留实现前的映射与验收要求。实际接口与边界见 [DESIGN.md](DESIGN.md)，命令见 [LAUNCHER.md](LAUNCHER.md)。尚未执行 Docker/付费 agent 搜索。

## 实现前的代码依据

检查日期：2026-11-18。HEAD：`06e82c5ca4cbb49c7fb7abaa386a4d93acf9b898`，另有用户工作树修改；HEAD 不能单独标识本稿检查的代码。后续实现/实验须保存实际源码快照及哈希，保留当前未提交修改。

| 位置 | 已确认行为 |
|---|---|
| [Scheduler](../../crates/simthesizer-core/src/schedulers/scheduler.rs) | 已可注册替换调度器；`schedule()` 没有时间、全局观察和显式唤醒结果 |
| [ChunkedPrefillScheduler](../../crates/simthesizer-core/src/schedulers/chunked_prefill.rs) | `batch_requests()` 取队列前缀；`done_iteration()` 按相同前缀推进和完成，不能只替换选取算法 |
| [SingleInstanceSystem](../../crates/simthesizer-core/src/systems/single_instance.rs) | `LlmSchedule` 空 batch 直接返回；enqueue 和 batch complete 才产生下一次调度 |
| [PdDisaggregationSystem](../../crates/simthesizer-core/src/systems/pd_disaggregation.rs) | `add_request_arrival()` 中轮转选择 P/D 并建立 KV plan；顶层通常发生在装载 workload 时 |
| [MultiInstanceSystem](../../crates/simthesizer-core/src/systems/multi_instance.rs) | 真正 arrival 回调中使用当前负载路由，可参考其事件结构 |
| [KvRuntime](../../crates/simthesizer-core/src/kv.rs) | `Produce` 或 `PrefillDone` 直接 `copy()`；复制创建时 pin，ACK 释放；就绪区间已索引 |
| [RooflangModelRunner](../../crates/simthesizer-core/src/model_runners/rooflang.rs) | D Read 依赖构图时已存在的 ACK；尚未发送的传输需稳定就绪门 |
| [资源域](../../crates/simthesizer-core/src/resources.rs) | `admit()` 按容量/槽位准入，`rates()` 共享速率；没有可调带宽优先级 |
| [Engine](../../crates/simthesizer-core/src/engine.rs) | 持有唯一时钟，结束检查资源、KV 与剩余节点；观察回调在同一时刻事件处理后运行 |
| [结果分析](../../scripts/results.py) | 验证 token 数，重建时间，计算 TTFT/TPOT/ITL 和服务吞吐；尚无 SLO 联合达标指标 |
| [检查入口](../../scripts/inspect_rooflang.py) | 默认仅指标与图，reports/full-graphs 选择性开启 |

相关已有设计：[整体架构](../../ARCHITECTURE.md)、[共享 Rooflang](../../architectures/rooflang-shared.md)、[验证](../../architectures/validation.md)、[验证工具](../../.agents/tools/validation.md)。架构文档是摘要；本稿关于路由时机和 batch 身份的判断以实际实现为准。

参考 launcher：`/mnt/nvme1n1/data/lmk/PROJECT/TEST2/rooflang/agents/find_pareto_frontier/{AGENTS.md,Dockerfile,run-agent.sh,entrypoint.sh}`。已静态阅读，未执行。其 Dockerfile 未提供本项目 Rust/uv 环境，Claude 安装段被注释；实际前端能力须在构建验收中确认。引用其组织方式，不复用凭据文件。

## 需求、所有者、接入与验收

R1–R10 来自用户确认的方案。表中为验收要求；已执行结果及未覆盖组合见 `output/serving-policy-validation/`，不把整个测试矩阵视为已经通过。

| ID / 需求 | 状态、触发与依赖 | 接入与成本边界 | 区分性验收 |
|---|---|---|---|
| R1 固定资源 | 配置/硬件文件在 run 创建时冻结；实例持有设备映射，资源域持有容量/速率 | launcher 验证冻结配置；候选只覆盖声明的策略参数，不改权重、显存利用上限、GPU ID、带宽或成本 | 候选增加设备/带宽、减小权重或改变精度均拒绝；GPU 共享按物理 ID 去重，不能把 P/D 分开少算 |
| R2 完整完成 | System 持有请求，Scheduler 持有已提交进度；完成沿 batch→子请求→根请求推进 | 保留 Engine/KV/resource drain；固定评测器比较原始输入与结果的全部 ID、任务内容和 token 数 | 漏掉一个长请求、少一个输出 token、重复完成、死锁、超时均不能评分为成功；零/一输出单独覆盖 |
| R3 当前状态路由 | PD arrival 事件中读取已到达状态，选 P/D 后创建计划；不迁移已有计划 | 将顶层预先路由移入原生 arrival，复用 multi_instance 的事件模式；成本仍由选定拓扑路径计算 | 前序请求改变实例负载时，后续路由能变化；同一历史、不同未来到达的案例，在分叉前决策一致 |
| R4 任意 batch 与唤醒 | Scheduler 保存权威队列和带身份 BatchPlan；batch 完成按 ID 提交；System 持有唤醒节点 | 统一 BatchPlan 向 runner 提供 shape/请求对应，替换队列前缀假设；保留轮次完成提交边界 | 队列 A/B/C 选择 A/C：B 不推进，A/C KV 与 token 不串号；无后续到达时，主动等待能按时唤醒；同刻多触发不重复执行 |
| R5 KV 发送控制 | KvRuntime 建立 pending intent、pin 和持久 ready gate；ACK 更新区间/引用并触发策略 | `kv.rs` 分离意图与发送；runner Read 依赖 coverage gate；源/目标预留、bytes、HBM/网络成本不变 | KV 在 P 完成后仍未发送，源不能释放且 D 不能读取；ACK 后就绪并唤醒；慢链路共享竞争、跨 TP 分片逐项守恒；whole/layer 基线覆盖 |
| R6 合法确定性策略 | 每次试验一份 policy memory；执行器根据触发快照检查动作后提交 | 新增策略边界，注册方式沿用项目组件模式；agent 只能编辑候选实现，不能持有核心可变状态 | 未到达请求、重复成员、负/越界 q_len、错误实例、重复发送、过去时间唤醒失败；同代码/输入/种子重现决策与 token 时序 |
| R7 SLO 与前沿 | results.py 权威计算；experiments 追加记录，frontier 是派生视图 | 按 DESIGN 公式扩展统一分析，archive 只做分组/支配；保留所有统计样本口径 | 手算例复核；单输出 TPOT 为 N/A；一个请求仅 TTFT 达标另一个仅 TPOT 达标时联合率为 0；分组隔离、等值、支配、不同取舍、不可行点均覆盖 |
| R8 负载与泛化 | 评测器生成到达变体、冻结数据划分；holdout 最终外部使用 | 保持内容和原生到达顺序；同一候选/案例身份才能复用；完整矩阵晋级 | m=2 的相对到达间隔减半，token/请求不变；同刻顺序保留；训练、验证、holdout 泄漏检查；100 请求结果不被标为持续容量 |
| R9 Docker 与恢复 | 宿主拥有容器生命周期/预算，候选副本与 artifacts 持久化 | 按内容身份编译/缓存；冻结 core 与 evaluator；外部干净复验；版本更新新建实验 | 宿主源码哈希不变；越界修改拒绝；受控中断后只补缺失案例；不兼容配置不能续用旧分数；并发 run 隔离、OOM/退出/超时正确分类 |
| R10 轻量与可解释输出 | 默认结果/指标/图；诊断模式决策关联原生节点；详细历史按需保存 | 复用 inspect/report/plot，禁止为每次试验建全图；记录实际墙钟/内存但不当 GPU 指标 | reports 开/关逐 token 结果一致；指标图存在；定向报告能定位策略动作、KV 等待与全局时间；存储预算触发后前沿证据仍可复现 |

核心路由、按 ID batch、定时唤醒、KV pin/就绪门已有原生测试。实际 Docker/搜索、专用策略解释时间线、磁盘配额及独立 holdout 尚未完成；不以已有测试替代这些覆盖。

## 开发顺序与完成门槛

### 本轮接入约束（2026-11-18）

用户已授权实现接口，并要求避免无用耗时。沿用默认实验的 known_trace 与零模拟调度开销；新增 R11：策略仅在语义事件触发，运行时为 Rust，无逐事件 LLM/Python 调用、轮询或重复预演。未启用策略和 reference 保留 FCFS/立即 KV 发送快路径；candidate 才构造只读队列与待发送观察。固定大小策略内存不持有核心状态。

PD 持有唯一共享 PolicyRuntime，向实例与 KV 域传入引用。candidate 路由在实际 arrival 执行；batch 选取与完成以稳定完成节点 ID 对应。KV pending 提前持有源引用，实际 ACK 按 layer/rank/head 的区间覆盖释放显式 condition 节点；未满足 condition 保持阻塞并参与原生 drain，不能靠额外虚拟延时隐藏依赖。

性能与语义验证使用 `output/serving-policy-validation/` 中的改造前源码/二进制、场景与完整 100 请求。先比较逐 token 时间，再比较配对墙钟和 RSS；不把一次计时波动当作性能结论。评测结果按完整身份复用，编译缓存不复用未知候选结果。尚未执行的组合仍须明确列出。

### D0 固定评测契约

落实 A1；填写 SLO、预算、场景/负载和划分。统一指标新增 goodput/达标率及输入完整性检查。先使用现有策略建立基线与负载曲线，保留逐 token 结果和图。R1/R2/R7/R8 的小案例先通过。

### D1 最小策略边界

core 的 `policies` 提供只读观察、明确动作、有界状态和动作校验；保留无策略入口。agent 只能编辑 candidate 子目录，不能修改 scheduler/kv/resource 或依赖。

先把当前轮转路由、FCFS/chunk 及 whole/layer 行为表示成参考策略。只改接入方式时，固定案例的逐 token 时间与原路径配对；若有差异，追到首个事件并解释，不以“均值接近”通过。检查同刻事件和多实例交互的次序是否发生变化。

### D2 调度与 KV 机制

实现 R3/R4/R5：真实 arrival 路由、显式 BatchPlan、定时/条件唤醒、PendingSend pin、覆盖就绪门。整套参考策略和 native 不变量通过后，再开放动态 chunk、batch 重排、KV 并发窗口和多实例负载路由。

策略能选择不提交，但执行器必须保留未完成工作、可触发的订阅和 drain 检查。不能因为没有新事件而提前写出成功结果。无可行配置/动作时报告失败，不通过缩短 workload 或放宽容量完成。

### D3 容器 agent 与归档

实现此目录的 Dockerfile、launcher、entrypoint、evaluate、archive。`evaluate` 调用固定分析实现；`archive` 从结果记录重建前沿。每个允许的静态参数/动作能力写入 manifest；候选不能修改 Cargo 依赖、core 或评测代码。

先用固定、可预期的候选检查循环和恢复，不依赖大模型找到优化才能验证工具。随后启动小预算 agent 试验；预算、停止、恢复由宿主控制，不能仅依赖 AGENTS 中的文字。

### D4 独立复验与展示

当前外部复验在宿主独立目录从冻结源码及候选重建，使用干净 Cargo target，重新核对结果哈希与指标。独立复验容器与 holdout 尚未接入。

在 D0–D4 完成前，不能宣称已完成可替换策略搜索。机制扩展另开有独立成本/正确性证据的版本，比较“旧基线 / 新机制默认策略 / 新机制搜索策略”。

## 验证执行计划

- 原生：相关 Rust 单测、小型场景、旧策略逐 token 配对、依赖/KV/resource drain；行为改动完成后运行仓库要求的格式与 workspace 检查。
- 分析：手算指标、完成性反例、负载变换、候选分组/支配与恢复身份的 Python 检查。
- 容器：固定工具链构建、只读边界、资源限制、断点恢复、独立复验及输出图。
- 代表案例覆盖 TP 1/2/4/8、P/D TP 不同、多实例、同 GPU 共享和慢链路；覆盖不等于真实 GPU 精度校准。每个未验证组合单列。
- 默认无 trace；不变量失败或新机制诊断才开启定向 native trace。结果 timing 不能重建未记录的内部历史。

启动器边界见 [LAUNCHER.md](LAUNCHER.md)。机制和基线对照验证不代表 agent 已取得优化收益。
