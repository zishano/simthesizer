# Qwen MoE 机制与策略自主研究

目标：固定模型语义、物理 GPU、负载、专家选择和成本口径，完整完成请求，提高系统吞吐和用户生成速度，同时满足 TTFT/TPOT。你的工作包括发现瓶颈、自主开发缺失机制、评测及反证，不能只扫描现有策略参数。

`research.workflow=bottleneck-v1` 时，宿主按 TTFT p99、TPOT p99、达标吞吐/GPU、总输出吞吐/GPU、平均单用户生成速度保留 Pareto 取舍。报告每个 case 的实际数值、百分比、模拟服务时间和尾部；固定输出/GPU 下服务时间与总吞吐互为倒数，不重复计为独立目标。达标吞吐改善但总吞吐不变、用户速度下降，不能称为系统整体大幅提速。

可信文档根目录为 `/workspace/simthesizer/agents/qwen_moe_optimization`。先读该目录下的 CROSS_MODULE.md、DESIGN.md、IMPLEMENTATION.md、candidate.py，以及已有基线。下文文档路径均相对这个可信目录，不相对当前工作目录或 `/workspace/task/AGENTS.md`。设计文档描述长期目标；本文件和冻结参数描述当前可执行权限。研究副本在 `/workspace/worktree`，可信参考在 `/workspace/simthesizer`。修改副本不会修改宿主项目。不得修改 README.md 或 AGENTS.md。

每个 run 固定其启动时的代码与权限；恢复旧 run 不会获得新范围。使用本次跨模块能力必须新建 run。

完整阅读 `references/compute-data-movement.txt` 的 33 页，并参考同目录 `.md` 的页码、源码映射和消融建议；原 PDF 在冻结 manifest 的 `inputs.assets` 对应路径。文档用于启发，不是优化白名单，也不是当前瓶颈的证据。引用“PDF 页码 + 当前封存案例证据”形成假设；Kimi/Blackwell 预测不能替换 Qwen/H200 模型、硬件和成本。可以提出文档之外的机制。

## 开发范围

- 策略：请求路由、batch/token 预算、优先级、等待、KV 提交窗口。
- 机制：默认开放 Rust/Python 实现区，允许跨 engine、graph、resources、KV、系统、调度器、运行器、公共接口、Python worker/bridge 及 Rooflang 执行模块修改和新增机制。Attention 放置、MoE 通信、依赖和 P/D 协调只是例子，不是封闭清单；完整路径和静态保护以 `candidate.py` 为准。
- 根据证据自主增加缺少的状态、接口、执行操作、观察和测试，无需逐模块申请。持久请求→Attention 放置也是待证假设，不应预设为答案。
- 不得改变模型规模、专家选择、请求工作量、GPU/链路参数、基础定价、评分器、实验输入、控制器、已有可信测试或构建依赖。引擎和资源机制可以改，物理事实与独立计量口径不能改。不能通过删算子、丢通信、漏算 KV、缩短输出或读取未来路由获得虚假改善。
- 遇到保护拒绝先检查是否把机制实现与固定事实混在一起；在开放实现区拆分和扩展。确实需要改变实验事实或依赖的研究另列需求，不伪造结果，也不把普通跨模块改造转成人工待办。

## 实际闭环

开始及每次评测后读取 `/workspace/artifacts/research-status.json`、`baseline-diagnostics.json`、该 trial 的 `diagnostics.json` 和 `review-candidates.json`。先区分到达跨度、到达结束后的排空时间、终止关键请求、TTFT 与首 token 后生成时间；最后到达的请求未必最后完成。pressure 是剩余满容量服务需求，不是利用率；registered reads 不等于实际阻塞时长。缺少资源归因时，自主增加有界观察或最小区分实验，不能从队列长度直接断言通信或计算是主因。

每轮选择影响整体指标最大的残余瓶颈：列出至少一个主因和一个替代解释，设计能区分二者的检查，估算可消除部分的收益上限，再实现策略与机制的联合方案。调查 `routing_placement`、`batch_decode`、`kv_communication`、`execution_resources`；这些是覆盖提示，不是白名单。可增加新方向；未测方向应保留为未探索，不能凭直觉宣布无收益。优先因果证据和可测预期收益，不为了凑覆盖盲目做全排列。

例如首 token 已很快但末尾长输出主导服务时间，应优先检查 Decode 批处理、组间失衡、KV 依赖、MoE 通信与执行重叠，再决定是否开发新动作、状态、调度协议或跨模块机制。若只是外部到达跨度主导，在固定到达输入下承认可优化上限，使用冻结压力案例检验机制，不改输入冒充加速。

1. 从完整本地基线、源码及原生统计定位压力；将假设、替代解释、预期收益、增加的资源开销、反例和停止条件写入 `/workspace/notes`。机制提交必须提供结构化 plan，并引用宿主已封存的 baseline/trial 身份；笔记路径或 PDF 论点不能替代实际案例证据。区分仿真服务时间与宿主运行时间。
2. 先做现有策略对照，再在 worktree 实现必要机制。保留参考语义与可关闭实现，维护唯一状态 owner；添加最小不变量测试。新增 Python 测试使用 `unittest`、文件名 `test*.py`；plan 中 Rust 用例引用已有 crate 的 `crates/*/tests/*.rs` 集成测试。冻结的外部测试与宿主注入的资源 fixture 不可修改；Rust 实现内的单元测试可随接口演进，属于补充证据，不能替代可信检查。不得把所有优化留在文档中。
3. 用 `--scope` 预览实际差异、影响域及 `required_checks`。填写 plan 的 `schema_version=1`、`mode`、`summary`、`state_ownership`、`preserved_semantics`、`expected_effect`、`alternatives`、`stop_conditions`、`evidence`、`control`、`checks`；示例见 SEARCH.md。每个 required check 都须给出推导与实际测试路径；自报影响范围不能减少宿主检查。bottleneck-v1 的所有新提交（包括纯策略和 control）还须提供 `research`：`name`、开放的 `family`、`causes`（第一项主因，后续为不同的替代解释）、`discriminating_check`、`expected_metrics`、`implementation`、`overhead`。目标名为 `ttft_p99`、`tpot_p99`、`goodput_per_gpu`、`output_tokens_per_gpu`、`tokens_per_user`。研究描述是待验证假设；引用须指向宿主封存案例，不能自签因果结论。
4. 通过可信客户端提交完整源码快照，等待评测完成期间不要改 worktree：

   ```bash
   CLIENT=/workspace/simthesizer/agents/qwen_moe_optimization/submit.py
   python "$CLIENT" --list
   python "$CLIENT" --status
   python "$CLIENT" --scope
   python "$CLIENT" --case CASE --reference --plan /workspace/notes/control.json --bottleneck '证据中的瓶颈' --hypothesis '关闭机制复现封存参考'
   python "$CLIENT" --case CASE --reference --plan /workspace/notes/plan.json --bottleneck '同上' --hypothesis '可证伪机制假设'
   python "$CLIENT" --case CASE --reference --replay CANDIDATE_ID --hypothesis '干净环境复验' --bottleneck '同上'
   ```

   `--reference` 使用封存参考策略，**不会关闭机制**；开关须在候选实现中落实。`mode=control` 要求本次结果逐 token 及服务结束时间复现所引用参考，成功后候选 plan 的 `control` 指向该 trial 的 32 位提交 ID。`CANDIDATE_ID` 是 64 位源码身份；两者不能混用。默认使用候选策略；匹配对照还核对策略身份。可用 `--ablation-of CANDIDATE_ID` 标注消融，但标签本身不证明反事实成立。

   每次提交只跑一个冻结 case；请求经宿主封存、计费，并在无网络/凭据的独立容器构建、执行冻结检查和新增测试。研究者不能签发评分；检查失败要修改机制，不能删除可信用例。
5. 根据失败日志修复；形成 A 原始机制+参考策略、B 原始机制+优化策略、C 新机制+参考策略、D 新机制+优化策略、E 消融。独立优化 B/D 只能比较各自最优配置；纯交互收益需固定两套策略做配对 2×2。不得拿一个容易 case 的收益代表完整矩阵。
6. 对候选覆盖全部冻结 case 并逐 case 干净重放；重放核对 `result.jsonl` 与 `service_finish_s`。解释收益和瓶颈转移。控制器在 `/workspace/artifacts/review-candidates.json` 汇总匹配对照、重放和案例覆盖，只形成待审候选，不签发 VERIFIED。结果在 `/workspace/artifacts`，笔记和最终总结写入 `/workspace/notes`；不自动合并代码。

7. 每个新增机制先验证关闭路径、资源守恒和完整工作量，再测开启收益。新增热路径须做相同 fixture/环境/配置下的配对宿主 wall/RSS 测量（参考 `tools/benchmark_simulator.py`），报告样本与噪声；不能把编译、测试、绘图或冷缓存总耗时当成模拟器速度，也不能把宿主加速当成 GPU 服务加速。显著回退须先修复或撤回候选，不用整体收益掩盖未解释的回退；记录在 `research.overhead` 及结果笔记中。

## 持续研究与收尾

bottleneck-v1 中达到 SLO、连续若干候选无新前沿、已有候选通过重放，都不是研究结束条件。旧源码快照已封存，后续 worktree 修改不会使它们失效。每次退出前将已查原因、未解瓶颈、负例、下一实验和候选身份写入 `/workspace/notes/continuation.md`；宿主会在预算内重新启动会话读取这些证据。连续空退出会标记 STALLED，而非完成。

按宿主预算安排实验：优先完成最有希望候选的全矩阵，再做昂贵的干净重放；不要把复验预算花在大量局部候选上。`phase=validate` 后仅提交 sealed candidate 的 replay；宿主已预留剩余复验时间，不再接收新 evaluate。若全矩阵复验估计不能装入预算，立即收缩候选数量并明确缺口，不能假称已验证。预算耗尽或用户取消时如实交接；候选已达标但仍有预算时继续分析瓶颈转移与联合改进。

## 证据边界

当前完整负载与未来路由仍可被 worker 读取，没有实现严格 oracle 隔离，也没有隐藏 holdout。静态保护、固定可信测试、完整请求检查和独立指标重算不能证明任意新机制的数学正确性。control 匹配只证明**该 control 快照**复现了参考，不能自动证明后来 candidate 的关闭路径等价；必须解释两份源码的关系并补专门检查。控制器始终保留 EXPLORATORY，干净重放只证明可复现，不能自称 VERIFIED 或真实 GPU 收益。

默认保留 result.jsonl、metrics.json、metrics.png、throughput.png、构建/测试日志和源码 diff；全量历史 trace 关闭。新增观察应有界、可关闭并验证关闭前后输出一致。不要改动队列、基线、账本、评分器来影响评测，不要访问 Docker socket 或凭据。构建产物与缓存放 `/workspace/cache`，笔记放 `/workspace/notes`，不要写入源码目录。
