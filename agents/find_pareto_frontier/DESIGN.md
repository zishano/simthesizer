# P/D 服务策略搜索设计

状态：2026-11-18。已接入原生策略、评测、前沿归档和外部复验；Docker 启动见 [LAUNCHER.md](LAUNCHER.md)。尚未执行付费 agent 搜索或验证优化收益。

目标：固定资源、完整完成请求，提高满足延迟目标的吞吐，保存不同延迟与吞吐取舍的策略。依据用户确认的上一轮方案；代码接入和验收见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。

## 1. 范围与输入

- 使用现有 shared Rooflang P/D 路径；支持范围沿用该后端：FP16 Llama 3.1 8B、TP 1/2/4/8、PP=EP=1、单 LLM 请求。模型、硬件与负载作为配置输入，不把接口写死为某一 TP 或 GPU。其他模型、请求 DAG 的 P/D 支持需另行补齐。
- 每次实验冻结模型、精度、物理 GPU 集合、显存上限、拓扑、P/D 实例划分及并行策略。不同部署分别评估；运行中不调整 TP 或迁移实例。
- agent 可修改策略程序与参数，搜索 P chunk、batch 成员/大小、等待与唤醒、KV 发送顺序/并发窗口、多实例路由。单 P/单 D 配置没有路由选择空间。
- 成本公式、模型算子、资源分配速率、KV 正确性、指标和工作负载不可由搜索 agent 修改。压缩、换出、抢占、迁移等新机制必须先独立实现并验证，再升级动作接口和实验版本。
- agent 在实验之间编写策略；运行时执行确定性原生策略，不逐事件调用大模型。

输入分三类，只有一处权威来源：

| 输入 | 来源与冻结方式 |
|---|---|
| 部署与基线 | 场景 TOML 及引用的硬件/模型文件；保存解析后配置、原文件和哈希 |
| 请求 | 原始 workload JSONL；冻结 ID、输入、输出、依赖、token 内容和顺序 |
| 实验约束 | [experiment.example.toml](experiment.example.toml) 的实填副本；冻结 SLO、负载网格、数据划分、动作范围和预算 |

已查看 `examples/workloads/realistic.jsonl` 的原始记录：每条含 `request_id`、`arrival_time`、`initial`、`sub_requests`；LLM 含 `input_tokens`、`output_tokens`、`next` 和可选 token IDs。示例前两条为输入/输出 25/619、19/119；它们不是通用负载分布的证据。

首版沿用 `known_trace`：现有模型用 workload 的实际输出长度预留 D KV，并决定结束。策略观察不直接提供实际剩余输出，但容量状态仍受已知长度影响；不得称为完全未知输出长度的在线模型。`bounded_max` 尚未实现，两种模式不可混合比较。

[experiment.toml](experiment.toml) 提供可修改的初始 SLO、预算和完整 100 请求训练负载；这些初值不是生产性能目标，也没有独立验证集或 holdout。

## 2. 评分与前沿

指标统一使用秒，复用 `scripts/results.py`；新增指标也归此模块，launcher 不另写一套公式。

一次评估先检查：输入请求与结果一一对应，输出 token 数及任务内容不变，所有请求完成，依赖/容量/KV 检查通过。超时、死锁、非法动作、OOM、漏请求或部分结果不能进入前沿。

对有输出的单 LLM 请求 i，设输出数为 O_i：

- TTFT_i = `token_latency[0]`。
- O_i > 1 时，TPOT_i = 其余 token 间隔的平均值；O_i = 1 时 TPOT 不适用，只检查 TTFT，不能填成 0。
- pass_i = TTFT_i ≤ F，且 TPOT 适用时 TPOT_i ≤ P。
- 达标率 = 达标请求数 / 有输出的请求数；须不低于预先配置的比例 α。
- T = 最后请求完成时刻 − 首个请求到达时刻，包含排队、空闲和尾部排空。
- goodput = Σ(O_i × pass_i) / T，单位 output token/s。
- goodput_per_gpu = goodput / 已分配物理 GPU 去重数量，包含全部 P/D 设备。

零输出请求仍必须完成，但不进入延迟达标分母；报告排除数量。没有有输出请求、T≤0 或 GPU 数未知时，评分不可用，不能进入前沿。ITL、请求 E2E、总吞吐、tokens/s/user 与 tokens/s/GPU 继续保留，不能用有效吞吐覆盖原指标。

例（仅说明公式）：4 张 GPU、T=2 秒，两请求分别输出 4、2 token，只有前者达标，则 goodput=2 token/s，goodput_per_gpu=0.5 token/s/GPU，达标率=50%；所有请求仍须完成。是否满足 α 由实验配置决定。

前沿按相同部署、workload、到达强度、SLO、模型假设和评测版本分组，目标为：TTFT p99 更低、TPOT p99 更低、goodput_per_gpu 更高。原始数值上所有维度不差且至少一个严格更好才构成支配；不先四舍五入。若整个输入案例均无多 token 输出，预先移除 TPOT 维度，记录原因。等值点以稳定候选 ID 去重，保留等值实现引用。

区分以下状态：

| 状态 | 处理 |
|---|---|
| INVALID / INCOMPLETE | 保存原因、日志与已有证据；不计为低分正常结果 |
| VALID_INFEASIBLE | 正确完成但 SLO 达标率不足；保留负载曲线，不进入该组可行前沿 |
| VALID_FEASIBLE | 正确完成且达标；可以成为待复验前沿点 |
| VERIFIED | 独立重建复验通过；只对已完成的案例矩阵声明覆盖 |

前沿保存多份代码快照/补丁及参数。旧基线始终保留；新策略无需独自覆盖所有旧点。新增取舍点不能表述成全面改善。不存在把不同案例均值压成唯一冠军的默认规则。

## 3. 负载与评估协议

1. 训练集可反复反馈；验证集用于晋级，也可能被适应，不称为未见数据。holdout 仅在搜索冻结后由外部复验器使用，内容不挂载给 agent；未提供时明确标注无独立泛化证据。
2. 由固定评测器生成到达变体：`a'_i = a_min + (a_i - a_min) / m`，m>0。m>1 增大到达密度；保持请求集合、长度、顺序和相同时间的稳定次序。原始与派生文件均记录哈希，agent 不得自行挑选或修改案例。
3. 同一 m 下基线和候选使用完全相同的派生负载。不同 m 分组比较；绘制完整曲线，并报告测试网格内可行点，不假定达标性严格单调。
4. 短案例用于筛选；声称覆盖某套负载前必须跑完全部规定案例。每次试验都排空请求；现实时间预算中止的试验为 INCOMPLETE。
5. 最高达标网格点只表示该有限负载的结果。宣称持续承载能力需额外固定长窗口/不同长度负载，并检查排队趋势；不能由一次 100 请求试验推断。
6. 相同代码/配置的仿真是确定性的；重复运行用于复验，不当作独立统计样本。变化来自预先声明的不同 workload 或种子，并采用配对比较。

## 4. 运行时策略契约

接口版本为 `serving-policy-v1`。候选实现 `route`、`batch`、`kv` 三个 Rust 函数，使用借用的当前观察与每次试验重置的有界字节状态：

```text
decide(trigger, observation, policy_memory)
    -> actions, next_policy_memory, wakeup
```

| 部分 | 约束 |
|---|---|
| trigger | 请求实际到达、实例空闲时的调度、KV 可发送/ACK、已登记的定时唤醒；KV 窗口满时直接等待 ACK |
| observation | 当前仿真时间、实例负载、当前就绪请求及提交进度、默认 batch/token 预算、显存上限、待发送 KV 及在途数、固定 SLO；尚无链路实时需求、逐请求 TTFT deadline 或资源可用量观察 |
| policy_memory | 有界启发式状态；每次试验重新初始化。队列、KV、容量和 token 进度仍只有原组件持有权威状态 |
| actions | Route(request,P,D)、Batch(instance,显式请求 ID 与 q_len)、SubmitKv(已存在传输意图 ID)、Wait(事件订阅或未来时间) |
| wakeup | 有未完成工作却不提交时必须有唤醒依据；时间须有限且大于 now，事件订阅须能被执行器验证 |

不暴露未到达请求、整份未来 DAG、未来完成时刻、真实剩余输出长度或结果文件。允许基于当前状态的估计，但估计不能冒充已观测事实。相同时间事件按原生执行顺序更新状态后决策；避免重复注册导致零时间自循环。

动作由执行器验证后提交：请求已到达且未完成、实例合法、无重复 batch 成员、q_len 合法、传输确实待发送、依赖和容量不被绕过。非法动作使试验失败；batch 按稳定完成节点 ID 提交，执行器保留显存准入和资源等待检查。

策略不持有 Graph/资源/KV 可变引用，不改虚拟时间、算子成本、传输字节数、完成标记或进度。执行器生成原生 DAG；算子完成可发布 KV，但 token 仍在原有 batch 完成边界提交。

### KV 延迟发送

```text
Produce → PendingSend → InFlight → ACK/Ready → Release
```

- Produce 为每个目的分片建立传输意图并持有源引用；PendingSend 同样阻止释放。发送时不能再次重复增加引用。
- 首版保留当前整体 KV 容量预留；策略不能虚构释放空间，也不改变传输分片/字节数。
- D 读取依赖持久的 layer/rank/token/head 就绪门，而非仅枚举已经创建的传输节点。尚未发送的 KV 也必须挡住读取；其他已就绪层可继续执行。
- 就绪门必须表达真实未满足的条件并参与未排空检查，不能用无前驱、可立即执行的占位节点冒充等待。
- ACK 在实际网络和 HBM 服务完成后发布就绪，释放对应引用并唤醒策略。producer 引用在 P 完成时释放；源数据须等全部引用清零才回收。
- SubmitKv 只控制未发送工作的顺序与并发；已启动传输继续按现有资源共享规则执行，不凭空获得带宽优先权或被免费抢占。

## 5. Docker 与自主搜索

参考 `/mnt/nvme1n1/data/lmk/PROJECT/TEST2/rooflang/agents/find_pareto_frontier/` 的独立副本、只读挂载、允许修改目录和恢复日志；不照搬模型优化目标、5% 门槛、单一最终代码状态或理论最优停止条件。

```text
冻结配置/源码 → 基线 → agent 提出假设与候选 → 正确性检查
                                  ↑                 ↓
                            反馈与前沿记录 ← 完整仿真/评分
                                                    ↓
                                    搜索冻结 → 独立复验 → 发布本地报告
```

- 镜像提供锁定的 Rust 工具链、Python 3.14、uv 和核心依赖；无需物理 GPU、CUDA 或 vLLM。固定镜像/agent CLI 版本，恢复时不自动升级。
- 副本只包含必要源码和显式数据，排除 `.git`、`output`、`target`、`.venv`、历史 runs 及凭据；不改宿主源码。Cargo 产物/缓存放独立可写目录，缓存命中须匹配身份。
- 仿真主体、成本、指标、任务和实验定义只读。候选策略目录可写；凭据单独传入，不进入镜像或源码快照。
- 宿主 launcher 管理容器、CPU/内存/进程和墙钟预算；agent 不持有 Docker socket。模拟 GPU 资源与容器 CPU/RAM 预算分别记录。
- 原生候选不是恶意代码沙箱：仅允许既定策略 API，禁止文件/网络/进程/时钟访问、unsafe、改评测环境或跨试验记忆；通过代码差异审查、动作校验、因果性案例和独立复验检查。只读挂载不等于证明任意 Rust 程序无法读取未来数据。
- agent 内部评测产物为 provisional。当前外部复验器在宿主独立目录，以干净 Cargo target 从冻结源码重建、运行和算分；已验证产物不挂载给 agent。独立复验容器尚未接入。
- 预算结束/无改进达到配置轮数时保留状态；中断试验不计完成。恢复核对所有身份，兼容时跳过已完成案例，重跑未完成案例；身份不同开启新实验。
- 不要求证明全局最优，不自动提交、推送或把候选合并回主项目。

一次 trial 是“一个候选 × 一个 workload × 一个到达倍率 × 一个种子”的一次完整模拟尝试。搜索 trial 预算包含基线、失败和重试；墙钟预算包含 agent、候选构建及评测，镜像构建和首次快照准备在此之外。最终独立复验使用另行冻结的 trial/墙钟预算；预算不足的点保留 provisional 状态，不能因搜索结束而自动标成 VERIFIED。

## 6. 目录与产物

目录结构：

```text
agents/find_pareto_frontier/
  DESIGN.md / IMPLEMENTATION.md
  AGENTS.md.template               # 渲染为每次 run 的只读任务
  experiment.example.toml
  Dockerfile / run-agent.sh / entrypoint.sh
  launcher.py / agent.env.example / LAUNCHER.md
  evaluate.py / archive.py / verify.py
  runs/<本地时间戳>[-碰撞序号]/
    task/manifest.json            # 输入/源码/依赖/预算与约束身份
    task/AGENTS.md
    source/ / inputs/ / baseline-policy/
    cache/                        # 可丢弃的构建与工具缓存
    agent.trace.log / supervisor.json
    artifacts/
      hypotheses.jsonl
      experiments.jsonl           # 追加式案例结果与状态，权威记录
      candidates/<内容哈希>/       # 策略、参数、父候选、可复现身份
      trials/<候选>/<案例>/        # result、metrics、配置、日志、指标图
      frontier.json / frontier.png  # 从 experiments 派生，可重建
    verified/                     # 外部复验器写入，不挂载给 agent
```

候选 ID 包含策略代码与参数哈希；评估身份另包含基础源码快照、策略接口、指标、派生 workload、场景、长度假设、种子和工具链身份。恢复不依赖一次临时 agent 会话。

默认保留 `result.jsonl`、`metrics.json`、`metrics.png`、`throughput.png`、配置/负载引用、日志和复现收据。失败的部分数据明确标注。按需对优胜或异常候选重跑诊断，使用已有 reports/选定轮次能力；完整 HTML 为额外选项，不对每个试验生成全量历史。详细诊断产物设保留预算；清理前保存摘要、来源哈希及复现方法，不能删除唯一的前沿证据。

`metrics.json` 含完成数、达标分子/分母、SLO、样本数及策略汇总；`experiments.jsonl` 引用它及原始结果身份。决策计数和函数墙钟在完成后写一次；参考策略沿用原生 KV 路径，没有 KV 回调计数。逐决策解释与专用策略时间线尚未实现，现有图历史仍可按需开启。

## 7. 交付判定与边界

实现验收以 [IMPLEMENTATION.md](IMPLEMENTATION.md) 的 R1–R10 为准。先使参考策略复现当前行为，再开放搜索；参数扫掠可用于初步检查，不能作为动态策略能力已完成的证据。

当前模型的调度、Python IPC 和策略计算不消耗虚拟时间；记录实际决策开销并施加执行预算。首版是忽略在线调度开销的仿真实验，不把复杂策略的计算当作真实系统免费收益。预测收益还受 Rooflang 成本、KV 预留和通信近似影响；内部一致性不等于硬件准确性。
