# Docker 启动脚本

适用于本项目 Linux 主机。启动器已接通原生策略、evaluate.py、archive.py 和宿主独立复验；预检会编译冻结源码并检查原生能力。尚未执行付费 agent 搜索。

## 文件职责

| 文件 | 实现职责 |
|---|---|
| [Dockerfile](Dockerfile) | Python 3.14.5、Rust 1.96.0、uv 0.11.19、Codex 0.166.0；安装核心 Python 依赖和 ruff，排除 vLLM；缓存 Cargo.lock 依赖 |
| [run-agent.sh](run-agent.sh) | 从任意目录定位项目，选用项目 Python，进入 launcher；宿主只需 Python 3.11+，无需安装 Codex |
| [launcher.py](launcher.py) | 参数与配置检查、最小构建上下文、源码/输入快照、只读任务、Docker argv、镜像身份校验、墙钟监督及恢复 |
| [entrypoint.sh](entrypoint.sh) | 检查挂载、初始化独立缓存、运行容器身份/评测器预检，最后启动 agent；不自动升级或同步依赖 |

依赖版本来自本机已安装工具/包元数据，镜像构建仍需实际确认其下载可用性。Codex 安装使用官方支持的 [npm 方式](https://learn.chatgpt.com/docs/codex/cli)，显式安装平台依赖；构建时检查版本及所需 CLI 参数。执行使用官方 [非交互模式](https://learn.chatgpt.com/docs/non-interactive-mode)。凭据不参与构建。

## 构建与检查

项目根目录提供 [run_rooflang_agent.sh](../../run_rooflang_agent.sh)，常用路径及构建选项集中在脚本开头，支持 `TMP_DOCKER_CONFIG` 和 `RUNTIME_ENV`。默认新运行加 `--build`；模型、资源、负载、SLO 和预算仍以实验 TOML 为准。

```sh
./run_rooflang_agent.sh --build-only
./run_rooflang_agent.sh --check
./run_rooflang_agent.sh --prepare-only
./run_rooflang_agent.sh
./run_rooflang_agent.sh --resume agents/find_pareto_frontier/runs/<时间目录>
```

首次配置步骤见 `./run_rooflang_agent.sh --help`。检查/准备不需要凭据；构建不需要实验文件。普通运行仍须通过下述配置与能力检查。

从项目根目录执行：

```sh
./agents/find_pareto_frontier/run-agent.sh --help
./agents/find_pareto_frontier/run-agent.sh --build-only
```

build-only 不读取实验参数或 API key，也不运行仿真/agent。仅将 Dockerfile、启动器、Python/Cargo manifests 与 lockfiles 送入临时构建上下文；不发送源码、output、workload、runs 或凭据。镜像带构建输入哈希，运行时校验标签并记录实际 image ID。

可配置 `--python-image`、`--uv-image`、`--node-image`、`--rust-version`、`--codex-version`、`--buildkit-image` 和 `--image`。Python 镜像必须提供 3.14；Codex 版本应同步写入实验配置。

构建使用项目专用 `docker-container` builder。builder 和构建步骤均使用 host 网络。根目录脚本默认先用 `socks5h://127.0.0.1:17890`；遇到明确的连接、DNS、超时或代理错误时，用 `http://10.250.0.3:7890` 重试一次。普通编译、配置或包缺失错误不触发重试；APT 的 exit 100 本身不视为网络错误。构建日志逐行输出。

镜像内 APT 使用 Debian 官方 HTTPS 源，禁用 HTTP 请求流水线，下载失败最多重试 3 次，连接/读超时为 30 秒；索引未完整更新时停止安装。保留 TLS 和软件源签名校验，不修改宿主的软件源。

可用 `BUILD_PROXY`、`BUILD_FALLBACK_PROXY` 覆盖地址；将前者设为空表示继承终端代理，后者设为空禁用回退。直接调用 run-agent.sh 时，对应参数为 `--build-proxy` 和 `--build-fallback-proxy`，未指定则继承终端代理且不回退。显式代理同步设置 HTTP/HTTPS/ALL 的大小写变量，保留 NO_PROXY；基础镜像拉取和构建依赖下载使用同一代理。不读取或修改 agent.env。

显式选择 builder 并 `--load` 到本地镜像库，不切换默认 builder、不修改已有 builder、不修改或重启 Docker 服务。代理设置仅作用于构建子进程，搜索容器仍按下文配置。

Buildx 客户端状态保存在 `output/agent-build/<UID>/<配置摘要>/`，不写入用户的默认 Buildx 配置目录。名称区分项目、用户、Docker daemon、BuildKit 镜像和代理配置；配置一致时复用构建缓存，代理变化时创建另一专用 builder。目录权限为仅当前用户可访问，避免暴露其中的代理配置。同项目构建串行执行。

首次启动 builder 仍需 BuildKit 镜像，默认 `moby/buildkit:buildx-stable-1`。Buildx 0.29.1 拉取失败时会尝试本地同名镜像；没有缓存且 daemon 无法下载时，启动器会报 bootstrap 错误，不回退到他人的 builder。可通过脚本的 `BUILDKIT_IMAGE` 或 `--buildkit-image` 指定已加载的镜像。项目源代码、任务副本和指标计算不受构建网络设置影响。

可手工检查镜像环境，不调用 API：

```sh
docker run --rm --network none simthesizer-pareto-agent:codex-0.166.0 --check-env
```

检查同时覆盖普通 shell 和 agent 使用的登录 shell：两者必须使用 `/opt/venv`，能读取 Python 依赖版本并找到 Cargo。镜像内 `/etc/profile.d/simthesizer.sh` 保持构建时的工具路径，避免登录配置切换到系统 Python。修改镜像后需重新构建并创建新 run；旧 run 保留原环境身份。

## 配置、准备、启动

已提供 [experiment.toml](experiment.toml) 及其 [原生场景](scenarios/shared-workload.toml)，无需再复制模板。场景是 `run_rooflang_shared_workload.sh` 实际参数的快照，不会自动随脚本变化：

| 参数 | P | D |
|---|---|---|
| GPU / 实例 / TP | H200 / 1 / 4 | H200 / 1 / 4 |
| max_batch | 32 | 32 |
| token budget | 128 | 32 |

共 8 张独立 GPU，模型为 FP16 Llama 3.1 8B；使用 shared 执行、whole KV 交接及预设硬件图。完整沿用 `realistic.jsonl` 的 100 条请求，关闭 reports/full_graphs。原脚本的 TP2 注释过时，本配置取实际 TP4 值。

初始设置为 TTFT≤5s、TPOT≤50ms、联合达标率≥95%，100 次试验/2 小时/单次 30 分钟；agent 模型为 `gpt-6.1-sol`。到达倍率、搜索边界、复验及容器预算也在 TOML 中明确列出，可按目标修改。当前只配置训练负载，没有独立验证集或 holdout。

```sh
./run_rooflang_agent.sh --check
```

GPU 型号/数量/并行划分来自 `inputs.scenario`，这里不重复维护。文件路径默认相对项目根目录，也可显式给绝对输入路径。创建其他实验可另存 `experiment.example.toml`，填写 REQUIRED 并设 `design_only=false`，再通过 `--experiment` 选择；不要用空模板覆盖已提供的配置。

场景和输入原样冻结；评测器只注入 policy 配置，并把 worker 命令指向冻结源码，保存实际运行配置。首版支持内联模型/拓扑及内置 Rooflang worker，不支持任意外部 worker 或额外诊断参数。

```sh
./agents/find_pareto_frontier/run-agent.sh \
  --prepare-only --experiment agents/find_pareto_frontier/experiment.toml
```

prepare-only 不调用 Docker/API。有效配置可先冻结为 `runs/YYYYMMDD-HHMMSS[-序号]/`，同时列出尚缺的运行能力。默认仅复制必要源码目录、场景模板和显式 workload/assets；排除历史报告、构建产物、宿主工具配置和凭据。核心未准备好的快照不能在补写宿主代码后原地升级；需新建 run。

启动：

```sh
./agents/find_pareto_frontier/run-agent.sh \
  --experiment agents/find_pareto_frontier/experiment.toml \
  --env-file /secure/path/agent.env
```

运行环境文件参考 [agent.env.example](agent.env.example)，只允许 key、服务地址和代理变量，不允许覆盖 PATH、Python/Cargo 路径或资源限制。搜索容器默认 host 网络，可访问宿主 `127.0.0.1`；可用 `AGENT_NETWORK=bridge` 或 `--network bridge` 切换。搜索容器的代理仍由 agent.env 显式配置，不自动继承终端代理。不会读取宿主登录状态。恢复旧实验沿用其冻结的网络模式。

## 容器内布局

| 容器路径 | 权限 / 含义 |
|---|---|
| `/workspace/simthesizer` | 只读源码副本 |
| `…/crates/simthesizer-core/src/policies/candidate` | 唯一可写的策略源码目录，提供 route/batch/kv 函数 |
| `/workspace/baseline-policy` | 原始候选源码，只读 |
| `/workspace/task`、`/workspace/inputs` | 只读 manifest、AGENTS、场景与声明的负载/资产 |
| `/workspace/artifacts` | 候选、结果、指标图、实验记录；不含外部 VERIFIED 结果 |
| `/workspace/cache` | Cargo target/registry、uv、绘图缓存；不作为性能证据 |
| `/home/agent`、`/tmp` | 限大小 tmpfs，按宿主 UID/GID 使用，每次重建 |

容器 rootfs 只读、无额外 capabilities、禁止提升权限；不挂载 Docker socket。原生执行器校验策略动作，候选没有 Graph 或资源的可变引用。`cargo` 离线使用预取缓存；变化的 lockfile/launcher 需要重建镜像。

## 搜索启动协议与恢复

进入容器后，固定评测器必须接受：

```text
python agents/find_pareto_frontier/evaluate.py --preflight --manifest /workspace/task/manifest.json
```

在单次试验时限内完成原生编译和能力检查，输出 JSON：`ready`、`enforces_trial_budget`、`enforces_trial_timeout` 为 true，`policy_api_version=serving-policy-v1`。容器外层总墙钟仍包含这段预检。`--check` 只检查配置与文件，不能代替该编译预检。

通过后才执行 `codex exec`，使用冻结 model/版本、JSONL 日志、临时会话和外部容器隔离。启动器不自动执行验证或赋予 VERIFIED 状态。agent exit 0 只记为 `agent_exited`。

```sh
./agents/find_pareto_frontier/run-agent.sh \
  --resume agents/find_pareto_frontier/runs/<时间目录> \
  --env-file /secure/path/agent.env
```

恢复校验所有只读文件、输入、任务与实际 image ID；保留候选修改，拒绝策略目录中的符号链接。锁防止并发评测；只复用身份与证据哈希均一致的已完成案例。

宿主累计监督 agent、候选编译和评测墙钟；Ctrl-C、SIGTERM、预算耗尽均先停止容器再记状态。评测器累计基线、失败及重试次数，并在单次超时时终止整个进程组。恢复不清零预算；缓存复用不重复仿真。搜索记录可由 agent 写入，因此试验计数依赖任务约束，宿主墙钟与容器资源限制独立强制执行。磁盘配额尚未实现。

退出时按本次记录的容器 ID 停止搜索容器，并尝试删除它；运行目录、结果和缓存保留。不执行全局容器清理，也不停止 Docker 服务。`BUILD_IMAGE=0` 只启动搜索容器，不启动构建 builder。

## 手工评测与外部复验

准备目录后，可使用项目 Python 执行；容器内对应根目录为 `/workspace`：

```sh
.venv/bin/python <run>/source/agents/find_pareto_frontier/evaluate.py --manifest <run>/task/manifest.json --preflight
.venv/bin/python <run>/source/agents/find_pareto_frontier/evaluate.py --manifest <run>/task/manifest.json --list-cases
.venv/bin/python <run>/source/agents/find_pareto_frontier/evaluate.py --manifest <run>/task/manifest.json --baseline
.venv/bin/python <run>/source/agents/find_pareto_frontier/evaluate.py --manifest <run>/task/manifest.json --candidate
.venv/bin/python <run>/source/agents/find_pareto_frontier/archive.py --artifacts <run>/artifacts
.venv/bin/python <run>/source/agents/find_pareto_frontier/verify.py --manifest <run>/task/manifest.json --candidate <候选SHA256>
```

`--case <ID>` 可选单个案例。使用 run 中冻结的工具，避免当前工作树变更后混用新旧指标。默认保留结果、指标、指标图、吞吐图和汇总策略耗时，不生成全量图历史。复验必须从搜索容器外执行：使用独立源码和干净 Cargo target，重新模拟并核对结果；输出在不挂载给 agent 的 `<run>/verified/`。它不等于独立验证集或实机精度验证。

## 验证范围

本地单测覆盖配置拒绝、快照隔离、受保护文件检查、挂载参数、任务渲染、预算累计、异常退出、项目 builder 隔离及代理回退。使用临时文件和模拟 Docker 命令，不启动真实容器。

```sh
bash -n agents/find_pareto_frontier/run-agent.sh agents/find_pareto_frontier/entrypoint.sh
.venv/bin/ruff check agents/find_pareto_frontier/launcher.py agents/find_pareto_frontier/test_launcher.py
.venv/bin/python -W error::ResourceWarning -m unittest discover -s agents/find_pareto_frontier -p test_launcher.py
```

空模板的 `--check` 返回 2，列出未填字段。默认配置已补齐接口文件；运行是否可用仍由原生编译预检判断。原生场景由现有生成器按参考脚本实参生成。

镜像构建和 `--check-env` 不代替容器挂载验收或 API 调用验证。原生机制和仿真对照证据保存在 `output/serving-policy-validation/`；此处不宣称 agent 已找到优化策略。
