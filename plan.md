# NanoRecon APP 后续修改计划（0.3.0）

状态（2026-10-05）：阶段 A、B、C 已完成；阶段 D 的一键安装包已做好。还差两项需要服务器开机时做：C1/C2 改动后的 GPU 复测，以及安装包的实机安装测试（见第 2 节）。复选框 `[x]` 表示已完成。基线：提交 `434fdf4`（APP 0.2.0）。

本文已取代 `SYSTEM-DESIGN.zh-CN.md`、`issue.md`、`IMPLEMENTATION.zh-CN.md`。这些文档和需求原文 `HumanAnalysis.md`、首版计划都可在 git 历史 `434fdf4` 中查看。

## 1. 使用范围与原则

| 项目 | 决定 |
|---|---|
| 用户 | 本实验室，加一两个相关实验室。需求由我们直接提出 |
| 运行环境 | Linux + NVIDIA GPU，不支持 CPU 推理 |
| 模型 | 只有一个：`GHSSHG/NanoRecon` @ `92be5d5aa05990eccc9edca33c263ee5ffc72739`，写死在代码里 |
| 验证 | 不配置 CI。代码用 rsync 复制到 GPU 服务器的 `~/Work/nanopore-app`，用服务器的 `~/myenv` 运行测试。Mac 本地只跑单元测试，不跑真实模型或 GPU 分析 |
| 技术栈 | 继续使用 Python + NumPy + JAX/Flax |

原则：

1. **只做已经提出的需求。** 不为"用户可能需要"增加选项、模式或兼容层。我们自己就是用户，需要时会再提。
2. **检查的取舍。** 一项检查只在以下情况保留：没有它，正常使用时会静默产生错误结果，或者出错后很难定位。不为以下情况写专门代码：人为构造的损坏文件、极少发生的并发竞态、多模型或多版本场景。人为损坏的文件以"内部错误"退出是可以接受的。
3. **先测量，再优化。** 性能和资源问题先在服务器的 GPU 上测量，再决定改不改，不提前设计。
4. **需求原文中的硬性要求不变：**
   - 不出现 SHA256 或其他哈希校验；
   - 输出中不含 MAE/RMSE 等训练指标；
   - DiVeQ 固定为最近码字硬量化（noise=0）；
   - 不整体复用训练代码；
   - 不丢弃短 read。

## 2. 现状与接下来的顺序

**现状（0.3.0）**

| 方面 | 状态 |
|---|---|
| 功能 | 六个命令可用：compress、decompress、ls-remote、pull、ls、info；参数只剩 A2 列出的那些 |
| 正确性 | 三个真实数据集上 Meta、顺序、长度零差异；与训练模型码字 100% 一致，同码字波形最大差 5.4e-7，APP 忠实复现了模型 |
| 测试 | 191 个单元测试（Mac 和服务器）、7 个 GPU 测试（服务器）全部通过；真实数据验证见 B3 |
| 压缩速度 | 瓶颈在 GPU（占 94–97% 的时间）；batch 越大越快 |
| 解压速度 | 已改为跨 read 装满 batch（C2）；改前短 read 数据有 91% 的计算浪费在补齐行上，改后只有文件最后一个 batch 有补齐行；GPU 上尚未复测 |
| 代码量 | `src/` 约 3300 行（C1 删除多线程减少约 270 行，C2 增加约 60 行） |
| 发布 | 一键安装包 `dist/nanorecon-0.3.0-linux-x86_64.sh`（56 KB）由 `tools/build_installer.sh` 构建；尚未在 Linux 上实装测试 |

**接下来的顺序**

1. [x] **C1** 删除多线程。
2. [x] **C2** 解压跨 read 组 batch。
3. [x] **C3** `--batch-size` 缺省值改为 64。
4. [x] **阶段 D**：一键安装包。
5. [ ] 服务器开机后（约 15 分钟）：
   - rsync 后跑单元测试和 `pytest tests/gpu`，确认 C1/C2 没有改变结果；
   - 用 `tools/validate.py` 在一个短 read 数据集上复测解压速度；
   - 以全新的 HOME 运行安装包，装好后对一个小 POD5 做 compress/decompress。

## 3. 保持不变的核心设计

阶段 A 没有改变本节的任何行为（已用 0.2.0 的输出逐项对照确认）。以后要改，必须单独提出。

**3.1 数据流**

```text
compress:   POD5 → 按顺序读 Meta 和长度 → 组成一组 read → 加载 ADC
            → 切窗、校准、min-max 归一化、短窗反射填充
            → GPU 编码 + 最近码字搜索 → 写回组内数组 → 按 read 顺序写入 .nrpod
decompress: .nrpod → 逐 read 读 Meta 和 centers/scales → 按 batch 读 codes → GPU 查码本 + decoder
            → 反归一化到 pA → 加权拼接 → 统一转回 int16 ADC → 带原 Meta 写入 POD5
```

batch 中的每一行都是一个独立的 chunk。压缩 batch 可以混合不同 read 的 chunk；不会把多个 chunk 拼成一条长 token 序列送进模型。

**3.2 编解码规则（CodecProfile，完整写入文件头）**

L=8192，O=144，H=8048，T=512，5000 Hz，codebook 65536，`uint16`，`minmax_pm1`（eps 1e-6），`reflect`，`shift_last`，`hard_v1`，`linear_edges_v1`，`rint_clip_int16`。代码中的常量为 `model_config.PROFILE`，详细规则见 [docs/format.md](docs/format.md) 第 7 节。

**3.3 切窗、归一化与拼接**

- **chunk 数：**
  - N=0 时 K=0，原样保留为空 read；
  - 0<N<L 时只有一个短窗；
  - 否则 `K = 1 + ceil((N−L)/H)`，末窗右对齐，起点为 N−L。
  - 例：N=16300 时起点为 0、8048、8108，最后两窗重叠 8132 点。
- **归一化：** `pA = (ADC + offset) × calibration_scale`。然后用该窗真实样本的 min/max 计算 center 和半极差。半极差 < eps 时整窗置 0。短窗先归一化，再反射填充（N=1 时重复填充）。
- **拼接：**
  - 权重 `w(t) = min(1, (t+1)/(O+1), (L−t)/(O+1))`；
  - 在 pA 空间按每个窗的实际起点累加 `Σ w·pA` 和 `Σ w`，再相除（所以三窗共同覆盖的区域也能正确处理）；
  - 最后统一做 `clip(rint(pA/scale − offset))`，转回 int16。

**3.4 中间文件（格式 1.0，详见 [docs/format.md](docs/format.md)）**

- **记录顺序：** `Preamble → FILE_HEADER → (GROUP → READ×n)* → END`。
- **FILE_HEADER：** 固定的模型身份（40 位 commit）、完整 profile、read 总数。
- **GROUP：** 组内去重的 RunInfo、pore_type、end_reason 表。
- **READ：** 104 字节二进制 Meta，然后是 `float32 centers[K]`、`float32 scales[K]`、`uint16 codes[K×T]`。
- **不保存的内容：** 原始 ADC、连续 latent、噪声或随机种子、哈希、训练指标、本机路径。

**3.5 数据所有权**

| 数据 | 归属 |
|---|---|
| 原始信号 | 当前 read 组 |
| 准备好的 chunk | 槽位池（或主线程） |
| GPU batch | 唯一的消费者 |
| 码字结果 | 组内数组，按 `offsets[read_slot] + chunk_seq` 定位，不依赖完成顺序 |
| 重建信号 | 当前正在解压的 read |

修改并发结构时，必须说明每份数据何时创建、由谁使用、何时释放。

**3.6 容易混淆的概念**

- `search_chunk_size=8192` 是码本搜索的分块大小，与信号 chunk 长度 L 无关。
- `istft_hop_length=16` 是 decoder 内部的帧步长，与切窗步长 H=8048 无关。
- read 的 `calibration_scale` 用于 ADC→pA 转换；chunk 的 `scale` 是归一化半极差。两者不是一回事。
- 配置中的 `diveq_sigma2=0.001` 只是训练事实，推理不使用。

## 4. 修改计划

### 阶段 A：精简（已完成）

**A0 对照基准**

- [x] 用 0.2.0 代码和替身模型生成一份基准输出（覆盖 0、1、2、短窗、8191/8192/8193、三窗口重叠等长度），阶段 A 每一步之后比较解析后的内容：文件头中的模型和 profile、每条 read 的 Meta/centers/scales/codes、重建 ADC。全部一致。
- [x] 确认无误后，按你的要求删除了这份基准测试和数据，不在仓库中保留旧版本的内容（提交 `12ee1fe` 中仍可查看）。
- 顺带：测试替身 `HashEngine` 原来用 blake2b 生成码字，现在改名为 `ContentEngine`，改用位置加权和。仓库中已没有任何哈希用法。

**A1 包改名**

- [x] `src/nanopore_app/` 改为 `src/nanorecon/`，继续使用 src 布局；pyproject、测试导入同步修改。
- 为什么包目录不能省：Python 的导入名就是目录名。如果文件直接放在 `src/` 下，`cli`、`models` 都会变成顶层模块，而 `types`、`io`、`signal` 会和标准库同名模块冲突。本轮在包目录内直接运行 Python 时就实际遇到了这个冲突：我们的 `types.py` 替换了标准库的 `types`，解释器启动即报错。

**A2 收缩 CLI，只支持 GPU**

精简后的完整接口：

```bash
nanorecon ls-remote
nanorecon pull
nanorecon ls
nanorecon info
nanorecon compress   input.pod5   -o output.nrpod   [--batch-size N] [--workers N] [--force] [-v]
nanorecon decompress output.nrpod -o restored.pod5  [--batch-size N] [--force] [-v]
```

| 已删除 | 替代 / 原因 |
|---|---|
| `--device`（含 `cpu` 路径） | 只用第一块可见 GPU；用 `CUDA_VISIBLE_DEVICES=1 nanorecon ...` 选卡 |
| `--attention` | 使用模型配置中的 cuDNN（与训练一致）。如果有 GPU 不支持，见 C4 |
| `--memory-budget-mib`、隐藏的 `--queue-capacity`、`--max-group-reads` | 改为内部常量，见 A4 |
| `--max-reads` | 需要小样本时，用 pod5 自带的 `pod5 subset` / `pod5 filter` 生成 |
| `--model`、`--revision`、`--model-path`、`info --remote`、`--hf-cache-dir`、`--home` | 只有一个模型，见 A3。缓存位置用 HF 标准环境变量（`HF_HOME`）控制 |
| `--json`，以及 help/version 的 JSON 捕获 | stdout 打印一行结果摘要，日志、进度和错误写 stderr |
| `-q` | 已删除；`-v` 保留，打印调试日志、模型耗时和 traceback |

- [x] 退出码只保留 0（成功）、1（错误）、2（参数错误）、130（中断）。
- [x] 错误类型只保留三个：`NanoReconError(message, hint)`、`UsageError`、`Cancelled`。hint 用于"请先 `nanorecon pull`"这类提示。
- [x] `--workers` 保留，C1 根据测量结果决定是否删除。
- [x] compress 和 decompress 的 `--batch-size` 缺省值分别定义（目前都是 16），由 C3 确定。
- 实现细节：`CompressOptions` 中保留 `queue_capacity`、`group_chunks`、`group_reads` 三个内部字段，只供测试缩小边界，不出现在命令行。

**A3 单模型的模型管理**

- [x] `model_config.py` 中固定 `REPO_ID`、`REVISION`、`CONFIG_FILE`、`WEIGHTS_FILE`，以及文件头用的 `MODEL`（模型身份）和 `PROFILE`。
- [x] `pull`：用 `hf_hub_download` 把两个文件下载到 HF 缓存，然后解析 config。下载会沿用 HF_HOME、HF_TOKEN 和代理设置；再次执行时直接复用缓存。
- [x] 已删除：
  - registry、ref 记录与迁移、commit 前缀解析；
  - `nanorecon-model.json` 和离线模型目录；
  - 对远端每个 revision 的"是否支持"判断；
  - pull 时的权重预检（权重结构在加载模型时完整核对）。
- [x] compress、decompress、ls、info 在 HF 缓存中定位文件；找不到时提示 `nanorecon pull`。离线机器：拷贝 HF 缓存目录，并设置 `HF_HOME`。
- [x] decompress 要求文件头的模型身份和 profile 与代码常量完全一致，否则报错并说明文件用的是哪个模型。文件头仍记录模型身份和 profile，将来换模型时可以再按文件头加载。
- [x] 各命令的输出：

  | 命令 | 显示内容 |
  |---|---|
  | `ls-remote` | 仓库名、固定 revision、该 revision 在远端是否还存在；远端 main 指向别的 commit 时提示 |
  | `ls` | 固定版本是否已下载、文件位置、大小 |
  | `info` | 模型身份、文件位置、网络关键参数、profile；注明训练用 DiVeQ σ² 不参与推理 |

- [x] `model_config.py` 精简为约 240 行：
  - 删除 `COMPAT_PROFILES`、`inference` 配置段解析、逐字段来源追踪和 profile 的通用校验；
  - `NetworkConfig` 仍从 `config.json` 解析（构造网络需要），但只保留训练工厂的缺省值，不再做字段类型和别名检查。
- [x] `docs/inference-profile.md` 的规则部分并入 `docs/format.md` 第 7 节后删除，"建议上游补 inference 段"的内容一并删除。
- 已对真实 Hub 运行 `ls-remote`、`ls`、`info`，结果正确；未在 Mac 上执行 `pull`。

**A4 去掉显式内存预算**（ISSUE-001、ISSUE-004）

- [x] 删除 `pipeline/budget.py`、`--memory-budget-mib`、解压的 `admit` 回调和逐 read 预算检查。
- [x] 压缩分组改用内部常量。满足以下任一条件时结束当前组：
  - 达到 `group_chunks` 个 chunk（初值 4096，约 66 MB 原始 ADC + 4 MB 码字）；
  - 达到 `group_reads` 条 read（初值 4096）；
  - 共享表接近格式上限。

  单条 read 超过上限时自成一组，不报错（有测试覆盖 40 万点、50 个 chunk 的 read）。初值由 C3 调整。
- [x] 解压不限制单条 read 的长度。一条 1000 万采样点的 read 约需 100 MB。
- [x] README 不再承诺内存上限；实测峰值将记入验证报告。

**A5 简化输出提交**（ISSUE-002）

- [x] 启动时检查：目标已存在且未给 `--force`，报错；输入和输出是同一个文件，报错。
- [x] 写入目标同目录下的临时文件 `.<name>.<pid>.nanorecon-tmp`（带 pid，避免两个进程撞名）。完成后 fsync，再用 `os.replace` 移到目标位置。失败或 Ctrl-C/SIGTERM 时删除临时文件。
- [x] 已删除硬链接提交、私有临时目录、目录 fsync，以及 pipeline 里重复的 fsync。
- 接受一个已知竞态：如果运行期间另一个进程创建了同名输出，它会被覆盖。实验室使用中可以忽略。

**A6 同步测试与文档**

- [x] 删除随功能删除的测试：registry、ref、远端支持判断、预算、JSON 输出、CPU 设备、硬链接退化，以及 80 次随机 POD5 损坏这类针对人为损坏输入的测试。
- [x] model_config、container 中改到的部分，顺手删除了只针对人为构造输入的检查（如 profile 的类型与范围校验、超大整数 epsilon、未实现的规则名）。其余已有的低成本检查保留。
- [x] README 按新 CLI 重写；`docs/validation.md` 改为"已验证 / 待 GPU 验证"两部分。
- [x] 版本号改为 0.3.0；中间格式仍为 1.0（格式没有变化）。
- [x] 提前做了 B3 的代码部分：`tests/integration/` 改为 `tests/gpu/`，见 B3。
- [x] 删除旧的 `tools/validate.py`：它依赖已删除的参数，已在 B3 中按新需求重写。

**验收：** Mac 上 `pytest`（193 个测试）全部通过；A0 对照一致；CLI 只剩 A2 列出的参数；pyflakes 检查无问题。

### 阶段 B：在服务器上验证（已完成）

**B0 做法**

- 不配置 CI。代码用 rsync 复制到 `potato:~/Work/nanopore-app`，用服务器的 `~/myenv` 运行（依赖版本与 pyproject 固定的完全一致）。用 `PYTHONPATH=src` 直接运行，不往 `~/myenv` 安装任何包。验证的中间文件放在服务器 `/tmp/nanorecon-validate`，用完即删。
- 复现命令和完整结果见 [docs/validation.md](docs/validation.md)。

**B1 服务器情况**（主机 `potato`，ssh 别名 `bioinformatics`，用户 `superbig`）

| 项目 | 情况 |
|---|---|
| 系统 | Ubuntu 24.04.3，EPYC 7663 56 核，440 GiB 内存，无作业调度器；另有其他用户日常使用 |
| GPU | 3 块 NVIDIA DRIVE-PG199-PROD（A100 级，compute capability 8.0，32 GB），驱动 570.195.03（CUDA 12.8）；没有系统 CUDA，也不需要 |
| Python | `~/myenv`：Python 3.12.3，jax 0.8.2 + CUDA 12 插件、flax 0.12.3、pod5 0.3.35、numpy 2.3.5、pytest |
| 磁盘 | `/home` 剩 1.2 TB，`/data_nvme` 剩 1.7 TB，`/data` 已用 92% |
| 真实数据 | `/data/nanopore` 下 1445 个 POD5，一般约 2 GB；最小的约 0.5–1 MB（`hereditary_cancer_positive_control_2025.11/raw/FC01/pod5/` 下的 `_135`、`_139`） |
| 训练仓库 | `~/Work/Nanopore-Reconstruction`（`97362cd`，`codec/models` 与 Mac 上的 `27d508d` 完全相同） |
| 模型 | 已执行 `nanorecon pull`，权重在 `~/.cache/huggingface` |

**B2 GPU 测试**

- [x] 单元测试 191 个通过；GPU 测试 7 个全部通过。
- [x] 与训练模型对照（cuDNN attention）：encoder 码字 100% 一致；同码字 decoder 最大差 5.4e-7。据此收紧测试容差：decoder < 1e-5；encoder 仍要求 ≥ 99.9%，以容许其他 GPU 上的近似并列翻转。
- [x] 合成 POD5 和真实 POD5 的 CLI 往返都通过；从文件解码与内存中的硬量化重建逐点一致；多线程与单线程压缩逐字节一致。

**B3 真实数据验证**（`tools/validate.py`）

- [x] 重写验证工具：按 chunk 数（而不是 read 数）截取子集；分别统计压缩和解压在各 batch 下的耗时、GPU 时间占比、补齐行比例和显存峰值；按区域统计重建误差；比较不同 batch 产生的码字。
- [x] 三个数据集，每个取约 1 万个 chunk：阳性对照（短 read 为主）、chrom_acc（长 read）、hereditary_cancer（63% 短 read）。主要结果：

  | 项目 | 结果 |
  |---|---|
  | Meta、顺序、长度 | 三个数据集全部零差异 |
  | 文件大小 | 比原 POD5 小 4.4–6.7 倍；短 read 多的数据压缩率低，因为每个 chunk 不论有多少真实信号都要存 512 个码字 |
  | 重建误差 | 平均 2.5–3.2 pA，约为单条 read 信号标准差的 10–16%；接缝比 read 内部高 8–16%，read 首尾边缘最多高 37% |
  | 压缩速度 | batch 16 / 64 / 256：330 / 738 / 1087 chunk/s；GPU 时间占 94–97% |
  | 解压速度 | 长 read 5.2 M 点/s（补齐行 7%）；短 read 为主的数据只有 0.38 M 点/s（补齐行 91%） |
  | 资源 | 主机内存约 2 GB；显存 1.1 / 1.7 / 4.9 GiB（batch 16 / 64 / 256） |

**B4 发现**

1. **GPU 0 掉线。** 在约 45 分钟、中间只有短暂间隔的高负载测试之后，GPU 0 报 Xid 79（GPU has fallen off the bus），驱动要求重启整机。你判断是温度过高，重启即可；不是程序错误。以后在这台机器上跑长任务时，可以用 `nvidia-smi dmon` 留意温度。
2. **码字与 batch 大小有关。** 同一 batch 下结果完全确定；不同 batch 之间有 0.6–0.8% 的码字不同（GPU 选用了不同的计算内核，近似并列的码字翻转），重建误差不变（2.6341 对 2.6340 pA）。与 DiVeQ 无关，推理中没有随机性。是否处理见 C6。
3. **worker 线程没有收益。** 12 亿点的长 read 子集上，主线程准备 460 秒，4 个 worker 469 秒。C1 的判断条件已满足。
4. **短 read 解压大部分算力浪费在补齐行上**（91%），需要做 C2。
5. **压缩 batch 越大越快。** 256 比 16 快 3.3 倍，显存 4.9 GiB。

**B5 剩余验证：跳过**

按你的决定，现有测试已经足够充分，以下几项不再做：teloseq 数据集、命令行整文件实测与中断测试、与训练仓库直接调用的速度对比。碱基识别对比属于模型本身的评估，模型已经测试过可以容忍，不在 APP 的范围内。

### 阶段 C：按测量结果调整（已完成，GPU 复测待服务器开机）

| 项 | 决定 |
|---|---|
| C1 删除多线程 | 做 |
| C2 解压跨 read 组 batch | 做 |
| C3 `--batch-size` 缺省值 | 做，改为 64 |
| C4 attention 后端 | 不做。都是较新的卡，服务器上已通过 |
| C5 拼接规则 | 不改。接缝只比 read 内部高 8–16%，乱改反而容易出问题 |
| C6 码字可复现性 | 不做。不同 batch 的码字差异对结果没有影响，batch 按需调整；docs/format.md 已有说明 |

**C1 删除多线程**（ISSUE-003）

实测压缩 94–97% 的时间花在 GPU 上，4 个 worker 线程与主线程准备 chunk 一样快。多线程没有收益，只增加开发和维护负担。

- [x] 删除 `ThreadedChunkEncoder`、`TaskCursor`、整个 `pipeline/sync.py`、`--workers` 参数，以及 `CompressOptions` 中的 `workers`、`queue_capacity`。
- [x] 保留现有的单线程路径：主线程准备 chunk、组 batch、调用 GPU、写回结果。
- [x] 删除只针对多线程的测试（线程泄漏、worker 异常传播、队列容量、完成顺序扰动等），README 和 `tests/gpu` 中去掉 `--workers`。
- [x] ISSUE-003（旧组被线程引用、没及时释放）随线程一起消失。
- 不做"GPU 计算时同时准备下一个 batch"的异步重叠：主机侧只占 3–6% 的时间，最多省这么多，不值得增加复杂度。
- 结果：`pipeline/compress.py` 由 456 行减到 257 行，删除了 `pipeline/sync.py`（71 行）。没有多线程后，chunk 按顺序准备、按顺序取回，组内结果直接按位置写入，原来为乱序结果准备的逐行登记和查重位图也一并删除。

**C2 解压跨 read 组 batch**（ISSUE-009）

- **问题：** 现在解压逐条 read 处理，一个 batch 只装同一条 read 的 chunk。一条 read 只有 1–2 个 chunk 时，batch 16 里其余 14–15 行都是补齐的空行，GPU 照样要算。
- **实测例子：** hereditary_cancer 子集有 6906 条 read、1 万个 chunk。现在要调用 GPU 6921 次（几乎每条 read 一次），91% 的行是空行，耗时 148 秒。改成跨 read 装满后，只需约 625 次调用（10000 ÷ 16），预计十几秒。长 read 数据只有每条 read 的最后一个 batch 有空行，影响较小（batch 16 时 7%，batch 64 时 29%），但同样受益。
- **来由：** 首版计划为了先求简单，把解压定为"一次处理一条 read"，跨 read 组 batch 列为以后的优化。从来没有"一个 batch 只能装一条 read"的要求；压缩一开始就是跨 read 组 batch 的。
- [x] 做法：
  - 按文件顺序读取 read，把各 read 的 chunk 依次填进同一个 batch，每行记下它属于哪条 read、是第几个 chunk；
  - batch 装满（或文件读完）时调用一次 decoder，把每行结果加到对应 read 的拼接累加器；
  - 某条 read 的 chunk 全部到齐，就转回 ADC，并按原顺序写出。空 read 也按顺序排队写出；
  - 同时处于"处理中"的 read 最多约 batch+1 条，内存仍然很小；文件仍然顺序读取。
- [x] 测试：
  - 无损替身往返测试改为在多种"压缩 batch × 解压 batch"组合下运行（各种长度下 ADC 逐点还原、顺序和 Meta 不变）；
  - 新增单元测试：96 条长短混合 read（含夹在中间的空 read），decoder 调用次数正好等于 `ceil(总 chunk 数 / batch)`，输出顺序和信号不变；
  - [ ] 服务器开机后跑一次单元测试和 GPU 测试，并复测解压速度。
- 实现：`pipeline/decompress.py` 中的 `_BatchDecoder` 负责装 batch 和按顺序交回完成的 read；`StoredRead.read_codes()` 一次读出一条 read 的全部码字（每个 chunk 约 1 KB）。

**C3 `--batch-size` 缺省值**

- batch 是一次 GPU 调用处理的 chunk 数（一个 chunk 是 8192 个采样点）。使用时随时可以用 `--batch-size` 调整；C3 只是决定不写这个参数时用多少。
- 实测：

  | batch | 压缩速度 | 显存 | 首次编译 |
  |---|---|---|---|
  | 16（现在的缺省值） | 330 chunk/s | 1.1 GiB | 约 5 秒 |
  | 64 | 738 chunk/s | 1.7 GiB | 约 5.5 秒 |
  | 256 | 1087 chunk/s | 4.9 GiB | 约 7 秒 |

- [x] compress 和 decompress 统一缺省为 64：比 16 快约 2.2 倍，显存不到 2 GiB，给同一台机器上的其他任务留足空间。GPU 空闲、想要更快时，手动加 `--batch-size 256`。C2 完成后，解压也能从大 batch 中受益。
- [x] README 写明这几个实测数字，以及是在哪块 GPU 上测得的。

### 阶段 D：Linux CLI 打包发布（一键安装包已做好，待 Linux 实装测试）

- [x] 整理文档：旧的分析和设计文档已由你清理；`docs/` 只剩 `format.md` 和 `validation.md`。
- 暂不建 GitHub 仓库（以后由你指定），暂不需要 LICENSE（内部使用）。

**D0 情况**

- 代码本身很小：wheel 约 50 KB。
- 依赖很大：jaxlib 80 MB、jax-cuda12-pjrt 154 MB，再加十几个 nvidia-* CUDA 库（cuDNN、cuBLAS 等）。首次安装约下载 2.5–3 GB，装好后每个环境约占 5 GB。
- CUDA 由 pip 随依赖带上，用户机器只需要 NVIDIA 驱动，不需要装 CUDA。
- 模型（420 MB）不打进安装包，由 `nanorecon pull` 从 Hugging Face 下载。

**D1 做法：一键安装包**（按你的选择）

- 和"游戏启动器"一样：安装包里只有我们的代码（wheel），庞大的依赖在安装时从 PyPI 下载。
- 与"用户手动 pip"相比，机制相同，区别是安装脚本替用户做完所有步骤：检查环境、建独立环境、装依赖、建命令链接、下载模型。
- 用户拿到一个文件 `nanorecon-0.3.0-linux-x86_64.sh`（56 KB），执行 `bash nanorecon-0.3.0-linux-x86_64.sh` 即可。没有仓库之前，用 `scp` 发给要用的人。
- 需要时也可以用 `--extract DIR` 只解出 wheel，自己用 pip 安装。

**D2 安装包的行为**（`packaging/install.sh`）

- [x] **安装前检查**（前三项不满足时给出原因并退出，不留下任何东西；GPU 一项只警告）：
  - Linux x86_64，glibc ≥ 2.27（JAX 的要求；对应 Ubuntu 18.04+、RHEL/Rocky 8+）；
  - 依次寻找 python3.13 / 3.12 / 3.11 / python3，要求版本在 3.11–3.13 之间且能创建 venv。Ubuntu 上缺 venv 时，提示安装 `python3.12-venv`；
  - 安装目录所在磁盘至少剩 8 GB（环境约 5 GB，pip 缓存约 3 GB）；
  - `nvidia-smi` 可用且驱动 ≥ 525。不满足时只警告，因为模型管理命令不需要 GPU。
- [x] **安装：**
  - 解出内置的 wheel；
  - 在 `~/.local/share/nanorecon/<版本>/` 创建 venv，执行 `pip install "<wheel>[cuda12]"`。这一步下载依赖，可复用 pip 缓存；代理和 PyPI 镜像的环境变量照常生效；完整输出写入 `~/.local/share/nanorecon/install-<版本>.log`；
  - 运行 `nanorecon --version` 自检；
  - 把 `~/.local/bin/nanorecon` 链接到这个版本。如果 `~/.local/bin` 不在 PATH 中，提示怎么加；
  - 执行 `nanorecon pull` 下载模型。下载模型失败不算安装失败，只提示稍后再执行一次。
- [x] **失败或 Ctrl-C：** 删除不完整的版本目录，保留日志，并提示日志位置。
- [x] **升级：** 运行新版本的安装包，装到新的版本目录并切换命令链接；旧版本保留，可用完整路径运行。
- [x] **卸载：** 帮助信息和安装结束时都给出命令：`rm -rf ~/.local/share/nanorecon ~/.local/bin/nanorecon`。

**D3 依赖锁定：不做**（按你的决定）

APP 之后不出大问题不会更新。只固定 pyproject 中的直接依赖版本，传递依赖由 pip 解析。

**D4 构建**

- [x] 版本号只保留一个来源：pyproject 通过 setuptools 的 dynamic version 读取 `nanorecon.__version__`，安装包文件名、安装目录和 `--version` 都从这里来。
- [x] `tools/build_installer.sh`：构建 wheel，把它附加到 `packaging/install.sh` 末尾，生成 `dist/nanorecon-<版本>-linux-x86_64.sh`。`dist/` 不进 git。
- [x] 已确认 wheel 只包含 `nanorecon/` 包和元数据，不含 tests、tools、docs。
- [x] 在 Mac 上已检查：
  - 两个脚本通过 `bash -n` 和 shellcheck；
  - `--help` 输出正确；在非 Linux 机器上会给出明确的拒绝提示；
  - `--extract` 解出的 wheel 与构建的完全相同；
  - 该 wheel 装进全新环境后，`nanorecon --version` 显示 0.3.0。
- [ ] 在 Linux 服务器上以全新的 HOME 实际安装一次（需要服务器开机）。

**D5 支持范围**（已写进 README）

| 项目 | 要求 |
|---|---|
| 系统 | Linux x86_64，glibc ≥ 2.27 |
| Python | 系统中有 3.11–3.13，并有 venv 模块（Ubuntu 24.04 自带 3.12） |
| GPU | NVIDIA，驱动支持 CUDA 12（525+）；较新的卡（服务器上的 A100 级已验证） |
| 显存 | 缺省 batch 64 约 1.7 GiB；batch 256 约 4.9 GiB |
| 磁盘 | 约 8 GB（环境约 5 GB，pip 缓存约 3 GB）；模型 420 MB |
| 网络 | 安装时能访问 PyPI（或镜像），`pull` 时能访问 huggingface.co（或代理） |

**D6 以后需要时再做的增强**

- 安装包内嵌 uv（体积变为约 20 MB），在没有合适 Python 的机器上自动下载一个 Python。
- 完整离线包（约 3 GB），给不能联网的机器。
- 指定安装目录，用于在服务器上做组内共享安装。

**D7 每次发布的检查清单**

1. Mac：`pytest`。
2. rsync 到服务器：单元测试，加上 `pytest tests/gpu`。
3. 改动涉及数值或性能时：用 `tools/validate.py` 在 2–3 个数据集上各跑约 1 万个 chunk，与 docs/validation.md 对比。
4. 修改 `src/nanorecon/__init__.py` 中的版本号，运行 `tools/build_installer.sh`。
5. 在服务器上以全新的 HOME（例如 `HOME=/tmp/nanorecon-install-test bash dist/nanorecon-<版本>-linux-x86_64.sh`）安装，再对一个小 POD5 做 compress/decompress。
6. 把安装包发给使用者；有仓库之后改为附在 GitHub Release 上。

**D8 换模型时**

修改 `REVISION`，发新版本。token 文件记录了模型身份，旧文件需要用旧版本 APP 解码。因为每个版本装在独立目录，旧版本一直可用：运行 `~/.local/share/nanorecon/<旧版本>/bin/nanorecon` 即可。

## 5. 问题清单的处理

| ID | 问题 | 处理 | 状态 |
|---|---|---|---|
| ISSUE-001 | 显式内存预算与使用要求不符 | 删除预算，改为内部按 chunk 数分组 | 已完成（A4） |
| ISSUE-002 | 输出提交退化路径的并发覆盖 | 简化为启动时检查 + `os.replace`；明确接受该竞态 | 已完成（A5） |
| ISSUE-003 | 已完成的压缩组未及时释放 | 随多线程一起删除 | 已完成（C1） |
| ISSUE-004 | Meta 与 RunInfo 索引的开销未计入预算 | 不再承诺字节预算，不单独处理；实测峰值记入报告 | 已关闭（A4） |
| ISSUE-005 | decoder 对照可能被跳过 | 拆开 encoder 与 decoder 的对照 | 已完成，GPU 上通过（B2） |
| ISSUE-006 | Linux GPU 未验收 | 在服务器上手工验证 | 已完成（B） |
| ISSUE-007 | 真实数据质量与接缝未验收 | 报告按区域统计；必要时调整拼接规则 | 已完成：三个数据集，接缝无需调整（C5） |
| ISSUE-008 | 依赖 pod5 私有缓存接口 | 固定 pod5 版本；升级时手动跑内存检查 | 接受，升级时处理 |
| ISSUE-009 | 组间停顿、逐 read 解码影响吞吐 | 已测量：压缩瓶颈在 GPU；解压浪费在补齐行，由 C2 解决 | 已完成（C2），GPU 复测待做 |
| ISSUE-010 | DiVeQ 噪声参数含义 | 属于训练侧事项，移出 APP 计划 | — |

## 6. 本轮不做

- CPU 推理；多模型或多版本管理；离线模型目录；跨模型版本解码。
- JSON 输出；可调的内存预算；隐藏的调参选项。
- 哈希或摘要校验；训练指标输出。
- 多 GPU、多进程、随机访问索引、二次熵压缩、目录批处理、Rust 重写。
- 推理时加 DiVeQ 噪声；训练侧修改（ISSUE-010 归训练仓库处理）；模型本身的质量评估（如碱基识别对比）。
- GPU 计算与主机侧工作的异步重叠（主机侧只占 3–10% 的时间）；cuDNN attention 的回退（C4）；修改拼接规则（C5）；固定压缩 batch 以求文件可复现（C6）。

以上需要时再提。

## 7. 0.3.0 完成标准

- [x] 六个命令可用，参数只有 A2 列出的那些；Mac 上单元测试通过。
- [x] 服务器上的 GPU 测试通过：
  - encoder 与 decoder 分别对照；
  - 真实 POD5 往返中，read 数、顺序、长度、Meta 零差异。
- [x] 验证报告包含按区域统计的质量、压缩率、吞吐和内存数据。
- [x] C1–C3 完成，Mac 上单元测试通过。
- [ ] 服务器上的单元测试和 GPU 测试在 C1/C2 改动后再次通过。
- [ ] 一键安装包按 D7 的清单在服务器上以全新的 HOME 安装成功，并完成一次 compress/decompress。
- [x] README 与实际 CLI 和安装方式一致（只写现状和用法，包括不同配置的推荐参数）。
