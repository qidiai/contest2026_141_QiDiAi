# 开发环境搭建（WSL2 + openvela）

> 适用：Windows 主机 + WSL2 Ubuntu。openvela 编译需要 Linux 环境。
> 脚本版见仓库根 `tools/wsl2-bootstrap.sh`（在 WSL2 内执行）。

## 0. 前提

- Windows 10/11 已安装 WSL2 + Ubuntu 22.04+。
- 磁盘预留 **≥ 60GB**（openvela 全量源码 + 编译产物较大）。
- 网络可访问 github.com / 组委会镜像。

## 1. 安装 WSL2（若未装）

```powershell
# 在 Windows PowerShell（管理员）中
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
```

## 2. 在 WSL2 内安装依赖

```bash
sudo apt update
sudo apt install -y repo git curl python3 python3-pip build-essential \
  gcc g++ gperf flex bison texinfo libssl-dev libncurses-dev \
  device-tree-compiler dfu-util gettext unzip xz-utils pkg-config \
  cmake ninja-build
# 校验 repo 工具
repo --version
```

> 若 `repo` 未随 apt 提供，手动安装：
> ```bash
> mkdir -p ~/.bin && curl https://storage.googleapis.com/git-repo-downloads/repo > ~/.bin/repo \
>   && chmod +x ~/.bin/repo && echo 'export PATH=$HOME/.bin:$PATH' >> ~/.bashrc
> ```

## 3. 拉取完整工程

```bash
# 选一个工作区目录（不要用 /mnt/c 下的 Windows 路径，IO 极慢）
mkdir -p ~/openvela && cd ~/openvela
repo init -u https://github.com/open-vela/contest2026_141_QiDiAi \
  -b dev-ai-contest-2026 -m contest2026_141_QiDiAi.xml
repo sync -c -j8
```

同步后结构：`~/openvela/contest2026_141_QiDiAi/`（你们自己的仓）+
外层 `nuttx/ apps/ packages/ vendor/ prebuilts/` 等 openvela 全量源码。
**只在 `contest2026_141_QiDiAi/` 里改代码**，其余由 manifest 软链接管。

## 4. 编译（在 openvela 工作区根目录，即你们仓的上一级）

```bash
cd ~/openvela
# 先用 menuconfig 开启建木 Demo：
./build.sh <board-config-path> menuconfig
#   路径：Application Configuration -> Demos ->
#         [*] Contest 2026 team 141 - QiDiAi 建木 on-device semantic index
./build.sh <board-config-path> -j8
```

`<board-config-path>` 与目标产物以**你所在赛道教程**为准：
- AI 硬件 / ESP32-S3 EYE：见 AI 硬件赛道教程导航（工具链 `xtensa-esp32s3-elf` 已在 prebuilts）。
- 手表 / SF32LB52 LCD：见对应板级教程。

## 5. 无板验证：QEMU 模拟器

openvela 提供 QEMU 模拟器（prebuilts/qemu），**没有实体板也能先验证编译与推理**：

```bash
./build.sh <qemu-board-config-path> -j8
# 按教程启动模拟器，进入 NSH shell 后执行：
nsh> jianmu
# 预期看到：离线嵌入 + 本地索引 + Top-K 语义检索结果
```

## 6. 常见问题

- **repo sync 慢 / 中断**：`repo sync -c -j8` 可重入，断点续传。
- **编译报 `sqrtf` 未定义**：确认配置开启了 libm（openvela 默认含）。
- **Windows 路径编译极慢**：务必在 WSL2 的 Linux 文件系统（`~/`）内操作，勿用 `/mnt/c`。
- **改了 app/jianmu 不生效**：确认 `contest2026_141_QiDiAi.xml` 的 linkfile 指向 `app/jianmu`，
  且 menuconfig 已开启 `LVX_USE_DEMO_CONTEST2026_141_JIANMU`。

## 7. V9v3 引擎与权重（建木核心）

`app/jianmu/` 已集成**真实 V9v3 引擎**（从 `jianmu-os/worm_knowledge_engine/c_core`
逐字复制，零依赖，仅 libc + libm）：

- `v9v3_infer.c` —— 推理核心（6 层 SourcePool Transformer，1024-d 输出）。
  **其 `main()` 已在本仓副本中改名为 `v9v3_infer_main()`**，以便和 `jianmu_main.c`
  共存于同一可执行文件。如需独立 CLI，单独编译 `v9v3_infer.c` 即可
  （`gcc -O2 -lm v9v3_infer.c`）。
- `v9v3_api.c` —— 公开 API：`v9v3_init / v9v3_embed / v9v3_get_pools /
  v9v3_free / v9v3_output_dim / v9v3_pool_dim`。
  **不要用 `v9v3_wrapper.c`**：它把 `model_init` 等声明成 `static`，与
  `v9v3_infer.c` 的外部符号冲突，链接会失败。
- `v9v3_weights.baize` —— 12.3MB 权重（3.15M 参数，与方案参数一致）。

### 权重部署（运行时必读）
`jianmu_main.c` 启动时调用 `v9v3_init(path)`，默认读 `/data/v9v3_weights.baize`
（可用 `jianmu <path>` 覆盖）。**权重文件必须在设备文件系统上可用**：
- QEMU：把 `.baize` 放进模拟器挂载的目录 / 打进 ROMFS 镜像。
- 真机：烧录后 `adb push` / 拷贝到对应分区，或修改 `V9V3_MODEL_PATH` 宏。
- 仓库内 `app/jianmu/v9v3_weights.baize` 仅为方便构建/版本对照；
  **最终提交建议 gitignore 或用 Git LFS**，避免把 12.3MB 二进制压进 git。

### ⚠️ RAM 约束（重要）
`v9v3_infer.c` 的 `model_init` 会把整份权重 **`malloc` 进 RAM（≈12.3MB）**。
- 在 QEMU / Linux 主机上完全没问题。
- 在**手表（SF32LB52）/ ESP32-S3 EYE 上通常放不下**（EYE 一般 8MB PSRAM < 12.3MB）。
  → **全量模型用于 QEMU/主机原型验证**；**真上可穿戴请用 V9v3-nano（<5MB）**，
  这点和方案里的轻量化演进规划一致。

### 主机侧先验证（不烧板）
仓库外的 `jianmu-os/worm_knowledge_engine/c_core/` 已带编译好的 Linux 二进制
（`v9v3_infer`、`libv9v3.so`）。在 WSL2 里即可验证权重与语义质量：

```bash
# 引擎自检（确定性 / 500 轮稳定性 / L2 范数）
/mnt/<盘>/.../c_core/v9v3_infer --test /mnt/<盘>/.../c_core/model/v9v3_weights.baize

# 用 libv9v3.so + Python ctypes 快速看语义相似度
python3 - <<'PY'
import ctypes, math
lib = ctypes.CDLL("/path/to/libv9v3.so")
lib.v9v3_init.argtypes=[ctypes.c_char_p]; lib.v9v3_init.restype=ctypes.c_int
lib.v9v3_embed.argtypes=[ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.v9v3_embed.restype=ctypes.c_int
def emb(t):
    o=(ctypes.c_float*1024)(); lib.v9v3_embed(t.encode(),o,1024); return list(o)
def cos(a,b):
    d=sum(x*y for x,y in zip(a,b)); return d/((sum(x*x for x in a)*sum(y*y for y in b))**0.5)
lib.v9v3_init(b"/path/to/v9v3_weights.baize")
print(cos(emb("找上次聊装修的朋友"), emb("上次聊装修的朋友 小王 微信")))  # ~0.74
print(cos(emb("找上次聊装修的朋友"), emb("健身计划 每周三跑步五公里")))  # ~0.25
PY
```
实测真实模型对中文语义的区分度明显（0.74 vs 0.25），远好于占位哈希。

### 链接常见坑
- 若报 `undefined reference to 'sqrtf' / 'expf' / 'tanhf'`：应用 Makefile 加
  `LDLIBS += -lm`、CMakeLists 给 `nuttx_add_application` 加 `LINK_FLAGS -lm`。
- 栈：索引的查询向量是 1024-d（4KB），`jianmu` 的 `STACKSIZE` 已设为 16384；
  文档向量已改为堆分配（`si_build` 内 `calloc`），不会爆栈。
