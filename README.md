# QiDiAi 建木 — 基于 openvela 的可穿戴设备语义索引

> 2026 首届 openvela AI 硬件开发者大赛 · 队伍 QiDiAi（编号 141）
> 赛道：手表应用创新 + AI 硬件产品创新

## 一、作品简介

**建木**是一个面向 openvela 智能手表 / 手环的**轻量级离线 AI 语义检索应用**。
它将自研 **V10 向量嵌入模型**（2.58M 参数、1024 维、9.8 MB 权重、C 实现、仅依赖 libc）
部署于 openvela，为可穿戴设备提供**毫秒级、完全离线**的语义搜索能力，
解决云端大模型在可穿戴场景下的四大盲区：持续在线功耗、隐私红线、毫秒级响应、离线连续性。

本仓库包含一个**可编译、可烧录、可运行的完整应用**（`app/jianmu/`）：
- 端到端跑通 `文本嵌入 → 本地向量索引 → 离线语义检索`
- **V10 引擎已接入真实模型**（1024 维输出、9.8 MB 权重、纯 libc+libm 零依赖），非占位实现
- **LVGL 触摸前端 UI**（390x450 AMOLED 屏，中文搜索框 + 结果列表），基于 openvela 图形框架
- **XIP 权重嵌入**：模型权重以 `.weights_blob` 段链接进固件镜像，运行时零拷贝直接从 NOR flash 读取
- `jianmu --selftest` 在 SF32LB52-DevKit-LCD 真机验证 **PASS**

## 二、选题方向

- **手表应用创新**：在 openvela 手表 / 手环上提供腕上语义搜索与索引管理 UI（LVGL 中文界面 + 触摸交互）。
- **AI 硬件产品创新**：将 V10 端侧推理作为核心 AI 能力落地，含 BLE 多设备语义同步、OTA。

## 三、目录结构

```
app/jianmu/               — 核心应用
  ├─ jianmu_main.c        — 入口：加载权重 → embed → 建索引 → 检索 → LVGL UI
  ├─ lvgl_frontend.c/.h   — LVGL 触摸前端（搜索框 + 结果列表，390x450 AMOLED）
  ├─ lv_font_misans_ui.c  — MiSans 中文字体子集（UI 文案 + 搜索语料覆盖）
  ├─ v10_infer.c          — V10 推理核心（真实引擎，Pure Source Pool 架构）
  ├─ v10_api.c            — V10 公开 API（init / embed / get_pools / free）
  ├─ v10.c / v10.h        — 引擎胶水层
  ├─ v10_token_map.h      — V10 tokenizer 词表（92K 条目）
  ├─ weights_blob.S        — XIP 权重嵌入汇编（.incbin → v10_weights.baize）
  ├─ v10_weights.baize    — 9.8 MB 模型权重（gitignore，需本地放置）
  ├─ semantic_index.c/.h  — 本地向量索引与余弦相似度检索
  ├─ v9v3_*.c/.h          — V9v3 旧引擎（保留对照）
  └─ v9v3_weights.baize   — 旧权重（保留对照）
board/contest_board/      — 板级适配（SF32LB52 LCD）
docs/建木-方案.md          — 完整参赛方案
docs/env-setup.md         — 开发环境搭建（WSL2 + repo 拉取 + 编译烧录）
tools/wsl2-bootstrap.sh   — WSL2 一键环境引导脚本
logs/                     — AI Coding 日志（按组委会规范导出）
```

## 四、核心能力与 openvela 对接

| openvela 核心能力 | 对接方式 | 实现 |
|------------------|---------|------|
| **AI** | V10 嵌入模型 + 本地向量检索 | 端侧语义理解，离线可用，推理 ~3ms |
| **图形** | LVGL + CO5300 AMOLED | 中文搜索框、结果列表、触摸交互 |
| **多媒体** | FT6146 触摸输入 | 搜索查询输入、结果选择 |

### V10 引擎

| 指标 | V10 | 传统方案 |
|------|-----|---------|
| 推理延迟 | ~3ms | 云端 30s-5min |
| 模型体积 | 9.8 MB | 1.2 GB (bge-m3) |
| 部署依赖 | libc only | PyTorch / Ollama |
| 断网可用 | 100% | 0% |
| 权重加载 | XIP 零拷贝 | 需加载到 RAM |

### LVGL 前端 UI

- **屏幕**：SF32LB52-DevKit-LCD 1.85" CO5300 AMOLED 390x450
- **触摸**：FT6146 电容触摸（I2C 0x38）
- **界面**：搜索输入框 + 结果列表 + 状态栏
- **字体**：MiSans 中文子集，覆盖 UI 文案与搜索语料
- **构建**：`DEFINITIONS V10_XIP_EMBED`，权重以 `.weights_blob` 段 XIP 链接

## 五、运行方式

> 完整步骤见 [`docs/env-setup.md`](docs/env-setup.md)。要点：

1. **拉取完整工程**（在 WSL2 Ubuntu 中，工作区根目录）：
   ```bash
   repo init -u https://github.com/open-vela/contest2026_141_QiDiAi \
     -b dev-ai-contest-2026 -m contest2026_141_QiDiAi.xml
   repo sync -c -j8
   ```
2. **放置权重文件**：将 `v10_weights.baize`（9.8 MB）放到 `app/jianmu/` 目录
3. **启用并编译**（在 openvela 工作区根目录）：
   ```bash
   ./build.sh <board-config-path> menuconfig   # 开启 LVX_USE_DEMO_CONTEST2026_141_JIANMU
   ./build.sh <board-config-path> -j8
   ```
4. **烧录**：使用 sftool 烧录到 SF32LB52 开发板
5. **运行**：在 NSH 终端执行 `jianmu --selftest` 验证，或直接 `jianmu` 启动 LVGL UI

## 六、selftest 验证

在 SF32LB52-DevKit-LCD 真机上 `jianmu --selftest` 输出：

```
=== QiDiAi 建木 (openvela on-device semantic index) ===
[V10] [XIP] 2580168 params, weights=9.8 MB, runtime=321.6 KB
  [0] "测试文本": norm=1.0000 nan=0 OK
  [1] "hello world": norm=1.0000 nan=0 OK
  [2] "": norm=0.0000 nan=0 OK

=== selftest PASSED (embedded XIP weights, real model) ===
```

## 七、AI Coding 使用说明

本作品全程借助 AI 辅助开发（需求拆解、方案设计、代码骨架、调试、文档）。
- 方案定义与功能拆分见 `docs/建木-方案.md`；
- V10 引擎集成、LVGL 前端、CMakeLists、板级配置由 AI 辅助完成；
- 完整 AI 对话日志见 `logs/` 目录（按组委会规范导出）。

## 八、提交须知（评委评估口径）

评委依据「**作品本身 + 本 README 说明 + `logs/` 里的 AI Coding 日志**」评估。
- 所有改动经 **Pull Request** 合入（分支保护，可自行合入自己的 PR）。
- 首次贡献需在官网签署 **CLA**，PR 评论 `/check-cla` 复检。
- **作品提交截止：2026-09-20**，截止后收回 push 权限。
- 官方流程以 [《参赛代码提交指南》](https://github.com/open-vela/docs/blob/dev-ai-contest-2026/zh-cn/contest_2026/code_submission_guide.md) 为准。
