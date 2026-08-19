#!/usr/bin/env bash
# ============================================================================
# QiDiAi 建木 — WSL2 开发环境一键引导脚本
# 用法（在 WSL2 Ubuntu 终端内执行）：
#   bash tools/wsl2-bootstrap.sh
# 脚本会：安装依赖 -> 拉取 openvela 全量源码 -> 打开建木 Demo 的 menuconfig
# 注意：repo sync 下载量大，请确保磁盘 ≥ 60GB 且网络可访问 github。
# ============================================================================
set -euo pipefail

WORKSPACE="${OPENVELA_WORKSPACE:-$HOME/openvela}"
MANIFEST_REPO="https://github.com/open-vela/contest2026_141_QiDiAi"
MANIFEST_BRANCH="dev-ai-contest-2026"
MANIFEST_FILE="contest2026_141_QiDiAi.xml"

echo "==> [1/4] 安装编译依赖"
sudo apt update
sudo apt install -y repo git curl python3 python3-pip build-essential \
  gcc g++ gperf flex bison texinfo libssl-dev libncurses-dev \
  device-tree-compiler dfu-util gettext unzip xz-utils pkg-config \
  cmake ninja-build

# 确保 repo 在 PATH（若 apt 未提供）
if ! command -v repo >/dev/null 2>&1; then
  mkdir -p ~/.bin
  curl -fsSL https://storage.googleapis.com/git-repo-downloads/repo > ~/.bin/repo
  chmod +x ~/.bin/repo
  grep -q '$HOME/.bin' ~/.bashrc || echo 'export PATH=$HOME/.bin:$PATH' >> ~/.bashrc
  export PATH="$HOME/.bin:$PATH"
fi

echo "==> [2/4] 创建工作区并拉取 openvela 全量工程"
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"
if [ ! -d .repo ]; then
  repo init -u "$MANIFEST_REPO" -b "$MANIFEST_BRANCH" -m "$MANIFEST_FILE"
fi
repo sync -c -j"$(nproc)"

echo "==> [3/4] 进入 openvela 工作区根目录"
cd "$WORKSPACE"

echo "==> [4/4] 提示下一步"
cat <<EOF

拉取完成。你的队伍仓在: $WORKSPACE/contest2026_141_QiDiAi/
后续步骤：
  1) 编译前开启建木 Demo：
       ./build.sh <board-config-path> menuconfig
       -> Application Configuration -> Demos ->
          [*] Contest 2026 team 141 - QiDiAi 建木 on-device semantic index
  2) 编译： ./build.sh <board-config-path> -j\$(nproc)
  3) 运行： 烧录开发板，或 QEMU 模拟器中执行 'jianmu'

<board-config-path> 与烧录方式以你所在赛道教程为准。
EOF
