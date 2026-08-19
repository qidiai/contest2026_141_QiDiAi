# contest_board — 板级适配

## 实板构建

本项目在 **SF32LB52-DevKit-LCD** 开发板上验证通过。实板构建使用 openvela 自带的板级配置：

```
vendor/sifli/boards/sf32lb52/sf32lb52_devkit_lcd/
```

### 关键配置（defconfig 要点）

以下配置在 SF32LB52-DevKit-LCD 的 `.config` 中启用：

```
# LVGL 图形框架
CONFIG_LVGL=y
CONFIG_LV_USE_NUTTX=y
CONFIG_LV_USE_DEV_XPT2046=n

# LCD (CO5300 AMOLED 390x450)
CONFIG_LCD=y
CONFIG_LCD_USING_CO5300=y

# 触摸 (FT6146 I2C)
CONFIG_INPUT=y
CONFIG_INPUT_FT6146=y
CONFIG_TOUCH_IRQ_PIN=31

# Jianmu 应用
CONFIG_EXAMPLES_JIANMU=y
CONFIG_LVX_USE_DEMO_CONTEST2026_141_JIANMU=y

# V10 XIP 权重嵌入
CONFIG_V10_XIP_EMBED=y

# I2C 调试
CONFIG_I2C_POLLED=y
CONFIG_DEBUG_I2C=y

# NSH
CONFIG_NSH_MAXARGUMENTS=12
```

### 编译命令

```bash
# 在 openvela 工作区根目录
./build.sh vendor/sifli/boards/sf32lb52/sf32lb52_devkit_lcd/configs/nsh menuconfig
./build.sh vendor/sifli/boards/sf32lb52/sf32lb52_devkit_lcd/configs/nsh -j8
```

### 烧录

```bash
sftool -c SF32LB52 -p COM3 -b 115200 --compat true \
  --connect-attempts 0 --before default_reset --after soft_reset \
  write_flash --no-compress "nuttx_xip.bin@0x12010000"
```

## QEMU 模拟

本目录下的 `configs/nsh/defconfig` 是 QEMU 占位配置，用于无硬件时的基础编译验证。
实板功能（LVGL UI、触摸、V10 XIP 推理）需要 SF32LB52 开发板。
