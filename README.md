# deepfake-orangepi
端侧AI语音深度伪造检测系统 | 国产Orange Pi部署 | 离线实时推理RTF 0.1569 · GUI可视化
&lt;p align="center"&gt;
  &lt;b&gt;国产边缘设备 · OpenGauss数据库 · 离线实时推理&lt;/b&gt;
&lt;/p&gt;

&lt;p align="center"&gt;
  &lt;img src="https://img.shields.io/badge/Python-3.8+-blue.svg" /&gt;
  &lt;img src="https://img.shields.io/badge/PyTorch-1.12+-orange.svg" /&gt;
  &lt;img src="https://img.shields.io/badge/Platform-Orange%20Pi-green.svg" /&gt;
  &lt;img src="https://img.shields.io/badge/License-MIT-yellow.svg" /&gt;
&lt;/p&gt;

---

## 📌 项目简介

本项目是一套面向**国产信创终端**的端侧AI语音深度伪造检测系统。针对语音克隆诈骗频发、云端方案高延迟且无法离线运行的痛点，将模型部署至 **Orange Pi 5 Plus** 国产边缘设备，实现从音频采集、模型推理到结果可视化的**全链路AI自动化闭环**。

系统支持**离线实时检测**，无需云端交互即可在资源受限场景下完成毫秒级推理，适用于金融客服、远程身份核验等隐私敏感场景。

&gt; **核心指标**：推理 RTF **0.1569**，基于 ASVspoof 数据集验证。

---

## ✨ 功能特性

- 🎙️ **实时音频鉴伪**：支持 `.wav`、`.flac` 等格式音频输入，端到端检测伪造语音
- 📊 **GUI 可视化界面**：PyQt5 构建的图形化界面，实时展示波形、频谱与检测结果
- 🗄️ **OpenGauss数据管理**：检测历史自动入库，支持查询、追溯与批量导出
- 🔒 **纯离线运行**：端侧完成全部推理，零云端依赖，保障数据隐私
- 🇨🇳 **国产平台适配**：针对 Orange Pi 5 Plus等 ARM 架构边缘设备进行深度优化

---

## 🏗️ 技术架构

核心逻辑流采用**多阶段长链推理架构**：
