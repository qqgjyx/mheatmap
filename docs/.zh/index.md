# <img src="../assets/mheatmap.png" width="40px" align="center" alt="mheatmap logo"> mheatmap

[![PyPI version](https://badge.fury.io/py/mheatmap.svg)](https://badge.fury.io/py/mheatmap)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个用于高级热图可视化和矩阵分析的Python包，
具有马赛克/比例热图、混淆矩阵后处理
和谱重排序功能。

---

## 🚀 特性

- **马赛克热图**  
  使用比例大小的单元格可视化矩阵值。  
  ![普通热图和马赛克热图的对比](../assets/basic_mosaic_heatmap.png)

- **自动模型校准 (AMC)**  
  对齐、掩码和混淆——一种用于后处理混淆矩阵的算法。

- **谱重排序**  
  基于谱分析重新排序矩阵。
  ![谱重排序示例](../assets/spectral_reordering.png)

- **RMS (反向合并/分割) 分析**  
  执行高级排列分析以探索矩阵结构。  
  ![RMS排列示例](../assets/rms_permutation.png)

---

## 📦 安装

### 从 PyPI 安装

```bash
pip install mheatmap
```

### 从源码安装

```bash
git clone https://github.com/qqgjyx/mheatmap.git
cd mheatmap
pip install .
```

## 📘 文档

全面的文档可在 `docs/` 目录中找到，
包括以下内容：

- [马赛克热图](docs/mosaic_heatmap.md)
- [AMC 后处理](docs/amc_postprocess.md)
- [谱重排序](docs/spectral_reordering.md)
- [RMS 分析](docs/rms_permutation.md)

## 🛠 贡献

我们欢迎贡献来改进 mheatmap！请按照以下步骤操作：

1. Fork 仓库
2. 创建一个新分支 (`feature-branch`)
3. 提交你的更改
4. 打开一个拉取请求

## 📝 许可证

本项目采用 MIT 许可证。
详见 [LICENSE](LICENSE) 文件。
