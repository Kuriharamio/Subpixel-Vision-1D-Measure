# 机器视觉课题 3 —— 跑道形空腔尺寸测量

> 用 C++17 + OpenCV，对工件上的跑道形 (stadium / discorectangle) 空腔做 1D 亚像素尺寸测量：r1、r2、d1、d2，并评估测量重复性。
>
> 完整算法原理、实验数据与分析见 [📘 设计报告 (报告.md)](报告.md)。

---

## 测量量

| 量 | 含义 | 直观比喻 |
|---|---|---|
| **r1** | 空腔左端圆弧半径 | 跑道左侧 180° 弧 |
| **r2** | 空腔右端圆弧半径 | 跑道右侧 180° 弧 |
| **d1** | 跑道形空腔外缘宽度 (外上→外下) | 整个操场的宽度 |
| **d2** | 内亮区宽度 (内上→内下) | 草坪 (内场) 的宽度 |

## 主要结果

跨 6 张图像重复性 CV ≤ **2.13%**；单图加噪 30 次的算法本身重复性 σ ≤ **0.11 px**，d1/d2 稳定到 **0.02 px (≈ 1/50 像素)** 量级。
单图端到端平均 **68.8 ms** (2592×1944)，纯 1D 测量主段 **< 15 ms**。

更详细的数据表格、算法原理、误差分析见 [报告](报告.md)。

---

## 目录结构

```
final_design/
├── README.md                  ← 本文件
├── 报告.md                    ← 设计报告 (按模板五节)
├── CMakeLists.txt
├── include/
│   ├── types.h                共用几何 / 结果数据结构
│   ├── locator.h              粗定位空腔
│   ├── measurement.h          1D 亚像素边缘 (Halcon measure_pos)
│   ├── geometric_fit.h        圆 / 直线拟合 + RANSAC
│   ├── pipeline.h             端到端测量流水线
│   └── visualization.h        工程式标注绘制
├── src/                       (与 include/ 对应的 6 个 .cpp)
├── imgs/                      输入图像 1.bmp ~ 6.bmp
└── results/                   自动生成
    ├── 1_result.png  ...      逐图测量可视化
    ├── measurements.csv       6 张图逐图测量值 + 计时
    └── noise_repeatability.csv  噪声重复性 30 次试验
```

---

## 环境要求 (Ubuntu 22.04)

* `g++` ≥ 11
* `cmake` ≥ 3.10
* OpenCV 4.x (Ubuntu 22.04 默认 apt 仓库即 4.5.4，足够)

一键安装：

```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev
```

## 编译

```bash
cd final_design
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

成功后可执行文件位于 `build/slot_measure`。

## 运行

把待测图像放进 `imgs/` (BMP / PNG / JPG 皆可)，然后：

```bash
cd final_design
./build/slot_measure                # 等价于 ./build/slot_measure imgs results
```

也可显式指定输入 / 输出目录：

```bash
./build/slot_measure <input_dir> <output_dir>
```

运行结束后：

* 控制台打印逐图测量值 + 跨图重复性统计 + 噪声重复性统计；
* `results/<name>_result.png` —— 每张图的工程式可视化；
* `results/measurements.csv` —— 全部测量值 + RMS + 耗时；
* `results/noise_repeatability.csv` —— 同图加噪 30 次的测量值。

## 控制台输出示例

```
========== 单图测量结果 ==========
image          r1(px)     r2(px)     d1(px)     d2(px)      rms_L      rms_R      t(ms)
-------------------------------------------------------------------------------------
1.bmp         169.174    169.165    350.234     94.270      0.594      0.616       83.0
2.bmp         168.912    167.912    348.220     93.554      0.634      0.575      146.5
3.bmp         166.963    163.634    342.246     98.953      0.641      0.703       62.6
4.bmp         169.078    168.606    349.340     95.223      0.634      0.548      114.6
5.bmp         169.725    168.927    350.107     95.342      0.745      0.614       64.3
6.bmp         164.838    163.676    340.172     97.508      0.642      0.579       62.5

========== 重复性统计 (N=6) ==========
r1   mean=168.1150  std=1.8622  range=4.8866  CV(%)=1.108
r2   mean=166.9867  std=2.6150  range=5.5311  CV(%)=1.566
d1   mean=346.7196  std=4.3777  range=10.0618 CV(%)=1.263
d2   mean= 95.8084  std=2.0390  range=5.3989  CV(%)=2.128

========== 噪声重复性 (1.bmp + N(0,4), N=30) ==========
r1   mean=169.1100  std=0.0156  CV(%)=0.009
r2   mean=169.1696  std=0.1091  CV(%)=0.064
d1   mean=350.1944  std=0.0128  CV(%)=0.004
d2   mean= 94.3688  std=0.0101  CV(%)=0.011
```

## 可视化图例

每张 `*_result.png` 上：

* **深红色圆 + 十字** — Taubin 圆拟合的 r1 / r2 (左、右弧)
* **深绿色小点** — 用于拟合圆的径向 1D 亚像素边缘点 (RANSAC 内点)
* **深橙色小点** — 60 条副轴扫描线上检测出的 外上 / 内上 / 内下 / 外下 亚像素点
* **深蓝色双向箭头** — d1，跑道整体宽度
* **深紫色双向箭头** — d2，内亮区宽度

---

## 算法概要

```
gray → [粗定位空腔位姿] ──┬─ [径向 1D 扫描 + 圆拟合]    → r1, r2
                         └─ [副轴 1D 扫描 + 4 边识别] → d1, d2
                                                    → 可视化结果图
```

* **1D 亚像素边缘**：沿扫描线做矩形积分 → 高斯一阶导 (DoG) 卷积 → 极大值 + 抛物线 3 点拟合，理论亚像素误差 < 0.05 px。
* **圆拟合**：Taubin 代数最小二乘 + RANSAC 抗外点 (内点阈值 1.5 px)。
* **4 边识别**：对每条副轴扫描线返回的所有亚像素峰，按 “最外侧 ±1” 选外缘、按 “最靠近中心 ±1” 选内缘，对内凸台上字符标记噪声鲁棒。

完整推导和代码注释见 [报告.md](报告.md) 第 3 节。

---

## 配置项 (`include/pipeline.h::PipelineConfig`)

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `edge_sigma` | 1.5 | 1D 高斯一阶导核 σ |
| `edge_threshold` | 4.0 | 边缘响应阈值 |
| `num_arc_lines` | 41 | 每条圆弧上的径向扫描线条数 |
| `num_width_lines` | 60 | 测 d1/d2 的副轴扫描线条数 |
| `width_margin` | 0.08 | 扫描线主轴两端的安全余量比例 |
| `scan_half_width` | 30.0 | 扫描线矩形积分半宽 (抑制字符标记) |
| `ransac_circle_inlier` | 1.5 | 圆拟合 RANSAC 内点阈值 (px) |
| `ransac_circle_iter` | 200 | RANSAC 迭代次数 |

---

## 一键复现命令

```bash
sudo apt install -y build-essential cmake libopencv-dev
git clone <this-repo>  # 或解压源码
cd final_design
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/slot_measure
```

完成后查看 [`results/`](results/) 目录与控制台输出即可。
