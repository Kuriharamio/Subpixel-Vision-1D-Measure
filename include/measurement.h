// measurement.h —— 1D 亚像素边缘检测 (Halcon-style measure_pos)
#pragma once
#include "types.h"
#include <opencv2/core.hpp>
#include <vector>

namespace slot {

// 沿一条 1D 扫描线采样并返回所有亚像素边缘点（按 t 升序）。
//   gray       : 输入灰度图
//   line       : 扫描线参数（中心、方向、半长、矩形积分半宽）
//   sigma      : 高斯一阶导（DoG）滤波核标准差
//   threshold  : 边缘强度阈值（|I'| 必须 >= 此值）
//   polarity   : 0 表示双向都接受；+1 表示仅暗→亮；-1 表示仅亮→暗
//   out        : 输出点（按 t 排序）
void measurePos(const cv::Mat& gray, const ScanLine& line,
                double sigma, double threshold, int polarity,
                std::vector<EdgePoint>& out);

// 在一条 1D profile（已等距采样到 1 像素步长）上提取亚像素峰位置。
// 对外暴露便于单元化复用与单测。
void subpixPeaksFromDerivative(const std::vector<double>& profile,
                               double sigma, double threshold, int polarity,
                               std::vector<EdgePoint>& peaks);

} // namespace slot
