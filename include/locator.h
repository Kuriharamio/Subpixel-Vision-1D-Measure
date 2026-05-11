// locator.h —— 粗定位模块：找到跑道形空腔并估计主轴
#pragma once
#include "types.h"
#include <opencv2/core.hpp>

namespace slot {

// 在灰度图中找到跑道形空腔的姿态（中心、主轴、长宽）。
//   gray  : 输入灰度图（CV_8UC1）
//   pose  : 输出
// 返回 true 表示成功。
//
// 实现思路：
//   1. 利用 Otsu 阈值得到工件（白色亮区）二值图；
//   2. 取工件内部的“低亮度连通域”作为空腔候选；
//   3. 用形状（长宽比、面积、长宽尺度）筛选出跑道形空腔；
//   4. 对该连通域做最小外接旋转矩形 + PCA 二次校正得到主轴。
bool locateCavity(const cv::Mat& gray, CavityPose& pose);

} // namespace slot
