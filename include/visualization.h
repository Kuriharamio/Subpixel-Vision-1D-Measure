// visualization.h —— 在结果图上绘制 r1/r2/d1/d2 的工程标注
#pragma once
#include "types.h"
#include <opencv2/core.hpp>

namespace slot {

// 在原图上叠加测量结果（圆 / 直径 / 通道宽度 + 箭头 + 文字）。
// 返回一张三通道图。
cv::Mat renderResult(const cv::Mat& gray, const MeasureResult& r);

} // namespace slot
