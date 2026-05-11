// pipeline.h —— 单张图像的端到端测量管线
#pragma once
#include "types.h"
#include <opencv2/core.hpp>
#include <string>

namespace slot {

struct PipelineConfig {
    // 1D measure_pos 相关
    double edge_sigma     = 1.5;   // DoG 标准差
    double edge_threshold = 4.0;   // 边缘响应阈值（灰度梯度）

    // 扫描线布局
    int    num_arc_lines   = 41;   // 圆弧两端各布多少条径向扫描线
    int    num_width_lines = 60;   // 沿主轴方向用于测 d1/d2 的扫描线数量
    double width_margin    = 0.08; // 扫描线在主轴两端各留出的比例（避开圆弧段）
    double scan_half_width = 30.0; // 扫描线垂直方向的积分半宽（像素，沿主轴方向平均，
                                   // 用以抑制内凸台上字符标记的局部干扰）

    // RANSAC
    double ransac_circle_inlier = 1.5; // 像素
    int    ransac_circle_iter   = 200;

    bool   verbose = false;
};

bool measureOne(const cv::Mat& gray,
                const PipelineConfig& cfg,
                MeasureResult& out);

} // namespace slot
