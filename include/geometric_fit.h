// geometric_fit.h —— 几何拟合（圆 / 直线 + RANSAC 剔除外点）
#pragma once
#include <opencv2/core.hpp>
#include <vector>

namespace slot {

// Taubin 代数最小二乘圆拟合（含一次重加权抑制噪声）。
//   pts    : 输入点（>= 3）
//   center : 输出圆心
//   radius : 输出半径
//   rms    : 输出 RMS 残差（像素）
// 返回 false 表示数值不稳定（如点几乎共线）。
bool fitCircleTaubin(const std::vector<cv::Point2d>& pts,
                     cv::Point2d& center, double& radius, double& rms);

// 用 RANSAC 把圆拟合中的外点剔除掉，再做一次 Taubin 拟合。
//   inlier_thresh : 内点的最大允许 |r_point - r_fit|（像素）
bool fitCircleRANSAC(const std::vector<cv::Point2d>& pts,
                     double inlier_thresh, int max_iter,
                     cv::Point2d& center, double& radius, double& rms,
                     std::vector<int>& inlier_idx);

// 简单的最小二乘水平直线 (y = a*x + b) 拟合 + RANSAC，
// 用于在长边段拟合上下外缘 / 内凸台缘，再换算各扫描线上的距离。
struct Line2D {
    cv::Point2d p0;     // 直线上一点
    cv::Point2d dir;    // 单位方向向量
};

bool fitLineLS(const std::vector<cv::Point2d>& pts, Line2D& line, double& rms);

bool fitLineRANSAC(const std::vector<cv::Point2d>& pts,
                   double inlier_thresh, int max_iter,
                   Line2D& line, double& rms, std::vector<int>& inlier_idx);

// 点到直线的有符号距离 (沿直线法线方向)
double signedDistanceToLine(const Line2D& line, const cv::Point2d& p);

} // namespace slot
