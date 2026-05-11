// types.h —— 测量管线中各模块共用的几何与结果数据结构
#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <string>

namespace slot {

// 1D 亚像素边缘点：含浮点位置（沿扫描线的弧长 t）、对应的 2D 坐标，
// 以及该点处的边缘强度 |I'|，用于按响应强弱筛选可靠点。
struct EdgePoint {
    double  t      = 0.0;        // 沿扫描线方向的亚像素位置（像素）
    cv::Point2d pos{0.0, 0.0};   // 在图像上的亚像素 (x, y)
    double  strength = 0.0;      // 边缘强度（梯度峰值绝对值）
    int     polarity = 0;        // +1: 由暗到亮，-1: 由亮到暗
};

// 一条 1D 扫描线（measure_pos 的入参）
struct ScanLine {
    cv::Point2d center;          // 中心点
    cv::Point2d dir;             // 单位方向向量（采样方向）
    double      half_length;     // 半长度（像素）
    double      half_width = 0.0;// 垂直方向积分半宽（>0 时按矩形积分得到 1D profile）
};

// 跑道形空腔的初定位结果
struct CavityPose {
    cv::Point2d center;          // 质心（图像坐标）
    cv::Point2d axis_x;          // 主轴方向（沿长边的单位向量）
    cv::Point2d axis_y;          // 副轴方向（轴向的垂直方向单位向量）
    double      length;          // 主轴方向半长（像素）
    double      width;           // 副轴方向半宽（像素）
    cv::RotatedRect rrect;       // 备份的旋转矩形（便于可视化）
};

// 一张图像的测量结果
struct MeasureResult {
    std::string image_name;
    double r1 = 0.0;             // 左侧圆弧半径（像素）
    double r2 = 0.0;             // 右侧圆弧半径（像素）
    double d1 = 0.0;             // 上侧通道平均宽度（像素）
    double d2 = 0.0;             // 下侧通道平均宽度（像素）

    cv::Point2d c1{0,0};         // 左圆心
    cv::Point2d c2{0,0};         // 右圆心

    // 用于绘图的辅助数据：所有亚像素边缘点
    std::vector<cv::Point2d> arc_left_pts;
    std::vector<cv::Point2d> arc_right_pts;
    std::vector<cv::Point2d> top_outer_pts;
    std::vector<cv::Point2d> top_inner_pts;
    std::vector<cv::Point2d> bot_inner_pts;
    std::vector<cv::Point2d> bot_outer_pts;

    // 每条扫描线返回的 d1/d2 测值，用于估计单图重复性（线间标准差）
    std::vector<double> d1_per_line;
    std::vector<double> d2_per_line;

    // 用于绘制 d1/d2 标注箭头的位置（一对端点）
    cv::Point2d d1_p_top, d1_p_bot;
    cv::Point2d d2_p_top, d2_p_bot;

    // 拟合残差（RMS，像素）
    double rms_arc_left  = 0.0;
    double rms_arc_right = 0.0;

    // 计时（毫秒）
    double elapsed_ms = 0.0;

    CavityPose pose;
};

} // namespace slot
