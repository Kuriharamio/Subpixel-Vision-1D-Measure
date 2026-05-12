/**
 * @file visualization.cpp
 * @brief 测量结果可视化模块
 * 
 * 本模块负责将测量结果叠加到原始图像上，生成直观易懂的可视化输出。
 * 
 * 可视化内容包括：
 *   1. 拟合圆：显示左/右半圆的拟合圆心和半径
 *   2. 亚像素边缘点：显示所有检测到的亚像素边缘点
 *   3. 尺寸标注：r1, r2, d1, d2 的双向箭头和数值标签
 *   4. 信息条：顶部显示图像名称、测量值和处理耗时
 * 
 * 颜色方案：
 *   深色配色（在亮工件上可读性更好）：
 *   - 深红：圆弧和半径标注
 *   - 深蓝：d1 标注（外缘宽度）
 *   - 深绿：圆弧亚像素点
 *   - 深紫：d2 标注（内亮区宽度）
 *   - 深橙：内/外边缘亚像素点
 * 
 * @see main.cpp - 调用本模块生成可视化结果
 * @see MeasureResult - 测量结果数据结构
 */

#include "visualization.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <sstream>
#include <iomanip>

namespace slot {

/**
 * @brief 绘制双向尺寸箭头
 * 
 * 在两点之间绘制双向箭头，用于标注尺寸。
 * 箭头长度由起点和终点之间的距离决定。
 * 
 * @param[out] img 输出图像
 * @param[in] a 箭头起点
 * @param[in] b 箭头终点
 * @param[in] color 箭头颜色（BGR 格式）
 * @param[in] thick 线条粗细
 */
static void drawDimArrow(cv::Mat& img, cv::Point2d a, cv::Point2d b,
                         const cv::Scalar& color, int thick = 2)
{
    // 绘制双向箭头（两端都有箭头）
    cv::arrowedLine(img, a, b, color, thick, cv::LINE_AA, 0, 0.02);
    cv::arrowedLine(img, b, a, color, thick, cv::LINE_AA, 0, 0.02);
}

/**
 * @brief 格式化浮点数为字符串
 * 
 * @param v 要格式化的数值
 * @param prec 小数位数
 * @return std::string 格式化后的字符串
 */
static std::string fmt(double v, int prec = 2) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(prec) << v;
    return oss.str();
}

/**
 * @brief 渲染测量结果到图像上
 * 
 * 将 MeasureResult 中的所有测量数据叠加到输入图像上，
 * 生成一个带有标注的可视化图像。
 * 
 * @param[in] gray 输入的灰度图像
 * @param[in] r 测量结果结构体
 * @return cv::Mat BGR 格式的可视化图像
 * 
 * 渲染顺序（从后到前）：
 *   1. 亚像素边缘点（小圆点）
 *   2. 拟合圆和圆心标记
 *   3. 半径标注和箭头
 *   4. 宽度标注和箭头
 *   5. 顶部信息条
 */
cv::Mat renderResult(const cv::Mat& gray, const MeasureResult& r)
{
    // ============================================================
    // 步骤 1：灰度图转 BGR 图（OpenCV 绘制需要 BGR 格式）
    // ============================================================
    cv::Mat bgr;
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);

    // ============================================================
    // 步骤 2：定义配色方案（深色在亮工件上可读性更好）
    // ============================================================
    const cv::Scalar DRED   (  0,   0, 180);   // 深红：圆 + r1/r2 标注
    const cv::Scalar DBLUE  (160,  60,   0);   // 深蓝：d1 箭头（外缘宽度）
    const cv::Scalar DGREEN (  0, 110,   0);   // 深绿：圆弧亚像素点
    const cv::Scalar DPURPLE( 90,   0, 110);   // 深紫：d2 箭头（内亮区宽度）
    const cv::Scalar DORANGE(  0,  80, 160);   // 深橙：内/外边缘亚像素点

    // ============================================================
    // 步骤 3：绘制亚像素边缘点（小圆点）
    // ============================================================
    auto drawPts = [&](const std::vector<cv::Point2d>& pts, const cv::Scalar& c) {
        for (auto& p : pts) 
            cv::circle(bgr, p, 2, c, cv::FILLED, cv::LINE_AA);
    };
    
    // 圆弧边缘点（深绿色）
    drawPts(r.arc_left_pts,  DGREEN);
    drawPts(r.arc_right_pts, DGREEN);
    
    // 直段边缘点（深橙色）
    drawPts(r.top_outer_pts, DORANGE);  // 外上边缘
    drawPts(r.top_inner_pts, DORANGE);  // 内上边缘
    drawPts(r.bot_outer_pts, DORANGE);  // 外下边缘
    drawPts(r.bot_inner_pts, DORANGE);  // 内下边缘

    // ============================================================
    // 步骤 4：绘制拟合圆和圆心标记
    // ============================================================
    // 左圆（深红色）
    cv::circle(bgr, r.c1, (int)std::round(r.r1), DRED, 2, cv::LINE_AA);
    cv::drawMarker(bgr, r.c1, DRED, cv::MARKER_CROSS, 14, 2);
    
    // 右圆（深红色）
    cv::circle(bgr, r.c2, (int)std::round(r.r2), DRED, 2, cv::LINE_AA);
    cv::drawMarker(bgr, r.c2, DRED, cv::MARKER_CROSS, 14, 2);

    // ============================================================
    // 步骤 5：绘制 r1/r2 半径箭头和标注
    // ============================================================
    /**
     * @brief 选择最左/最右的边缘点用于绘制半径箭头
     * 
     * @param pts 边缘点集
     * @param c 圆心
     * @param left true=最左点，false=最右点
     * @return cv::Point2d 选中的边缘点
     */
    auto pickRayPt = [&](const std::vector<cv::Point2d>& pts, cv::Point2d c, bool left) {
        if (pts.empty()) return c;
        cv::Point2d best = pts.front();
        double best_dx = left ? -1e9 : +1e9;
        for (auto& p : pts) {
            double dx = p.x - c.x;
            if ((left && dx < best_dx) || (!left && dx > best_dx)) { 
                best_dx = dx; 
                best = p; 
            }
        }
        return best;
    };
    
    // r1 标注（左圆）
    if (!r.arc_left_pts.empty()) {
        cv::Point2d p_arc = pickRayPt(r.arc_left_pts, r.c1, true);
        cv::arrowedLine(bgr, r.c1, p_arc, DRED, 2, cv::LINE_AA, 0, 0.04);
        cv::putText(bgr, "r1=" + fmt(r.r1) + " px",
                    cv::Point((int)((r.c1.x + p_arc.x)/2 - 70), (int)(r.c1.y - 15)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, DRED, 2, cv::LINE_AA);
    }
    
    // r2 标注（右圆）
    if (!r.arc_right_pts.empty()) {
        cv::Point2d p_arc = pickRayPt(r.arc_right_pts, r.c2, false);
        cv::arrowedLine(bgr, r.c2, p_arc, DRED, 2, cv::LINE_AA, 0, 0.04);
        cv::putText(bgr, "r2=" + fmt(r.r2) + " px",
                    cv::Point((int)((r.c2.x + p_arc.x)/2 - 50), (int)(r.c2.y + 45)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, DRED, 2, cv::LINE_AA);
    }

    // ============================================================
    // 步骤 6：绘制 d1/d2 宽度箭头
    // ============================================================
    // d1：跨越整个外缘（深蓝色）
    drawDimArrow(bgr, r.d1_p_top, r.d1_p_bot, DBLUE, 2);
    
    // d2：跨越内亮区（深紫色）
    drawDimArrow(bgr, r.d2_p_top, r.d2_p_bot, DPURPLE, 2);

    // ============================================================
    // 步骤 7：绘制 d1 标注文字
    // ============================================================
    // 位置：在跑道空腔的上方灰色通道内，箭头右侧
    cv::Point2d d1_label_anchor(
        r.d1_p_top.x + 16,                  // 箭头右侧
        r.d1_p_top.y + 40                   // 外上之下 ~40 px，仍在灰色通道里
    );
    cv::putText(bgr, "d1=" + fmt(r.d1) + " px",
                cv::Point((int)d1_label_anchor.x, (int)d1_label_anchor.y),
                cv::FONT_HERSHEY_SIMPLEX, 0.85, DBLUE, 2, cv::LINE_AA);

    // ============================================================
    // 步骤 8：绘制 d2 标注文字
    // ============================================================
    // 位置：在内亮区中心，d2 箭头左侧（与 d1 标注呈对称构图）
    cv::Point2d cavity_center = 0.5 * (r.d2_p_top + r.d2_p_bot);
    std::string d2_text = "d2=" + fmt(r.d2) + " px";
    int baseline = 0;
    cv::Size d2_sz = cv::getTextSize(d2_text, cv::FONT_HERSHEY_SIMPLEX, 0.85, 2, &baseline);
    cv::Point2d d2_label_anchor(
        cavity_center.x - 16 - d2_sz.width,   // 紫线左侧，按文字宽度对齐
        cavity_center.y + 8
    );
    cv::putText(bgr, d2_text,
                cv::Point((int)d2_label_anchor.x, (int)d2_label_anchor.y),
                cv::FONT_HERSHEY_SIMPLEX, 0.85, DPURPLE, 2, cv::LINE_AA);

    // ============================================================
    // 步骤 9：绘制顶部信息条
    // ============================================================
    // 信息条内容：图像名 + 测量值 + 处理耗时
    std::string info = r.image_name +
        "    r1=" + fmt(r.r1) + "  r2=" + fmt(r.r2) +
        "    d1=" + fmt(r.d1) + "  d2=" + fmt(r.d2) +
        "    t=" + fmt(r.elapsed_ms, 1) + " ms";
    
    // 绘制黑色背景矩形
    cv::rectangle(bgr, cv::Point(0, 0), cv::Point(bgr.cols, 36),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    
    // 绘制白色文字
    cv::putText(bgr, info, cv::Point(12, 26),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    return bgr;
}

} // namespace slot