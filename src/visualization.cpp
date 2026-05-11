// visualization.cpp —— 把测量结果叠加到图像上：圆弧 + 半径箭头 + d1/d2 标注

#include "visualization.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <sstream>
#include <iomanip>

namespace slot {

static void drawDimArrow(cv::Mat& img, cv::Point2d a, cv::Point2d b,
                         const cv::Scalar& color, int thick = 2)
{
    // 双向箭头
    cv::arrowedLine(img, a, b, color, thick, cv::LINE_AA, 0, 0.02);
    cv::arrowedLine(img, b, a, color, thick, cv::LINE_AA, 0, 0.02);
}

static std::string fmt(double v, int prec = 2) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(prec) << v;
    return oss.str();
}

cv::Mat renderResult(const cv::Mat& gray, const MeasureResult& r)
{
    cv::Mat bgr;
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);

    // 深色配色（深红、深蓝、深绿、深紫、深橙）：在亮工件上可读性更好
    const cv::Scalar DRED   (  0,   0, 180);   // 深红，圆 + r 标注
    const cv::Scalar DBLUE  (160,  60,   0);   // 深蓝，d1 箭头 (外缘)
    const cv::Scalar DGREEN (  0, 110,   0);   // 深绿，圆弧亚像素点
    const cv::Scalar DPURPLE( 90,   0, 110);   // 深紫色，d2 箭头 (内边)
    const cv::Scalar DORANGE(  0,  80, 160);   // 深橙，内 / 外边沿亚像素点

    // 圆弧 / 通道亚像素点（深色小点）
    auto drawPts = [&](const std::vector<cv::Point2d>& pts, const cv::Scalar& c) {
        for (auto& p : pts) cv::circle(bgr, p, 2, c, cv::FILLED, cv::LINE_AA);
    };
    drawPts(r.arc_left_pts,  DGREEN);
    drawPts(r.arc_right_pts, DGREEN);
    drawPts(r.top_outer_pts, DORANGE);
    drawPts(r.top_inner_pts, DORANGE);
    drawPts(r.bot_outer_pts, DORANGE);
    drawPts(r.bot_inner_pts, DORANGE);

    // 拟合的左 / 右圆
    cv::circle(bgr, r.c1, (int)std::round(r.r1), DRED, 2, cv::LINE_AA);
    cv::circle(bgr, r.c2, (int)std::round(r.r2), DRED, 2, cv::LINE_AA);
    cv::drawMarker(bgr, r.c1, DRED, cv::MARKER_CROSS, 14, 2);
    cv::drawMarker(bgr, r.c2, DRED, cv::MARKER_CROSS, 14, 2);

    // r1 / r2 箭头（圆心 → 弧上的某个亚像素点）
    auto pickRayPt = [&](const std::vector<cv::Point2d>& pts, cv::Point2d c, bool left) {
        if (pts.empty()) return c;
        cv::Point2d best = pts.front();
        double best_dx = left ? -1e9 : +1e9;
        for (auto& p : pts) {
            double dx = p.x - c.x;
            if ((left && dx < best_dx) || (!left && dx > best_dx)) { best_dx = dx; best = p; }
        }
        return best;
    };
    if (!r.arc_left_pts.empty()) {
        cv::Point2d p_arc = pickRayPt(r.arc_left_pts, r.c1, true);
        cv::arrowedLine(bgr, r.c1, p_arc, DRED, 2, cv::LINE_AA, 0, 0.04);
        cv::putText(bgr, "r1=" + fmt(r.r1) + " px",
                    cv::Point((int)((r.c1.x + p_arc.x)/2 - 70), (int)(r.c1.y - 15)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, DRED, 2, cv::LINE_AA);
    }
    if (!r.arc_right_pts.empty()) {
        cv::Point2d p_arc = pickRayPt(r.arc_right_pts, r.c2, false);
        cv::arrowedLine(bgr, r.c2, p_arc, DRED, 2, cv::LINE_AA, 0, 0.04);
        cv::putText(bgr, "r2=" + fmt(r.r2) + " px",
                    cv::Point((int)((r.c2.x + p_arc.x)/2 - 50), (int)(r.c2.y + 45)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, DRED, 2, cv::LINE_AA);
    }

    // d1 / d2 箭头：不同深色 + 水平错位防互相遮挡
    drawDimArrow(bgr, r.d1_p_top, r.d1_p_bot, DBLUE,   2);
    drawDimArrow(bgr, r.d2_p_top, r.d2_p_bot, DPURPLE, 2);

    // d1 标注：放在跑道空腔的上方灰色通道内，文字位于箭头右侧不与之重叠
    cv::Point2d d1_label_anchor(
        r.d1_p_top.x + 16,                  // 箭头右侧
        r.d1_p_top.y + 40                   // 外上之下 ~40 px，仍在灰色通道里
    );
    cv::putText(bgr, "d1=" + fmt(r.d1) + " px",
                cv::Point((int)d1_label_anchor.x, (int)d1_label_anchor.y),
                cv::FONT_HERSHEY_SIMPLEX, 0.85, DBLUE, 2, cv::LINE_AA);

    // d2 标注：放在内亮区中心，文字位于 d2 箭头左侧 (与右侧 d1 标注呈对称构图)
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

    // 顶部信息条
    std::string info = r.image_name +
        "    r1=" + fmt(r.r1) + "  r2=" + fmt(r.r2) +
        "    d1=" + fmt(r.d1) + "  d2=" + fmt(r.d2) +
        "    t=" + fmt(r.elapsed_ms, 1) + " ms";
    cv::rectangle(bgr, cv::Point(0, 0), cv::Point(bgr.cols, 36),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(bgr, info, cv::Point(12, 26),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    return bgr;
}

} // namespace slot
