// pipeline.cpp —— 单张图像的端到端测量管线
//
// 测量策略（与跑道形空腔几何相匹配）：
//   r1 / r2：在跑道形空腔的左右两端，从估计弧心向外沿径向铺设 num_arc_lines 条
//            1D 扫描线，每条线只会跨越外缘 (暗→亮) 一条边，得到一个亚像素点，
//            用 Taubin + RANSAC 拟合圆得到半径。
//   d1 / d2：在跑道直段铺设 num_width_lines 条垂直 (副轴方向) 1D 扫描线，
//            每条线沿轴向跨越 4 条边：外上 → 内上 → 内下 → 外下，
//            分别得到 4 个亚像素 t 值。
//              d1 = t(内上) - t(外上)   上侧通道宽度
//              d2 = t(外下) - t(内下)   下侧通道宽度
//            对所有有效扫描线求平均得到最终 d1, d2，并记录线间标准差用于重复性。

#include "pipeline.h"
#include "locator.h"
#include "measurement.h"
#include "geometric_fit.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>

namespace slot {

// 从一条扫描线返回的所有亚像素峰中，挑出语义上的 4 条边：
//   外上 (-1)：扫描线最外侧的暗→亮反向 (工件 → 灰带)
//   外下 (+1)：扫描线最外侧的暗→亮  (灰带 → 工件)
//   内上 (+1)：内亮区的上边界，对应灰带 → 亮内区 中“最靠近内亮区”的强阶跃
//              (即在 (外上, 0) 中 t 最大的 +1 强峰)
//   内下 (-1)：内亮区的下边界，对应 在 (0, 外下) 中 t 最小的 -1 强峰
//
// 语义对应：
//   d1 = 外下.t - 外上.t  →  跑道形空腔的整体宽度（操场宽度）
//   d2 = 内下.t - 内上.t  →  内亮区（草地）宽度
//
// 之所以选择“最内侧”的内边（而非最外侧），是因为部分零件上灰带与内亮区之间
// 存在亮度渐变带（中间亮度），其外侧 +1 峰对应渐变带的起点 ≠ 真实的内凸台缘；
// 选最内侧的 +1 峰才是亮内区真正的入口。
static bool pickFourEdges(std::vector<EdgePoint>& peaks,
                          EdgePoint& e_outer_top, EdgePoint& e_inner_top,
                          EdgePoint& e_inner_bot, EdgePoint& e_outer_bot)
{
    if (peaks.size() < 4) return false;
    std::sort(peaks.begin(), peaks.end(),
              [](const EdgePoint& a, const EdgePoint& b){ return a.t < b.t; });

    // 外上：左端最外侧的 -1 强峰
    int i_ot = -1;
    for (int j = 0; j < (int)peaks.size(); ++j) {
        if (peaks[j].polarity == -1) { i_ot = j; break; }
    }
    // 外下：右端最外侧的 +1 强峰
    int i_ob = -1;
    for (int j = (int)peaks.size() - 1; j >= 0; --j) {
        if (peaks[j].polarity == +1) { i_ob = j; break; }
    }
    if (i_ot < 0 || i_ob < 0 || i_ot >= i_ob) return false;

    // 内上：(外上, 0) 中 t 最大的 +1 峰 (=最靠近中心)
    int i_it = -1;
    for (int j = i_ob - 1; j > i_ot; --j) {
        if (peaks[j].t >= 0) continue;
        if (peaks[j].polarity == +1) { i_it = j; break; }
    }
    // 内下：(0, 外下) 中 t 最小的 -1 峰 (=最靠近中心)
    int i_ib = -1;
    for (int j = i_ot + 1; j < i_ob; ++j) {
        if (peaks[j].t <= 0) continue;
        if (peaks[j].polarity == -1) { i_ib = j; break; }
    }
    if (i_it < 0 || i_ib < 0 || i_it >= i_ib) return false;

    e_outer_top = peaks[i_ot];
    e_inner_top = peaks[i_it];
    e_inner_bot = peaks[i_ib];
    e_outer_bot = peaks[i_ob];
    return true;
}

bool measureOne(const cv::Mat& gray,
                const PipelineConfig& cfg,
                MeasureResult& out)
{
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();

    CV_Assert(gray.type() == CV_8UC1);

    // ===== 1) 粗定位空腔 =====
    CavityPose pose;
    if (!locateCavity(gray, pose)) {
        std::cerr << "[pipeline] locateCavity failed\n";
        return false;
    }
    out.pose = pose;

    const cv::Point2d cx = pose.axis_x;            // 主轴单位向量
    const cv::Point2d cy = pose.axis_y;            // 副轴单位向量
    const double L = pose.length;                  // 主轴半长
    const double W = pose.width;                   // 副轴半宽

    // ===== 2) r1, r2：径向扫描线 + 圆拟合 =====
    auto fitOneArc = [&](double cx_off, bool left,
                         std::vector<cv::Point2d>& arc_pts,
                         cv::Point2d& center, double& radius, double& rms) -> bool
    {
        cv::Point2d arc_center = pose.center + cx_off * cx;

        // 径向扫描方向角度：以主轴 cx 方向为 0°，左弧取 [+90°, +270°] (覆盖左半圆)，
        // 右弧取 [-90°, +90°] (覆盖右半圆)。为避开与直段的衔接处，
        // 实际取 [+95°, +265°] / [-85°, +85°]。
        const int N = cfg.num_arc_lines;
        const double a_deg_lo = left ?  95.0 : -85.0;
        const double a_deg_hi = left ? 265.0 :  85.0;

        std::vector<cv::Point2d> pts;
        pts.reserve(N);

        const double r_inner = W * 0.4;    // 扫描线靠近圆心的起点
        const double r_outer = W * 1.6;    // 远端，必须超过外缘

        for (int i = 0; i < N; ++i) {
            double a_deg = a_deg_lo + (a_deg_hi - a_deg_lo) * i / (N - 1);
            double a = a_deg * CV_PI / 180.0;
            // 在 (cx, cy) 局部坐标下方向 = (cos a, sin a)
            cv::Point2d dir = std::cos(a) * cx + std::sin(a) * cy;

            ScanLine sl;
            sl.center = arc_center + 0.5 * (r_inner + r_outer) * dir;
            sl.dir    = dir;
            sl.half_length = 0.5 * (r_outer - r_inner);
            sl.half_width  = cfg.scan_half_width;

            std::vector<EdgePoint> peaks;
            measurePos(gray, sl, cfg.edge_sigma, cfg.edge_threshold, +1, peaks);
            if (peaks.empty()) continue;
            // 取最靠近圆心 (t 最小) 的强边
            auto it = std::min_element(peaks.begin(), peaks.end(),
                [](const EdgePoint& a, const EdgePoint& b){ return a.t < b.t; });
            if (it->strength < cfg.edge_threshold) continue;
            pts.push_back(it->pos);
        }

        if (pts.size() < 5) return false;

        std::vector<int> in_idx;
        if (!fitCircleRANSAC(pts, cfg.ransac_circle_inlier, cfg.ransac_circle_iter,
                             center, radius, rms, in_idx))
            return false;

        // 仅保留内点用于可视化
        std::vector<cv::Point2d> in_pts;
        for (int k : in_idx) in_pts.push_back(pts[k]);
        arc_pts = std::move(in_pts);
        return true;
    };

    // 左/右弧心的 x_offset = ±(L - W)
    double cx_off_L = -(L - W);
    double cx_off_R = +(L - W);

    if (!fitOneArc(cx_off_L, true,  out.arc_left_pts,  out.c1, out.r1, out.rms_arc_left)) {
        std::cerr << "[pipeline] left arc fit failed\n";
        return false;
    }
    if (!fitOneArc(cx_off_R, false, out.arc_right_pts, out.c2, out.r2, out.rms_arc_right)) {
        std::cerr << "[pipeline] right arc fit failed\n";
        return false;
    }

    // ===== 3) d1, d2：副轴方向 4 边检测 =====
    {
        const int M = cfg.num_width_lines;
        // 直段范围：|x_offset| < L - W - margin
        double half_span = (L - W) * (1.0 - cfg.width_margin);
        if (half_span < W * 0.2) half_span = W * 0.2;

        out.d1_per_line.clear();
        out.d2_per_line.clear();

        // 累积内/外顶/底点用于可视化
        out.top_outer_pts.clear(); out.top_inner_pts.clear();
        out.bot_outer_pts.clear(); out.bot_inner_pts.clear();

        std::vector<double> d1s, d2s;

        for (int i = 0; i < M; ++i) {
            double off = -half_span + 2.0 * half_span * i / (M - 1);
            cv::Point2d c = pose.center + off * cx;

            ScanLine sl;
            sl.center = c;
            sl.dir    = cy;                 // 副轴方向（向下）
            sl.half_length = W * 1.6;        // 必须覆盖外上 / 外下
            sl.half_width  = cfg.scan_half_width;

            std::vector<EdgePoint> peaks;
            measurePos(gray, sl, cfg.edge_sigma, cfg.edge_threshold, 0, peaks);

            EdgePoint eOT, eIT, eIB, eOB;
            if (!pickFourEdges(peaks, eOT, eIT, eIB, eOB)) continue;

            // d1 = 整个跑道形空腔的宽度  (外上 → 外下)
            // d2 = 内亮区的宽度          (内上 → 内下)
            double d1 = eOB.t - eOT.t;
            double d2 = eIB.t - eIT.t;
            if (d1 <= 0 || d2 <= 0) continue;
            // 合理性：d2 应严格小于 d1；d1 应在 (W, 4W) 之间
            if (d2 >= d1) continue;
            if (d1 < W * 1.5 || d1 > W * 2.5) continue;

            d1s.push_back(d1);
            d2s.push_back(d2);
            out.d1_per_line.push_back(d1);
            out.d2_per_line.push_back(d2);

            out.top_outer_pts.push_back(eOT.pos);
            out.top_inner_pts.push_back(eIT.pos);
            out.bot_inner_pts.push_back(eIB.pos);
            out.bot_outer_pts.push_back(eOB.pos);
        }

        if (d1s.empty() || d2s.empty()) {
            std::cerr << "[pipeline] width measurement failed: no valid scan lines\n";
            return false;
        }

        // 用截断均值（去掉最大/最小 10%）抵抗个别异常
        auto trimmedMean = [](std::vector<double> v, double trim) {
            std::sort(v.begin(), v.end());
            int k = (int)std::floor(v.size() * trim);
            double s = 0;
            int cnt = 0;
            for (int i = k; i < (int)v.size() - k; ++i) { s += v[i]; cnt++; }
            return s / std::max(1, cnt);
        };
        out.d1 = trimmedMean(d1s, 0.1);
        out.d2 = trimmedMean(d2s, 0.1);

        // 标注箭头：d1 跨越整个外缘（用中-左位置）；d2 跨越内亮区（用中-右位置），
        // 横向错位使二者不互相遮挡。
        if (!out.top_outer_pts.empty()) {
            int N = (int)out.top_outer_pts.size();
            int idx_d1 = N / 3;       // 偏左
            int idx_d2 = 2 * N / 3;   // 偏右
            out.d1_p_top = out.top_outer_pts[idx_d1];
            out.d1_p_bot = out.bot_outer_pts[idx_d1];
            out.d2_p_top = out.top_inner_pts[idx_d2];
            out.d2_p_bot = out.bot_inner_pts[idx_d2];
        }
    }

    auto t1 = clk::now();
    out.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

} // namespace slot
