/**
 * @file pipeline.cpp
 * @brief 单张图像的端到端测量管线实现
 * 
 * 本模块实现了跑道形空腔尺寸测量的核心算法：
 * 
 * 测量参数：
 *   - r1, r2: 左右两端半圆的半径（通过径向扫描 + 圆拟合获得）
 *   - d1: 空腔整体宽度（通过副轴方向 4 边检测获得）
 *   - d2: 内亮区宽度（通过副轴方向 4 边检测获得）
 * 
 * 算法原理：
 *   1. 粗定位阶段：使用图像处理技术找到空腔的大致位置和方向
 *   2. 弧半径测量：在跑道形空腔的左右两端，从估计弧心向外沿径向铺设多条
 *      1D 扫描线，每条线跨越外缘得到一个亚像素边缘点，然后用 Taubin 圆拟合
 *      + RANSAC 去除离群点，得到圆半径。
 *   3. 宽度测量：在跑道直段铺设多条垂直于主轴的扫描线，每条线跨越 4 条边：
 *      外上 → 内上 → 内下 → 外下，分别得到 4 个亚像素 t 值，计算宽度。
 * 
 * @see measurement.h - 亚像素边缘检测实现
 * @see geometric_fit.h - 圆拟合实现
 * @see locator.h - 空腔粗定位实现
 */

#include "pipeline.h"
#include "locator.h"
#include "measurement.h"
#include "geometric_fit.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>

namespace slot {

/**
 * @brief 从扫描线的亚像素峰中挑选出语义上的 4 条边
 * 
 * 跑道形空腔在副轴方向上形成 4 条可区分的边缘：
 * 
 * 图像结构示意图（从上到下）：
 *   背景（暗）| 外上(-1) | 灰带 | 内上(+1) | 亮内区 | 内下(-1) | 灰带 | 外下(+1) | 背景（暗）
 * 
 * 边缘极性说明：
 *   - -1 (暗→亮)：从暗区进入亮区的边缘
 *   - +1 (亮→暗)：从亮区进入暗区的边缘
 * 
 * 边选择策略：
 *   - 外上：最左侧（t 最小）的 -1 强峰
 *   - 外下：最右侧（t 最大）的 +1 强峰
 *   - 内上：在 (外上, 0) 区间内，t 最大的 +1 峰（最靠近中心）
 *   - 内下：在 (0, 外下) 区间内，t 最小的 -1 峰（最靠近中心）
 * 
 * 之所以选择"最内侧"的内边（而非最外侧），是因为部分零件上灰带与内亮区之间
 * 存在亮度渐变带（中间亮度），其外侧 +1 峰对应渐变带的起点，而非真实的内凸台缘。
 * 选最内侧的 +1 峰才是亮内区真正的入口。
 * 
 * @param[in] peaks 一条扫描线检测到的所有亚像素边缘点（按 t 值排序）
 * @param[out] e_outer_top 外上边缘点（跑道空腔上边缘的最外侧点）
 * @param[out] e_inner_top 内上边缘点（亮内区上边界）
 * @param[out] e_inner_bot 内下边缘点（亮内区下边界）
 * @param[out] e_outer_bot 外下边缘点（跑道空腔下边缘的最外侧点）
 * @return bool 是否成功找到 4 条边
 * 
 * @note 调用此函数前，peaks 应已按 t 值升序排列
 */
static bool pickFourEdges(std::vector<EdgePoint>& peaks,
                          EdgePoint& e_outer_top, EdgePoint& e_inner_top,
                          EdgePoint& e_inner_bot, EdgePoint& e_outer_bot)
{
    // 检查是否有足够的边缘点（至少需要 4 个）
    if (peaks.size() < 4) return false;
    
    // 按 t 值（亚像素位置）升序排序
    std::sort(peaks.begin(), peaks.end(),
              [](const EdgePoint& a, const EdgePoint& b){ return a.t < b.t; });

    // ============================================================
    // 步骤 1：找到外上边缘（左端最外侧的 -1 强峰）
    // ============================================================
    int i_ot = -1;
    for (int j = 0; j < (int)peaks.size(); ++j) {
        if (peaks[j].polarity == -1) { 
            i_ot = j; 
            break; 
        }
    }
    
    // ============================================================
    // 步骤 2：找到外下边缘（右端最外侧的 +1 强峰）
    // ============================================================
    int i_ob = -1;
    for (int j = (int)peaks.size() - 1; j >= 0; --j) {
        if (peaks[j].polarity == +1) { 
            i_ob = j; 
            break; 
        }
    }
    
    // 验证找到的外边缘是否有效
    if (i_ot < 0 || i_ob < 0 || i_ot >= i_ob) return false;

    // ============================================================
    // 步骤 3：找到内上边缘（外上, 0 中 t 最大的 +1 峰）
    // ============================================================
    int i_it = -1;
    for (int j = i_ob - 1; j > i_ot; --j) {
        // 只考虑 t < 0 的区域（在外上和中点之间）
        if (peaks[j].t >= 0) continue;
        if (peaks[j].polarity == +1) { 
            i_it = j; 
            break; 
        }
    }
    
    // ============================================================
    // 步骤 4：找到内下边缘（0, 外下 中 t 最小的 -1 峰）
    // ============================================================
    int i_ib = -1;
    for (int j = i_ot + 1; j < i_ob; ++j) {
        // 只考虑 t > 0 的区域（在中点和外下之间）
        if (peaks[j].t <= 0) continue;
        if (peaks[j].polarity == -1) { 
            i_ib = j; 
            break; 
        }
    }
    
    // 验证找到的内边缘是否有效
    if (i_it < 0 || i_ib < 0 || i_it >= i_ib) return false;

    // 输出结果
    e_outer_top = peaks[i_ot];
    e_inner_top = peaks[i_it];
    e_inner_bot = peaks[i_ib];
    e_outer_bot = peaks[i_ob];
    return true;
}

/**
 * @brief 对单张图像执行完整的跑道形空腔尺寸测量
 * 
 * 测量管线执行以下步骤：
 * 1. 粗定位空腔：确定空腔中心位置、主轴方向、长宽参数
 * 2. 弧半径测量：在左右两端各铺设径向扫描线，用 RANSAC 圆拟合得到 r1, r2
 * 3. 宽度测量：在直段铺设垂直扫描线，通过 4 边检测得到 d1, d2
 * 
 * @param[in] gray 输入的灰度图像（必须是 CV_8UC1 类型）
 * @param[in] cfg 测量管线配置参数
 * @param[out] out 测量结果结构体，包含所有测量值和中间数据
 * @return bool 测量是否成功
 * 
 * @see PipelineConfig - 配置参数说明
 * @see MeasureResult - 结果结构体说明
 */
bool measureOne(const cv::Mat& gray,
                const PipelineConfig& cfg,
                MeasureResult& out)
{
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();  // 开始计时

    // 确保输入是单通道灰度图像
    CV_Assert(gray.type() == CV_8UC1);

    // ============================================================
    // 阶段 1：粗定位空腔
    // ============================================================
    CavityPose pose;  // 空腔位姿：包含中心、轴向、长宽
    
    if (!locateCavity(gray, pose)) {
        std::cerr << "[pipeline] locateCavity failed\n";
        return false;
    }
    out.pose = pose;

    // 提取关键几何参数
    const cv::Point2d cx = pose.axis_x;   ///< 主轴单位向量（跑道长度方向）
    const cv::Point2d cy = pose.axis_y;   ///< 副轴单位向量（跑道宽度方向）
    const double L = pose.length;          ///< 主轴半长（弧心到中心的距离）
    const double W = pose.width;            ///< 副轴半宽（弧的半径估计）

    // ============================================================
    // 阶段 2：弧半径测量（r1, r2）
    // ============================================================
    /**
     * @brief 单个半圆的拟合 lambda 函数
     * 
     * 从估计的弧心出发，沿径向铺设多条扫描线，每条线检测外缘边缘点，
     * 最后用 RANSAC + Taubin 圆拟合得到精确的圆心和半径。
     * 
     * @param cx_off 弧心在主轴方向上的偏移量（正值向右，负值向左）
     * @param left 是否为左半圆（决定扫描角度范围）
     * @param[out] arc_pts 有效的内点（用于可视化）
     * @param[out] center 拟合得到的圆心
     * @param[out] radius 拟合得到的半径
     * @param[out] rms 拟合的均方根误差
     * @return bool 拟合是否成功
     */
    auto fitOneArc = [&](double cx_off, bool left,
                         std::vector<cv::Point2d>& arc_pts,
                         cv::Point2d& center, double& radius, double& rms) -> bool
    {
        // 计算当前弧的圆心位置
        cv::Point2d arc_center = pose.center + cx_off * cx;

        // 径向扫描方向角度设置：
        // 以主轴 cx 方向为 0°，左弧覆盖 [+95°, +265°]（避开与直段衔接处）
        // 右弧覆盖 [-85°, +85°]
        const int N = cfg.num_arc_lines;
        const double a_deg_lo = left ?  95.0 : -85.0;
        const double a_deg_hi = left ? 265.0 :  85.0;

        std::vector<cv::Point2d> pts;
        pts.reserve(N);

        // 扫描线的起止半径（相对于弧心）
        const double r_inner = W * 0.4;    // 起点：靠近圆心（避开直段衔接）
        const double r_outer = W * 1.6;    // 终点：必须超过外缘

        // 沿径向方向铺设扫描线
        for (int i = 0; i < N; ++i) {
            // 计算当前扫描线的角度
            double a_deg = a_deg_lo + (a_deg_hi - a_deg_lo) * i / (N - 1);
            double a = a_deg * CV_PI / 180.0;
            
            // 在 (cx, cy) 局部坐标系下的方向向量
            cv::Point2d dir = std::cos(a) * cx + std::sin(a) * cy;

            // 构建扫描线
            ScanLine sl;
            sl.center = arc_center + 0.5 * (r_inner + r_outer) * dir;
            sl.dir    = dir;
            sl.half_length = 0.5 * (r_outer - r_inner);
            sl.half_width  = cfg.scan_half_width;

            // 执行亚像素边缘检测（只检测正向边缘，即暗→亮）
            std::vector<EdgePoint> peaks;
            measurePos(gray, sl, cfg.edge_sigma, cfg.edge_threshold, +1, peaks);
            if (peaks.empty()) continue;
            
            // 选取最靠近圆心（t 最小）的强边缘点
            auto it = std::min_element(peaks.begin(), peaks.end(),
                [](const EdgePoint& a, const EdgePoint& b){ return a.t < b.t; });
            if (it->strength < cfg.edge_threshold) continue;
            pts.push_back(it->pos);
        }

        // 需要至少 5 个有效点才能拟合圆
        if (pts.size() < 5) return false;

        // RANSAC 圆拟合
        std::vector<int> in_idx;
        if (!fitCircleRANSAC(pts, cfg.ransac_circle_inlier, cfg.ransac_circle_iter,
                             center, radius, rms, in_idx))
            return false;

        // 仅保留内点用于后续可视化
        std::vector<cv::Point2d> in_pts;
        for (int k : in_idx) in_pts.push_back(pts[k]);
        arc_pts = std::move(in_pts);
        return true;
    };

    // 左/右弧心的 x_offset = ±(L - W)
    // L 是主轴半长，W 是副轴半宽，弧心位于 (L-W) 处
    double cx_off_L = -(L - W);  // 左弧圆心偏左
    double cx_off_R = +(L - W);  // 右弧圆心偏右

    // 拟合左半圆
    if (!fitOneArc(cx_off_L, true,  out.arc_left_pts,  out.c1, out.r1, out.rms_arc_left)) {
        std::cerr << "[pipeline] left arc fit failed\n";
        return false;
    }
    
    // 拟合右半圆
    if (!fitOneArc(cx_off_R, false, out.arc_right_pts, out.c2, out.r2, out.rms_arc_right)) {
        std::cerr << "[pipeline] right arc fit failed\n";
        return false;
    }

    // ============================================================
    // 阶段 3：宽度测量（d1, d2）
    // ============================================================
    {
        const int M = cfg.num_width_lines;  // 扫描线数量
        
        // 直段有效范围：排除与圆弧衔接的部分
        double half_span = (L - W) * (1.0 - cfg.width_margin);
        if (half_span < W * 0.2) half_span = W * 0.2;  // 确保最小范围

        // 清空用于可视化的点容器
        out.d1_per_line.clear();
        out.d2_per_line.clear();
        out.top_outer_pts.clear(); 
        out.top_inner_pts.clear();
        out.bot_outer_pts.clear(); 
        out.bot_inner_pts.clear();

        std::vector<double> d1s, d2s;  // 累积所有扫描线的测量结果

        // 沿主轴方向铺设 M 条扫描线
        for (int i = 0; i < M; ++i) {
            // 计算当前扫描线的位置（从 -half_span 到 +half_span）
            double off = -half_span + 2.0 * half_span * i / (M - 1);
            cv::Point2d c = pose.center + off * cx;

            // 构建副轴方向的扫描线
            ScanLine sl;
            sl.center = c;
            sl.dir    = cy;                 // 副轴方向（垂直于主轴）
            sl.half_length = W * 1.6;       // 必须覆盖外上 / 外下
            sl.half_width  = cfg.scan_half_width;

            // 执行边缘检测（检测所有极性的边缘）
            std::vector<EdgePoint> peaks;
            measurePos(gray, sl, cfg.edge_sigma, cfg.edge_threshold, 0, peaks);

            // 从检测到的边缘中挑选 4 条关键边
            EdgePoint eOT, eIT, eIB, eOB;
            if (!pickFourEdges(peaks, eOT, eIT, eIB, eOB)) continue;

            // 计算 d1 和 d2
            // d1 = 整个跑道形空腔的宽度（外上 → 外下）
            // d2 = 内亮区的宽度（内上 → 内下）
            double d1 = eOB.t - eOT.t;
            double d2 = eIB.t - eIT.t;
            
            // 有效性检查：宽度必须为正
            if (d1 <= 0 || d2 <= 0) continue;
            
            // 合理性检查：
            // 1. d2 应严格小于 d1（内亮区在空腔内部）
            // 2. d1 应在 (1.5W, 2.5W) 范围内（空腔宽度与副轴半宽的关系）
            if (d2 >= d1) continue;
            if (d1 < W * 1.5 || d1 > W * 2.5) continue;

            // 累积有效测量结果
            d1s.push_back(d1);
            d2s.push_back(d2);
            out.d1_per_line.push_back(d1);
            out.d2_per_line.push_back(d2);

            // 保存边缘点用于后续可视化
            out.top_outer_pts.push_back(eOT.pos);
            out.top_inner_pts.push_back(eIT.pos);
            out.bot_inner_pts.push_back(eIB.pos);
            out.bot_outer_pts.push_back(eOB.pos);
        }

        // 检查是否获得足够的有效扫描线
        if (d1s.empty() || d2s.empty()) {
            std::cerr << "[pipeline] width measurement failed: no valid scan lines\n";
            return false;
        }

        // ============================================================
        // 使用截断均值（Trimmed Mean）聚合结果
        // 去掉最大和最小各 10% 的值，以抵抗个别异常扫描线的影响
        // ============================================================
        auto trimmedMean = [](std::vector<double> v, double trim) {
            std::sort(v.begin(), v.end());
            int k = (int)std::floor(v.size() * trim);
            double s = 0;
            int cnt = 0;
            for (int i = k; i < (int)v.size() - k; ++i) { 
                s += v[i]; 
                cnt++; 
            }
            return s / std::max(1, cnt);
        };
        out.d1 = trimmedMean(d1s, 0.1);
        out.d2 = trimmedMean(d2s, 0.1);

        // ============================================================
        // 设置标注箭头位置（用于可视化）
        // d1 跨越整个外缘（用中-左位置），d2 跨越内亮区（用中-右位置）
        // 横向错位使二者不互相遮挡
        // ============================================================
        if (!out.top_outer_pts.empty()) {
            int N = (int)out.top_outer_pts.size();
            int idx_d1 = N / 3;       // 偏左的扫描线
            int idx_d2 = 2 * N / 3;   // 偏右的扫描线
            out.d1_p_top = out.top_outer_pts[idx_d1];
            out.d1_p_bot = out.bot_outer_pts[idx_d1];
            out.d2_p_top = out.top_inner_pts[idx_d2];
            out.d2_p_bot = out.bot_inner_pts[idx_d2];
        }
    }

    // 计算处理耗时
    auto t1 = clk::now();
    out.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

} // namespace slot