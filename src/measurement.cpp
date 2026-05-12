/**
 * @file measurement.cpp
 * @brief 1D 亚像素边缘提取模块
 * 
 * 本模块实现了 Halcon 的 measure_pos 算子的等价功能，用于从图像中提取亚像素精度的边缘点。
 * 
 * 算法原理：
 *   1. 沿扫描线方向以 1 像素步长采样原图灰度，得到 1D profile I[t]。
 *   2. 为提升信噪比，采用"矩形积分"模式：沿副轴方向以 ±half_width 做双线性插值并求平均。
 *   3. 用离散高斯一阶导核 g'(t; σ) 对 I[t] 做卷积，得到导数响应 D[t]。
 *   4. 对 |D[t]| 找局部极大，保留 |D| >= threshold 的峰值。
 *   5. 在每个极大值处用抛物线拟合三点，得到亚像素峰位。
 *   6. 输出 EdgePoint：包含亚像素位置 t、边缘强度 strength 和极性 (sign(D[t]))。
 * 
 * 亚像素精度来源：
 *   在局部极大值处，用 D[t-1], D[t], D[t+1] 三点抛物线拟合，
 *   得到 t* = t + 0.5 * (D[t-1] - D[t+1]) / (D[t-1] - 2D[t] + D[t+1])
 *   理论亚像素误差 < 0.05 像素。
 * 
 * @see pipeline.cpp - 调用本模块进行边缘检测
 * @see types.h - ScanLine 和 EdgePoint 定义
 */

#include "measurement.h"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <vector>

namespace slot {

/**
 * @brief 生成离散高斯一阶导核
 * 
 * 高斯一阶导数的解析形式为：
 *   g'(t) = -(t/σ²) * (1/(σ√(2π))) * exp(-t²/(2σ²))
 * 
 * 离散化处理：
 *   - 在 [-r, r] 范围内采样，其中 r = ceil(3σ)
 *   - 归一化使得对单位幅值阶跃的响应峰值 = 1
 * 
 * @param sigma 高斯核的标准差（控制平滑程度）
 * @return std::vector<double> 离散化的高斯一阶导核（长度 = 2r+1）
 * 
 * @note 归一化条件：∑(-g'[i] * i) = 1，这使得对灰度差为 1 的理想阶跃，
 *       响应峰值正好为 1（强度单位等价于灰度差）
 */
static std::vector<double> makeGaussianDerivKernel(double sigma) {
    // 计算核半径：覆盖 ±3σ 范围
    int r = std::max(1, (int)std::ceil(3.0 * sigma));
    std::vector<double> g(2*r + 1);
    
    double s2 = sigma * sigma;
    double inv = 1.0 / (sigma * std::sqrt(2.0 * CV_PI));
    double sum_abs_t = 0.0;
    
    // 计算离散核值
    for (int i = -r; i <= r; ++i) {
        double t = (double)i;
        // 高斯一阶导数公式
        double v = -(t / s2) * inv * std::exp(-t*t / (2.0 * s2));
        g[i + r] = v;
        sum_abs_t += std::fabs(v) * std::fabs(t);
    }
    
    // 归一化：让卷积对幅值为 1 的阶跃响应峰值 == 1
    // 这确保"强度"单位等价于灰度差
    double norm = 0.0;
    for (int i = -r; i <= r; ++i) 
        norm += -g[i + r] * (double)i;  // = ∫ -t g'(t) dt = 1（连续情况）
    if (norm > 1e-12) 
        for (auto& v : g) v /= norm;
    return g;
}

/**
 * @brief 双线性插值采样
 * 
 * 在图像的亚像素位置 (x, y) 处进行采样，使用双线性插值提高精度。
 * 超出图像范围的采样返回 0。
 * 
 * @param img 输入图像（必须是 CV_8UC1）
 * @param x 采样点的 x 坐标（亚像素精度）
 * @param y 采样点的 y 坐标（亚像素精度）
 * @return double 采样得到的灰度值
 */
static inline double bilinearSample(const cv::Mat& img, double x, double y) {
    // 边界检查：超出图像范围返回 0
    if (x < 0 || y < 0 || x > img.cols - 1 || y > img.rows - 1) return 0.0;
    
    // 获取四个最近邻像素的整数坐标
    int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
    int x1 = std::min(x0 + 1, img.cols - 1);
    int y1 = std::min(y0 + 1, img.rows - 1);
    
    // 计算双线性权重
    double dx = x - x0, dy = y - y0;
    
    // 获取四个角的像素值
    double v00 = img.at<uchar>(y0, x0);
    double v01 = img.at<uchar>(y0, x1);
    double v10 = img.at<uchar>(y1, x0);
    double v11 = img.at<uchar>(y1, x1);
    
    // 双线性插值
    return (1-dx)*(1-dy)*v00 + dx*(1-dy)*v01 + (1-dx)*dy*v10 + dx*dy*v11;
}

/**
 * @brief 从一阶导数 profile 中提取亚像素边缘峰
 * 
 * 算法步骤：
 * 1. 用高斯一阶导核对 profile 做卷积，得到导数响应 D[t]
 * 2. 找到 |D[t]| 的局部极大值（边缘候选）
 * 3. 根据极性过滤（可选）
 * 4. 用抛物线拟合三点得到亚像素精度
 * 
 * @param[in] profile 输入的 1D 灰度信号
 * @param[in] sigma 高斯平滑参数（控制边缘检测的灵敏度）
 * @param[in] threshold 边缘强度阈值（用于滤除弱边缘）
 * @param[in] polarity 边缘极性过滤：
 *        0 = 接受所有极性
 *        +1 = 只接受正向边缘（暗→亮）
 *        -1 = 只接受负向边缘（亮→暗）
 * @param[out] peaks 输出的亚像素边缘点列表
 * 
 * 抛物线拟合原理：
 *   设峰值为抛物线 y = a(t-t*)² + b 的形式
 *   离散三点满足：y1 = a((t-1)-t*)² + b, y2 = a(t-t*)² + b, y3 = a((t+1)-t*)² + b
 *   解得：t* = t + 0.5 * (y1 - y3) / (y1 - 2y2 + y3)
 */
void subpixPeaksFromDerivative(const std::vector<double>& profile,
                               double sigma, double threshold, int polarity,
                               std::vector<EdgePoint>& peaks)
{
    peaks.clear();
    const int N = (int)profile.size();
    if (N < 5) return;

    // 生成高斯一阶导核并进行卷积
    auto g = makeGaussianDerivKernel(sigma);
    int r = (int)g.size() / 2;

    // 计算导数响应 D = profile * g'
    std::vector<double> D(N, 0.0);
    for (int t = 0; t < N; ++t) {
        double s = 0.0;
        for (int k = -r; k <= r; ++k) {
            // 边界处理：用最近的有效值填充
            int idx = std::clamp(t - k, 0, N - 1);
            s += profile[idx] * g[k + r];
        }
        D[t] = s;
    }

    // 极大值检测 + 抛物线亚像素精化
    for (int t = 1; t < N - 1; ++t) {
        double absD = std::fabs(D[t]);
        
        // 强度阈值过滤
        if (absD < threshold) continue;
        
        // 局部极值检测（基于绝对值）
        if (absD < std::fabs(D[t-1]) || absD < std::fabs(D[t+1])) continue;
        
        // 极性过滤
        int sgn = (D[t] > 0) ? +1 : -1;
        if (polarity != 0 && sgn != polarity) continue;

        // 抛物线亚像素精化
        double y1 = D[t-1], y2 = D[t], y3 = D[t+1];
        double denom = (y1 - 2.0*y2 + y3);
        double dt = 0.0;
        if (std::fabs(denom) > 1e-12) {
            dt = 0.5 * (y1 - y3) / denom;
            // 限制亚像素偏移在 ±1 像素范围内
            if (dt < -1.0 || dt > 1.0) dt = 0.0;
        }
        
        // 构建边缘点
        EdgePoint p;
        p.t = (double)t + dt;  // 亚像素位置（相对于扫描线起点）
        p.strength = absD;      // 边缘强度
        p.polarity = sgn;       // 边缘极性
        peaks.push_back(p);
    }
}

/**
 * @brief 在指定扫描线上执行亚像素边缘检测
 * 
 * 这是对外的主要接口，封装了从扫描线构建到边缘输出的完整流程。
 * 
 * 扫描线模型：
 *   扫描线由中心点、方向向量和长度/宽度参数定义。
 *   - center: 扫描线的中心点
 *   - dir: 主方向单位向量（扫描方向）
 *   - half_length: 半长（扫描线从 center ± half_length）
 *   - half_width: 半宽（用于矩形积分模式）
 * 
 * 矩形积分模式：
 *   为了提高信噪比，在每个采样位置，沿副轴方向（扫描线的垂线）
 *   累积 ±half_width 范围内的像素值并求平均。
 *   这相当于将一条窄矩形 ROI 投影到主轴上。
 * 
 * @param[in] gray 输入的灰度图像（必须是 CV_8UC1 类型）
 * @param[in] line 扫描线参数
 * @param[in] sigma 高斯平滑参数（控制边缘检测的灵敏度）
 * @param[in] threshold 边缘强度阈值（用于滤除弱边缘）
 * @param[in] polarity 边缘极性过滤（0=所有，+1=暗→亮，-1=亮→暗）
 * @param[out] out 输出的亚像素边缘点列表
 * 
 * @see ScanLine - 扫描线参数定义
 * @see EdgePoint - 边缘点数据结构
 */
void measurePos(const cv::Mat& gray, const ScanLine& line,
                double sigma, double threshold, int polarity,
                std::vector<EdgePoint>& out)
{
    out.clear();
    CV_Assert(gray.type() == CV_8UC1);

    // ============================================================
    // 步骤 1：设置采样参数
    // ============================================================
    
    // 采样长度：以 1px 步长，覆盖 [-half_length, +half_length]
    int N = (int)std::round(2.0 * line.half_length) + 1;
    if (N < 5) return;
    
    double t0 = -line.half_length;  // 扫描线起点的偏移

    // 副轴单位向量（与主轴正交）
    cv::Point2d n(-line.dir.y, line.dir.x);

    // 矩形积分的宽度（采样点数）
    int W = std::max(1, (int)std::round(2.0 * line.half_width) + 1);
    double w0 = -line.half_width;  // 副轴方向的起点偏移

    // ============================================================
    // 步骤 2：沿扫描线方向采样灰度 profile
    // ============================================================
    
    std::vector<double> profile(N, 0.0);
    for (int i = 0; i < N; ++i) {
        double t = t0 + (double)i;  // 当前采样点的主轴坐标
        double sum = 0.0;
        int cnt = 0;
        
        // 沿副轴方向积分（矩形积分模式）
        for (int j = 0; j < W; ++j) {
            double w = (line.half_width > 0) 
                ? (w0 + (double)j * (W > 1 ? (2.0*line.half_width/(W-1)) : 0.0)) 
                : 0.0;
            // 计算当前采样点的图像坐标
            double x = line.center.x + t * line.dir.x + w * n.x;
            double y = line.center.y + t * line.dir.y + w * n.y;
            sum += bilinearSample(gray, x, y);
            cnt++;
        }
        profile[i] = sum / std::max(1, cnt);  // 取平均值
    }

    // ============================================================
    // 步骤 3：1D 边缘提取
    // ============================================================
    
    std::vector<EdgePoint> peaks;
    subpixPeaksFromDerivative(profile, sigma, threshold, polarity, peaks);

    // ============================================================
    // 步骤 4：将 t（索引坐标）转换为沿 dir 方向的有符号位置
    // ============================================================
    
    for (auto& p : peaks) {
        // 转换索引偏移为实际坐标偏移
        double t_signed = p.t + t0;
        
        // 反算为图像坐标
        p.pos.x = line.center.x + t_signed * line.dir.x;
        p.pos.y = line.center.y + t_signed * line.dir.y;
        
        // 更新 t 为有符号的实际位置
        p.t = t_signed;
    }
    out = std::move(peaks);
}

} // namespace slot