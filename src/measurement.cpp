// measurement.cpp —— 1D 亚像素边缘提取（Halcon measure_pos 的等价实现）
//
// 算法关键点：
//   (1) 沿扫描线方向以 1 像素步长采样原图灰度，得到 1D profile I[t]。
//       为提升信噪比，采用 "矩形积分" 模式：在每个采样位置，沿副轴方向
//       (扫描线垂线) 以 ±half_width 做双线性插值并求平均，等价于把矩形 ROI
//       投影到主轴。
//   (2) 用离散高斯一阶导核 g'(t; σ) 对 I[t] 做卷积，得到 D[t] = (I * g')[t]。
//   (3) 对 |D[t]| 找局部极大；保留 |D| >= threshold 的峰值。
//   (4) 在每个极大值处用抛物线拟合三点 (t-1, D[t-1]), (t, D[t]), (t+1, D[t+1])，
//       得到亚像素峰位 t* = t + 0.5 * (D[t-1] - D[t+1]) / (D[t-1] - 2 D[t] + D[t+1]).
//       这是 Halcon 等系统所用的标准做法，理论亚像素误差 < 0.05 px。
//   (5) 输出 EdgePoint：t* 反算为图像坐标 + 强度 + 极性 (sign(D[t*]))。

#include "measurement.h"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <vector>

namespace slot {

// 离散高斯一阶导核 g'(t) = -(t/σ²) * (1/(σ √(2π))) * exp(-t²/(2σ²))
// 在 [-r, r] 内采样，r = ceil(3σ)。归一化使得 sum(t * g'(t)) = -1，
// 即对一个理想阶跃 H(t) (上升) 的响应在 t=0 处等于 +1（强度即等价灰度差）。
static std::vector<double> makeGaussianDerivKernel(double sigma) {
    int r = std::max(1, (int)std::ceil(3.0 * sigma));
    std::vector<double> g(2*r + 1);
    double s2 = sigma * sigma;
    double inv = 1.0 / (sigma * std::sqrt(2.0 * CV_PI));
    double sum_abs_t = 0.0;
    for (int i = -r; i <= r; ++i) {
        double t = (double)i;
        double v = -(t / s2) * inv * std::exp(-t*t / (2.0 * s2));
        g[i + r] = v;
        sum_abs_t += std::fabs(v) * std::fabs(t);
    }
    // 归一化：让卷积对幅值为 1 的阶跃响应峰值 == 1
    double norm = 0.0;
    for (int i = -r; i <= r; ++i) norm += -g[i + r] * (double)i; // = ∫ -t g'(t) dt = 1（连续）
    if (norm > 1e-12) for (auto& v : g) v /= norm;
    return g;
}

static inline double bilinearSample(const cv::Mat& img, double x, double y) {
    if (x < 0 || y < 0 || x > img.cols - 1 || y > img.rows - 1) return 0.0;
    int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
    int x1 = std::min(x0 + 1, img.cols - 1);
    int y1 = std::min(y0 + 1, img.rows - 1);
    double dx = x - x0, dy = y - y0;
    double v00 = img.at<uchar>(y0, x0);
    double v01 = img.at<uchar>(y0, x1);
    double v10 = img.at<uchar>(y1, x0);
    double v11 = img.at<uchar>(y1, x1);
    return (1-dx)*(1-dy)*v00 + dx*(1-dy)*v01 + (1-dx)*dy*v10 + dx*dy*v11;
}

void subpixPeaksFromDerivative(const std::vector<double>& profile,
                               double sigma, double threshold, int polarity,
                               std::vector<EdgePoint>& peaks)
{
    peaks.clear();
    const int N = (int)profile.size();
    if (N < 5) return;

    // 卷积 D = profile * g'
    auto g = makeGaussianDerivKernel(sigma);
    int r = (int)g.size() / 2;

    std::vector<double> D(N, 0.0);
    for (int t = 0; t < N; ++t) {
        double s = 0.0;
        for (int k = -r; k <= r; ++k) {
            int idx = std::clamp(t - k, 0, N - 1); // 边界用复制
            s += profile[idx] * g[k + r];
        }
        D[t] = s;
    }

    // 极大值检测 + 抛物线亚像素精化
    for (int t = 1; t < N - 1; ++t) {
        double absD = std::fabs(D[t]);
        if (absD < threshold) continue;
        // 局部极值 (在绝对值意义上)
        if (absD < std::fabs(D[t-1]) || absD < std::fabs(D[t+1])) continue;
        // 极性过滤
        int sgn = (D[t] > 0) ? +1 : -1;
        if (polarity != 0 && sgn != polarity) continue;

        // 抛物线亚像素：对带符号的 D 拟合，保证三点形成同号峰
        double y1 = D[t-1], y2 = D[t], y3 = D[t+1];
        double denom = (y1 - 2.0*y2 + y3);
        double dt = 0.0;
        if (std::fabs(denom) > 1e-12) {
            dt = 0.5 * (y1 - y3) / denom;
            if (dt < -1.0 || dt > 1.0) dt = 0.0;
        }
        EdgePoint p;
        p.t = (double)t + dt;
        p.strength = absD;
        p.polarity = sgn;
        peaks.push_back(p);
    }
}

void measurePos(const cv::Mat& gray, const ScanLine& line,
                double sigma, double threshold, int polarity,
                std::vector<EdgePoint>& out)
{
    out.clear();
    CV_Assert(gray.type() == CV_8UC1);

    // 采样长度：以 1px 步长，覆盖 [-half_length, +half_length]
    int N = (int)std::round(2.0 * line.half_length) + 1;
    if (N < 5) return;
    double t0 = -line.half_length;

    // 副轴单位向量（与主轴正交，已由调用方保证）
    cv::Point2d n(-line.dir.y, line.dir.x);

    // 矩形积分（沿副轴累加并平均）
    int W = std::max(1, (int)std::round(2.0 * line.half_width) + 1);
    double w0 = -line.half_width;

    std::vector<double> profile(N, 0.0);
    for (int i = 0; i < N; ++i) {
        double t = t0 + (double)i;
        double sum = 0.0;
        int cnt = 0;
        for (int j = 0; j < W; ++j) {
            double w = (line.half_width > 0) ? (w0 + (double)j * (W > 1 ? (2.0*line.half_width/(W-1)) : 0.0)) : 0.0;
            double x = line.center.x + t * line.dir.x + w * n.x;
            double y = line.center.y + t * line.dir.y + w * n.y;
            sum += bilinearSample(gray, x, y);
            cnt++;
        }
        profile[i] = sum / std::max(1, cnt);
    }

    // 1D 边缘提取
    std::vector<EdgePoint> peaks;
    subpixPeaksFromDerivative(profile, sigma, threshold, polarity, peaks);

    // 把 t (索引坐标) 还原成沿 dir 方向的有符号位置 (t = t_index + t0)
    for (auto& p : peaks) {
        double t_signed = p.t + t0;
        p.pos.x = line.center.x + t_signed * line.dir.x;
        p.pos.y = line.center.y + t_signed * line.dir.y;
        p.t = t_signed;
    }
    out = std::move(peaks);
}

} // namespace slot
