/**
 * @file geometric_fit.cpp
 * @brief 几何拟合模块：圆拟合 + 直线拟合 + RANSAC 离群点剔除
 * 
 * 本模块实现了用于测量管线的高精度几何拟合算法：
 * 
 * 1. 圆拟合 - Taubin (1991) 代数法
 *    这是一种数值稳定的圆拟合方法，优于简单的最小二乘法。
 *    核心思想是将圆拟合问题转化为多项式求根问题。
 * 
 * 2. 直线拟合 - Total Least Squares (TLS) / SVD 方法
 *    通过奇异值分解求协方差矩阵的最大特征向量，即为直线方向。
 * 
 * 3. RANSAC 随机抽样一致性
 *    用于在存在离群点的情况下鲁棒估计模型参数。
 *    通过多次随机抽样，找出内点最多的模型。
 * 
 * @see pipeline.cpp - 使用本模块进行圆拟合（r1, r2）
 */

#include "geometric_fit.h"
#include <opencv2/core.hpp>
#include <random>
#include <cmath>

namespace slot {

/**
 * @brief 使用 Taubin 方法拟合圆
 * 
 * Taubin (1991) 圆拟合算法是一种代数拟合法，相比于简单最小二乘法
 * 具有更好的数值稳定性和几何意义。
 * 
 * 算法原理：
 *   1. 将点集中心化，简化计算
 *   2. 计算二阶和三阶矩
 *   3. 构建四次多项式 P(x) = A3*x³ + A2*x² + A1*x + A0
 *   4. 使用 Newton-Raphson 方法求最小正解
 *   5. 从 x 值计算圆心坐标和半径
 * 
 * @param[in] pts 输入的点集（至少需要 3 个点）
 * @param[out] center 拟合得到的圆心
 * @param[out] radius 拟合得到的半径
 * @param[out] rms 拟合的均方根误差（点到圆的平均距离偏差）
 * @return bool 拟合是否成功
 * 
 * @note 中心化处理：将原点移到点集的质心，提高数值稳定性
 * @note Newton-Raphson 迭代：最多 99 次或收敛（|Δx| < 1e-12）
 */
bool fitCircleTaubin(const std::vector<cv::Point2d>& pts,
                     cv::Point2d& center, double& radius, double& rms)
{
    const int n = (int)pts.size();
    if (n < 3) return false;

    // ============================================================
    // 步骤 1：中心化（将原点移到点集质心）
    // ============================================================
    double mx = 0, my = 0;
    for (auto& p : pts) { mx += p.x; my += p.y; }
    mx /= n; my /= n;

    // ============================================================
    // 步骤 2：计算中心化后的各阶矩
    // ============================================================
    // Mxx, Myy: 二阶中心矩
    // Mxy: 协方差矩
    // Mxz, Myz, Mzz: 三阶矩
    double Mxx=0, Myy=0, Mxy=0, Mxz=0, Myz=0, Mzz=0;
    for (auto& p : pts) {
        double X = p.x - mx, Y = p.y - my;
        double Z = X*X + Y*Y;
        Mxx += X*X; Myy += Y*Y; Mxy += X*Y;
        Mxz += X*Z; Myz += Y*Z; Mzz += Z*Z;
    }
    // 计算均值
    Mxx/=n; Myy/=n; Mxy/=n; Mxz/=n; Myz/=n; Mzz/=n;

    // ============================================================
    // 步骤 3：构建特征多项式系数
    // ============================================================
    double Mz = Mxx + Myy;
    double Cov_xy = Mxx*Myy - Mxy*Mxy;
    double Var_z  = Mzz - Mz*Mz;
    double A3 = 4.0 * Mz;
    double A2 = -3.0*Mz*Mz - Mzz;
    double A1 = Var_z*Mz + 4.0*Cov_xy*Mz - Mxz*Mxz - Myz*Myz;
    double A0 = Mxz*(Mxz*Myy - Myz*Mxy) + Myz*(Myz*Mxx - Mxz*Mxy) - Var_z*Cov_xy;
    double A22 = A2 + A2;
    double A33 = A3 + A3 + A3;

    // ============================================================
    // 步骤 4：Newton-Raphson 迭代求最小正解
    // ============================================================
    // P(x) = A3*x³ + A2*x² + A1*x + A0 = 0
    // dP/dx = A1 + 2*A2*x + 3*A3*x²
    double x = 0.0, y = A0;  // x: 当前估计, y: P(x)
    for (int it = 0; it < 99; ++it) {
        double Dy = A1 + x*(A22 + A33*x);  // dP/dx
        double xnew = x - y / Dy;          // Newton 迭代
        if (!std::isfinite(xnew) || std::fabs(xnew - x) < 1e-12) break;
        double ynew = A0 + xnew*(A1 + xnew*(A2 + xnew*A3));  // 计算新的 P(x)
        if (std::fabs(ynew) >= std::fabs(y)) break;  // 如果没有改善则停止
        x = xnew; y = ynew;
    }

    // ============================================================
    // 步骤 5：从 x 计算圆心坐标和半径
    // ============================================================
    double DET = x*x - x*Mz + Cov_xy;
    if (std::fabs(DET) < 1e-12) return false;
    double Xc = (Mxz*(Myy - x) - Myz*Mxy) / (2.0 * DET);
    double Yc = (Myz*(Mxx - x) - Mxz*Mxy) / (2.0 * DET);

    // 转换回原坐标系
    center.x = Xc + mx;
    center.y = Yc + my;
    radius = std::sqrt(Xc*Xc + Yc*Yc + Mz + 2.0*x);

    // ============================================================
    // 步骤 6：计算 RMS 拟合误差
    // ============================================================
    double s = 0.0;
    for (auto& p : pts) {
        double d = std::hypot(p.x - center.x, p.y - center.y) - radius;
        s += d*d;
    }
    rms = std::sqrt(s / n);
    return std::isfinite(radius) && radius > 0;
}

/**
 * @brief 使用 RANSAC 进行鲁棒圆拟合
 * 
 * RANSAC（Random Sample Consensus）是一种从包含离群点的数据中
 * 估计模型参数的迭代方法。本函数使用 RANSAC 找到最佳圆拟合。
 * 
 * 算法流程：
 *   1. 随机选择 3 个点（确定一个圆）
 *   2. 计算所有点到该圆的距离，统计内点数量
 *   3. 重复 max_iter 次，保留内点最多的模型
 *   4. 用所有内点重新拟合圆（Taubin 方法）
 * 
 * @param[in] pts 输入的点集（至少需要 3 个点）
 * @param[in] inlier_thresh 内点阈值（点到圆的距离 < 此值视为内点）
 * @param[in] max_iter 最大迭代次数
 * @param[out] center 拟合得到的圆心
 * @param[out] radius 拟合得到的半径
 * @param[out] rms 拟合的均方根误差
 * @param[out] inlier_idx 内点的索引向量
 * @return bool 拟合是否成功（至少需要 5 个内点）
 * 
 * @see fitCircleTaubin - 用于内点二次拟合
 */
bool fitCircleRANSAC(const std::vector<cv::Point2d>& pts,
                     double inlier_thresh, int max_iter,
                     cv::Point2d& center, double& radius, double& rms,
                     std::vector<int>& inlier_idx)
{
    const int n = (int)pts.size();
    if (n < 3) return false;

    // 随机数生成器（固定种子确保可重复性）
    std::mt19937 rng(20260512);
    std::uniform_int_distribution<int> dist(0, n - 1);

    int best_cnt = 0;
    std::vector<int> best_inliers;

    // ============================================================
    // RANSAC 主循环
    // ============================================================
    for (int it = 0; it < max_iter; ++it) {
        // 随机选择 3 个不同的点
        int i1 = dist(rng), i2 = dist(rng), i3 = dist(rng);
        if (i1 == i2 || i1 == i3 || i2 == i3) continue;

        // ============================================================
        // 三点确定圆（解析解）
        // ============================================================
        double ax = pts[i1].x, ay = pts[i1].y;
        double bx = pts[i2].x, by = pts[i2].y;
        double cx = pts[i3].x, cy = pts[i3].y;
        
        // 计算圆的参数
        double d = 2.0 * (ax*(by - cy) + bx*(cy - ay) + cx*(ay - by));
        if (std::fabs(d) < 1e-9) continue;  // 三点共线
        
        // 圆心坐标（使用外接圆公式的向量形式）
        double ux = ((ax*ax + ay*ay)*(by - cy) + (bx*bx + by*by)*(cy - ay) + (cx*cx + cy*cy)*(ay - by)) / d;
        double uy = ((ax*ax + ay*ay)*(cx - bx) + (bx*bx + by*by)*(ax - cx) + (cx*cx + cy*cy)*(bx - ax)) / d;
        double r = std::hypot(ax - ux, ay - uy);  // 半径
        if (!std::isfinite(r) || r <= 0) continue;

        // ============================================================
        // 统计内点
        // ============================================================
        std::vector<int> in;
        in.reserve(n);
        for (int k = 0; k < n; ++k) {
            double dr = std::fabs(std::hypot(pts[k].x - ux, pts[k].y - uy) - r);
            if (dr <= inlier_thresh) in.push_back(k);  // 距离小于阈值视为内点
        }
        if ((int)in.size() > best_cnt) {
            best_cnt = (int)in.size();
            best_inliers = std::move(in);
        }
    }

    // 需要至少 5 个内点才能拟合可靠的圆
    if (best_cnt < 5) return false;

    // ============================================================
    // 使用所有内点进行最终拟合
    // ============================================================
    std::vector<cv::Point2d> in_pts;
    in_pts.reserve(best_inliers.size());
    for (int k : best_inliers) in_pts.push_back(pts[k]);

    if (!fitCircleTaubin(in_pts, center, radius, rms)) return false;
    inlier_idx = std::move(best_inliers);
    return true;
}

/**
 * @brief 使用 Total Least Squares (TLS) 拟合直线
 * 
 * TLS 方法通过奇异值分解（SVD）求解，能够同时考虑 x 和 y 方向的噪声，
 * 优于普通的最小二乘法（只考虑 y 方向的噪声）。
 * 
 * 算法原理：
 *   1. 计算点集的质心
 *   2. 构建协方差矩阵 [[sxx, sxy], [sxy, syy]]
 *   3. 求协方差矩阵的最大特征向量，即为直线方向
 *   4. 直线通过质心
 * 
 * @param[in] pts 输入的点集（至少需要 2 个点）
 * @param[out] line 拟合得到的直线（包含方向和参考点）
 * @param[out] rms 拟合的均方根误差（点到直线的垂直距离）
 * @return bool 拟合是否成功
 * 
 * @see fitLineRANSAC - 带有 RANSAC 的鲁棒版本
 */
bool fitLineLS(const std::vector<cv::Point2d>& pts, Line2D& line, double& rms)
{
    const int n = (int)pts.size();
    if (n < 2) return false;
    
    // ============================================================
    // 步骤 1：计算质心
    // ============================================================
    double mx = 0, my = 0;
    for (auto& p : pts) { mx += p.x; my += p.y; }
    mx /= n; my /= n;

    // ============================================================
    // 步骤 2：计算协方差矩阵
    // ============================================================
    double sxx = 0, syy = 0, sxy = 0;
    for (auto& p : pts) {
        double dx = p.x - mx, dy = p.y - my;
        sxx += dx*dx; syy += dy*dy; sxy += dx*dy;
    }

    // ============================================================
    // 步骤 3：求协方差矩阵的特征值和特征向量
    // ============================================================
    // 协方差矩阵的特征方程：λ² - trace(A)*λ + det(A) = 0
    // 最大特征值对应的特征向量即为直线方向
    double tr = sxx + syy;
    double det = sxx*syy - sxy*sxy;
    double tmp = std::sqrt(std::max(0.0, tr*tr/4.0 - det));
    double l1 = tr/2.0 + tmp;  // 最大特征值
    double dx, dy;
    
    // 计算最大特征值对应的特征向量
    if (std::fabs(sxy) > 1e-12) {
        dx = l1 - syy;  // (sxx - λ, sxy) 在 λ=l1 时与 (λ - syy, sxy) 成比例
        dy = sxy;
    } else if (sxx >= syy) {
        dx = 1; dy = 0;  // 水平直线
    } else {
        dx = 0; dy = 1;  // 垂直直线
    }
    
    // 归一化方向向量
    double L = std::hypot(dx, dy);
    line.dir = cv::Point2d(dx / L, dy / L);
    line.p0  = cv::Point2d(mx, my);  // 直线通过质心

    // ============================================================
    // 步骤 4：计算 RMS 误差（点到直线的垂直距离）
    // ============================================================
    double s = 0.0;
    cv::Point2d nrm(-line.dir.y, line.dir.x);  // 法向量
    for (auto& p : pts) {
        double d2 = (p.x - mx)*nrm.x + (p.y - my)*nrm.y;  // 点到直线的带符号距离
        s += d2*d2;
    }
    rms = std::sqrt(s / (double)n);
    return true;
}

/**
 * @brief 使用 RANSAC 进行鲁棒直线拟合
 * 
 * @param[in] pts 输入的点集
 * @param[in] inlier_thresh 内点阈值（点到直线的距离 < 此值视为内点）
 * @param[in] max_iter 最大迭代次数
 * @param[out] line 拟合得到的直线
 * @param[out] rms 拟合的均方根误差
 * @param[out] inlier_idx 内点的索引向量
 * @return bool 拟合是否成功（至少需要 4 个内点）
 * 
 * @see fitLineLS - 用于内点二次拟合
 */
bool fitLineRANSAC(const std::vector<cv::Point2d>& pts,
                   double inlier_thresh, int max_iter,
                   Line2D& line, double& rms, std::vector<int>& inlier_idx)
{
    const int n = (int)pts.size();
    if (n < 2) return false;
    
    // 随机数生成器
    std::mt19937 rng(20260512);
    std::uniform_int_distribution<int> dist(0, n - 1);

    int best_cnt = 0;
    std::vector<int> best_inliers;

    // ============================================================
    // RANSAC 主循环
    // ============================================================
    for (int it = 0; it < max_iter; ++it) {
        // 随机选择 2 个点确定一条直线
        int i1 = dist(rng), i2 = dist(rng);
        if (i1 == i2) continue;
        
        // 计算方向向量和法向量
        cv::Point2d d = pts[i2] - pts[i1];
        double L = std::hypot(d.x, d.y);
        if (L < 1e-6) continue;  // 两点太近
        d.x /= L; d.y /= L;
        cv::Point2d nrm(-d.y, d.x);

        // 统计内点
        std::vector<int> in;
        in.reserve(n);
        for (int k = 0; k < n; ++k) {
            double d2 = std::fabs((pts[k].x - pts[i1].x)*nrm.x + (pts[k].y - pts[i1].y)*nrm.y);
            if (d2 <= inlier_thresh) in.push_back(k);
        }
        if ((int)in.size() > best_cnt) {
            best_cnt = (int)in.size();
            best_inliers = std::move(in);
        }
    }

    // 需要至少 4 个内点
    if (best_cnt < 4) return false;
    
    // 使用内点进行最终拟合
    std::vector<cv::Point2d> in_pts;
    in_pts.reserve(best_inliers.size());
    for (int k : best_inliers) in_pts.push_back(pts[k]);
    if (!fitLineLS(in_pts, line, rms)) return false;
    inlier_idx = std::move(best_inliers);
    return true;
}

/**
 * @brief 计算点到直线的带符号距离
 * 
 * @param[in] line 直线参数
 * @param[in] p 待计算的点
 * @return double 带符号距离（正负表示点在法向量的哪一侧）
 * 
 * @note 距离 = (p - p0) · n，其中 n 是单位法向量
 */
double signedDistanceToLine(const Line2D& line, const cv::Point2d& p)
{
    cv::Point2d nrm(-line.dir.y, line.dir.x);
    return (p.x - line.p0.x) * nrm.x + (p.y - line.p0.y) * nrm.y;
}

} // namespace slot