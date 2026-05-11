// geometric_fit.cpp —— 圆 / 直线 拟合 + RANSAC
//
// 圆拟合采用 Taubin (1991) 代数法，对几何噪声估计良好且数值稳定。
// 直线采用 TLS (奇异值分解) 形式。

#include "geometric_fit.h"
#include <opencv2/core.hpp>
#include <random>
#include <cmath>

namespace slot {

bool fitCircleTaubin(const std::vector<cv::Point2d>& pts,
                     cv::Point2d& center, double& radius, double& rms)
{
    const int n = (int)pts.size();
    if (n < 3) return false;

    // 中心化
    double mx = 0, my = 0;
    for (auto& p : pts) { mx += p.x; my += p.y; }
    mx /= n; my /= n;

    // 矩
    double Mxx=0, Myy=0, Mxy=0, Mxz=0, Myz=0, Mzz=0;
    for (auto& p : pts) {
        double X = p.x - mx, Y = p.y - my;
        double Z = X*X + Y*Y;
        Mxx += X*X; Myy += Y*Y; Mxy += X*Y;
        Mxz += X*Z; Myz += Y*Z; Mzz += Z*Z;
    }
    Mxx/=n; Myy/=n; Mxy/=n; Mxz/=n; Myz/=n; Mzz/=n;

    double Mz = Mxx + Myy;
    double Cov_xy = Mxx*Myy - Mxy*Mxy;
    double Var_z  = Mzz - Mz*Mz;
    double A3 = 4.0 * Mz;
    double A2 = -3.0*Mz*Mz - Mzz;
    double A1 = Var_z*Mz + 4.0*Cov_xy*Mz - Mxz*Mxz - Myz*Myz;
    double A0 = Mxz*(Mxz*Myy - Myz*Mxy) + Myz*(Myz*Mxx - Mxz*Mxy) - Var_z*Cov_xy;
    double A22 = A2 + A2;
    double A33 = A3 + A3 + A3;

    // Newton 求 P(x)=A3 x^3 + A2 x^2 + A1 x + A0 的最小正根
    double x = 0.0, y = A0;
    for (int it = 0; it < 99; ++it) {
        double Dy = A1 + x*(A22 + A33*x);
        double xnew = x - y / Dy;
        if (!std::isfinite(xnew) || std::fabs(xnew - x) < 1e-12) break;
        double ynew = A0 + xnew*(A1 + xnew*(A2 + xnew*A3));
        if (std::fabs(ynew) >= std::fabs(y)) break;
        x = xnew; y = ynew;
    }

    double DET = x*x - x*Mz + Cov_xy;
    if (std::fabs(DET) < 1e-12) return false;
    double Xc = (Mxz*(Myy - x) - Myz*Mxy) / (2.0 * DET);
    double Yc = (Myz*(Mxx - x) - Mxz*Mxy) / (2.0 * DET);

    center.x = Xc + mx;
    center.y = Yc + my;
    radius = std::sqrt(Xc*Xc + Yc*Yc + Mz + 2.0*x);

    // RMS
    double s = 0.0;
    for (auto& p : pts) {
        double d = std::hypot(p.x - center.x, p.y - center.y) - radius;
        s += d*d;
    }
    rms = std::sqrt(s / n);
    return std::isfinite(radius) && radius > 0;
}

bool fitCircleRANSAC(const std::vector<cv::Point2d>& pts,
                     double inlier_thresh, int max_iter,
                     cv::Point2d& center, double& radius, double& rms,
                     std::vector<int>& inlier_idx)
{
    const int n = (int)pts.size();
    if (n < 3) return false;

    std::mt19937 rng(20260512);
    std::uniform_int_distribution<int> dist(0, n - 1);

    int best_cnt = 0;
    std::vector<int> best_inliers;

    for (int it = 0; it < max_iter; ++it) {
        int i1 = dist(rng), i2 = dist(rng), i3 = dist(rng);
        if (i1 == i2 || i1 == i3 || i2 == i3) continue;

        // 三点确定圆（解析）
        double ax = pts[i1].x, ay = pts[i1].y;
        double bx = pts[i2].x, by = pts[i2].y;
        double cx = pts[i3].x, cy = pts[i3].y;
        double d = 2.0 * (ax*(by - cy) + bx*(cy - ay) + cx*(ay - by));
        if (std::fabs(d) < 1e-9) continue;
        double ux = ((ax*ax + ay*ay)*(by - cy) + (bx*bx + by*by)*(cy - ay) + (cx*cx + cy*cy)*(ay - by)) / d;
        double uy = ((ax*ax + ay*ay)*(cx - bx) + (bx*bx + by*by)*(ax - cx) + (cx*cx + cy*cy)*(bx - ax)) / d;
        double r = std::hypot(ax - ux, ay - uy);
        if (!std::isfinite(r) || r <= 0) continue;

        // 计内点
        std::vector<int> in;
        in.reserve(n);
        for (int k = 0; k < n; ++k) {
            double dr = std::fabs(std::hypot(pts[k].x - ux, pts[k].y - uy) - r);
            if (dr <= inlier_thresh) in.push_back(k);
        }
        if ((int)in.size() > best_cnt) {
            best_cnt = (int)in.size();
            best_inliers = std::move(in);
        }
    }

    if (best_cnt < 5) return false;

    std::vector<cv::Point2d> in_pts;
    in_pts.reserve(best_inliers.size());
    for (int k : best_inliers) in_pts.push_back(pts[k]);

    if (!fitCircleTaubin(in_pts, center, radius, rms)) return false;
    inlier_idx = std::move(best_inliers);
    return true;
}

bool fitLineLS(const std::vector<cv::Point2d>& pts, Line2D& line, double& rms)
{
    const int n = (int)pts.size();
    if (n < 2) return false;
    double mx = 0, my = 0;
    for (auto& p : pts) { mx += p.x; my += p.y; }
    mx /= n; my /= n;

    double sxx = 0, syy = 0, sxy = 0;
    for (auto& p : pts) {
        double dx = p.x - mx, dy = p.y - my;
        sxx += dx*dx; syy += dy*dy; sxy += dx*dy;
    }
    // 协方差矩阵 [[sxx, sxy],[sxy, syy]] 的最大特征向量即直线方向
    double tr = sxx + syy;
    double det = sxx*syy - sxy*sxy;
    double tmp = std::sqrt(std::max(0.0, tr*tr/4.0 - det));
    double l1 = tr/2.0 + tmp; // 最大特征值
    double dx, dy;
    if (std::fabs(sxy) > 1e-12) {
        dx = l1 - syy;
        dy = sxy;
    } else if (sxx >= syy) {
        dx = 1; dy = 0;
    } else {
        dx = 0; dy = 1;
    }
    double L = std::hypot(dx, dy);
    line.dir = cv::Point2d(dx / L, dy / L);
    line.p0  = cv::Point2d(mx, my);

    // RMS 法向距离
    double s = 0.0;
    cv::Point2d nrm(-line.dir.y, line.dir.x);
    for (auto& p : pts) {
        double d2 = (p.x - mx)*nrm.x + (p.y - my)*nrm.y;
        s += d2*d2;
    }
    rms = std::sqrt(s / (double)n);
    return true;
}

bool fitLineRANSAC(const std::vector<cv::Point2d>& pts,
                   double inlier_thresh, int max_iter,
                   Line2D& line, double& rms, std::vector<int>& inlier_idx)
{
    const int n = (int)pts.size();
    if (n < 2) return false;
    std::mt19937 rng(20260512);
    std::uniform_int_distribution<int> dist(0, n - 1);

    int best_cnt = 0;
    std::vector<int> best_inliers;

    for (int it = 0; it < max_iter; ++it) {
        int i1 = dist(rng), i2 = dist(rng);
        if (i1 == i2) continue;
        cv::Point2d d = pts[i2] - pts[i1];
        double L = std::hypot(d.x, d.y);
        if (L < 1e-6) continue;
        d.x /= L; d.y /= L;
        cv::Point2d nrm(-d.y, d.x);

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

    if (best_cnt < 4) return false;
    std::vector<cv::Point2d> in_pts;
    in_pts.reserve(best_inliers.size());
    for (int k : best_inliers) in_pts.push_back(pts[k]);
    if (!fitLineLS(in_pts, line, rms)) return false;
    inlier_idx = std::move(best_inliers);
    return true;
}

double signedDistanceToLine(const Line2D& line, const cv::Point2d& p)
{
    cv::Point2d nrm(-line.dir.y, line.dir.x);
    return (p.x - line.p0.x) * nrm.x + (p.y - line.p0.y) * nrm.y;
}

} // namespace slot
