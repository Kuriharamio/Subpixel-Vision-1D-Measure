// locator.cpp —— 粗定位跑道形空腔（位置 + 主轴）
//
// 算法：
//   (1) 对灰度图做高斯平滑 → Otsu 二值化 → 形态学开闭运算，得到工件外轮廓掩码 M_part；
//   (2) 在工件内部寻找低亮度连通域：先对 ROI 取 (gray < part_mean - k*sigma) 得到候选掩码，
//       再用 connectedComponentsWithStats 取最大且形状似 "跑道形" 的连通域；
//   (3) 对该连通域做 PCA：质心 + 协方差矩阵 → 主轴方向、长短半轴；
//   (4) 用 minAreaRect 作长宽估计的另一来源做交叉校验。
//
// 该模块只负责粗定位（提供主轴和扫描线参考）。后续 1D 亚像素测量不依赖二值边界。

#include "locator.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <iostream>

namespace slot {

bool locateCavity(const cv::Mat& gray, CavityPose& pose) {
    CV_Assert(gray.type() == CV_8UC1);

    // ---- (1) 工件分割 ----
    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(0,0), 1.5);

    cv::Mat part_bin;
    cv::threshold(blur, part_bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // 形态学清理
    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9,9));
    cv::morphologyEx(part_bin, part_bin, cv::MORPH_CLOSE, k);
    cv::morphologyEx(part_bin, part_bin, cv::MORPH_OPEN,  k);

    // 取最大白色连通域作为工件
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(part_bin, labels, stats, centroids, 8);
    int best = -1, best_area = 0;
    for (int i = 1; i < n; ++i) {
        int a = stats.at<int>(i, cv::CC_STAT_AREA);
        if (a > best_area) { best_area = a; best = i; }
    }
    if (best < 0) return false;

    cv::Mat part_mask = (labels == best);

    // ---- (2) 空腔候选：工件内部的暗区 ----
    // 先得到工件内的中位亮度，再用 (gray < median - delta) 找暗区
    cv::Mat part_pix;
    blur.copyTo(part_pix, part_mask);
    std::vector<uchar> vals;
    vals.reserve(cv::countNonZero(part_mask));
    for (int y = 0; y < blur.rows; ++y) {
        const uchar* m = part_mask.ptr<uchar>(y);
        const uchar* p = blur.ptr<uchar>(y);
        for (int x = 0; x < blur.cols; ++x) if (m[x]) vals.push_back(p[x]);
    }
    if (vals.empty()) return false;
    std::nth_element(vals.begin(), vals.begin() + vals.size()/2, vals.end());
    int med = vals[vals.size()/2];

    // 空腔像素：在工件内部 + 灰度比中位低一定阈值
    cv::Mat cavity_mask = (blur < (med - 25)) & part_mask;

    // 形态学清理，去掉小颗粒
    cv::Mat k2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7));
    cv::morphologyEx(cavity_mask, cavity_mask, cv::MORPH_OPEN, k2);
    cv::morphologyEx(cavity_mask, cavity_mask, cv::MORPH_CLOSE, k2);

    // 选最大暗区
    n = cv::connectedComponentsWithStats(cavity_mask, labels, stats, centroids, 8);
    best = -1; best_area = 0;
    for (int i = 1; i < n; ++i) {
        int a = stats.at<int>(i, cv::CC_STAT_AREA);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        // 长宽比要 > 2（跑道形）
        double ratio = std::max(w, h) / std::max(1.0, (double)std::min(w, h));
        if (a > best_area && ratio > 1.8) { best_area = a; best = i; }
    }
    if (best < 0) return false;

    cv::Mat cav = (labels == best);

    // ---- (3) PCA 求主轴 ----
    std::vector<cv::Point2f> pts;
    pts.reserve(best_area);
    for (int y = 0; y < cav.rows; ++y) {
        const uchar* m = cav.ptr<uchar>(y);
        for (int x = 0; x < cav.cols; ++x) if (m[x]) pts.emplace_back((float)x, (float)y);
    }
    if (pts.size() < 50) return false;

    cv::Mat data(static_cast<int>(pts.size()), 2, CV_64F);
    for (size_t i = 0; i < pts.size(); ++i) {
        data.at<double>((int)i, 0) = pts[i].x;
        data.at<double>((int)i, 1) = pts[i].y;
    }
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 2);

    cv::Point2d c(pca.mean.at<double>(0,0), pca.mean.at<double>(0,1));
    cv::Point2d e0(pca.eigenvectors.at<double>(0,0), pca.eigenvectors.at<double>(0,1));
    cv::Point2d e1(pca.eigenvectors.at<double>(1,0), pca.eigenvectors.at<double>(1,1));

    // 让主轴方向指向 +x（保证可重复性）
    if (e0.x < 0) e0 = -e0;
    // 右手坐标系
    e1 = cv::Point2d(-e0.y, e0.x);

    // 将所有点投影到主/副轴，取 99% 分位作为半长 / 半宽估计
    std::vector<double> proj0, proj1;
    proj0.reserve(pts.size());
    proj1.reserve(pts.size());
    for (auto& p : pts) {
        double dx = p.x - c.x, dy = p.y - c.y;
        proj0.push_back(dx * e0.x + dy * e0.y);
        proj1.push_back(dx * e1.x + dy * e1.y);
    }
    auto quantileAbs = [](std::vector<double>& v, double q) {
        for (auto& x : v) x = std::fabs(x);
        size_t k = std::min(v.size() - 1, (size_t)(v.size() * q));
        std::nth_element(v.begin(), v.begin() + k, v.end());
        return v[k];
    };
    double half_len = quantileAbs(proj0, 0.995);
    double half_wid = quantileAbs(proj1, 0.995);

    pose.center  = c;
    pose.axis_x  = e0;
    pose.axis_y  = e1;
    pose.length  = half_len;
    pose.width   = half_wid;

    pose.rrect   = cv::minAreaRect(pts);

    return true;
}

} // namespace slot
