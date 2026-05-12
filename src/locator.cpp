/**
 * @file locator.cpp
 * @brief 跑道形空腔粗定位模块
 * 
 * 本模块负责在图像中快速定位跑道形空腔的位置和方向，为后续的精确亚像素测量提供：
 *   - 空腔中心位置 (center)
 *   - 主轴方向 (axis_x) 和副轴方向 (axis_y)
 *   - 主轴半长 (length) 和副轴半宽 (width)
 * 
 * 算法流程：
 *   1. 工件分割：使用 Otsu 自动阈值分割 + 形态学处理，提取工件外轮廓
 *   2. 空腔检测：在工件内部寻找暗区连通域，筛选出跑道形状的候选区域
 *   3. PCA 分析：对空腔区域像素做主成分分析，得到质心、主轴方向和长宽估计
 *   4. 交叉校验：使用 minAreaRect 作为辅助参考
 * 
 * @note 本模块只负责粗定位，不进行精确的亚像素测量
 * @see pipeline.cpp - 使用本模块结果的测量管线
 */

#include "locator.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <iostream>

namespace slot {

/**
 * @brief 在图像中粗定位跑道形空腔
 * 
 * 通过图像处理技术找到空腔的位置、主轴方向和长宽参数。
 * 这些参数将用于后续的精确亚像素测量。
 * 
 * @param[in] gray 输入的灰度图像（必须是 CV_8UC1 类型）
 * @param[out] pose 空腔位姿结构体，包含中心、轴向和长宽
 * @return bool 是否成功定位
 * 
 * 算法详解：
 * 
 * 阶段1 - 工件分割：
 *   - 对图像进行高斯平滑（σ=1.5）以减少噪声
 *   - 使用 Otsu 自动阈值方法进行二值化
 *   - 通过形态学开闭运算去除小的空洞和噪点
 *   - 提取最大连通域作为工件掩码
 * 
 * 阶段2 - 空腔候选检测：
 *   - 计算工件内部像素的中位灰度值
 *   - 在工件区域内，找灰度低于 (中位 - 25) 的暗区
 *   - 通过形态学清理去除小的噪点
 *   - 提取最大且长宽比 > 1.8 的连通域作为空腔候选
 * 
 * 阶段3 - PCA主轴分析：
 *   - 收集空腔掩码内所有像素的坐标
 *   - 计算协方差矩阵并进行特征分解
 *   - 主特征向量作为主轴方向（跑道长度方向）
 *   - 次特征向量作为副轴方向（跑道宽度方向）
 *   - 计算 99.5% 分位数作为半长和半宽的估计
 * 
 * @see CavityPose - 输出结构体定义
 */
bool locateCavity(const cv::Mat& gray, CavityPose& pose) {
    CV_Assert(gray.type() == CV_8UC1);

    // ============================================================
    // 阶段1：工件分割
    // ============================================================
    
    // 高斯平滑：使用 5x5 核，σ=1.5，减少图像噪声
    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(0,0), 1.5);

    // Otsu 自动阈值二值化：自动找到最佳分割阈值
    cv::Mat part_bin;
    cv::threshold(blur, part_bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // 形态学处理：
    // - 闭运算：填充小的空洞
    // - 开运算：去除小的噪点
    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9,9));
    cv::morphologyEx(part_bin, part_bin, cv::MORPH_CLOSE, k);
    cv::morphologyEx(part_bin, part_bin, cv::MORPH_OPEN,  k);

    // 提取最大白色连通域作为工件
    // connectedComponentsWithStats 会标记所有连通域并返回统计信息
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(part_bin, labels, stats, centroids, 8);
    
    int best = -1, best_area = 0;
    for (int i = 1; i < n; ++i) {  // 从1开始，跳过背景（label=0）
        int a = stats.at<int>(i, cv::CC_STAT_AREA);
        if (a > best_area) { best_area = a; best = i; }
    }
    if (best < 0) return false;

    // 创建工件掩码（二值图像，1=工件，0=背景）
    cv::Mat part_mask = (labels == best);

    // ============================================================
    // 阶段2：空腔候选检测（在工件内部的暗区）
    // ============================================================
    
    // 计算工件内像素的灰度值
    cv::Mat part_pix;
    blur.copyTo(part_pix, part_mask);
    std::vector<uchar> vals;
    vals.reserve(cv::countNonZero(part_mask));
    
    // 收集所有工件像素的灰度值
    for (int y = 0; y < blur.rows; ++y) {
        const uchar* m = part_mask.ptr<uchar>(y);
        const uchar* p = blur.ptr<uchar>(y);
        for (int x = 0; x < blur.cols; ++x) 
            if (m[x]) vals.push_back(p[x]);
    }
    if (vals.empty()) return false;
    
    // 计算中位灰度值（使用 nth_element 高效选择）
    std::nth_element(vals.begin(), vals.begin() + vals.size()/2, vals.end());
    int med = vals[vals.size()/2];

    // 空腔像素定义：在工件内部 + 灰度比中位低 25 个单位
    // 这利用了跑道形空腔是暗区的特点
    cv::Mat cavity_mask = (blur < (med - 25)) & part_mask;

    // 形态学清理
    cv::Mat k2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7));
    cv::morphologyEx(cavity_mask, cavity_mask, cv::MORPH_OPEN, k2);
    cv::morphologyEx(cavity_mask, cavity_mask, cv::MORPH_CLOSE, k2);

    // 选择最大且形状接近跑道形（长宽比 > 1.8）的连通域
    n = cv::connectedComponentsWithStats(cavity_mask, labels, stats, centroids, 8);
    best = -1; best_area = 0;
    for (int i = 1; i < n; ++i) {
        int a = stats.at<int>(i, cv::CC_STAT_AREA);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        // 长宽比要 > 1.8（跑道形通常是细长的）
        double ratio = std::max(w, h) / std::max(1.0, (double)std::min(w, h));
        if (a > best_area && ratio > 1.8) { best_area = a; best = i; }
    }
    if (best < 0) return false;

    // 创建空腔掩码
    cv::Mat cav = (labels == best);

    // ============================================================
    // 阶段3：PCA 主轴分析
    // ============================================================
    
    // 收集空腔掩码内所有像素的坐标
    std::vector<cv::Point2f> pts;
    pts.reserve(best_area);
    for (int y = 0; y < cav.rows; ++y) {
        const uchar* m = cav.ptr<uchar>(y);
        for (int x = 0; x < cav.cols; ++x) 
            if (m[x]) pts.emplace_back((float)x, (float)y);
    }
    if (pts.size() < 50) return false;  // 需要足够的点才能做 PCA

    // 构建数据矩阵（每行一个点，列1=x，列2=y）
    cv::Mat data(static_cast<int>(pts.size()), 2, CV_64F);
    for (size_t i = 0; i < pts.size(); ++i) {
        data.at<double>((int)i, 0) = pts[i].x;
        data.at<double>((int)i, 1) = pts[i].y;
    }
    
    // 执行 PCA：提取主成分
    // DATA_AS_ROW 表示每行是一个样本
    // 2 表示保留全部两个主成分
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 2);

    // 提取分析结果
    // mean: 所有点的质心
    // eigenvectors: 特征向量（行0=主轴方向，行1=副轴方向）
    cv::Point2d c(pca.mean.at<double>(0,0), pca.mean.at<double>(0,1));
    cv::Point2d e0(pca.eigenvectors.at<double>(0,0), pca.eigenvectors.at<double>(0,1));
    cv::Point2d e1(pca.eigenvectors.at<double>(1,0), pca.eigenvectors.at<double>(1,1));

    // 确保主轴方向指向 +x 方向（保证结果的可重复性）
    if (e0.x < 0) e0 = -e0;
    // 构建右手坐标系（副轴 = 主轴逆时针旋转 90°）
    e1 = cv::Point2d(-e0.y, e0.x);

    // 计算每个点到主轴和副轴的投影距离
    std::vector<double> proj0, proj1;
    proj0.reserve(pts.size());
    proj1.reserve(pts.size());
    for (auto& p : pts) {
        double dx = p.x - c.x, dy = p.y - c.y;
        proj0.push_back(dx * e0.x + dy * e0.y);  // 主轴投影
        proj1.push_back(dx * e1.x + dy * e1.y);  // 副轴投影
    }
    
    /**
     * @brief 计算向量绝对值的 q 分位数
     * 
     * 用于估计半长和半宽。使用 99.5% 分位数可以排除异常点的影响。
     * 
     * @param v 输入向量（将被修改）
     * @param q 分位数（0 < q < 1）
     * @return double 绝对值的 q 分位数
     */
    auto quantileAbs = [](std::vector<double>& v, double q) {
        // 转换为绝对值
        for (auto& x : v) x = std::fabs(x);
        // 找到第 q 分位数的位置
        size_t k = std::min(v.size() - 1, (size_t)(v.size() * q));
        std::nth_element(v.begin(), v.begin() + k, v.end());
        return v[k];
    };
    
    // 计算半长和半宽（99.5% 分位数）
    double half_len = quantileAbs(proj0, 0.995);
    double half_wid = quantileAbs(proj1, 0.995);

    // 填充输出结构体
    pose.center  = c;      // 空腔质心
    pose.axis_x  = e0;     // 主轴单位向量
    pose.axis_y  = e1;     // 副轴单位向量（与主轴正交）
    pose.length  = half_len;  // 主轴半长
    pose.width   = half_wid;   // 副轴半宽

    // 额外计算最小面积外接矩形（作为交叉校验）
    pose.rrect   = cv::minAreaRect(pts);

    return true;
}

} // namespace slot