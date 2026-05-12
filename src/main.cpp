/**
 * @mainpage 跑道形空腔尺寸测量程序
 * 
 * 本程序用于测量跑道形（矩形两端带半圆）空腔的尺寸参数：
 *   - r1, r2: 左右两端半圆的半径
 *   - d1: 空腔整体宽度
 *   - d2: 内亮区宽度
 * 
 * 处理流程：
 *   1. 遍历指定目录中的所有图像
 *   2. 对每张图像执行亚像素边缘检测和几何拟合
 *   3. 输出测量结果到 CSV 文件和控制台
 *   4. 进行重复性分析（无噪声和有噪声）
 * 
 * @author 自动生成
 * @date 2026
 */

#include "pipeline.h"
#include "visualization.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <numeric>
#include <iomanip>
#include <algorithm>
#include <random>

namespace fs = std::filesystem;

/**
 * @brief 聚合统计数据结构
 * 
 * 存储一组数值的统计特征：
 *   - mean: 算术平均值
 *   - stddev: 标准差（样本标准差）
 *   - range: 极差（最大值 - 最小值）
 *   - min_v: 最小值
 *   - max_v: 最大值
 */
struct AggStats {
    double mean = 0;     ///< 算术平均值
    double stddev = 0;  ///< 样本标准差
    double range = 0;   ///< 极差（最大值 - 最小值）
    double min_v = 0;   ///< 最小值
    double max_v = 0;   ///< 最大值
};

/**
 * @brief 计算一组数值的聚合统计量
 * 
 * 使用贝塞尔校正计算样本标准差（除以 n-1 而非 n），
 * 以获得对总体标准差的无偏估计。
 * 
 * @param v 输入的数值向量
 * @return AggStats 包含均值、标准差、极差、最小/最大值的结构体
 */
static AggStats aggregate(const std::vector<double>& v) {
    AggStats s;
    if (v.empty()) return s;
    
    // 计算总和
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    s.mean = sum / v.size();
    
    // 计算平方和（用于标准差计算）
    double ss = 0;
    for (double x : v) ss += (x - s.mean) * (x - s.mean);
    
    // 样本标准差（使用 n-1 作为除数）
    s.stddev = std::sqrt(ss / std::max(1.0, (double)v.size() - 1.0));
    
    // 找出最小值和最大值
    auto [mn, mx] = std::minmax_element(v.begin(), v.end());
    s.min_v = *mn; 
    s.max_v = *mx; 
    s.range = s.max_v - s.min_v;
    return s;
}

/**
 * @brief 程序入口点
 * 
 * 主函数负责以下任务：
 * 1. 解析命令行参数（可选：图像目录和输出目录）
 * 2. 遍历图像目录，处理所有支持的图像格式
 * 3. 对每张图像执行测量并保存结果
 * 4. 输出测量结果的统计摘要
 * 5. 进行噪声重复性测试以评估算法稳定性
 * 
 * @param argc 命令行参数个数
 * @param argv 命令行参数数组
 *        argv[1]: 图像目录路径（默认："imgs"）
 *        argv[2]: 输出目录路径（默认："results"）
 * @return 程序退出码（0=成功，1=无图像文件）
 */
int main(int argc, char** argv) {
    // ============================================================
    // 第1步：解析命令行参数
    // ============================================================
    std::string img_dir = "imgs";   ///< 输入图像目录（默认：程序目录下的 imgs）
    std::string out_dir = "results"; ///< 输出结果目录（默认：程序目录下的 results）
    
    // 支持通过命令行覆盖默认路径
    if (argc > 1) img_dir = argv[1];
    if (argc > 2) out_dir = argv[2];

    // 确保输出目录存在（如果不存在则创建）
    fs::create_directories(out_dir);

    // ============================================================
    // 第2步：扫描图像目录，收集所有支持的图像文件
    // ============================================================
    std::vector<fs::path> files;
    
    // 遍历目录中的所有文件和子目录
    for (auto& e : fs::directory_iterator(img_dir)) {
        // 跳过非普通文件（如子目录、链接等）
        if (!e.is_regular_file()) continue;
        
        auto p = e.path();
        std::string ext = p.extension().string();
        
        // 将扩展名转换为小写（确保大小写不敏感）
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        
        // 只保留 BMP、PNG、JPG 格式的图像文件
        if (ext == ".bmp" || ext == ".png" || ext == ".jpg") {
            files.push_back(p);
        }
    }
    
    // 按文件名排序，确保处理顺序一致
    std::sort(files.begin(), files.end());

    // 检查是否找到了图像文件
    if (files.empty()) {
        std::cerr << "No images found in " << img_dir << "\n";
        return 1;  // 没有图像文件则退出
    }

    // ============================================================
    // 第3步：初始化测量管线和输出文件
    // ============================================================
    slot::PipelineConfig cfg;  ///< 测量管线配置参数

    // 创建 CSV 文件用于存储测量结果
    std::ofstream csv(out_dir + "/measurements.csv");
    csv << "image,r1_px,r2_px,d1_px,d2_px,d1_std_px,d2_std_px,"
           "rms_arc_left_px,rms_arc_right_px,elapsed_ms\n";

    // 用于累积所有图像的测量结果（用于后续统计分析）
    std::vector<double> r1s, r2s, d1s, d2s, times;

    // ============================================================
    // 第4步：打印表头，准备输出单图测量结果
    // ============================================================
    std::cout << "\n========== 单图测量结果 ==========\n";
    std::cout << std::left
              << std::setw(10) << "image"
              << std::right
              << std::setw(11) << "r1(px)"    // 左圆半径
              << std::setw(11) << "r2(px)"    // 右圆半径
              << std::setw(11) << "d1(px)"    // 整体宽度
              << std::setw(11) << "d2(px)"    // 内亮区宽度
              << std::setw(11) << "rms_L"     // 左圆拟合 RMS
              << std::setw(11) << "rms_R"     // 右圆拟合 RMS
              << std::setw(11) << "t(ms)"     // 处理耗时（毫秒）
              << "\n";
    std::cout << std::string(85, '-') << "\n";

    // ============================================================
    // 第5步：逐张处理图像
    // ============================================================
    for (auto& fp : files) {
        // 读取图像为灰度图
        cv::Mat gray = cv::imread(fp.string(), cv::IMREAD_GRAYSCALE);
        if (gray.empty()) {
            std::cerr << "Failed to read " << fp << "\n";
            continue;  // 跳过无法读取的图像
        }

        // 初始化测量结果结构
        slot::MeasureResult res;
        res.image_name = fp.filename().string();
        
        // 执行亚像素测量
        if (!slot::measureOne(gray, cfg, res)) {
            std::cerr << "[main] measure failed: " << fp << "\n";
            continue;  // 跳过测量失败的图像
        }

        // 生成测量结果可视化图像
        cv::Mat vis = slot::renderResult(gray, res);
        
        // 保存可视化结果到输出目录
        std::string out_img = out_dir + "/" + fp.stem().string() + "_result.png";
        cv::imwrite(out_img, vis);

        // 计算 d1/d2 的线间标准差（衡量同一张图中扫描线之间的一致性）
        AggStats sd1 = aggregate(res.d1_per_line);
        AggStats sd2 = aggregate(res.d2_per_line);

        // 将测量结果写入 CSV 文件
        csv << res.image_name << ","
            << res.r1 << "," << res.r2 << ","
            << res.d1 << "," << res.d2 << ","
            << sd1.stddev << "," << sd2.stddev << ","
            << res.rms_arc_left << "," << res.rms_arc_right << ","
            << res.elapsed_ms << "\n";

        // 累积测量结果用于统计
        r1s.push_back(res.r1);
        r2s.push_back(res.r2);
        d1s.push_back(res.d1);
        d2s.push_back(res.d2);
        times.push_back(res.elapsed_ms);

        // 在控制台打印单图结果
        std::cout << std::left  << std::setw(10) << res.image_name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(11) << res.r1
                  << std::setw(11) << res.r2
                  << std::setw(11) << res.d1
                  << std::setw(11) << res.d2
                  << std::setw(11) << res.rms_arc_left
                  << std::setw(11) << res.rms_arc_right
                  << std::setw(11) << std::setprecision(1) << res.elapsed_ms
                  << "\n";
    }
    csv.close();  // 关闭 CSV 文件

    // ============================================================
    // 第6步：输出重复性统计（无噪声条件下）
    // ============================================================
    
    /**
     * @brief 打印统计信息 lambda 函数
     * 
     * 打印指定参数的均值、标准差、极差、最值和变异系数(CV)
     * CV = (标准差/均值)×100%，用于衡量测量的相对稳定性
     * 
     * @param name 参数名称
     * @param v 参数的多次测量值向量
     */
    auto print = [&](const char* name, const std::vector<double>& v) {
        AggStats s = aggregate(v);
        std::cout << std::left  << std::setw(10) << name
                  << std::right << std::fixed << std::setprecision(4)
                  << " mean=" << std::setw(10) << s.mean
                  << "  std=" << std::setw(8)  << s.stddev
                  << "  range=" << std::setw(8) << s.range
                  << "  min=" << std::setw(10) << s.min_v
                  << "  max=" << std::setw(10) << s.max_v
                  << "  CV(%)=" << std::setprecision(3) 
                  << (s.mean != 0 ? 100.0 * s.stddev / std::fabs(s.mean) : 0.0)
                  << "\n";
    };

    std::cout << "\n========== 重复性统计 (N=" << r1s.size() << ") ==========\n";
    print("r1", r1s);     // 左圆半径的重复性
    print("r2", r2s);     // 右圆半径的重复性
    print("d1", d1s);     // 整体宽度的重复性
    print("d2", d2s);     // 内亮区宽度的重复性
    print("time_ms", times);  // 处理耗时的重复性

    // ============================================================
    // 第7步：噪声重复性测试
    // ============================================================
    /** 
     * 对第一张图像反复添加高斯噪声并测量，评估算法对噪声的鲁棒性
     * 这对于评估算法在实际工业环境中的稳定性非常重要
     */
    if (!files.empty()) {
        // 读取第一张图像用于噪声测试
        cv::Mat g0 = cv::imread(files.front().string(), cv::IMREAD_GRAYSCALE);
        
        const int N_TRIALS = 30;      ///< 噪声测试重复次数
        const double NOISE_STD = 4.0; ///< 灰度噪声标准差（像素值范围0-255，约1.5%动态范围）
        
        // 用于存储噪声测试结果
        std::vector<double> r1n, r2n, d1n, d2n, tn;

        // 设置随机数生成器（使用固定种子确保可重复性）
        std::mt19937 rng(20260512);
        std::normal_distribution<double> nd(0.0, NOISE_STD);

        // 执行 N_TRIALS 次噪声测试
        for (int t = 0; t < N_TRIALS; ++t) {
            // 创建噪声图像
            cv::Mat noisy = g0.clone();
            
            // 逐像素添加高斯噪声
            for (int y = 0; y < noisy.rows; ++y) {
                uchar* p = noisy.ptr<uchar>(y);
                for (int x = 0; x < noisy.cols; ++x) {
                    double v = (double)p[x] + nd(rng);  // 添加噪声
                    // 将结果限制在有效像素值范围内 [0, 255]
                    p[x] = (uchar)std::clamp((int)std::round(v), 0, 255);
                }
            }
            
            // 对噪声图像执行测量
            slot::MeasureResult res;
            res.image_name = files.front().filename().string();
            if (!slot::measureOne(noisy, cfg, res)) continue;
            
            // 累积结果
            r1n.push_back(res.r1);
            r2n.push_back(res.r2);
            d1n.push_back(res.d1);
            d2n.push_back(res.d2);
            tn.push_back(res.elapsed_ms);
        }

        // 将噪声重复性测试结果写入 CSV 文件
        std::ofstream cn(out_dir + "/noise_repeatability.csv");
        cn << "trial,r1_px,r2_px,d1_px,d2_px,elapsed_ms\n";
        for (size_t i = 0; i < r1n.size(); ++i)
            cn << i << "," << r1n[i] << "," << r2n[i] << "," << d1n[i] << "," << d2n[i] << "," << tn[i] << "\n";

        // 打印噪声重复性统计
        std::cout << "\n========== 噪声重复性 (" << files.front().filename().string()
                  << " + N(0," << NOISE_STD << "), N=" << r1n.size() << ") ==========\n";
        print("r1", r1n);
        print("r2", r2n);
        print("d1", d1n);
        print("d2", d2n);
        print("time_ms", tn);
    }

    // ============================================================
    // 第8步：输出结果文件路径提示
    // ============================================================
    std::cout << "\n结果文件: " << out_dir << "/measurements.csv,  "
              << out_dir << "/noise_repeatability.csv\n";
    std::cout << "可视化结果: " << out_dir << "/*_result.png\n";

    return 0;  // 程序正常结束
}
