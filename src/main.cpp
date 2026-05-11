// main.cpp —— 跑道形空腔尺寸测量主程序
//   * 处理 imgs/*.bmp 中所有图像
//   * 每张图保存结果可视化到 results/<name>_result.png
//   * 把每张图的 r1/r2/d1/d2、计时、拟合 RMS 输出到 results/measurements.csv
//   * 控制台同时打印逐图结果 + 总体统计 (均值, 标准差, 极差)，用于重复性分析
//   * 选取一张图，注入高斯噪声 N 次，评估算法对噪声的稳定性 (重复性)

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

struct AggStats {
    double mean = 0, stddev = 0, range = 0, min_v = 0, max_v = 0;
};
static AggStats aggregate(const std::vector<double>& v) {
    AggStats s;
    if (v.empty()) return s;
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    s.mean = sum / v.size();
    double ss = 0;
    for (double x : v) ss += (x - s.mean) * (x - s.mean);
    s.stddev = std::sqrt(ss / std::max(1.0, (double)v.size() - 1.0));
    auto [mn, mx] = std::minmax_element(v.begin(), v.end());
    s.min_v = *mn; s.max_v = *mx; s.range = s.max_v - s.min_v;
    return s;
}

int main(int argc, char** argv) {
    std::string img_dir = "imgs";
    std::string out_dir = "results";
    if (argc > 1) img_dir = argv[1];
    if (argc > 2) out_dir = argv[2];

    fs::create_directories(out_dir);

    std::vector<fs::path> files;
    for (auto& e : fs::directory_iterator(img_dir)) {
        if (!e.is_regular_file()) continue;
        auto p = e.path();
        std::string ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".bmp" || ext == ".png" || ext == ".jpg") files.push_back(p);
    }
    std::sort(files.begin(), files.end());

    if (files.empty()) {
        std::cerr << "No images found in " << img_dir << "\n";
        return 1;
    }

    slot::PipelineConfig cfg;

    std::ofstream csv(out_dir + "/measurements.csv");
    csv << "image,r1_px,r2_px,d1_px,d2_px,d1_std_px,d2_std_px,"
           "rms_arc_left_px,rms_arc_right_px,elapsed_ms\n";

    std::vector<double> r1s, r2s, d1s, d2s, times;

    std::cout << "\n========== 单图测量结果 ==========\n";
    std::cout << std::left
              << std::setw(10) << "image"
              << std::right
              << std::setw(11) << "r1(px)"
              << std::setw(11) << "r2(px)"
              << std::setw(11) << "d1(px)"
              << std::setw(11) << "d2(px)"
              << std::setw(11) << "rms_L"
              << std::setw(11) << "rms_R"
              << std::setw(11) << "t(ms)"
              << "\n";
    std::cout << std::string(85, '-') << "\n";

    for (auto& fp : files) {
        cv::Mat gray = cv::imread(fp.string(), cv::IMREAD_GRAYSCALE);
        if (gray.empty()) {
            std::cerr << "Failed to read " << fp << "\n";
            continue;
        }
        slot::MeasureResult res;
        res.image_name = fp.filename().string();
        if (!slot::measureOne(gray, cfg, res)) {
            std::cerr << "[main] measure failed: " << fp << "\n";
            continue;
        }

        cv::Mat vis = slot::renderResult(gray, res);
        std::string out_img = out_dir + "/" + fp.stem().string() + "_result.png";
        cv::imwrite(out_img, vis);

        // d1/d2 单图标准差
        AggStats sd1 = aggregate(res.d1_per_line);
        AggStats sd2 = aggregate(res.d2_per_line);

        csv << res.image_name << ","
            << res.r1 << "," << res.r2 << ","
            << res.d1 << "," << res.d2 << ","
            << sd1.stddev << "," << sd2.stddev << ","
            << res.rms_arc_left << "," << res.rms_arc_right << ","
            << res.elapsed_ms << "\n";

        r1s.push_back(res.r1);
        r2s.push_back(res.r2);
        d1s.push_back(res.d1);
        d2s.push_back(res.d2);
        times.push_back(res.elapsed_ms);

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
    csv.close();

    // ---- 重复性分析 ----
    auto print = [&](const char* name, const std::vector<double>& v) {
        AggStats s = aggregate(v);
        std::cout << std::left  << std::setw(10) << name
                  << std::right << std::fixed << std::setprecision(4)
                  << " mean=" << std::setw(10) << s.mean
                  << "  std=" << std::setw(8)  << s.stddev
                  << "  range=" << std::setw(8) << s.range
                  << "  min=" << std::setw(10) << s.min_v
                  << "  max=" << std::setw(10) << s.max_v
                  << "  CV(%)=" << std::setprecision(3) << (s.mean != 0 ? 100.0 * s.stddev / std::fabs(s.mean) : 0.0)
                  << "\n";
    };

    std::cout << "\n========== 重复性统计 (N=" << r1s.size() << ") ==========\n";
    print("r1", r1s);
    print("r2", r2s);
    print("d1", d1s);
    print("d2", d2s);
    print("time_ms", times);
    // ---- 噪声重复性测试：选一张图反复加噪声并测量 ----
    if (!files.empty()) {
        cv::Mat g0 = cv::imread(files.front().string(), cv::IMREAD_GRAYSCALE);
        const int N_TRIALS = 30;
        const double NOISE_STD = 4.0;        // 灰度高斯噪声标准差 (~1.5% 动态范围)
        std::vector<double> r1n, r2n, d1n, d2n, tn;

        std::mt19937 rng(20260512);
        std::normal_distribution<double> nd(0.0, NOISE_STD);

        for (int t = 0; t < N_TRIALS; ++t) {
            cv::Mat noisy = g0.clone();
            for (int y = 0; y < noisy.rows; ++y) {
                uchar* p = noisy.ptr<uchar>(y);
                for (int x = 0; x < noisy.cols; ++x) {
                    double v = (double)p[x] + nd(rng);
                    p[x] = (uchar)std::clamp((int)std::round(v), 0, 255);
                }
            }
            slot::MeasureResult res;
            res.image_name = files.front().filename().string();
            if (!slot::measureOne(noisy, cfg, res)) continue;
            r1n.push_back(res.r1);
            r2n.push_back(res.r2);
            d1n.push_back(res.d1);
            d2n.push_back(res.d2);
            tn.push_back(res.elapsed_ms);
        }

        std::ofstream cn(out_dir + "/noise_repeatability.csv");
        cn << "trial,r1_px,r2_px,d1_px,d2_px,elapsed_ms\n";
        for (size_t i = 0; i < r1n.size(); ++i)
            cn << i << "," << r1n[i] << "," << r2n[i] << "," << d1n[i] << "," << d2n[i] << "," << tn[i] << "\n";

        std::cout << "\n========== 噪声重复性 (" << files.front().filename().string()
                  << " + N(0," << NOISE_STD << "), N=" << r1n.size() << ") ==========\n";
        print("r1", r1n);
        print("r2", r2n);
        print("d1", d1n);
        print("d2", d2n);
        print("time_ms", tn);
    }

    std::cout << "\n结果文件: " << out_dir << "/measurements.csv,  "
              << out_dir << "/noise_repeatability.csv\n";
    std::cout << "可视化结果: " << out_dir << "/*_result.png\n";

    return 0;
}
