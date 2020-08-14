//
// Created by himalaya on 3/25/20 at 12:23 PM.
//

#ifndef DIFFU_INPAINTING_INPAINTING2D_H
#define DIFFU_INPAINTING_INPAINTING2D_H

#include "base.h"

struct MyPoint2d {
    MyPoint2d() {}
    MyPoint2d(int i, int j): i(i), j(j) {}
    int i;
    int j;
};

class Inpainting2d {
    void __preProcess(std::vector<dtype> &pre, std::vector<dtype> &cur, const cv::Mat &input_img, const cv::Mat &mask);
    void __preProcess(std::vector<dtype> &d_pre, std::vector<dtype> &d_cur, std::vector<dtype> &pre, std::vector<dtype> &cur, const cv::Mat &input_img, const cv::Mat &mask);
    void __genResult(const std::vector<dtype> &arr, cv::Mat &output_img);
    void __genResult(const std::vector<dtype> &arr, cv::Mat &output_img, const cv::Mat &input_img);
    void __genMask(const cv::Mat &input_img);
    void __findComputeArea();
    template <class T>
    T __getVal(const std::vector<T> &arr, int i, int j);
    static dtype __getSum(const cv::Mat &input_img, const cv::Mat &mask);
    std::vector<MyPoint2d> __computed_area;
    int __n_iterations;
    dtype __time_step;
    cv::Mat __mask;
public:
    Inpainting2d(int height, int width);
    int h, w;
    unsigned long index(int i, int j);
    bool isValidRange (int i, int j);
    void setIterationTimes(int n);
    void setTimeStep(dtype ts);
    void heatDiffusion(const cv::Mat &input_img, cv::Mat &output_img, const cv::Mat &mask);
    void anisotropicDiffusion(const cv::Mat &input_img, cv::Mat &output_img, const cv::Mat &mask);
    void acceleratedTVDiffusion(const cv::Mat &input_img, cv::Mat &output_img, const cv::Mat &mask);
    void acceleratedBeltramiDiffusion(const cv::Mat &input_img, cv::Mat &output_img, const cv::Mat &mask);
};

inline unsigned long Inpainting2d::index(int i, int j) {
    return i * w + j;
}

inline bool Inpainting2d::isValidRange(int i, int j) {
    return i >= 0 && i < h && j >= 0 && j < w;
}

template <class T>
inline T Inpainting2d::__getVal(const std::vector<T> &arr, int i, int j) {
    return this->isValidRange(i, j) ? arr[this->index(i, j)] : 0;
}
#endif //DIFFU_INPAINTING_INPAINTING2D_H
