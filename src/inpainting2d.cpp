//
// Created by himalaya on 3/25/20 at 12:23 PM.
//

#include "inpainting2d.h"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;


ostream &operator<< (ostream &os, const MyPoint2d &point) {
    os << point.i << ", " << point.j;
    return os;
}
Inpainting2d::Inpainting2d(int height, int width):
        h(height), w(width) {

}


void Inpainting2d::__preProcess(std::vector<dtype> &pre, std::vector<dtype> &cur, const cv::Mat &input_img,
                                const cv::Mat &mask) {

    // consider 16bit image for now
    if (input_img.depth() != 2) {

        cerr << "Invalid image type: " << input_img.depth() << endl;
        exit(EXIT_FAILURE);
    }
//    cv::imshow("mask", mask);
//    cv::waitKey(0);
//    cv::destroyAllWindows();
    pre.resize(this->h * this->w);
    cur.resize(this->h * this->w);
//    cv::Mat test = (cv::Mat_<uint16_t>(3, 4) << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
    pre.assign((d_bits *)input_img.datastart, (d_bits *) input_img.dataend);
    cur.assign((d_bits *)input_img.datastart, (d_bits *) input_img.dataend);
//    int k = 0;
//    for (int i = 0; i < this->h; i++) {
//        for (int j = 0; j < this->w; j++) {
//            pre[k++] = input_img.at<uint16_t>(i, j);
//        }
//    }
//    k = 0;
//    for (int i = 0; i < this->h; i++) {
//        for (int j = 0; j < this->w; j++) {
//            cur[k++] = input_img.at<uint16_t>(i, j);
//        }
//    }

//    cv::Mat test(h, w, DTYPE, pre.data());
//    test.convertTo(test, CV_16UC1);
//    cv::imwrite("../res/test.png", test);
//    cout << test << endl;
//    cur.assign((d_bits *)test.datastart, (d_bits *) test.dataend);
//    cur[5] = 0.1;
//    for (auto i : cur) {
//        cout << i << ", ";
//    }
//    cout << endl;
//    cv::Mat test_res = cv::Mat(3, 4, DTYPE);
//    memcpy(test_res.data, cur.data(), cur.size() * sizeof(dtype));
//    cout << test_res << endl;
//    test_res.convertTo(test_res, CV_16UC1);
//    cout << test_res << endl;

//    cv::Mat mask_thresh;
//    cv::threshold(mask, mask_thresh, 125, 255, cv::THRESH_BINARY);
////    cv::imshow("mask", mask_thresh);
////    cv::waitKey(0);
////    cv::destroyAllWindows();
//    cv::Mat mask_area;
//    cv::findNonZero(mask_thresh, mask_area);
////    cout << mask_area.cols << " " << mask_area.rows << " " << (mask_area.dataend - mask_area.datastart) / sizeof(cv::Point) << endl;
//    this->__computed_area.reserve(mask_area.rows);
//    for (int i = 0; i < mask_area.rows; i++) {
//        auto point = mask_area.at<cv::Point>(i);
//        this->__computed_area.emplace_back(MyPoint2d(point.y, point.x));
//    }
    this->__genMask(input_img);
    this->__findComputeArea();
}

void Inpainting2d::__preProcess(std::vector<dtype> &d_pre, std::vector<dtype> &d_cur, std::vector<dtype> &pre, std::vector<dtype> &cur,
                                const cv::Mat &input_img, const cv::Mat &mask) {
    if (input_img.depth() != 2) {
        cerr << "Invalid image type: " << input_img.depth() << endl;
        exit(EXIT_FAILURE);
    }
    d_pre.resize(this->h * this->w, 0);
    d_cur.resize(this->h * this->w, 0);
    pre.resize(this->h * this->w);
    cur.resize(this->h * this->w);
//    cv::Mat test = (cv::Mat_<uint16_t>(3, 4) << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
    pre.assign((d_bits *)input_img.datastart, (d_bits *) input_img.dataend);
    cur.assign((d_bits *)input_img.datastart, (d_bits *) input_img.dataend);

//    cv::Mat mask_thresh;
//    cv::threshold(mask, mask_thresh, 125, 255, cv::THRESH_BINARY);
//    cv::Mat mask_area;
//    cv::findNonZero(mask_thresh, mask_area);
//    this->__computed_area.reserve(mask_area.rows);
//    for (int i = 0; i < mask_area.rows; i++) {
//        auto point = mask_area.at<cv::Point>(i);
//        this->__computed_area.emplace_back(MyPoint2d(point.y, point.x));
//    }
//    this->__genMask(input_img);
//    this->__findComputeArea();
}

void Inpainting2d::__genResult(const std::vector<dtype> &arr, cv::Mat &output_img) {
    output_img = cv::Mat(this->h, this->w, DTYPE);
    assert(arr.size() == output_img.total());
    memcpy(output_img.data, arr.data(), arr.size() * sizeof(dtype));
    output_img.convertTo(output_img, CV_16UC1);
//    output_img = cv::Mat(this->h, this->w, CV_16UC1);
//    k = 0;
//    for (int i = 0; i < this->h; i++) {
//        for (int j = 0; j < this->w; j++) {
//            output_img.at<uint16_t>(i, j) = (uint16_t)pre[k++];
//        }
//    }
//    cv::imshow("result", output_img);
//    cv::waitKey(0);
}

void Inpainting2d::__genResult(const std::vector<dtype> &arr, cv::Mat &output_img, const cv::Mat &input_img) {
    output_img = cv::Mat(this->h, this->w, DTYPE);
    assert(arr.size() == output_img.total());
    memcpy(output_img.data, arr.data(), arr.size() * sizeof(dtype));
    output_img.convertTo(output_img, CV_16UC1);
    output_img = input_img - output_img;
}

dtype Inpainting2d::__getSum(const cv::Mat &input_img, const cv::Mat &mask) {
    cv::Mat temp_for_sum;
    cv::Mat temp_mask;
    mask.convertTo(temp_mask, input_img.depth());
//    cv::imshow("temp mask", temp_mask);
//    cv::waitKey(0);
//    cv::destroyAllWindows();
    cv::multiply(input_img, temp_mask, temp_for_sum);
    return cv::sum(temp_for_sum)[0] / 255;
}

void Inpainting2d::__genMask(const cv::Mat &input_img) {
    this->__mask = input_img == 0;
//    cv::imshow("temp mask", this->__mask);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void Inpainting2d::__findComputeArea() {
    cv::Mat mask_area;
    cv::findNonZero(this->__mask, mask_area);
//    cout << mask_area.cols << " " << mask_area.rows << " " << (mask_area.dataend - mask_area.datastart) / sizeof(cv::Point) << endl;
    this->__computed_area.reserve(mask_area.rows);
    for (int i = 0; i < mask_area.rows; i++) {
        auto point = mask_area.at<cv::Point>(i);
        this->__computed_area.emplace_back(MyPoint2d(point.y, point.x));
    }
}

void Inpainting2d::heatDiffusion(const cv::Mat &input_img, cv::Mat &output_img, const cv::Mat &mask) {
    vector<dtype> pre, cur;

    this->__preProcess(pre, cur, input_img, mask);
    dtype pre_sum = this->__getSum(input_img, mask);

    constexpr dtype THRESH = 1;
    /**** heat diffusion ****/
    for (int n = 0; n < this->__n_iterations; n++) {
        dtype cur_sum = 0;
        for (const auto &point: this->__computed_area) {
            const auto &i = point.i;
            const auto &j = point.j;
//            assert(this->isValidRange(i, j));
//            cout << input_img.at<uint16_t>(i, j) << " " << pre[this->index(i, j)] << endl;
            dtype res = 0;
            auto idx = this->index(i + 1, j);
            res += (isValidRange(i + 1, j)) ? pre[this->index(i + 1, j)] : 0;
            res += (isValidRange(i - 1, j)) ? pre[this->index(i - 1, j)] : 0;
            res += (isValidRange(i, j + 1)) ? pre[this->index(i, j + 1)] : 0;
            res += (isValidRange(i, j - 1)) ? pre[this->index(i, j - 1)] : 0;
            res -= 4 * pre[this->index(i, j)];
            cur[this->index(i, j)] = pre[this->index(i, j)] + this->__time_step * res;
            cur_sum += cur[this->index(i, j)];
        }
        swap(pre, cur);
        cout << abs(cur_sum - pre_sum) << endl;
        if (abs(cur_sum - pre_sum) > THRESH) {
            pre_sum = cur_sum;
        }
        else {
            break;
        }
    }
    this->__genResult(pre, output_img);
}

void Inpainting2d::anisotropicDiffusion(const cv::Mat &input_img, cv::Mat &output_img, const cv::Mat &mask) {
    vector<dtype> pre, cur;
    this->__preProcess(pre, cur, input_img, mask);
    dtype pre_sum = this->__getSum(input_img, mask);

    constexpr dtype THRESH = 1;
    constexpr dtype FLOAT_ERR = 1e-10;
    /**** heat diffusion ****/
    for (int n = 0; n < this->__n_iterations; n++) {
        dtype cur_sum = 0;
        for (const auto &point: this->__computed_area) {
            const auto &i = point.i;
            const auto &j = point.j;
//            assert(this->isValidRange(i, j));
//            cout << input_img.at<uint16_t>(i, j) << " " << pre[this->index(i, j)] << endl;
//            dtype Ixx = pre[this->index(i, j + 1)] + pre[this->index(i, j - 1)] - 2 * pre[this->index(i, j)];
//            dtype Iyy = pre[this->index(i + 1, j)] + pre[this->index(i - 1, j)] - 2 * pre[this->index(i, j)];
//            dtype Ixy = (pre[this->index(i + 1, j + 1)] + pre[this->index(i - 1, j - 1)]
//                        - pre[this->index(i - 1, j + 1)] - pre[this->index(i + 1, j - 1)]) / 4.;
//            dtype Ix = (pre[this->index(i, j + 1)] - pre[this->index(i, j - 1)]) / 2.;
//            dtype Iy = (pre[this->index(i + 1, j)] - pre[this->index(i - 1, j)]) / 2.;
            dtype Ixx = this->__getVal(pre, i, j + 1) + this->__getVal(pre, i, j - 1) - 2 * pre[this->index(i, j)];
            dtype Iyy = this->__getVal(pre, i + 1, j) + this->__getVal(pre, i - 1, j) - 2 * pre[this->index(i, j)];
            dtype Ixy = (this->__getVal(pre, i + 1, j + 1) + this->__getVal(pre, i - 1, j - 1)
                         - this->__getVal(pre, i - 1, j + 1) - this->__getVal(pre, i + 1, j - 1)) / 4.;
            dtype Ix = (this->__getVal(pre, i, j + 1) - this->__getVal(pre, i, j - 1)) / 2.;
            dtype Iy = (this->__getVal(pre, i + 1, j) - this->__getVal(pre, i - 1, j)) / 2.;
            dtype delta;
            if (abs(Ix) > FLOAT_ERR || abs(Iy) > FLOAT_ERR) {
                delta = (pow(Ix, 2) * Iyy + pow(Iy, 2) * Ixx - 2 * Ix * Iy * Ixy) / pow(Ix * Ix + Iy * Iy, 1);
            }
            else {
                delta = 0;
            }
            cur[this->index(i, j)] = pre[this->index(i, j)] + this->__time_step * delta;
            cur_sum += cur[this->index(i, j)];
            if (isinf(cur_sum)) {
                cout << cur[this->index(i, j)] << endl;
            }
        }
        swap(pre, cur);
        cout << "Energy Diff: " << abs(cur_sum - pre_sum) << endl;
        if (abs(cur_sum - pre_sum) > THRESH) {
            pre_sum = cur_sum;
        }
        else {
            break;
        }
    }

    this->__genResult(pre, output_img);
}

void Inpainting2d::acceleratedTVDiffusion(const cv::Mat &input_img, cv::Mat &output_img, const cv::Mat &mask) {
    vector<dtype> pre, cur, d_cur, d_pre;
    __preProcess(d_pre, d_cur, pre, cur, input_img, mask);
    dtype pre_sum = this->__getSum(input_img, mask);

    constexpr dtype THRESH = 1;
    constexpr dtype FLOAT_ERR = 1e-10;
    dtype a = 0;
    const auto dt = this->__time_step;
    /*** TV regularization accelerated PDE ***/
    for (int n = 0; n < this->__n_iterations; n++) {
        dtype cur_sum = 0;
        for (const auto &point: this->__computed_area) {
            const auto &i = point.i;
            const auto &j = point.j;
            dtype Ixx = pre[this->index(i, j + 1)] + pre[this->index(i, j - 1)] - 2 * pre[this->index(i, j)];
            dtype Iyy = pre[this->index(i + 1, j)] + pre[this->index(i - 1, j)] - 2 * pre[this->index(i, j)];
            dtype Ixy = (pre[this->index(i + 1, j + 1)] + pre[this->index(i - 1, j - 1)]
                         - pre[this->index(i - 1, j + 1)] - pre[this->index(i + 1, j - 1)]) / 4.;
            dtype Ix = (pre[this->index(i, j + 1)] - pre[this->index(i, j - 1)]) / 2.;
            dtype Iy = (pre[this->index(i + 1, j)] - pre[this->index(i - 1, j)]) / 2.;
            dtype delta;
            if (abs(Ix) > FLOAT_ERR || abs(Iy) > FLOAT_ERR) {
                delta = -(pow(Ix, 2) * Iyy + pow(Iy, 2) * Ixx - 2 * Ix * Iy * Ixy) / pow(Ix * Ix + Iy * Iy, 1.5);
            }
            else {
                delta = 0;
            }
            d_cur[this->index(i, j)] = (2 - a * dt) / (2 + a * dt) * d_pre[this->index(i, j)]
                    - (2 * dt * dt) / (2 + a * dt) *delta;
            cur[this->index(i, j)] = pre[this->index(i, j)] + d_cur[this->index(i, j)];
            cur_sum += cur[this->index(i, j)];
            if (isinf(cur_sum) || isnan(cur_sum)) {
                cout << cur[this->index(i, j)] << endl;
                exit(EXIT_FAILURE);
            }
        }
        swap(cur, pre);
        swap(d_cur, d_pre);
        cout << "Energy Diff: " << abs(cur_sum - pre_sum) << endl;
        if (abs(cur_sum - pre_sum) > THRESH) {
            pre_sum = cur_sum;
        }
        else {
            break;
        }
    }

    this->__genResult(pre, output_img);
}



void Inpainting2d::acceleratedBeltramiDiffusion(const cv::Mat &input_img, cv::Mat &output_img, const cv::Mat &mask) {
    cv::Mat temp;
    this->setIterationTimes(1);
    this->heatDiffusion(input_img, temp, mask);
//    cv::inpaint(input_img, (input_img == 0), temp, 5, cv::INPAINT_TELEA);

    this->setIterationTimes(2000);
//    this->setTimeStep(0.4);
    vector<dtype> pre, cur, d_cur, d_pre;
    this->__preProcess(d_pre, d_cur, pre, cur, temp, mask);
//    dtype pre_sum = this->__getSum(input_img, mask);
    dtype pre_sum = 0;

    constexpr dtype THRESH = 1e-8;
    constexpr dtype FLOAT_ERR = 1e-10;
    constexpr dtype BETA = 1.;
    constexpr dtype lambda = 0;
    dtype a = 2. * sqrt(M_PI * M_PI + lambda);
    dtype dt_scale = 0.5;
    const auto dt = 2. / sqrt(8 * BETA + lambda) * dt_scale;


//    cv::threshold(mask, mask, 125, 255, cv::THRESH_BINARY);
//    cv::imshow("temp mask", mask);
//    cv::waitKey(0);
//    cv::destroyAllWindows();

//    /*** test Beltrami Denoising for whole image ***/
//    this->__computed_area.resize(this->h * this->w);
//    this->__computed_area.clear();
//    for (int i = 1; i < this->h - 1; i++) {
//        for (int j = 1; j < this->w - 1; j++) {
//            this->__computed_area.emplace_back(MyPoint2d(i, j));
//        }
//    }
    /**** 2st-order accelerated Beltrami Regularization ****/
    for (int n = 0; n < this->__n_iterations; n++) {
        dtype cur_sum = 0;
        for (const auto &point: this->__computed_area) {
            const auto &i = point.i;
            const auto &j = point.j;
            dtype Ixx = this->__getVal(pre, i, j + 1) + this->__getVal(pre, i, j - 1) - 2 * pre[this->index(i, j)];
            dtype Iyy = this->__getVal(pre, i + 1, j) + this->__getVal(pre, i - 1, j) - 2 * pre[this->index(i, j)];
            dtype Ixy = (this->__getVal(pre, i + 1, j + 1) + this->__getVal(pre, i - 1, j - 1)
                         - this->__getVal(pre, i - 1, j + 1) - this->__getVal(pre, i + 1, j - 1)) / 4.;
            dtype Ix = (this->__getVal(pre, i, j + 1) - this->__getVal(pre, i, j - 1)) / 2.;
            dtype Iy = (this->__getVal(pre, i + 1, j) - this->__getVal(pre, i - 1, j)) / 2.;
            dtype delta;
//            if (abs(Ix) > FLOAT_ERR || abs(Iy) > FLOAT_ERR) {
                delta = -BETA * ((1 + pow(BETA, 2) * pow(Ix, 2)) * Iyy + (1 + pow(BETA, 2) * pow(Iy, 2)) * Ixx - 2 * pow(BETA, 2) * Ix * Iy * Ixy)
                        / pow(1 + pow(BETA, 2) * (Ix * Ix + Iy * Iy), 1.5);
//            }
//            else {
//                delta = 0;
//            }
//            delta += lambda * (pre[this->index(i, j)] - ori[this->index(i, j)]);
            d_cur[this->index(i, j)] = (2 - a * dt) / (2 + a * dt) * d_pre[this->index(i, j)]
                                       - (2 * dt * dt) / (2 + a * dt) * delta;
//            d_cur[this->index(i, j)] = - dt * delta;
            cur[this->index(i, j)] = pre[this->index(i, j)] + d_cur[this->index(i, j)];
//            cur_sum += cur[this->index(i, j)];
            cur_sum += sqrt(1 + BETA * BETA * (Ix * Ix + Iy * Iy)) / BETA;
            if (isinf(cur_sum) || isnan(cur_sum)) {
                cout << cur[this->index(i, j)] << endl;
                exit(EXIT_FAILURE);
            }
        }
        swap(cur, pre);
        swap(d_cur, d_pre);
        cout << "Energy Diff: " << cur_sum - pre_sum << endl;
        if (abs(cur_sum - pre_sum) > THRESH) {
            pre_sum = cur_sum;
        }
        else {
//            break;
        }
    }

    this->__genResult(pre, output_img);
    cout << cv::sum(output_img) - cv::sum(input_img) << endl;
}

void Inpainting2d::setIterationTimes(int n) {
    this->__n_iterations = n;
}

void Inpainting2d::setTimeStep(dtype ts) {
    this->__time_step = ts;
}
