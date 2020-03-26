//
// Created by himalaya on 3/25/20 at 12:23 PM.
//

#include "inpainting2d.h"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <memory>
#include <opencv2/highgui.hpp>

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

    cv::Mat mask_thresh;
    cv::threshold(mask, mask_thresh, 125, 255, cv::THRESH_BINARY);
//    cv::imshow("mask", mask_thresh);
//    cv::waitKey(0);
//    cv::destroyAllWindows();
    cv::Mat mask_area;
    cv::findNonZero(mask_thresh, mask_area);
//    cout << mask_area.cols << " " << mask_area.rows << " " << (mask_area.dataend - mask_area.datastart) / sizeof(cv::Point) << endl;
    this->__computed_area.reserve(mask_area.rows);
    for (int i = 0; i < mask_area.rows; i++) {
        auto point = mask_area.at<cv::Point>(i);
        this->__computed_area.emplace_back(MyPoint2d(point.y, point.x));
    }
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

dtype Inpainting2d::__getSum(const cv::Mat &input_img, const cv::Mat &mask) {
    cv::Mat temp_for_sum;
    cv::Mat temp_mask;
    mask.convertTo(temp_mask, input_img.depth());
//    cv::imshow("temp mask", temp_mask);
//    cv::waitKey(0);
//    cv::destroyAllWindows();
    cv::multiply(input_img, temp_mask, temp_for_sum);
    return cv::sum(temp_for_sum)[0];
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
            dtype res = pre[this->index(i + 1, j)] + pre[this->index(i - 1, j)]
                    + pre[this->index(i, j + 1)] + pre[this->index(i, j - 1)] - 4 * pre[this->index(i, j)];
            cur[this->index(i, j)] = pre[this->index(i, j)] + this->__time_step * res;
            cur_sum += cur[this->index(i, j)];
        }
        swap(pre, cur);
//        cout << abs(cur_sum - pre_sum) << endl;
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
            dtype Ixx = pre[this->index(i, j + 1)] + pre[this->index(i, j - 1)] - 2 * pre[this->index(i, j)];
            dtype Iyy = pre[this->index(i + 1, j)] + pre[this->index(i - 1, j)] - 2 * pre[this->index(i, j)];
            dtype Ixy = (pre[this->index(i + 1, j + 1)] + pre[this->index(i - 1, j - 1)]
                        - pre[this->index(i - 1, j + 1)] - pre[this->index(i + 1, j - 1)]) / 4.;
            dtype Ix = (pre[this->index(i, j + 1)] - pre[this->index(i, j - 1)]) / 2.;
            dtype Iy = (pre[this->index(i + 1, j)] - pre[this->index(i - 1, j)]) / 2.;
            dtype delta;
            if (abs(Ix) > FLOAT_ERR || abs(Iy) > FLOAT_ERR) {
                delta = (pow(Ix, 2) * Iyy + pow(Iy, 2) * Ixx + Ix * Iy * Ixy) / (Ix * Ix + Iy * Iy);
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
//        cout << abs(cur_sum - pre_sum) << endl;
        if (abs(cur_sum - pre_sum) > THRESH) {
            pre_sum = cur_sum;
        }
        else {
            break;
        }
    }

    this->__genResult(pre, output_img);
}

void Inpainting2d::setIterationTimes(int n) {
    this->__n_iterations = n;
}

void Inpainting2d::setTimeStep(dtype ts) {
    this->__time_step = ts;
}
