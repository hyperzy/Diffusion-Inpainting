//
// Created by himalaya on 3/8/20 at 2:58 PM.
//

#include "camera.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <fstream>
#include <iostream>
using namespace std;
using namespace cv;

ImageBase::ImageBase():
    __operate_img(nullptr), r_set(false), t_set(false), extent_set(false),
    i_set(false) {

}

struct CallbackParams {
    cv::Rect rect;
    cv::Mat dst;
    std::string window_name;
    const cv::Mat &src;
    CallbackParams(const cv::Mat &src__img): src(src__img) {}
};
/**
 * @brief OpenCV mouse event interrupt.
 * @param event Mouse event
 * @param x Horizontal image location
 * @param y Vertical image location
 * @param flags
 * @param params
 */
static void onMouse(int event, int x, int y, int flags, void *params) {
    CallbackParams *parameter = (CallbackParams *)params;
    Rect &rect = parameter->rect;
    const Mat &src = parameter->src;
    Mat &dst = parameter->dst;
    string &window_name = parameter->window_name;
    if (event == EVENT_LBUTTONDOWN)
    {
        parameter->rect = Rect(x, y , 0 , 0);
        src.copyTo(dst);
        circle(dst, Point(x, y), 2, Scalar(255, 0, 0), FILLED);
        imshow(window_name, dst);
    }
//    // renew rectangular
//    else if (event == EVENT_MOUSEMOVE && !(flags & EVENT_FLAG_LBUTTON))
//    {
//        src.copyTo(dst);
//        imshow(window_name,dst);
//    }
        // draw rect when mouse is moving
    else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))
    {
        src.copyTo(dst);
        rectangle(dst, Point(rect.x, rect.y), Point(x, y), Scalar(255, 0, 0), 2);
        imshow(window_name, dst);
    }
    else if (event == EVENT_LBUTTONUP)
    {
        src.copyTo(dst);
        rectangle(dst, Point(rect.x, rect.y), Point(x, y), Scalar(255, 0, 0), 2);
        rect.width = abs(x - rect.x);
        rect.height = abs(y - rect.y);
        rect.x = min(rect.x, x);
        rect.y = min(rect.y, y);
        imshow(window_name, dst);
    }
}

void ImageBase::_showImage() {
    if (!this->__operate_img || !this->__operate_img->data) {
        cerr << "Empty Img" << endl;
    }
    else {
        namedWindow("img");
        imshow("img", *(this->__operate_img));
        waitKey(0);
        destroyAllWindows();
    }
}

void ImageBase::_selectROI() {
    if (!this->__operate_img || !this->__operate_img->data) {
        cerr << "Empty Img" << endl;
    }
    else {
        struct CallbackParams parameters(*(this->__operate_img));
        parameters.window_name = "ROI Selection";
        namedWindow(parameters.window_name, 0);
        setMouseCallback(parameters.window_name, onMouse, (void *)&parameters);
        imshow(parameters.window_name, parameters.src);
        while (true) {
            // press Enter to finish
            if (waitKey(0) == 13) break;
        }
        destroyAllWindows();
        this->__top_left << parameters.rect.x, parameters.rect.y;
        this->__bottom_right << parameters.rect.x + parameters.rect.width, parameters.rect.y + parameters.rect.height;
        this->extent_set = true;
    }
}
void ImageBase::loadImage(const std::string &filepath, int flag) {
    try {
        this->__img = imread(filepath, flag);
        this->__width = this->__img.cols;
        this->__height = this->__img.rows;
        if (!this->__img.data) {
            throw runtime_error("Invalid Image Path.");
        }
    }catch (exception const &e) {
        cerr << "Exception: " << e.what() << endl;
    }
}

void ImageBase::loadPose(const std::string &filepath) {
    ifstream fs(filepath, ifstream::in);
    try {
        if (!fs.is_open()) {
            throw runtime_error("Invalid Parameter Path.");
        }
        dtype mat[16];
        for (auto &entry: mat) {
            fs >> entry;
        }
        fs.close();
        Eigen::Map<Eigen::Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> P(mat, 4, 4);
        this->setPose(P.topRows(3));
//        Mat3 r = P.block<3, 3>(0, 0);
//        Vec3 c = P.block<3, 1>(0, 3);
//        Eigen::Matrix<dtype, 3, 4> pose;
//        this->__rotation = r;
//        this->__translation = -r * c;
//        pose.block<3, 3>(0, 0) = r;
//        pose.block<3, 1>(0, 3) = this->__translation;
//        this->__pose = pose;
//        this->r_set = this->t_set = true;
//        cout << this->__pose << endl;
    }catch (exception const &e) {
        cout << "Exception: " << e.what() << endl;
    }
}

void ImageBase::setPose(const Eigen::Matrix<dtype, 3, 4> &pose) {
    this->__pose = pose;
    this->__rotation = this->__pose.block<3, 3>(0, 0);
    this->__translation = this->__pose.block<3, 1>(0, 3);
    this->r_set = this->t_set = true;
}

void ImageBase::loadIntrinsic(const std::string &filepath) {
    ifstream fs(filepath, ifstream::in);
    if (!fs.is_open()) {
        cerr << "Invalid Parameter Path." << endl;
    }
    else {
        dtype mat[9];
        for (auto &entry: mat) {
            fs >> entry;
        }
        fs.close();

        this->__intrinsic = Eigen::Map<Eigen::Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(mat, 3, 3);
        this->i_set = true;
    }
}

void ImageBase::setIntrinsic(const Mat3 &intrinsic) {
    this->__intrinsic = intrinsic;
    this->i_set = true;
}


void ImageBase::saveROI(std::ofstream &fout, int idx) {
    if (!fout.is_open()) {
        cerr << "Wrong file path." << endl;
    }
    else {
        fout << idx << endl;
        fout << this->__top_left << endl;
        fout << this->__bottom_right << endl;
        fout << endl;
    }
}

void ImageBase::loadROI(std::ifstream &fin, int idx) {
    // todo: id and idx are used for information validation. e.g. id != idx blabla
    int id, x1, y1, x2, y2;
    if (!fin.is_open()) {
        cerr << "Wrong file path." << endl;
    }
    else {
        fin >> id >> x1 >> y1 >> x2 >> y2;
        this->__top_left << x1, y1;
        this->__bottom_right << x2, y2;
        this->extent_set = true;
    }
}

const Mat& ImageBase::getImage() const {
    return this->__img;
}

Eigen::Matrix<dtype, 3, 4> ImageBase::getPose() const {
    return this->__pose;
}


Mat3 ImageBase::getIntrinsic() const {
    if (!this->i_set) cerr << "Intrinsic matrix is not set." << endl;
    return this->__intrinsic;
}

Mat3 ImageBase::getRotation() const {
    if (!this->r_set) cerr << "Rotation Matrix Is Not Set." << endl;
    return this->__rotation;
}

Vec3 ImageBase::getTranslation() const {
    if (!this->t_set) cerr << "Translation Vector Is Not Set." << endl;
    return this->__translation;
}

Vec2 ImageBase::topLeft() const {
    return this->__top_left;
}

Vec2 ImageBase::bottomRight() const {
    return this->__bottom_right;
}

void ImageBase::transformInertia(const Mat3 &rotation, const Vec3 &translation) {
    // be careful about the execution order
    this->__translation = this->__rotation * translation + this->__translation;
    this->__rotation = this->__rotation * rotation;
    Eigen::Matrix<dtype, 3, 4> pose;
    pose.block<3, 3>(0, 0) = this->__rotation;
    pose.rightCols(1) = this->__translation;
    this->__pose = pose;
}

DepthImage::DepthImage():_max_depth(0) {
}

void DepthImage::loadImage(const std::string &filepath) {
    this->__max_inf = false;
//    try {
//        this->__img = imread(filepath, CV_LOAD_IMAGE_ANYDEPTH);
//        if (!this->__img.data) {
//            throw runtime_error("Invalid Image Path.");
//        }
//        double min_v;
//        minMaxIdx(this->__img, &min_v, &this->_max_depth);
//    }catch (exception const &e) {
//        cerr << "Exception: " << e.what() << endl;
//    }
    // load the image
    cv::Mat depth_map = cv::imread( filepath, -1 );
    if (!depth_map.data) cerr << "Invalid Image Path." << endl;
    double min_v;
    auto inf_notation = std::numeric_limits<double>::infinity();
    minMaxIdx(depth_map, &min_v, &this->_max_depth);
//    minMaxIdx(depth_map, &min_v, &this->_max_depth);
    if (this->_max_depth == inf_notation) {
        cv::cvtColor( depth_map, depth_map, COLOR_RGB2GRAY );
        this->__max_inf = true;
        Mat mask = depth_map == inf_notation;
        depth_map.setTo(INF, mask);
    }
    // convert to meters
    depth_map.convertTo( depth_map, DTYPEC1, 0.001 );
    minMaxIdx(depth_map, &min_v, &this->_max_depth);
    this->__img = depth_map.clone();
    this->__width = this->__img.cols;
    this->__height = this->__img.rows;
}

void DepthImage::_genDisplayImg(cv::Mat &dst) {
    if (!this->__img.data) {
        cerr << "Empty Depth Image." << endl;
    }
    else {
        convertScaleAbs(this->__img, dst, 255 / this->_max_depth);
    }
}

void DepthImage::showImage() {
    Mat img_for_show;
    if (this->__max_inf) {
        Mat mask = (this->__img == this->_max_depth);
        img_for_show = this->__img.clone();
        img_for_show.setTo(std::numeric_limits<double>::infinity(), mask);
//        this->__operate_img = &(this->__img);
        this->__operate_img = &img_for_show;
    }
    else {
        this->_genDisplayImg(img_for_show);
        this->__operate_img = &img_for_show;
    }
    this->_showImage();
    this->__operate_img = nullptr;
}

void DepthImage::denoieseImage() {
    ximgproc::anisotropicDiffusion(this->__img, this->__img, 0.2, 1.1, 50);
}

double DepthImage::getMaxDepth() const {
    return this->_max_depth;
}

void DepthImage::selectROI() {
    Mat img_for_show;
    if (isinf(this->_max_depth)) {
        this->__operate_img = &(this->__img);
    }
    else {
        this->_genDisplayImg(img_for_show);
        this->__operate_img = &img_for_show;
    }
    this->_selectROI();
    this->__operate_img = nullptr;
}

void DepthImage::loadKinect(const std::string &filepath) {
    this->__max_inf = false;
    try {
        this->__img = imread(filepath, CV_LOAD_IMAGE_ANYDEPTH);
        if (!this->__img.data) {
            throw runtime_error("Invalid Image Path.");
        }
        double min_v;
        minMaxIdx(this->__img, &min_v, &this->_max_depth);
    }catch (exception const &e) {
        cerr << "Exception: " << e.what() << endl;
    }
    this->__width = this->__img.cols;
    this->__height = this->__img.rows;
}

RGBImage::RGBImage() {
}

void RGBImage::loadImage(const std::string &filepath, int flag) {
    try {
        this->__img = imread(filepath, flag);
        this->__width = this->__img.cols;
        this->__height = this->__img.rows;
        if (!this->__img.data) {
            throw runtime_error("Invalid Image Path.");
        }
    }catch (exception const &e) {
        cerr << "Exception: " << e.what() << endl;
    }
}

void RGBImage::showImage() {
    this->__operate_img = &(this->__img);
    this->_showImage();
    this->__operate_img = nullptr;
}

double RGBImage::getMaxDepth() const {
    cerr << "No depth for RGB image." << endl;
}

void RGBImage::selectROI() {
    this->__operate_img = &(this->__img);
    this->_selectROI();
    this->__operate_img = nullptr;
}

void RGBImage::loadKinect(const std::string &filepath) {
    cerr << "Cannot load depth image for RGB image object" << endl;
    exit(EXIT_FAILURE);
}
