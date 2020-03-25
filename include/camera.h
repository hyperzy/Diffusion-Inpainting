//
// Created by himalaya on 3/8/20 at 2:58 PM.
//

#ifndef SDF2SDF_CAMERA_H
#define SDF2SDF_CAMERA_H
#include "base.h"
#include <string>
#include <opencv2/imgcodecs/imgcodecs_c.h>

class ImageBase {
protected:
    cv::Mat __img;
    const cv::Mat *__operate_img;
    Eigen::Matrix<dtype, 3, 4> __pose;
    Mat3 __intrinsic;
    Mat3 __rotation;
    Vec3 __translation;
    bool r_set, t_set, i_set;
    Vec2 __top_left, __bottom_right;
    bool extent_set;
    int __width, __height;
    ImageBase();
    void _selectROI();
    void _showImage();
public:
    int id;
    virtual void loadImage(const std::string &filepath, int flag);
    virtual void loadPose(const std::string &filepath);
    virtual void setPose(const Eigen::Matrix<dtype, 3, 4> &pose);
    virtual void loadIntrinsic(const std::string &filepath);
    virtual void setIntrinsic(const Mat3 &intrinsic);
    virtual void showImage() = 0;
    virtual double getMaxDepth() const = 0;
    virtual void selectROI() = 0;
    void saveROI(std::ofstream &fout, int idx);
    void loadROI(std::ifstream &fin, int idx);
    const cv::Mat& getImage() const;
    Eigen::Matrix<dtype, 3, 4> getPose() const;
    Mat3 getIntrinsic() const;
    Mat3 getRotation() const;
    Vec3 getTranslation() const;
    Vec2 topLeft() const;
    Vec2 bottomRight() const;
    int getWidth() const;
    int getHeight() const;
    /**
     * @brief Transform inertial coordinate system from cam0 to the one centered at the central position.
     * Use if only if necessary.
     * @param rotation Rotation matrix mapping coordinate in cam0 system into new inertia system.
     * @param translation Translation vector mapping coordinate in cam0 system into new inertia system.
     */
    void transformInertia(const Mat3 &rotation, const Vec3 &translation);
};

class DepthImage: public ImageBase {
    double _max_depth;
//    cv::Mat __img_for_show;
    void _genDisplayImg(cv::Mat &dst);
    bool __max_inf;
public:
    DepthImage();
    void loadImage(const std::string &filepath);
    void showImage() override;
    void denoieseImage();
    double getMaxDepth() const override;
    void selectROI() override;
};

class RGBImage: public ImageBase {
public:
    RGBImage();
    void loadImage(const std::string &filepath, int flag = CV_LOAD_IMAGE_ANYCOLOR);
    void showImage() override;
    double getMaxDepth() const override;
    void selectROI() override;
};

inline int ImageBase::getWidth() const {
    return this->__width;
}

inline int ImageBase::getHeight() const {
    return this->__height;
}


#endif //SDF2SDF_CAMERA_H
