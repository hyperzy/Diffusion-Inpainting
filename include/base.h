//
// Created by himalaya on 3/8/20 at 2:58 PM.
//

#ifndef SDF2SDF_BASE_H
#define SDF2SDF_BASE_H

/**
 *  Convention:
 *      Vec2 represents 2D point and 1st, 2nd entry correspond to x, y respectively
 *      Vec3 re[resents 3D point and 1st, 2nd, 3rd entry correspond to x, y, z respectively.
 *
 */
#include <opencv2/core.hpp>
#include <vector>
#include <Eigen/Eigen>

#define DEBUG_MODE
#undef DEBUG_MODE

typedef float dtype;
//typedef vtkFloatArray vtkDtypeArray;
//typedef cv::Point3f Point3;
typedef Eigen::Matrix<dtype, 3, 3> Mat3;
typedef Eigen::Vector<dtype, 4> Vec4;
typedef Eigen::Vector<dtype, 3> Vec3;
typedef Eigen::Vector<dtype, 2> Vec2;
//typedef Eigen::
#define DTYPE CV_32F
#define DTYPEC1 CV_32FC1
constexpr dtype INF = 5e10;
#define FAR 1
typedef unsigned short IdxType;
typedef unsigned short DimUnit;

#endif //SDF2SDF_BASE_H
