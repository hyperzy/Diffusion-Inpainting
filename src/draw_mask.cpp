//
// Created by himalaya on 3/10/20 at 4:01 PM.
//

#include "base.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "draw_mask.h"
using namespace std;
using namespace cv;

Mat img, img_for_show, out;
string window_name = "depth";
static void onMouse(int event, int x, int y, int flags, void *params) {
    if (event == EVENT_LBUTTONDOWN || (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))) {
        circle(img_for_show, Point(x, y), 3, Scalar(255, 0, 0), FILLED);
        circle(out, Point(x, y), 3, Scalar(255, 255, 255), FILLED);
        imshow(window_name, img_for_show);
    }
//    else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON)) {}

}
void genMask(const string &filepath, const string &output_path) {
    img = imread(filepath, CV_LOAD_IMAGE_ANYDEPTH);
    double min_d, max_d;
    minMaxIdx(img, &min_d, &max_d);
    convertScaleAbs(img, img_for_show, 255. / max_d);
    img_for_show.convertTo(img_for_show, CV_8UC1);
    namedWindow(window_name, 0);
    setMouseCallback(window_name, onMouse);
    imshow(window_name, img_for_show);
    out = Mat::zeros(img.rows, img.cols, img_for_show.type());
    while (1) {
        if (waitKey(0) == 13) break;
    }
    destroyAllWindows();
    imwrite(output_path, out);
}
