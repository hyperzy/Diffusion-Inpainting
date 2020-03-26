//
// Created by himalaya on 3/25/20 at 11:49 AM.
//

#include "base.h"
#include "camera.h"
#include "draw_mask.h"
#include "inpainting2d.h"
#include <memory>
#include <opencv2/imgcodecs.hpp>

using namespace std;

int main(int argc, char *argv[]) {
    auto d_img(make_unique<DepthImage>());
    string depth_path(argv[1]);
    d_img->loadKinect(depth_path);
    d_img->showImage();

//    genMask(depth_path);

    string mask_path(argv[2]);
    cv::Mat mask = cv::imread(mask_path, 0);
    auto inpaint(make_unique<Inpainting2d>(mask.rows, mask.cols));

    cv::Mat ans;
    inpaint->heatDiffusion(d_img->getImage(), ans, mask);
    string output_path(argv[3]);
    cv::imwrite(output_path, ans);
    auto res_img(make_unique<DepthImage>());
    res_img->loadKinect(output_path);
    res_img->showImage();
    return 0;
}
