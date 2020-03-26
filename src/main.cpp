//
// Created by himalaya on 3/25/20 at 11:49 AM.
//

#include "base.h"
#include "camera.h"
#include "draw_mask.h"
#include "inpainting2d.h"
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {
    if (argc != 7) {
        cout << "Usage: " << endl;
        cout << "         heat: ./diffu_inpainting heat [number of iteration (int)] [time step (float)] [input img path] [mask path] [output img path]" << endl;
        cout << "  anisotropic: ./diffu_inpainting anisotropic [number of iteration (int)] [time step (float)] [input img path] [mask path] [output img path]" << endl;
        exit(EXIT_FAILURE);
    }
    auto d_img(make_unique<DepthImage>());
    string depth_path(argv[4]);
    d_img->loadKinect(depth_path);
    d_img->showImage();

//    genMask(depth_path);

    string mask_path(argv[5]);
    cv::Mat mask = cv::imread(mask_path, 0);
    auto inpaint(make_unique<Inpainting2d>(mask.rows, mask.cols));

    cv::Mat ans;
    inpaint->setIterationTimes(stoi(string(argv[2])));
    inpaint->setTimeStep(stof(string(argv[3])));
    if (string(argv[1]) == "heat") {
        inpaint->heatDiffusion(d_img->getImage(), ans, mask);
    }
    else if (string(argv[1]) == "anisotropic") {
        inpaint->anisotropicDiffusion(d_img->getImage(), ans, mask);
    }
    else {
        cout << "Unknown method" << endl;
        exit(EXIT_FAILURE);
    }
    string output_path(argv[6]);
    cv::imwrite(output_path, ans);
    auto res_img(make_unique<DepthImage>());
    res_img->loadKinect(output_path);
    res_img->showImage();
    return 0;
}
