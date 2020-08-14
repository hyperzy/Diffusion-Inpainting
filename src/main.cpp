//
// Created by himalaya on 3/25/20 at 11:49 AM.
//

#include "base.h"
#include "camera.h"
#include "draw_mask.h"
#include "inpainting2d.h"
#include <memory>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/rgbd.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, char *argv[]) {
    if (argc < 4) {
        cout << "Usage: " << endl;
        cout << "         heat: ./diffu_inpainting heat [number of iteration (int)] [time step (float)] [input img path] [mask path] [output img path]" << endl;
        cout << "  anisotropic: ./diffu_inpainting anisotropic [number of iteration (int)] [time step (float)] [input img path] [mask path] [output img path]" << endl;
        cout << "    draw mask: ./diffu_inpainting draw [input img path] [output mask path]" << endl;
        exit(EXIT_FAILURE);
    }
//    genMask(depth_path);
    if (string(argv[1]) == "draw") {
        genMask(string(argv[2]), string(argv[3]));
        exit(EXIT_SUCCESS);
    }
    else {
        auto d_img(make_unique<DepthImage>());
        string depth_path(argv[4]);
        d_img->loadKinect(depth_path);
        d_img->showImage();

        /*** try opencv depth cleaner ****/
        cv::rgbd::DepthCleaner *depthc = new cv::rgbd::DepthCleaner(CV_16U, 5, cv::rgbd::DepthCleaner::DEPTH_CLEANER_NIL);
        cv::Mat out(d_img->getHeight(), d_img->getWidth(), CV_16U);
        depthc->operator()(d_img->getImage(), out);
//        cv::Mat out = d_img->getImage();
//        out = d_img->getImage();
//        string mask_path(argv[5]);
//        cv::Mat mask = cv::imread(mask_path, 0);
//        cv::threshold(mask, mask, 125, 255, CV_THRESH_BINARY);
        cv::inpaint(out, (out == 0), out, 5, cv::INPAINT_TELEA);
        string output_path(argv[6]);
        cv::imwrite(output_path, out);
        auto res_img(make_unique<DepthImage>());
        res_img->loadKinect(output_path);
        res_img->showImage();



//         string mask_path(argv[5]);
//         cv::Mat mask = cv::imread(mask_path, 0);
//         auto inpaint(make_unique<Inpainting2d>(d_img->getHeight(), d_img->getWidth()));
//
//         cv::Mat ans;
//         inpaint->setIterationTimes(stoi(string(argv[2])));
//         inpaint->setTimeStep(stof(string(argv[3])));
//         if (string(argv[1]) == "heat") {
//             inpaint->heatDiffusion(d_img->getImage(), ans, mask);
//         } else if (string(argv[1]) == "anisotropic") {
//             inpaint->anisotropicDiffusion(d_img->getImage(), ans, mask);
//         } else if (string(argv[1]) == "acceleratedTV") {
//             inpaint->acceleratedTVDiffusion(d_img->getImage(), ans, mask);
//         } else if (string(argv[1]) == "acceleratedBel") {
//             // pre_process missing area first
//             cv::Mat in;
////             inpaint->heatDiffusion(d_img->getImage(), in, mask);
//
//             inpaint->acceleratedBeltramiDiffusion(d_img->getImage(), ans, mask);
//         } else {
//                 cout << "Unknown method" << endl;
//                 exit(EXIT_FAILURE);
//         }
//         string output_path(argv[6]);
//         cv::imwrite(output_path, ans);
//         auto res_img(make_unique<DepthImage>());
//         res_img->loadKinect(output_path);
//         res_img->showImage();
        return 0;
    }
}
