//
// Created by himalaya on 3/25/20 at 11:49 AM.
//

#include "base.h"
#include "camera.h"
#include "draw_mask.h"
#include <memory>
using namespace std;

int main(int argc, char *argv[]) {
    auto d_img(make_unique<DepthImage>());
    string depth_path(argv[1]);
    d_img->loadImage(depth_path);
    d_img->showImage();

    genMask(depth_path);
    return 0;
}
