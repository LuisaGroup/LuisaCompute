//
// Created by Mike Smith on 2020/11/21.
//

#include <iostream>
#include <vector>
#include <memory>
#include <unordered_set>

#include <opencv2/opencv.hpp>
#include <OpenImageIO/imagecache.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/ustring.h>

class Resource {

};

class Buffer : Resource {

};

class ResourceManager {

private:

public:

};

template<typename T>
class ResourceSlot {

public:

};

int main() {
    
//    cv::Mat image{720, 1280, CV_8UC3, cv::Scalar::all(127.0)};
//    cv::imwrite("test.png", image);
    
    auto cache = OIIO::ImageCache::create();
    cache->attribute("max_memory_MB", 500.0f);
    cache->attribute("autotile", 64);
    cache->attribute("automip", true);
    
    std::array<float, 16 * 16 * 3> pixels{};
    
    OIIO::ustring filename{"test.tx"};
    auto image_spec = cache->imagespec(filename);
    std::cout << image_spec->format.c_str() << std::endl;
    if (cache->get_pixels(filename, 0, 0, -8, 8, 0, 16, 0, 1, OIIO::TypeFloat, pixels.data())) {
        for (auto p : pixels) { std::cout << p << " "; }
        std::cout << std::endl;
    }
    
    OIIO::ImageCache::destroy(cache);
}
