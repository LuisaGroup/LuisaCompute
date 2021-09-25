//
// Created by Mike Smith on 2021/9/17.
//

#include <iostream>
#include <asio.hpp>
#include <opencv2/opencv.hpp>
#include <core/logging.h>
#include <core/basic_types.h>

int main() {
    asio::io_context io_context;
    asio::ip::tcp::endpoint endpoint{asio::ip::address_v4::from_string("127.0.0.1"), 13};
    asio::ip::tcp::socket socket(io_context);
    socket.connect(endpoint);
    asio::error_code error;
    cv::Mat image{512, 512, CV_8UC4, cv::Scalar::all(0)};
    asio::write(socket, asio::buffer("client"), error);
    while (!error) {
        if (asio::read(socket, asio::buffer(image.data, 512u * 512u * 4u), error); !error) {
            cv::imshow("Display", image);
            cv::waitKey(1);
        }
    }
}
