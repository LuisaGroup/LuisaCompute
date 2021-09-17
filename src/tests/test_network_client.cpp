//
// Created by Mike Smith on 2021/9/17.
//

#include <iostream>
#include <asio.hpp>
#include <core/logging.h>

int main() {
    asio::io_context io_context;
    asio::ip::tcp::endpoint endpoint{asio::ip::address_v4::from_string("127.0.0.1"), 13};
    asio::ip::tcp::socket socket(io_context);
    socket.connect(endpoint);
    for (;;) {
        std::array<char, 128> buffer{};
        asio::error_code error;
        auto len = socket.read_some(asio::buffer(buffer), error);
        if (error == asio::error::eof) {
            break;// Connection closed cleanly by peer.
        } else if (error) {
            LUISA_ERROR_WITH_LOCATION("Error {}.", error.message());
        }
        LUISA_INFO("Receive: {}", std::string_view{buffer.data(), len});
    }
}
