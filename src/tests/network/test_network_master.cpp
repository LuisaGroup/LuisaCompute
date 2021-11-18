//
// Created by Mike Smith on 2021/9/17.
//

#include <ctime>
#include <memory>
#include <span>
#include <queue>
#include <mutex>
#include <thread>
#include <deque>
#include <condition_variable>

#include <asio.hpp>

#include <core/basic_types.h>
#include <core/allocator.h>
#include <core/logging.h>

#include <network/render_server.h>

using namespace luisa;
using namespace luisa::compute;

int main() {
    auto server = RenderServer::create(12345u, 23456u);
    server->run();
}
