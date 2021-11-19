//
// Created by Mike Smith on 2021/9/17.
//

#include <network/render_server.h>

using namespace luisa;
using namespace luisa::compute;

int main() {
    auto server = RenderServer::create(12345u, 23456u);
    server->run();
}
