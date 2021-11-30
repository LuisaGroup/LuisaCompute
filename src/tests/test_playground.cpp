//
// Created by Mike Smith on 2021/9/3.
//

#include <iostream>
#include <nlohmann/json.hpp>
#include <ast/op.h>
#include <core/logging.h>

int main() {

    using namespace luisa;
    using namespace luisa::compute;

    CallOpSet ops;
    ops.mark(CallOp::BINDLESS_BUFFER_READ);
    ops.mark(CallOp::SELECT);
    ops.mark(CallOp::ACOS);

    auto properties = nlohmann::json::object();
    LUISA_INFO("is_object: {}", properties.is_object());
    LUISA_INFO("Properties: {}", properties.dump());

    auto json = nlohmann::json::parse("{}");
    LUISA_INFO("Index: {}", json.value("index", 0));
    for (auto iter = ops.begin(); iter != ops.end(); iter++) {
        LUISA_INFO("Op: {}", to_underlying(*iter));
    }
}
