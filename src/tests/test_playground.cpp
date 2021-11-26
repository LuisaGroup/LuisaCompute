//
// Created by Mike Smith on 2021/9/3.
//

#include <iostream>

#include <ast/op.h>
#include <core/logging.h>

int main() {

    using namespace luisa;
    using namespace luisa::compute;

    CallOpSet ops;
    ops.mark(CallOp::BINDLESS_BUFFER_READ);
    ops.mark(CallOp::SELECT);
    ops.mark(CallOp::ACOS);

    for (auto iter = ops.begin(); iter != ops.end(); iter++) {
        LUISA_INFO("Op: {}", to_underlying(*iter));
    }
}
