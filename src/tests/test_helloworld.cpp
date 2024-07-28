#include <iostream>
#include <chrono>
#include <numeric>

#include <luisa/core/clock.h>
#include <luisa/core/fiber.h>
#include <luisa/core/dynamic_module.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/context.h>
#include <luisa/ast/interface.h>
#include <luisa/dsl/syntax.h>

#include "../backends/common/c_codegen/codegen_utils.h"
using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context ctx(argv[0]);
    Device device = ctx.create_default_device();    
}
