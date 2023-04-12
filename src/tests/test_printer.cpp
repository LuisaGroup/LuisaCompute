//
// Created by Mike Smith on 2022/11/26.
//

#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <dsl/syntax.h>
#include <dsl/sugar.h>
#include <dsl/printer.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);
    Printer printer{device};

    Kernel2D kernel = [&]() noexcept {
        UInt2 coord = dispatch_id().xy();
        $if(coord.x == coord.y) {
            Float2 v = make_float2(coord) / make_float2(dispatch_size().xy());
            printer.info_with_location("v = ({}, {})", v.x, v.y);
        };
    };
    Shader2D<> shader = device.compile(kernel);
    Stream stream = device.create_stream();
    stream << printer.reset()
           << shader().dispatch(128u, 128u)
           << printer.retrieve()
           << synchronize();
}
