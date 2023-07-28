#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/printer.h>

using namespace luisa;
using namespace luisa::compute;

struct MyStruct {
    float2 a;
    uint2 b;
};
LUISA_STRUCT(MyStruct, a, b) {};
int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);
    Printer printer{device};

    Kernel2D kernel = [&]() noexcept {
        UInt2 coord = dispatch_id().xy();
        $if (coord.x == coord.y) {
            Float2 v = make_float2(coord) / make_float2(dispatch_size().xy());
            Var<MyStruct> s;
            s.a = v;
            s.b = coord;
            printer.info_with_location("s = {}", s);
        };
    };
    Shader2D<> shader = device.compile(kernel);
    Stream stream = device.create_stream();
    stream << printer.reset()
           << shader().dispatch(128u, 128u);
    stream << printer.retrieve()
           << synchronize();
}
