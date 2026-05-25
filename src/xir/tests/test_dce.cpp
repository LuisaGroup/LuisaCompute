#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

int main(int argc, char *argv[]) {

    using namespace luisa;
    using namespace luisa::compute;

    Kernel1D kernel = [](BufferUInt buffer, UInt x) noexcept {
        $for (i, 10u) {
            $switch (x) {
                $case (0u) {
                    buffer->write(0u, 1u);
                    x += 1u;
                };
                $case (1u) {
                    buffer->write(0u, 2u);
                    x += 2u;
                };
                $default {
                    $unreachable;
                };
            };
        };
    };

    auto context = Context{luisa::current_executable_path()};
    auto device = context.create_device("cuda");
    auto shader = device.compile(kernel);
}
