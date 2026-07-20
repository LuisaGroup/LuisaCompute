// Hello World example demonstrating basic buffer operations
// and half-precision floating point support.
// Note: no offline/reference image comparison is added because this example does not render an image.

#include <luisa/luisa-compute.h>

#include <cmath>

using namespace luisa;
using namespace luisa::compute;

// Test structure with half-precision float
struct Test1 {
    half a;
    uint16_t b;
};
LUISA_STRUCT(Test1, a, b) {};

int main(int argc, char *argv[]) {
    auto executable = argc > 0 && argv != nullptr && argv[0] != nullptr ? argv[0] : "";
    if (argc <= 1 || argv == nullptr || argv[1] == nullptr || argv[1][0] == '\0') {
        LUISA_INFO("Usage: {} <backend>.", executable);
        return 1;
    }

    // Create context and device
    Context ctx(executable);
    Device device = ctx.create_device(argv[1]);

    // Create a buffer to store Test1 structure
    auto bf = device.create_buffer<Test1>(1);
    Test1 t{};

    // Compile a kernel that writes to the buffer
    auto s = device.compile<1>([&]() {
        Var<Test1> tt;
        tt.a = 1.5f;
        tt.b = 66;
        tt.b = tt.b + tt.b;
        bf->write(0, tt);
    });

    // Create stream and dispatch kernel
    auto stream = device.create_stream();
    stream << s().dispatch(1) << bf.copy_to(luisa::span{&t, 1}) << synchronize();

    // Output and validate the result. This keeps the example readable while
    // making its mirrored test_helloworld target a real conformance check.
    LUISA_INFO("{}, {}", static_cast<float>(t.a), static_cast<int>(t.b));
    constexpr auto expected_a = 1.5f;
    constexpr auto expected_b = 132u;
    if (std::abs(static_cast<float>(t.a) - expected_a) > 1.0e-3f ||
        t.b != expected_b) {
        LUISA_WARNING("Hello-world result mismatch: expected ({}, {}), got ({}, {}).",
                      expected_a, expected_b, static_cast<float>(t.a), t.b);
        return 1;
    }
    return 0;
}
