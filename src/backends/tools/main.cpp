// lc_compile_builtin: AOT-compile a DSL kernel to .dxil (DX) or .spv (VK).
//
// Creates a headless device, defines a kernel, and uses the compile_only
// shader option to save the compiled bytecode to a destination file.
//
// Usage:
//   lc-compile-builtin <backend> <destination> [<kernel_name>]
//   e.g.: lc-compile-builtin dx ./output.dxil
//         lc-compile-builtin vk ./output.spv
//
// The <backend> selects the device plugin ("dx", "vk", ...).
// The <destination> is an absolute or relative path for the output bytecode.
// The optional <kernel_name> overrides the default "test_builtin" name printed
// in codegen dumps.

#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

namespace {

constexpr auto kDefaultKernelName = "test_builtin";

[[nodiscard]] std::filesystem::path
resolve_destination(std::string_view dest) {
    auto p = std::filesystem::path{dest};
    if (p.is_absolute()) { return p; }
    return std::filesystem::absolute(p);
}

void print_usage(const char *exe) {
    LUISA_INFO(
        "Usage: {} <backend> <destination> [<kernel_name>]\n"
        "  backend     - device plugin name (e.g. dx, vk)\n"
        "  destination - output file path for the compiled bytecode\n"
        "  kernel_name - optional shader name (default: {})",
        exe, kDefaultKernelName);
}

} // namespace

int main(int argc, char *argv[]) {
    if (argc < 3) {
        print_usage(argc > 0 ? argv[0] : "lc-compile-builtin");
        return EXIT_FAILURE;
    }

    auto backend = std::string_view{argv[1]};
    auto dest = resolve_destination(argv[2]);
    auto kernel_name = argc > 3 ? std::string{argv[3]} : kDefaultKernelName;

    // Create a headless device — no window, no display extensions.
    const char *exe = (argc > 0 && argv[0]) ? argv[0] : "";
    Context context{exe};
    DeviceConfig config{};
    config.headless = true;
    Device device = context.create_device(backend, &config);
    LUISA_INFO("Created headless device: backend={}", backend);

    // Ensure the output directory exists.
    if (auto parent = dest.parent_path(); !parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }

    auto filename = luisa::string{dest.string()};
    LUISA_INFO("Compiling kernel '{}' -> {}", kernel_name, filename);

    // Use compile_only to save the bytecode without creating a shader object.
    auto kernel = Kernel1D{[](BufferVar<float> buffer) noexcept {
        set_block_size(64u, 1u, 1u);
        buffer.write(dispatch_id().x,
                      cast<float>(dispatch_id().x) * 2.0f);
    }};
    ShaderOption option{
        .compile_only = true,
        .name = std::move(filename),
    };
    auto info = device.compile<1>(kernel, option);
    LUISA_ASSERT(!info.valid(),
                  "compile_only should return an invalid handle.");

    if (std::filesystem::is_regular_file(dest)) {
        LUISA_INFO("Bytecode saved to {} ({} bytes)",
                   dest.string(),
                   std::filesystem::file_size(dest));
        return EXIT_SUCCESS;
    }
    LUISA_ERROR("Bytecode was not written to {}", dest.string());
    return EXIT_FAILURE;
}
