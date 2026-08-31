#include <charconv>
#include <filesystem>
#include <fstream>
#include <string_view>

#include <luisa/core/logging.h>
#include <luisa/runtime/rhi/resource.h>

#include "metal_air_pipeline.h"
#include "metal_xir_pipeline.h"
#include "metal4_ios_path_tracing_kernel.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::metal;

namespace {

[[nodiscard]] MetalAIRVersion parse_version(std::string_view text) noexcept {
    MetalAIRVersion version{};
    uint32_t *components[] = {
        &version.major, &version.minor, &version.patch};
    for (auto component = 0u; component < 3u; component++) {
        auto separator = text.find('.');
        auto token = text.substr(0u, separator);
        if (token.empty()) { return {}; }
        auto [end, error] = std::from_chars(
            token.data(), token.data() + token.size(), *components[component]);
        if (error != std::errc{} || end != token.data() + token.size()) {
            return {};
        }
        if (separator == std::string_view::npos) { break; }
        text.remove_prefix(separator + 1u);
    }
    return version;
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 4) {
        LUISA_INFO(
            "Usage: {} <output.metallib> [iOS deployment version] [iOS SDK version]",
            argv[0]);
        return 2;
    }
    auto deployment = parse_version(argc >= 3 ? argv[2] : "26.0");
    auto sdk = parse_version(argc >= 4 ? argv[3] : "26.4");
    if (deployment.major == 0u || sdk.major == 0u) {
        LUISA_WARNING("Invalid iOS deployment or SDK version.");
        return 2;
    }

    auto kernel = make_ios_path_tracing_kernel();
    auto option = ShaderOption{
        .enable_cache = false,
        .enable_fast_math = true,
        .enable_debug_info = false,
        .compile_only = true,
        .name = "luisa_ios_path_tracing"};
    auto module = metal_translate_ast_to_xir(
        kernel.function()->function(), option);
    auto target = metal_air_target_for_ios(deployment, sdk);
    auto air = metal_codegen_air(*module, option, target);
    constexpr auto expected_root_argument_size = 32u;
    if (air.library.empty() ||
        air.root_argument_size != expected_root_argument_size) {
        LUISA_WARNING(
            "Unexpected iOS path tracing AIR output: metallib={} bytes, root={} bytes (expected {}).",
            air.library.size(), air.root_argument_size,
            expected_root_argument_size);
        return 1;
    }

    auto output_path = std::filesystem::path{argv[1]};
    std::error_code error;
    if (auto parent = output_path.parent_path(); !parent.empty()) {
        std::filesystem::create_directories(parent, error);
        if (error) {
            LUISA_WARNING("Failed to create '{}': {}.",
                          parent.string(), error.message());
            return 1;
        }
    }
    std::ofstream output{output_path, std::ios::binary};
    output.write(reinterpret_cast<const char *>(air.library.data()),
                 static_cast<std::streamsize>(air.library.size()));
    if (!output) {
        LUISA_WARNING("Failed to write '{}'.", output_path.string());
        return 1;
    }
    LUISA_INFO(
        "Generated iOS AIR path tracer: '{}' ({} bytes, root={} bytes, block=8x8x1, target=iOS {}.{}.{}, SDK {}.{}).",
        output_path.string(), air.library.size(), air.root_argument_size,
        deployment.major, deployment.minor, deployment.patch,
        sdk.major, sdk.minor);
    return 0;
}
