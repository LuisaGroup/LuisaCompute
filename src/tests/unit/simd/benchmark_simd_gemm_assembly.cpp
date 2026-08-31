// Emits the optimized host assembly for the exact DSL kernel used by
// benchmark_simd_gemm. This is a diagnostic benchmark, not a production
// assembly provider.

#include "benchmark_simd_gemm_kernel.h"

#include "simd_compiler.h"

#include <charconv>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>

using namespace luisa::compute;

int main(int argc, char *argv[]) {
    if (argc != 2 || argv[1] == nullptr) {
        std::cerr << "Usage: " << (argc > 0 ? argv[0] : "benchmark")
                  << " <simd-width>\n";
        return 1;
    }
    auto text = std::string_view{argv[1]};
    auto width = uint32_t{0u};
    auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), width);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != text.data() + text.size() ||
        (width != 1u && width != 2u && width != 4u &&
         width != 8u && width != 16u)) {
        std::cerr << "SIMD width must be 1, 2, 4, 8, or 16\n";
        return 1;
    }

    auto kernel = simd::benchmark::make_gemm_kernel();
    auto compiled = simd::compile_simd_kernel(
        kernel.function()->function(), width,
        "luisa_simd_gemm_w" + std::to_string(width),
        false, true);
    if (!compiled.succeeded() || compiled.assembly.empty()) {
        for (auto &&diagnostic : compiled.diagnostics) {
            std::cerr << diagnostic << '\n';
        }
        return 2;
    }
    std::cout << compiled.assembly;
    return 0;
}
