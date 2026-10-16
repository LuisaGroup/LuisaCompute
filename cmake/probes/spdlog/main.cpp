#if defined(LUISA_COMPUTE_SYSTEM_DEPENDENCY_MISSING)
#error "The system spdlog package is unavailable"
#else

#include <vector>

#include <luisa/core/stl/format.h>
#include <spdlog/spdlog.h>

template<size_t N>
void probe_half_formatters() {
    // Exercise the public formatter contract rather than merely proving that
    // spdlog's imported target can format primitive types. Some packaged
    // spdlog/fmt combinations reject Luisa's half-vector formatter even though
    // a trivial spdlog program compiles.
    auto vector_text = luisa::to_string(luisa::Vector<luisa::half, N>{});
    auto matrix_text = luisa::to_string(luisa::Matrix<luisa::half, N>{});
    spdlog::info("LuisaCompute system dependency probe: {} {}",
                 vector_text, matrix_text);
}

int main() {
    probe_half_formatters<2u>();
    probe_half_formatters<3u>();
    probe_half_formatters<4u>();
    auto range_text = luisa::format(
        FMT_STRING("{}"), std::vector<size_t>{1u, 2u, 3u});
    spdlog::info("LuisaCompute range formatter probe: {}", range_text);
    return 0;
}

#endif
