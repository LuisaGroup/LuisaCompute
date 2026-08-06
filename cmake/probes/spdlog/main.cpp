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
    return 0;
}
