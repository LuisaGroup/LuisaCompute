// Built ONLY against the pinned historical source by build_legacy_exporter.py.
// The legacy examples and lowering are not compiled into the current library.
#define main legacy_unused_benchmark_main
#ifdef LUISA_LEGACY_SIZED
#include "legacy_tile_bench_parameterized.cpp"
#else
#include "examples/compute/tile_bench.cpp"
#endif
#undef main
#include <luisa/ast/ast2json.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <charconv>

int main(int argc, char *argv[]) {
    constexpr auto sized =
#ifdef LUISA_LEGACY_SIZED
        true;
#else
        false;
#endif
    if (argc != (sized ? 6 : 3)) {
        std::cerr << "emit_legacy_tile <operation> <output-prefix>\n";
        std::cerr << "emit_legacy_tile_sized <operation> <output-prefix> M N K\n";
        return 2;
    }
#ifdef LUISA_LEGACY_SIZED
    auto parse = [](const char *text) {
        auto input = std::string_view{text};
        int value{};
        auto result = std::from_chars(input.data(), input.data() + input.size(), value);
        return result.ec == std::errc{} && result.ptr == input.data() + input.size() && value > 0 && value <= 16384 ? value : 0;
    };
    auto m = parse(argv[3]), n = parse(argv[4]), k = parse(argv[5]);
    if (!m || !n || !k || int64_t{m} * n > (1ll << 24) || int64_t{m} * k > (1ll << 24) || int64_t{n} * k > (1ll << 24)) { return 2; }
    legacy_shape = {m, n, k};
#endif
    luisa::log_level_error();
    auto op = luisa::string_view{argv[1]};
    auto path = std::filesystem::path{std::string{argv[2]} + ".ast.json"};
    if (std::filesystem::exists(path)) { return 2; }
    auto emit = [&](auto function) {
        auto captured = luisa::compute::tile::jit(function).compile();
        auto config = luisa::compute::TileToKernelConfig{
            .use_tensor = false, .use_pipeline = true, .pipeline_use_async_copy = false, .pipeline_copy_warps = 0u};
        auto lowered = luisa::compute::tile_to_kernel(captured.function(), config);
        auto json = luisa::compute::to_json(luisa::compute::Function{lowered.function.get()});
        std::ofstream output{path};
        output << json;
        if (!output) { return 2; }
        auto block = luisa::compute::Function{lowered.function.get()}.block_size();
        std::cout << "{\"operation\":\"" << op << "\",\"dispatch\":[" << lowered.dispatch_size.x << ',' << lowered.dispatch_size.y
                  << "],\"block\":[" << block.x << ',' << block.y << ',' << block.z << ']'
                  << ",\"dimensions_parameterized\":" << (sized ? "true" : "false")
                  << ",\"use_tensor\":false,\"use_pipeline\":true,\"async_copy\":false,\"copy_warps\":0}\n";
        return 0;
    };
    if (op == "copy") { return emit(bench_copy); }
    if (op == "add") { return emit(bench_add); }
    if (op == "saxpy") { return emit(bench_saxpy); }
    if (op == "clamp") { return emit(bench_clamp); }
    if (op == "exp") { return emit(bench_unary_op<0>); }
    if (op == "rmsnorm") { return emit(bench_rms_norm); }
    if (op == "sum") { return emit(bench_reduce_op<0>); }
    if (op == "max") { return emit(bench_reduce_op<1>); }
    if (op == "min") { return emit(bench_reduce_op<2>); }
    if (op == "abssum") { return emit(bench_reduce_op<3>); }
    if (op == "absmax") { return emit(bench_reduce_op<4>); }
    if (op == "cumsum") { return emit(bench_scan); }
    if (op == "cumsum1d") { return emit(bench_scan_1d); }
    if (op == "cummax") { return emit(bench_cummax); }
    if (op == "transpose") { return emit(bench_transpose); }
    if (op == "gemm") { return emit(bench_gemm_4096); }
    if (op == "gemm_fp16") { return emit(bench_gemm); }
    return 2;
}
