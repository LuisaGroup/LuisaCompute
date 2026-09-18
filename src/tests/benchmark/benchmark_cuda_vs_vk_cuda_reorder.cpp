// Comparison benchmark: "small independent dispatch batch, NVIDIA GPU" along
// two different submission routes.
//
//   A  cuda backend, dispatched directly .................. NO reordering
//      The batch is recorded as ordinary DSL dispatches on a `cuda` device. The
//      CUDA backend has no command-reordering pass at all (see
//      src/backends/cuda/cuda_command_encoder.cpp: every command becomes one
//      cuLaunchKernel on the stream's single CUstream), so the dispatches are
//      strictly serialized by CUDA stream semantics. This is the baseline the
//      Vulkan runtime is supposed to beat.
//
//   B  vk runtime + imported CUDA kernels, reorder OFF .... baseline, 1 layer/cmd
//      The same DSL kernel is compiled on a *paired* cuda device (LUID-matched,
//      see VkCudaInterop::cuda_device_index()) and imported into the Vulkan
//      device with VkCudaInterop::create_cuda_kernel(), i.e. the kernels are
//      real CUDA modules launched inside a Vulkan command buffer through
//      VK_NV_cuda_kernel_launch. With reordering disabled every command keeps a
//      layer of its own, so the backend emits a barrier per command.
//
//   C  vk runtime + imported CUDA kernels, reorder ON ..... the candidate
//      Identical commands, but the reorder pass merges the hazard-free batch
//      into as few barrier-free layers as it can (ideally one). This is the
//      configuration that should beat A: A pays a serialized launch per
//      dispatch, C pays one barrier for the whole batch.
//
//   D  vk runtime + Vulkan-compiled kernel, reorder ON .... diagnostic
//      The same DSL source compiled by the Vulkan backend itself (so the launch
//      is a native vkCmdDispatch, not a CUDA module) and reordered. B/C vs. D
//      separates "cost of the vk<->CUDA launch route" from "cost of ordering".
//      Note D is not bit-comparable to A: a different code generator computes
//      sin/cos/exp/... slightly differently, so only a relative tolerance is
//      checked there.
//
// Batch shape is the one from benchmark_command_reorder.cpp: `dispatches`
// independent dispatches, each reading the whole of one shared read-only input
// buffer and writing its own `threads`-wide sub-range of one output buffer
// (`Buffer::view().subview(...)`), so the batch has no real hazards. Each thread
// runs a long serial chain of transcendentals plus a block-local shared-memory
// reduction, i.e. a batch costs milliseconds of *device* time while host-side
// recording stays negligible, which keeps the comparison focused on ordering
// overhead.
//
// Usage
// -----
//   benchmark_cuda_vs_vk_cuda_reorder vk [mode] [dispatches] [threads] [iters] [rounds] [verbose]
//
// Defaults: mode=0, dispatches=16, threads=256, iters=4096, rounds=5, verbose=0.
// The first argument must be "vk" (the route under test); the cuda device used
// for compiling/importing is created automatically from the same context.
//
// mode bit 0 selects the destination pattern, bit 1 the input footprint:
//   mode 0 (disjoint out, shared in): dispatch j writes sub-range j and every
//     dispatch reads the whole input range -> no real hazards, ideal layer 1.
//   mode 1 (rotate out, shared in): dispatch j writes sub-range (j % 16), i.e.
//     genuine write-after-write chains, ideal layer count ceil(dispatches / 16).
//   mode 2 / 3: same as 0 / 1, except that dispatch j reads its own input
//     sub-range instead of the shared one. This is the control that isolates the
//     effect of VkCudaInterop's fail-closed state isolation (see the note below):
//     with per-dispatch inputs the imported CUDA launches stop colliding with
//     each other and can merge into one layer as well.
//
// A note on the vk + CUDA route and reordering
// -------------------------------------------
// `CudaKernelLaunchCommand::requires_resource_state_isolation()` is true: the
// backend cannot know what an opaque CUDA module touches (it gets raw device
// addresses), so the reorder pass marks EVERY resource argument of an imported
// kernel - including one declared READ - as an exclusive access over its range.
// Consequence in mode 0/1: all dispatches share the read-only input buffer, so
// each of them "writes" the same range and the batch is pushed back to one layer
// per command, exactly the serialized shape reordering was meant to avoid. Mode
// 2/3 removes that collision; the gap between the mode 0 and the mode 2 numbers
// is what that fail-closed isolation costs.
//
// Examples:
//   xmake run benchmark_cuda_vs_vk_cuda_reorder vk
//   xmake run benchmark_cuda_vs_vk_cuda_reorder vk 0 64 256 4096 7
//   xmake run benchmark_cuda_vs_vk_cuda_reorder vk 2   # private inputs
//   xmake run benchmark_cuda_vs_vk_cuda_reorder vk 0 16 256 1 9 1
//     ^ iters=1 removes almost all device work, so what is left is launch and
//       submission overhead only; verbose=1 makes the Vulkan backend log
//       "command reorder: N commands -> M layers (reorder on)" for every batch,
//       which is the first thing to look at when C does not beat A.
//
// Reading the result
// ------------------
// The verdict line compares medians: gain = median(A) / median(C); anything below
// 1.05x is reported as "not meaningfully faster" together with the diagnosis of
// why (a 1-2% spread between two GPU-saturating routes is a tie). Groups
// alternate every round so that clock ramping and thermal drift hit them equally.
// The "submit" column is the host-side cost of recording + queueing the batch
// (measured before the wait), which tells a device-side ordering effect apart from
// a host-side one.
//
// Reference outcome (RTX 5070 Ti Laptop, driver 596.13, CUDA 13.2, release build,
// 16 dispatches x 256 threads x 4096 iterations, medians in ms):
//
//                      A cuda     B vk+cuda    C vk+cuda    D vk native   A/C
//   mode 0, shared in  3.972      4.007        3.957        0.541         1.00x
//   mode 2, private in 3.969      4.010        0.318        0.535        12.48x
//   mode 0, iters=1    0.213      0.173        0.111        0.076         1.91x
//   mode 2, iters=1    0.179      0.163        0.069        0.074         2.58x
//
// i.e. the reordered vk+CUDA route only wins once the dispatches stop sharing a
// resource range (see the isolation note above); the verbose layer counts confirm
// it directly - mode 0 logs "16 commands -> 16 layers (reorder on)" for the CUDA
// launches and "16 commands -> 1 layers" for the native ones, mode 2 logs "-> 1
// layers" for both.
//
// Correctness
// -----------
// B and C must reproduce A bit for bit (same CUDA module, same arguments, only
// the barrier structure differs), and every dispatch must own its sub-range.
// Failure there means the merged layers are racing, and any timing win is
// meaningless.
//
// Caveats
// -------
// * Shaders are compiled with ShaderOption{.enable_cache = false}, exactly like
//   benchmark_command_reorder.cpp does: every run compiles from scratch, so no
//   on-disk cache traffic can perturb the measured groups, and a cache entry written
//   by a differently configured build is out of the picture. The cache round trip
//   itself is covered by test_shader_cache_round_trip (and, for the imported
//   CUDA kernels specifically, by test_vk_cuda_kernel_launch - both compile with the
//   cache on).
// * The two routes do not run on identical memory: A uses cudaMalloc'd buffers,
//   B/C/D use Vulkan buffers imported into CUDA (VkCudaInterop::create_buffer),
//   which is required for vkCmdCudaLaunchKernelNV to pass raw device addresses.
//   That is part of what "the vk + CUDA route" costs, so it is measured as-is.
// * Absolute times are only comparable within one route: A/B/C/D do not all run
//   the same machine code. On this GPU the CUDA build of the reference kernel is
//   the cheaper one - serialized it costs ~0.25 ms per dispatch (group A) against
//   ~0.48 ms for the Vulkan build (the reorder-off numbers of
//   benchmark_command_reorder) - which is why the merged CUDA-launch batch (C in
//   mode 2/3) can come out ahead of the merged native batch (D) as well. Use B vs
//   C for "ordering inside the CUDA module", and D as an ordering probe.
#include "ut/ut.hpp"// boost.ut cfg used by test_device.h
#include "test_device.h"
#include <luisa/backends/ext/command_reorder_ext.h>
#include <luisa/backends/ext/vk_cuda_interop.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/syntax.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <sstream>
using namespace luisa;
using namespace luisa::compute;

namespace {

/// Sub-ranges that mode 1 (rotate) cycles over.
constexpr size_t kRotateRanges = 16u;
/// Threads per block for the heavy kernel; also the size of the block-local
/// shared-memory reduction.
constexpr uint32_t kBlockSize = 32u;
/// Relative deviation the Vulkan-compiled kernel (group D) is allowed to show
/// against the CUDA-compiled reference. Transcendental implementations differ in
/// the last bits and the kernel feeds them back into a recurrence, so a small
/// drift is expected; anything near this bound means broken ordering.
constexpr float kNativeTolerance = 2e-2f;

/// Destination pattern of the batch (mode bit 0).
constexpr size_t kOutputRotate = 1u;
/// Input footprint (mode bit 1): every dispatch reads its own input sub-range
/// instead of one shared read-only range.
constexpr size_t kPrivateInput = 2u;

[[nodiscard]] luisa::string_view mode_name(size_t mode) {
    static constexpr std::array<luisa::string_view, 4> kNames{
        "disjoint outputs + shared input",
        "rotated outputs + shared input",
        "disjoint outputs + private inputs",
        "rotated outputs + private inputs"};
    return kNames[mode & 3u];
}

struct Options {
    size_t mode{0u};
    size_t dispatches{16u};
    size_t threads{256u};
    size_t iters{4096u};
    size_t rounds{5u};
};

[[nodiscard]] size_t parse_uint(const char *text, const char *name, size_t minimum) {
    auto input = std::string_view{text};
    size_t value{};
    auto result = std::from_chars(input.data(), input.data() + input.size(), value);
    LUISA_ASSERT(result.ec == std::errc{} && result.ptr == input.data() + input.size() && value >= minimum,
                 "{} must be an integer >= {}, got \"{}\".", name, minimum, text);
    return value;
}
[[nodiscard]] size_t parse_positive(const char *text, const char *name) {
    return parse_uint(text, name, 1u);
}

struct Statistics {
    double min{};
    double median{};
    double mean{};
    double max{};
};

[[nodiscard]] Statistics summarize(luisa::vector<double> samples) {
    LUISA_ASSERT(!samples.empty(), "No samples to summarize.");
    std::sort(samples.begin(), samples.end());
    auto sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    return {samples.front(),
            samples[samples.size() / 2u],
            sum / static_cast<double>(samples.size()),
            samples.back()};
}

/// Emit one `dispatches`-long batch. `emit(list, src_view, dst_range_index, j)`
/// appends the j-th dispatch; the route-specific lambdas in main() differ only in
/// which shader handle and which buffers they bind, so the batch shape is
/// guaranteed to be identical for every group.
template<typename Emit>
[[nodiscard]] CommandList build_batch(const Options &opt, const Buffer<float> &src,
                                      Emit &&emit) {
    CommandList list;
    list.reserve(opt.dispatches, 0u);
    for (auto j = size_t{0}; j < opt.dispatches; j++) {
        auto dst_range = (opt.mode & kOutputRotate) != 0u ? j % kRotateRanges : j;
        // Shared input: one whole read-only range, so the batch has no input-side
        // hazard at all. Private input: dispatch j gets its own sub-range.
        auto in_view = (opt.mode & kPrivateInput) != 0u ? src.view().subview(j * opt.threads, opt.threads) : src.view();
        emit(list, in_view, dst_range, j);
    }
    return list;
}

/// Batch wall-clock samples (submission + wait) and the host-side part of each
/// batch (recording + queueing, measured before the wait).
struct GroupTiming {
    luisa::vector<double> batch;
    luisa::vector<double> submit;
};

/// Submit `rounds` batches built by `build_batch`. `warmup_rounds` extra batches
/// run first so that first-submission work (pipeline/module setup on the driver
/// side) is not attributed to a group.
template<typename Builder>
[[nodiscard]] GroupTiming measure_group(Stream &stream, Builder &&build_batch,
                                        size_t rounds, size_t warmup_rounds = 0u) {
    for (auto w = size_t{0}; w < warmup_rounds; w++) {
        stream << build_batch().commit() << synchronize();
    }
    GroupTiming timing;
    timing.batch.reserve(rounds);
    timing.submit.reserve(rounds);
    for (auto r = size_t{0}; r < rounds; r++) {
        Clock clock;
        stream << build_batch().commit();
        auto submit = clock.toc();
        stream << synchronize();
        timing.batch.emplace_back(clock.toc());
        timing.submit.emplace_back(submit);
    }
    return timing;
}

/// Append `more` to `samples`.
inline void append_samples(luisa::vector<double> &samples, const luisa::vector<double> &more) {
    samples.insert(samples.end(), more.begin(), more.end());
}

[[nodiscard]] luisa::vector<float4> download(Stream &stream, const Buffer<float4> &dst) {
    luisa::vector<float4> host(dst.size());
    stream << dst.copy_to(luisa::span{host}) << synchronize();
    return host;
}

[[nodiscard]] bool bitwise_equal(luisa::span<const float4> a, luisa::span<const float4> b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

/// Largest relative deviation from `reference`, used for the Vulkan-compiled
/// group whose math results are not expected to be bit-identical.
[[nodiscard]] float max_relative_deviation(luisa::span<const float4> value,
                                           luisa::span<const float4> reference) {
    LUISA_ASSERT(value.size() == reference.size(), "Comparison buffers differ in size.");
    auto worst = 0.0f;
    for (auto i = size_t{0}; i < value.size(); i++) {
        auto components = std::array{&value[i].x, &value[i].y, &value[i].z, &value[i].w};
        auto reference_components = std::array{&reference[i].x, &reference[i].y, &reference[i].z, &reference[i].w};
        for (auto c = size_t{0}; c < components.size(); c++) {
            auto deviation = std::abs(*components[c] - *reference_components[c]) /
                             (1.0f + std::abs(*reference_components[c]));
            worst = std::max(worst, deviation);
        }
    }
    return worst;
}

struct Row {
    luisa::string_view name;
    Statistics batch;
    Statistics submit;
};

void print_row(const Row &row) {
    std::ostringstream cells;
    auto fmt = [&cells](const Statistics &s) {
        cells << std::fixed << std::setprecision(3) << std::setw(10) << s.min
              << std::setw(10) << s.median << std::setw(10) << s.mean << std::setw(10) << s.max;
    };
    fmt(row.batch);
    cells << " |";
    fmt(row.submit);
    LUISA_INFO("{:<38}{}", row.name, cells.str());
}

/// A latency-bound transcendental chain: every iteration feeds the next one, so
/// a single thread only keeps its own dependency chain busy and the dispatch as
/// a whole is a good candidate to overlap with its neighbours. The block-local
/// shared-memory reduction adds real per-block work without introducing any
/// cross-dispatch dependency. Kept identical to the one in
/// benchmark_command_reorder.cpp so the numbers of the two benchmarks relate.
[[nodiscard]] auto heavy_kernel() noexcept {
    return Kernel1D{[](BufferFloat input, BufferFloat4 output,
                       Var<uint> iterations, Var<uint> salt) noexcept {
        set_block_size(kBlockSize, 1u, 1u);
        auto tid = dispatch_id().x;
        auto lane = thread_x();
        Shared<float> scratch{kBlockSize};
        Var<float> x = input.read(tid) + salt.cast<float>() * 0.03125f;
        Var<float> y = x * 0.5f + 1.0f;
        Var<float4> acc = make_float4(0.f);
        for (auto i : dynamic_range(iterations)) {
            auto s = sin(x);
            auto c = cos(y);
            auto p = s * c;
            Var<float> r0 = sqrt(abs(x) + 1.0f);
            acc += make_float4(s, c, p, r0);
            Var<float> t0 = exp(-abs(s));
            Var<float> t1 = log(abs(y) + 1.0f);
            Var<float> t2 = atan2(s, c);
            Var<float> t3 = p * p + i.cast<float>() * 0.000125f;
            acc += make_float4(t0, t1, t2, t3);
            x = x * 0.5f + s;
            y = y * 0.5f + c;
        }
        scratch[lane] = length(acc);
        sync_block();
        Var<float> reduction = 0.f;
        for (auto k = 0u; k < kBlockSize; k++) {
            reduction += scratch[k];
        }
        output.write(tid, acc + reduction);
    }};
}

}// namespace

int main(int argc, char *argv[]) {
    // Keep the backend check on raw argv (strcmp): benchmark_command_reorder.cpp
    // documents that string_view comparisons over argv get miscompiled in unity
    // builds of this TU family.
    if (argc < 2 || argv[1] == nullptr || std::strcmp(argv[1], "vk") != 0) {
        LUISA_ERROR_WITH_LOCATION(
            "Usage: {} vk [mode] [dispatches] [threads] [iters] [rounds] [verbose] "
            "(this benchmark always needs the vk backend plus the cuda backend).",
            argc > 0 ? argv[0] : "benchmark_cuda_vs_vk_cuda_reorder");
        return 1;
    }
    Options opt;
    if (argc > 2) {
        opt.mode = parse_uint(argv[2], "mode", 0u);
    }
    if (argc > 3) {
        opt.dispatches = parse_positive(argv[3], "dispatches");
    }
    if (argc > 4) {
        opt.threads = parse_positive(argv[4], "threads");
    }
    if (argc > 5) {
        opt.iters = parse_positive(argv[5], "iters");
    }
    if (argc > 6) {
        opt.rounds = parse_positive(argv[6], "rounds");
    }
    auto verbose = argc > 7 ? parse_positive(argv[7], "verbose") != 0u : false;
    LUISA_ASSERT(opt.mode <= 3u,
                 "mode bit 0: 0 = disjoint outputs, 1 = outputs rotated over 16 sub-ranges; "
                 "mode bit 1: 0 = one shared read-only input, 1 = per-dispatch input ranges.");
    LUISA_ASSERT(opt.dispatches <= 1024u,
                 "dispatches must be <= 1024 to bound buffer memory and batch size.");
    LUISA_ASSERT(opt.threads % kBlockSize == 0u,
                 "threads must be a multiple of the kernel block size ({}).", kBlockSize);

    if (verbose) {
        log_level_verbose();
    } else {
        log_level_info();
    }

    // ---- devices ----------------------------------------------------------------
    auto [context, device] = test::create_device(argc, argv);
    auto *reorder_ext = device.extension<CommandReorderExt>();
    auto *cuda_ext = device.extension<VkCudaInterop>();
    if (reorder_ext == nullptr || cuda_ext == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "The vk backend must be built with the Vulkan command-reorder pass and "
            "CUDA interop (lc_vk_cuda_interop): CommandReorderExt = {}, VkCudaInterop = {}.",
            reorder_ext != nullptr ? "present" : "missing",
            cuda_ext != nullptr ? "present" : "missing");
        return 1;
    }
    if (!cuda_ext->cuda_kernel_launch_supported()) {
        LUISA_ERROR_WITH_LOCATION(
            "VK_NV_cuda_kernel_launch is not enabled on this Vulkan device, so the "
            "vk + CUDA-kernel route cannot be measured.");
        return 1;
    }
    auto cuda_index = cuda_ext->cuda_device_index();
    if (cuda_index < 0) {
        LUISA_ERROR_WITH_LOCATION("No CUDA device matches this Vulkan device (LUID lookup failed).");
        return 1;
    }
    auto cuda_backend_installed = false;
    for (auto &&name : context.installed_backends()) {
        if (name == "cuda") { cuda_backend_installed = true; }
    }
    if (!cuda_backend_installed) {
        LUISA_ERROR_WITH_LOCATION("The cuda backend is not installed; nothing to compare against.");
        return 1;
    }
    // Pair the cuda device with the same physical GPU as the Vulkan one: the
    // compiled module image targets that GPU's compute capability, and
    // VK_NV_cuda_kernel_launch requires the two devices to be the same adapter.
    DeviceConfig cuda_config{.device_index = static_cast<size_t>(cuda_index)};
    auto cuda_device = context.create_device("cuda", &cuda_config);
    if (!cuda_device) {
        LUISA_ERROR_WITH_LOCATION("Failed to create the cuda device at index {}.", cuda_index);
        return 1;
    }
    auto stream = device.create_stream();
    auto cuda_stream = cuda_device.create_stream();
    LUISA_INFO("Routes: A = {} backend device #{}; B/C/D = {} backend device #{} "
               "(same physical adapter).",
               cuda_device.backend_name(), cuda_index,
               device.backend_name(), 0u);

    // Probe the runtime switch: a process-wide override would silently turn C
    // into a second copy of B.
    reorder_ext->set_command_reorder_enabled(true);
    if (!reorder_ext->command_reorder_enabled()) {
        LUISA_WARNING("Command reordering is forced off by LUISA_DISABLE_COMMAND_REORDER, "
                      "which overrides the runtime switch: groups B and C would measure the "
                      "same serialized baseline.");
    }

    // ---- buffers ------------------------------------------------------------------
    // Input: read-only and shared by every dispatch of the batch, so it never
    // splits the batch into separate layers. Output: one disjoint sub-range per
    // dispatch (mode 0) or sixteen sub-ranges written round-robin (mode 1).
    auto range_count = (opt.mode & kOutputRotate) != 0u ? kRotateRanges : opt.dispatches;
    auto src_size = (opt.mode & kPrivateInput) != 0u ? opt.dispatches * opt.threads : opt.threads;
    auto dst_size = range_count * opt.threads;
    luisa::vector<float> host_src(src_size);
    for (auto i = size_t{0}; i < src_size; i++) {
        host_src[i] = std::sin(static_cast<float>(i) * 0.017f) * 2.f + 0.75f;
    }
    // A: plain CUDA memory.
    auto cuda_src = cuda_device.create_buffer<float>(src_size);
    auto cuda_dst = cuda_device.create_buffer<float4>(dst_size);
    // B/C/D: Vulkan memory imported into CUDA. Required for the imported kernels,
    // which take raw device addresses; harmless (and desirable) for group D, so
    // all Vulkan groups run on the same memory.
    auto vk_src = cuda_ext->create_buffer<float>(src_size);
    auto vk_dst = cuda_ext->create_buffer<float4>(dst_size);
    {
        luisa::vector<float4> zero_dst(dst_size, float4{0.f, 0.f, 0.f, 0.f});
        cuda_stream << cuda_src.copy_from(luisa::span{host_src})
                    << cuda_dst.copy_from(luisa::span{zero_dst})
                    << synchronize();
        stream << vk_src.copy_from(luisa::span{host_src})
               << vk_dst.copy_from(luisa::span{zero_dst})
               << synchronize();
    }

    // ---- shaders ------------------------------------------------------------------
    // Cache bypass, see the "Caveats" block at the top of this file.
    auto cuda_shader = cuda_device.compile(heavy_kernel(), ShaderOption{.enable_cache = false});
    if (!cuda_shader) {
        LUISA_ERROR_WITH_LOCATION("Compiling the DSL kernel on the cuda backend failed.");
        return 1;
    }
    auto vk_cuda_shader = cuda_ext->create_cuda_kernel(cuda_shader);
    if (!vk_cuda_shader) {
        LUISA_ERROR_WITH_LOCATION(
            "Importing the CUDA kernel into the Vulkan runtime failed "
            "(VkCudaInterop::create_cuda_kernel).");
        return 1;
    }
    // Imported CUDA kernel with the read/write intent baked into the signature:
    // the reorder pass needs it to keep the shared input buffer non-exclusive and
    // the per-dispatch output ranges exclusive, exactly like the DSL does.
    auto vk_cuda_kernel = vk_cuda_shader.kernel<
        vk_cuda_interop::CudaArg<Buffer<float>, Usage::READ>,
        vk_cuda_interop::CudaArg<Buffer<float4>, Usage::WRITE>,
        uint32_t, uint32_t>();
    auto vk_shader = device.compile(heavy_kernel(), ShaderOption{.enable_cache = false});
    if (!vk_shader) {
        LUISA_ERROR_WITH_LOCATION("Compiling the DSL kernel on the vk backend failed.");
        return 1;
    }
    LUISA_INFO("Imported CUDA kernel: handle {}, compiled block size {} (dispatch uses the "
               "exact thread count, so grid = ceil(threads / block)).",
               vk_cuda_shader.handle(), vk_cuda_shader.block_size().x);

    auto iters = static_cast<uint32_t>(opt.iters);
    auto threads = static_cast<uint32_t>(opt.threads);
    // Route A: DSL dispatch on the cuda backend (one cuLaunchKernel per command,
    // strictly serialized by the single CUstream behind it).
    auto build_cuda_batch = [&]() {
        return build_batch(opt, cuda_src, [&](CommandList &list, const BufferView<float> &in_view, size_t dst_range, size_t j) {
            auto dst_view = cuda_dst.view().subview(dst_range * opt.threads, opt.threads);
            list << cuda_shader(in_view, dst_view, iters,
                                static_cast<uint32_t>(j))
                        .dispatch(threads);
        });
    };
    // Route B/C: imported CUDA kernel launched inside the Vulkan command buffer.
    auto build_vk_cuda_batch = [&]() {
        return build_batch(opt, vk_src, [&](CommandList &list, const BufferView<float> &in_view, size_t dst_range, size_t j) {
            auto dst_view = vk_dst.view().subview(dst_range * opt.threads, opt.threads);
            list << vk_cuda_kernel(in_view, dst_view, iters,
                                   static_cast<uint32_t>(j))
                        .dispatch(threads);
        });
    };
    // Route D: same DSL source, compiled by the Vulkan backend.
    auto build_vk_native_batch = [&]() {
        return build_batch(opt, vk_src, [&](CommandList &list, const BufferView<float> &in_view, size_t dst_range, size_t j) {
            auto dst_view = vk_dst.view().subview(dst_range * opt.threads, opt.threads);
            list << vk_shader(in_view, dst_view, iters,
                              static_cast<uint32_t>(j))
                        .dispatch(threads);
        });
    };

    LUISA_INFO("Batch shape: mode {} ({}), {} dispatches x {} threads x {} iterations "
               "({} threads/batch, block size {}).",
               opt.mode, mode_name(opt.mode),
               opt.dispatches, opt.threads, opt.iters,
               opt.dispatches * opt.threads, kBlockSize);
    // What the reorder pass can do with that shape. The CUDA-launch groups are
    // capped by the fail-closed state isolation of imported kernels: with a
    // shared input range every dispatch "writes" that range, so the layers cannot
    // merge at all; with private inputs the two routes are expected to behave the
    // same way. Rerun with verbose=1 to see the actual layer counts.
    if ((opt.mode & kPrivateInput) != 0u) {
        LUISA_INFO("Expected layers (reorder on): native vk ~{}, imported CUDA launch ~{}.",
                   (opt.mode & kOutputRotate) != 0u ? (opt.dispatches + kRotateRanges - 1u) / kRotateRanges : 1u,
                   (opt.mode & kOutputRotate) != 0u ? (opt.dispatches + kRotateRanges - 1u) / kRotateRanges : 1u);
    } else {
        LUISA_INFO("Expected layers (reorder on): native vk ~{}, imported CUDA launch {} "
                   "(one per command: resource state isolation makes every kernel 'write' "
                   "the shared input range).",
                   (opt.mode & kOutputRotate) != 0u ? (opt.dispatches + kRotateRanges - 1u) / kRotateRanges : 1u,
                   opt.dispatches);
    }

    // ---- correctness reference -----------------------------------------------------
    // Strictly ordered reference from the route that cannot reorder anything.
    reorder_ext->set_command_reorder_enabled(false);
    cuda_stream << build_cuda_batch().commit() << synchronize();
    auto reference = download(cuda_stream, cuda_dst);

    // ---- measurement ----------------------------------------------------------------
    // Warm every group up once (the first submission of a shader also touches
    // driver-side pipeline/module setup, which must not be attributed to a group),
    // then alternate the four groups every round so that clock ramping and thermal
    // drift hit them equally.
    constexpr auto kWarmupRounds = 2u;
    // The warm-up samples are discarded: they only exist to move first-submission
    // work out of the measured groups.
    static_cast<void>(measure_group(cuda_stream, build_cuda_batch, 0u, kWarmupRounds));
    reorder_ext->set_command_reorder_enabled(false);
    static_cast<void>(measure_group(stream, build_vk_cuda_batch, 0u, kWarmupRounds));
    reorder_ext->set_command_reorder_enabled(true);
    static_cast<void>(measure_group(stream, build_vk_cuda_batch, 0u, kWarmupRounds));
    static_cast<void>(measure_group(stream, build_vk_native_batch, 0u, kWarmupRounds));

    GroupTiming a;
    GroupTiming b;
    GroupTiming c;
    GroupTiming d;
    for (auto round = size_t{0}; round < opt.rounds; round++) {
        auto a_now = measure_group(cuda_stream, build_cuda_batch, 1u);
        reorder_ext->set_command_reorder_enabled(false);
        auto b_now = measure_group(stream, build_vk_cuda_batch, 1u);
        reorder_ext->set_command_reorder_enabled(true);
        auto c_now = measure_group(stream, build_vk_cuda_batch, 1u);
        auto d_now = measure_group(stream, build_vk_native_batch, 1u);
        append_samples(a.batch, a_now.batch);
        append_samples(a.submit, a_now.submit);
        append_samples(b.batch, b_now.batch);
        append_samples(b.submit, b_now.submit);
        append_samples(c.batch, c_now.batch);
        append_samples(c.submit, c_now.submit);
        append_samples(d.batch, d_now.batch);
        append_samples(d.submit, d_now.submit);
        LUISA_INFO("round {}: A(cuda) {:.3f} ms | B(vk+cuda, off) {:.3f} ms | "
                   "C(vk+cuda, on) {:.3f} ms | D(vk native, on) {:.3f} ms",
                   round + 1, a_now.batch.front(), b_now.batch.front(),
                   c_now.batch.front(), d_now.batch.front());
    }

    // ---- validation ------------------------------------------------------------------
    // B and C run on the same imported module as A, so bit-exactness is required;
    // it is also what proves the merged layers did not race.
    reorder_ext->set_command_reorder_enabled(true);
    stream << build_vk_cuda_batch().commit() << synchronize();
    auto vk_cuda_reordered = download(stream, vk_dst);
    reorder_ext->set_command_reorder_enabled(false);
    stream << build_vk_cuda_batch().commit() << synchronize();
    auto vk_cuda_serialized = download(stream, vk_dst);
    reorder_ext->set_command_reorder_enabled(true);
    stream << build_vk_native_batch().commit() << synchronize();
    auto vk_native = download(stream, vk_dst);
    reorder_ext->set_command_reorder_enabled(true);

    auto reference_span = luisa::span<const float4>{reference};
    auto all_finite = std::all_of(
        vk_cuda_reordered.begin(), vk_cuda_reordered.end(), [](float4 v) {
            return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) && std::isfinite(v.w);
        });
    auto span_at = [&](luisa::span<const float4> s, size_t j) {
        return s.subspan(j * opt.threads, opt.threads);
    };
    // Mode 1 rewrites the same sub-ranges repeatedly, so only the bitwise equality
    // with the strictly ordered reference proves ordering there.
    auto distinct_ranges = true;
    if ((opt.mode & kOutputRotate) == 0u) {
        for (auto j = size_t{1}; j < opt.dispatches; j++) {
            if (bitwise_equal(span_at(reference_span, j - 1u),
                              span_at(reference_span, j))) {
                distinct_ranges = false;
                break;
            }
        }
    }
    auto reordered_matches = bitwise_equal(luisa::span<const float4>{vk_cuda_reordered}, reference_span);
    auto serialized_matches = bitwise_equal(luisa::span<const float4>{vk_cuda_serialized}, reference_span);
    auto native_deviation = max_relative_deviation(luisa::span<const float4>{vk_native}, reference_span);

    // ---- report ------------------------------------------------------------------------
    auto a_batch = summarize(a.batch);
    auto b_batch = summarize(b.batch);
    auto c_batch = summarize(c.batch);
    auto d_batch = summarize(d.batch);
    LUISA_INFO("===========================================================================");
    LUISA_INFO("{:<38}{:>10}{:>10}{:>10}{:>10} |{:>10}{:>10}{:>10}{:>10}",
               "batch (ms) / host submit (ms)", "min", "median", "mean", "max",
               "min", "median", "mean", "max");
    print_row({"A cuda backend (no reorder)", a_batch, summarize(a.submit)});
    print_row({"B vk+cuda kernel, reorder OFF", b_batch, summarize(b.submit)});
    print_row({"C vk+cuda kernel, reorder ON", c_batch, summarize(c.submit)});
    print_row({"D vk native kernel, reorder ON", d_batch, summarize(d.submit)});
    LUISA_INFO("===========================================================================");
    LUISA_INFO("Reorder effect inside the vk+CUDA route (B/C): {:.2f}x", b_batch.median / c_batch.median);
    LUISA_INFO("vs. the cuda backend (A/C)                 : {:.2f}x", a_batch.median / c_batch.median);
    LUISA_INFO("vk+CUDA route vs native vk route (D/C)     : {:.2f}x", d_batch.median / c_batch.median);
    LUISA_INFO("Checks: vk+CUDA reordered == cuda reference = {}, strictly ordered == "
               "reference = {}, outputs finite = {}, one distinct sub-range per dispatch "
               "= {}, vk native max relative deviation = {:.3e} (< {:.1e} required).",
               reordered_matches, serialized_matches, all_finite, distinct_ranges,
               native_deviation, kNativeTolerance);

    auto failed = false;
    if (!reordered_matches || !serialized_matches || !all_finite || !distinct_ranges ||
        native_deviation > kNativeTolerance) {
        LUISA_ERROR_WITH_LOCATION(
            "Comparison validation FAILED: both Vulkan CUDA-launch groups must reproduce "
            "the strictly ordered cuda-backend result exactly, produce finite outputs and "
            "write one distinct sub-range per dispatch; the Vulkan-compiled group must stay "
            "within the numerical tolerance.");
        failed = true;
    }
    // "Faster" has to mean something: a 1-2% spread between two routes that both
    // saturate the GPU is a tie, not a win.
    constexpr auto kMeaningfulGain = 1.05;
    if (a_batch.median < kMeaningfulGain * c_batch.median) {
        LUISA_WARNING(
            "The reordered vk+CUDA route is NOT meaningfully faster than the plain cuda "
            "backend (A {:.3f} ms vs C {:.3f} ms, {:.2f}x). Diagnosis:",
            a_batch.median, c_batch.median, a_batch.median / c_batch.median);
        if (b_batch.median / c_batch.median < kMeaningfulGain) {
            LUISA_WARNING(
                "  1) reordering changed nothing inside this route (B {:.3f} ms vs C {:.3f} "
                "ms, {:.2f}x): the batch still runs as one layer per command, so C is back "
                "in the same strictly serialized regime as A. This is the fail-closed rule "
                "of imported kernels, not a barrier-cost accident - "
                "CudaKernelLaunchCommand::requires_resource_state_isolation() is true, so "
                "the reorder pass marks EVERY argument of every dispatch (including the one "
                "declared READ) as an exclusive access over its range. With the shared "
                "input range of mode {} all {} dispatches therefore 'write' the same range "
                "and cannot share a layer. Rerun with mode {} (private input ranges, same "
                "device work): the CUDA launches merge the way the native ones do and C "
                "drops below A. Verbose runs show the counts directly.",
                b_batch.median, c_batch.median, b_batch.median / c_batch.median,
                opt.mode, opt.dispatches, opt.mode ^ kPrivateInput);
        } else {
            LUISA_WARNING(
                "  1) reordering did help inside this route (B {:.3f} ms -> C {:.3f} ms, "
                "{:.2f}x), yet C is still level with A: the layers merged but the batch was "
                "not barrier-bound. Either the dispatches already overlap on the CUDA stream, "
                "or one dispatch fills the GPU (raise the dispatch count, or lower the thread "
                "count per dispatch to leave room).",
                b_batch.median, c_batch.median, b_batch.median / c_batch.median);
        }
        if (c_batch.median > 2.0 * d_batch.median) {
            LUISA_WARNING(
                "  2) the same Vulkan runtime launching a Vulkan-compiled kernel (D) needs "
                "only {:.3f} ms for the identical batch, {:.2f}x less than C, so the Vulkan "
                "backend, its argument encoding and its submission path are not what costs "
                "here: the difference is entirely the missing layer merge of point 1). Note "
                "that A/B/C/D are not fully comparable across code generators - the "
                "Vulkan-compiled kernel is not the same machine code as the CUDA one - so "
                "read D as an ordering probe, not as a faster kernel.",
                d_batch.median, c_batch.median / d_batch.median);
        } else {
            LUISA_WARNING(
                "  2) D (Vulkan-compiled, reordered) is {:.3f} ms, within 2x of C: with the "
                "layers merged both launch routes reach the same regime, so the remaining "
                "time is per-dispatch device work (median / {} dispatches = {:.3f} ms for C).",
                d_batch.median, opt.dispatches,
                c_batch.median / static_cast<double>(opt.dispatches));
        }
        LUISA_WARNING(
            "  3) median divided by the dispatch count: A {:.3f} ms, B {:.3f} ms, C {:.3f} ms, "
            "D {:.3f} ms. A, B and C agree because all three run one dispatch after another; "
            "D's number is a merged batch spread over the dispatches, i.e. the payoff of "
            "actually overlapping them.",
            a_batch.median / static_cast<double>(opt.dispatches),
            b_batch.median / static_cast<double>(opt.dispatches),
            c_batch.median / static_cast<double>(opt.dispatches),
            d_batch.median / static_cast<double>(opt.dispatches));
    } else {
        LUISA_INFO("Comparison PASSED: the reordered vk+CUDA route beats the plain cuda "
                   "backend by {:.2f}x (A {:.3f} ms vs C {:.3f} ms; inside the route, B/C "
                   "= {:.2f}x, and D/C = {:.2f}x).",
                   a_batch.median / c_batch.median, a_batch.median, c_batch.median,
                   b_batch.median / c_batch.median, d_batch.median / c_batch.median);
    }
    // The verdict is printed above; the resources go through their normal
    // destructors, which is also what keeps this benchmark honest about the
    // teardown paths (Context/device/stream destruction plus the imported CUDA
    // module and its Vulkan-side wrapper) that the vk + CUDA route uses.
    return failed ? 1 : 0;
}
