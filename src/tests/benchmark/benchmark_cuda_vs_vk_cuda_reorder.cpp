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
//   E  cuda backend, CUDA graph, submitted as one graph launch .... candidate
//      Group A's exact batch (same CUDA module, same arguments, same buffers)
//      built once into a CUDA graph (CudaGraphExt) and replayed with a single
//      cuGraphLaunch per iteration. Building the graph (dependency analysis +
//      cuGraphAddKernelNode + cuGraphInstantiate) is an expensive one-off host
//      cost - instantiation alone runs into milliseconds on some drivers - so it
//      is CHARGED TO THE GRAPH GROUP'S HOST SUBMIT instead of being reported out
//      of band: the submit column of E carries build / rounds on top of the
//      per-replay cuGraphLaunch, i.e. exactly what a caller amortising one graph
//      over `rounds` batches pays. The un-charged replay cost is printed too, so
//      the "replays to pay back the build" line stays like-for-like.
// The interesting comparisons are E vs. A on the replay path (what one
// graph launch buys over N cuLaunchKernel submissions) and E vs. C/D (the
// reorder routes against the graph route).
//
// F  cuda backend, batch split over several streams ............ the hand-rolled
//                                                  alternative (needs K ~ N)
// Group A's exact batch, arguments and buffers once more, but the dispatches
// are dealt round-robin over `streams` dedicated CUDA streams (each from
// device.create_stream(), so each owns its own CUstream) instead of all being
// queued on one. This is the "just use more streams" answer to the same
// problem: nothing analyses dependencies, the caller merely asserts that the
// batch is independent (mode 0/2 guarantees it) and hands K queues to the
// device. Every stream still serialises its own share, so the achievable
// overlap is bounded by the stream count instead of by the hazard graph, and
// nothing at all is gained once two dispatches of the batch must be ordered.
// It is therefore compared against the two routes that DO resolve ordering,
// the reorder pass (C) and the CUDA graph (E).
//
// MEASURED RESULT (with the fix described below in place): F now behaves exactly
// as the name promises - the batch overlaps, and how much it overlaps is decided
// by K alone. Mode 0, 16 dispatches, 7 rounds (ms): A 6.31, B 6.52, C 0.558,
// D 0.475, E(graph) 0.584, F 1.67 at K = 4 (A/F = 3.8x); F 0.536 at K = 16, i.e.
// one stream per dispatch (A/F = 11.8x, C/F = 1.01x, E/F = 1.09x) - so one
// stream per dispatch puts F level with the two routes that DO analyse
// dependencies. Fewer streams degrade F as ceil(N/K): K = 2/4/8/16 take
// 3.20/1.62/0.86/0.54 ms on that batch, which is exactly "each stream serialises
// its own share, streams run side by side". Mode 1 (genuine WAW chains) is still
// a race, because F has no way to detect a hazard at all.
//
// WHY THE NUMBERS USED TO BE FLAT, AND WHAT CHANGED. Before the fix F was as fast
// as A whatever K was, i.e. the streams never overlapped: this backend wrapped
// every single submission in a CUcontext switch (CUDADevice::with_handle's
// ContextGuard did cuCtxPushCurrent/cuCtxPopCurrent around each dispatch, see
// src/backends/cuda/cuda_device.h). On this driver a context switch between two
// launches closes the driver's submission batch, and the resulting per-stream
// batches are then executed strictly one after another, so K streams were just K
// serial batches. A standalone driver-API probe (same kernel, 16 launches on 16
// non-blocking CUstreams) pins it down: nothing between launches = 1.0 ms, a
// push/pop (or even a redundant cuCtxSetCurrent) around every launch = 13-15 ms,
// a cuCtxGetCurrent around every launch = 1.0 ms, one push/pop around the whole
// batch = 1.0 ms. ContextGuard now queries the current context (cheap, and unlike
// a set it is not a submission boundary) and only switches when the calling
// thread is not already running on ours, which keeps back-to-back submissions
// switch-free; a thread that had a *different* context still gets it back. The
// probe is what justified the change, and F is what keeps it honest.
// Caveat: the order of commands on different streams is undefined, so in
// mode 1 (rotating writes - genuine write-after-write chains between
// dispatches) the multi-stream batch is not a valid execution of the batch at
// all, it is a race: whichever dispatch finishes last wins its sub-range. The
// bit-exactness check is hence only applied to the hazard-free modes; in mode
// 1 the group is timed, but its output is reported as (deliberately) unchecked
// - and unlike the reorder pass, F has no mechanism to even detect that.
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
// benchmark_cuda_vs_vk_cuda_reorder vk [mode] [dispatches] [threads] [iters] [rounds] [verbose] [streams]
//
// Defaults: mode=0, dispatches=16, threads=256, iters=4096, rounds=5, verbose=0,
// streams=4 (the number of CUDA streams group F deals the batch over; 1 stream
// degenerates to group A plus the cost of the round-robin recording).
// The first argument must be "vk" (the route under test); the cuda device used
// for compiling/importing is created automatically from the same context.
//
// mode bit 0 selects the destination pattern, bit 1 the input footprint:
//   mode 0 (disjoint out, shared in): dispatch j writes sub-range j and every
//     dispatch reads the whole input range -> no real hazards, ideal layer 1.
//   mode 1 (rotate out, shared in): dispatch j writes sub-range (j % 16), i.e.
//     genuine write-after-write chains, ideal layer count ceil(dispatches / 16).
//   mode 2 / 3: same as 0 / 1, except that dispatch j reads its own input
//     sub-range instead of the shared one. This is the control that isolates
//     the input-sharing effect (see the note below): with per-dispatch inputs
//     the input side cannot merge layers either way, so all modes behave the
//     same.
//
// A note on the vk + CUDA route and reordering
// -------------------------------------------
// `CudaKernelLaunchCommand` declares per-argument usages, and the
// reorder pass tracks every argument with its DECLARED usage - the
// declaration itself is the contract: a declared READ is trusted to be
// read-only (concurrent reads of one range do not race, so dispatches that
// share a read-only input merge into a single barrier-free layer), and a
// declared WRITE is an exclusive access over its range, so RAW/WAW/WAR
// chains serialize exactly as they do for native dispatches. This holds for
// raw (hand-written) function handles too: a kernel that writes through a
// pointer it declared READ violates its declaration and must not be
// launched with it. The vk stream's ResourceBarrier has always recorded
// these launches by declared usage (kComputeRead vs kComputeUAV), so the
// reorder layer assignment is consistent with the actual barrier contract.
// Mode 2/3 (private inputs) removes the input-side sharing, which is what
// isolates the effect of the declared-usages rule.
//
// Examples:
//   xmake run benchmark_cuda_vs_vk_cuda_reorder vk
//   xmake run benchmark_cuda_vs_vk_cuda_reorder vk 0 64 256 4096 7
//   xmake run benchmark_cuda_vs_vk_cuda_reorder vk 2   # private inputs
//   xmake run benchmark_cuda_vs_vk_cuda_reorder vk 0 16 256 4096 7 0 16
// ^ group F with one CUDA stream per dispatch: the most favourable
//   multi-stream shape the batch admits (no stream serialises two dispatches),
//   i.e. the number to quote when arguing that more streams alone suffice.
//   xmake run benchmark_cuda_vs_vk_cuda_reorder vk 0 16 256 1 9 1
// ^ iters=1 removes almost all device work, so what is left is launch and
//   submission overhead only; verbose=1 makes the Vulkan backend log
//   "command reorder: N commands -> M layers (reorder on)" for every batch,
//   which is the first thing to look at when C does not beat A.
//
// Reading the result
// ------------------
// The verdict line compares medians: gain = median(A) / median(C); anything below
// 1.05x is reported as "not meaningfully faster" together with the diagnosis of
// why (a 1-2% spread between two GPU-saturating routes is a tie). Groups
// alternate every round so that clock ramping and thermal drift hit them equally.
// The "submit" column is the host-side cost of recording + queueing the batch
// (measured before the wait), which tells a device-side ordering effect apart from
// a host-side one. For the CUDA-graph group (E) it also carries the amortised
// share of the one-off graph build, see below. For the multi-stream group (F) it
// is the cost of recording all `streams` command lists and handing them to their
// streams - which is exactly what group A pays, just split over K lists.
//
// The CUDA-graph group (E) is special: its batch is fixed by the graph, so a
// replay's "submit" is one cuGraphLaunch and its "batch" is the whole graph
// running on the GPU. Building the graph (dependency analysis + node creation +
// instantiate) is an expensive one-off host cost, so it is measured on its own
// and CHARGED TO THE GROUP ITSELF: E's submit samples get build / rounds added,
// which is the per-batch host cost of a caller that builds the graph once and
// replays it `rounds` times. The report also prints the un-charged replay cost,
// the total host time of the route (build + all replays), and the amortisation of
// the build against the per-replay host saving versus A. If CudaGraphExt is
// unavailable or the batch cannot be represented as a graph, group E is skipped
// and its rows are simply omitted.
//
// How to read the multi-stream group (F): it is the same batch as A, so it must
// not be read as "a better A" but as "what a caller gets when it parallelises the
// batch by hand". The stream count, not the hazard graph, bounds the overlap - F
// spreads the batch over K queues and can only hope each queue's share is long
// enough to keep the device busy - so unlike the reorder pass and the CUDA graph
// it can neither merge the batch to one layer nor resolve a real hazard: in mode 1
// (rotating writes) its K streams would simply race over the shared sub-ranges,
// which is why its output is only verified in the hazard-free modes. Getting F
// close to C therefore takes K in the order of the dispatch count, whatever that
// costs in stream management - the point the group exists to make.
//
// Reference outcome (RTX 5070 Ti Laptop, driver 596.13, CUDA 13.2, release build,
// 16 dispatches x 256 threads x 4096 iterations, medians in ms):
//
//                      A cuda     B vk+cuda    C vk+cuda    D vk native   A/C
//   mode 0, shared in  3.973      4.003        0.318        0.537        12.49x
//   mode 2, private in 3.977      4.003        0.321        0.538        12.41x
//   mode 0, iters=1    0.191      0.162        0.067        0.069         2.80x
//   mode 2, iters=1    0.202      0.167        0.071        0.070         2.86x
//
// E (one cuGraphLaunch per replay of A's batch) is not in the table above (that
// table predates it). Its graph is built as an explicit DAG whose edges are only
// the real RAW/WAW/WAR dependencies (see CudaGraphExtImpl::_create_graph), so on
// the device it behaves like the reordered vk routes - the independent dispatches
// of the batch overlap - and its wall time tracks C/D rather than A. (The older
// claim that "a CUDA graph replays the recorded launches in stream order and never
// overlaps them" describes stream capture, which this backend does not use.) On
// the host a replay is one cuGraphLaunch, and the one-off build (dependency
// analysis + instantiate; ~0.15-0.3 ms for 16 nodes, up to ~1 ms for 64 on this
// machine, milliseconds on slower drivers) is charged to the graph group's submit
// column as build / rounds. The graph replay is checked bit-exact against the
// strictly ordered reference.
//
// F (K CUDA streams, round-robin) is not in the table either. Since each stream
// serialises its own share, one dispatch per stream is the best it can ever do,
// and the batch overlaps only as much as K allows; it runs on the same batch,
// arguments and buffers as A, so A/F is "what a hand-rolled stream split buys over
// one stream" and C/F, E/F are "how much of the reorder/graph win is left to the
// caller's own bookkeeping". Its output is verified bit-exact in the hazard-free
// modes only (see the mode-1 caveat above).
//
// i.e. with arguments tracked by their declared usages the reordered vk+CUDA
// route wins in every mode, shared input or not; the verbose layer counts
// confirm it directly - both mode 0 and mode 2 log "16 commands -> 1 layers
// (reorder on)" for the imported CUDA launches and for the native ones. Mode 1
// with 64 dispatches (genuine write-after-write chains) logs "64 commands ->
// 4 layers" for both routes, matching the 16 rotating ranges, and still
// validates bit-exactly against the strictly ordered reference - declared
// READ merging does not weaken write tracking.
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
// itself is covered by test_vk_shader_cache / test_fallback_shader_cache (and, for the
// imported CUDA kernels specifically, by test_vk_cuda_kernel_launch - all of them
// compile with the cache on).
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
#include <luisa/backends/ext/cuda/cuda_graph_ext.h>
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
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <optional>
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

/// Number of CUDA streams the multi-stream group (F) deals the batch over when
/// the `streams` argument is not given. Four is a realistic "hand-rolled
/// parallelism" number; raise it (up to `dispatches` = one stream per dispatch,
/// the most favourable shape this route admits) to see how far more streams
/// alone get.
constexpr size_t kMultiStreams = 4u;
/// Ceiling on the stream count: each CUDA stream costs a stream object, an
/// upload/download pool and a callback thread, so the sweep is bounded.
constexpr size_t kMaxStreams = 64u;

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
    /// Streams the multi-stream group (F) splits the batch over.
    size_t streams{kMultiStreams};
};

[[nodiscard]] size_t parse_uint(const char *text, const char *name, size_t minimum) {
    auto input = luisa::string_view{text};
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
    luisa::sort(samples.begin(), samples.end());
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

/// Emit the same batch as `build_batch`, but deal the dispatches round-robin
/// over `streams` command lists (dispatch j goes to list j % streams). The
/// per-dispatch `emit` lambda is shared with `build_batch`, so the two
/// expressions of the batch - one list on one stream, K lists on K streams -
/// are guaranteed to contain exactly the same commands.
template<typename Emit>
[[nodiscard]] luisa::vector<CommandList> build_batch_multi(const Options &opt,
                                                          const Buffer<float> &src,
                                                          size_t streams,
                                                          Emit &&emit) {
    LUISA_ASSERT(streams > 0u, "the multi-stream batch needs at least one stream.");
    luisa::vector<CommandList> lists;
    lists.resize(streams);
    auto per_stream = (opt.dispatches + streams - 1u) / streams;
    for (auto &list : lists) { list.reserve(per_stream, 0u); }
    for (auto j = size_t{0}; j < opt.dispatches; j++) {
        auto dst_range = (opt.mode & kOutputRotate) != 0u ? j % kRotateRanges : j;
        auto in_view = (opt.mode & kPrivateInput) != 0u ? src.view().subview(j * opt.threads, opt.threads) : src.view();
        emit(lists[j % streams], in_view, dst_range, j);
    }
    return lists;
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

/// Submit `rounds` batches split over several CUDA streams: every stream gets its
/// own command list from `build_lists`, all of them are recorded and committed
/// before any wait, and the batch is over when the last stream has drained. The
/// split between `submit` and `batch - submit` is therefore "record + queue K
/// lists" versus "wait for all K queues", the same split `measure_group` draws on
/// one stream.
template<typename Builder>
[[nodiscard]] GroupTiming measure_multi_stream(luisa::span<Stream *const> streams,
                                               Builder &&build_lists,
                                               size_t rounds, size_t warmup_rounds = 0u) {
    LUISA_ASSERT(!streams.empty(), "the multi-stream group needs at least one stream.");
    auto submit_all_batches = [&](luisa::vector<CommandList> &lists) {
        LUISA_ASSERT(lists.size() == streams.size(),
                     "one command list per stream is required.");
        for (auto i = size_t{0}; i < streams.size(); i++) {
            *streams[i] << lists[i].commit();
        }
    };
    auto drain_all_streams = [&] {
        // Waiting stream by stream is enough: each wait is independent and the
        // last one to finish is what bounds the batch.
        for (auto i = size_t{0}; i < streams.size(); i++) {
            *streams[i] << synchronize();
        }
    };
    for (auto w = size_t{0}; w < warmup_rounds; w++) {
        auto lists = build_lists();
        submit_all_batches(lists);
        drain_all_streams();
    }
    GroupTiming timing;
    timing.batch.reserve(rounds);
    timing.submit.reserve(rounds);
    for (auto r = size_t{0}; r < rounds; r++) {
        Clock clock;
        auto lists = build_lists();
        submit_all_batches(lists);
        auto submit = clock.toc();
        drain_all_streams();
        timing.batch.emplace_back(clock.toc());
        timing.submit.emplace_back(submit);
    }
    return timing;
}

/// Append `more` to `samples`.
inline void append_samples(luisa::vector<double> &samples, const luisa::vector<double> &more) {
    samples.insert(samples.end(), more.begin(), more.end());
}

/// Submit `rounds` replays of an instantiated CUDA graph. `ext->launch` is a
/// single cuGraphLaunch (the host-side cost of one graph submission), and the
/// device executes the whole captured batch as one unit, so the split between
/// `submit` and `batch - submit` is exactly "one launch on the CPU" versus
/// "everything the GPU does" - the same split `measure_group` draws for the
/// record-and-submit-per-command routes.
template<typename Launch>
[[nodiscard]] GroupTiming measure_graph(Stream &stream, Launch &&launch_exec,
                                        size_t rounds, size_t warmup_rounds = 0u) {
    for (auto w = size_t{0}; w < warmup_rounds; w++) {
        launch_exec();
        stream << synchronize();
    }
    GroupTiming timing;
    timing.batch.reserve(rounds);
    timing.submit.reserve(rounds);
    for (auto r = size_t{0}; r < rounds; r++) {
        Clock clock;
        launch_exec();
        auto submit = clock.toc();
        stream << synchronize();
        timing.batch.emplace_back(clock.toc());
        timing.submit.emplace_back(submit);
    }
    return timing;
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
            "Usage: {} vk [mode] [dispatches] [threads] [iters] [rounds] [verbose] [streams] "
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
    if (argc > 8) {
        opt.streams = parse_positive(argv[8], "streams");
    }
    LUISA_ASSERT(opt.mode <= 3u,
                 "mode bit 0: 0 = disjoint outputs, 1 = outputs rotated over 16 sub-ranges; "
                 "mode bit 1: 0 = one shared read-only input, 1 = per-dispatch input ranges.");
    LUISA_ASSERT(opt.dispatches <= 1024u,
                 "dispatches must be <= 1024 to bound buffer memory and batch size.");
    LUISA_ASSERT(opt.threads % kBlockSize == 0u,
                 "threads must be a multiple of the kernel block size ({}).", kBlockSize);
    LUISA_ASSERT(opt.streams <= kMaxStreams,
                 "streams must be <= {} (each stream costs a stream object and a callback thread).",
                 kMaxStreams);

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
    // Group F: `opt.streams` dedicated CUDA streams to deal the same batch over.
    // Each one is a full device stream of its own (its own CUstream, created
    // exactly like cuda_stream above), so the group measures what a caller that
    // parallelises the batch by hand actually gets out of the backend.
    luisa::vector<Stream> multi_streams;
    multi_streams.reserve(opt.streams);
    for (auto i = size_t{0}; i < opt.streams; i++) {
        multi_streams.emplace_back(cuda_device.create_stream());
    }
    luisa::vector<Stream *> multi_stream_ptrs;
    multi_stream_ptrs.reserve(multi_streams.size());
    for (auto &s : multi_streams) { multi_stream_ptrs.emplace_back(&s); }
    LUISA_INFO("Routes: A = {} backend device #{}; B/C/D = {} backend device #{} "
               "(same physical adapter); F = A's batch dealt over {} cuda streams.",
               cuda_device.backend_name(), cuda_index,
               device.backend_name(), 0u, opt.streams);

    // Probe the runtime switch: a process-wide override would silently turn C
    // into a second copy of B.
    reorder_ext->set_command_reorder_enabled(true);
    if (!reorder_ext->command_reorder_enabled()) {
        LUISA_WARNING("Command reordering is forced off by LUISA_DISABLE_COMMAND_REORDER, "
                      "which overrides the runtime switch: groups B and C would measure the "
                      "same serialized baseline.");
    }
    // The CUDA-graph route (group E) captures group A's batch on the cuda
    // device. The extension is cuda-only; if it is missing the graph group is
    // skipped rather than failing the whole comparison.
    auto *graph_ext = cuda_device.extension<CudaGraphExt>();
    if (graph_ext == nullptr) {
        LUISA_WARNING("CudaGraphExt is not available on the cuda device: the CUDA-graph "
                      "route (group E) is skipped.");
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
    // Imported CUDA kernel with the read/write intent baked into the
    // signature: the reorder pass tracks the shared input buffer as the
    // declared read and the per-dispatch output ranges as declared writes,
    // exactly like the DSL does, so the batch can merge into one layer.
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
    // Route F: the same cuda batch, dealt round-robin over `streams` cuda streams.
    // Nothing analyses dependencies here - the caller asserts that the batch is
    // independent (true for the hazard-free modes) and each stream only has to
    // keep its own share in order.
    auto build_cuda_multi_batch = [&]() {
        return build_batch_multi(opt, cuda_src, opt.streams,
                                 [&](CommandList &list, const BufferView<float> &in_view,
                                     size_t dst_range, size_t j) {
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
    // What the reorder pass can do with that shape: the declared-usages
    // rule tracks the shared read-only input as a read for every launch
    // route, so both the imported-CUDA and the native batches are expected
    // to merge the same way (declared writes still serialize per range
    // chain). Rerun with verbose=1 to see the actual layer counts.
    LUISA_INFO("Expected layers (reorder on): native vk ~{}, imported CUDA launch ~{}.",
               (opt.mode & kOutputRotate) != 0u ? (opt.dispatches + kRotateRanges - 1u) / kRotateRanges : 1u,
               (opt.mode & kOutputRotate) != 0u ? (opt.dispatches + kRotateRanges - 1u) / kRotateRanges : 1u);

    // ---- correctness reference -----------------------------------------------------
    // Strictly ordered reference from the route that cannot reorder anything.
    reorder_ext->set_command_reorder_enabled(false);
    cuda_stream << build_cuda_batch().commit() << synchronize();
    auto reference = download(cuda_stream, cuda_dst);

    // ---- CUDA graph (route E) ------------------------------------------------------
    // Group A's exact batch, captured once into a CUDA graph on the cuda
    // device. Capture + instantiation is an expensive one-off host cost (it
    // is measured and reported separately, and never inside a timed round),
    // but every replay afterwards is a single cuGraphLaunch for the whole
    // batch - that is the trade-off this group quantifies: host time of the
    // build and of each replay versus the device time, against the
    // per-command submission of groups A/C/D.
    luisa::optional<CudaGraphInstance> graph;
    luisa::optional<CudaGraphExecInstance> graph_exec;
    double graph_build_ms{0.0};
    if (graph_ext != nullptr) {
        Clock build_clock;
        auto captured = graph_ext->create_graph(build_cuda_batch());
        if (captured.handle().handle != CudaGraphExt::invalid_handle) {
            auto instantiated = graph_ext->instantiate(captured.handle().handle);
            if (instantiated.handle().handle != CudaGraphExt::invalid_handle) {
                graph.emplace(std::move(captured));
                graph_exec.emplace(std::move(instantiated));
            }
        }
        graph_build_ms = build_clock.toc();
        if (!graph_exec) {
            graph.reset();
            graph_ext = nullptr;// disable route E
            LUISA_WARNING("Capturing / instantiating the CUDA graph failed "
                          "(dispatches of non-native shaders cannot be captured); "
                          "the CUDA-graph route (group E) is skipped.");
        } else {
            LUISA_INFO("CUDA graph built: dependency analysis + instantiate on host = {:.3f} ms "
                       "(one-off, charged to the graph group's host submit as build / rounds; "
                       "{} dispatch nodes).",
                       graph_build_ms, opt.dispatches);
        }
    }
    const auto graph_exec_handle = graph_exec ? graph_exec->handle().handle : CudaGraphExt::invalid_handle;

    // ---- measurement ----------------------------------------------------------------
    // Warm every group up once (the first submission of a shader also touches
    // driver-side pipeline/module setup, which must not be attributed to a group),
    // then alternate all groups every round so that clock ramping and thermal
    // drift hit them equally.
    constexpr auto kWarmupRounds = 2u;
    // The warm-up samples are discarded: they only exist to move first-submission
    // work out of the measured groups.
    static_cast<void>(measure_group(cuda_stream, build_cuda_batch, 0u, kWarmupRounds));
    static_cast<void>(measure_multi_stream(multi_stream_ptrs, build_cuda_multi_batch,
                                          0u, kWarmupRounds));
    reorder_ext->set_command_reorder_enabled(false);
    static_cast<void>(measure_group(stream, build_vk_cuda_batch, 0u, kWarmupRounds));
    reorder_ext->set_command_reorder_enabled(true);
    static_cast<void>(measure_group(stream, build_vk_cuda_batch, 0u, kWarmupRounds));
    static_cast<void>(measure_group(stream, build_vk_native_batch, 0u, kWarmupRounds));
    const auto have_graph = graph_exec.has_value();
    auto launch_graph = [&] { graph_ext->launch(graph_exec_handle, cuda_stream.handle()); };
    if (have_graph) {
        static_cast<void>(measure_graph(cuda_stream, launch_graph, 0u, kWarmupRounds));
    }

    GroupTiming a;
    GroupTiming b;
    GroupTiming c;
    GroupTiming d;
    GroupTiming e;
    GroupTiming f;
    for (auto round = size_t{0}; round < opt.rounds; round++) {
        auto a_now = measure_group(cuda_stream, build_cuda_batch, 1u);
        auto f_now = measure_multi_stream(multi_stream_ptrs, build_cuda_multi_batch, 1u);
        reorder_ext->set_command_reorder_enabled(false);
        auto b_now = measure_group(stream, build_vk_cuda_batch, 1u);
        reorder_ext->set_command_reorder_enabled(true);
        auto c_now = measure_group(stream, build_vk_cuda_batch, 1u);
        auto d_now = measure_group(stream, build_vk_native_batch, 1u);
        auto e_now = have_graph ? measure_graph(cuda_stream, launch_graph, 1u) : GroupTiming{};
        append_samples(a.batch, a_now.batch);
        append_samples(a.submit, a_now.submit);
        append_samples(b.batch, b_now.batch);
        append_samples(b.submit, b_now.submit);
        append_samples(c.batch, c_now.batch);
        append_samples(c.submit, c_now.submit);
        append_samples(d.batch, d_now.batch);
        append_samples(d.submit, d_now.submit);
        append_samples(e.batch, e_now.batch);
        append_samples(e.submit, e_now.submit);
        append_samples(f.batch, f_now.batch);
        append_samples(f.submit, f_now.submit);
        LUISA_INFO("round {}: A(cuda) {:.3f} ms | B(vk+cuda, off) {:.3f} ms | "
                   "C(vk+cuda, on) {:.3f} ms | D(vk native, on) {:.3f} ms | "
                   "E(cuda graph) {:.3f} ms | F(cuda, {} streams) {:.3f} ms",
                   round + 1, a_now.batch.front(), b_now.batch.front(),
                   c_now.batch.front(), d_now.batch.front(),
                   have_graph ? e_now.batch.front() : 0.0,
                   opt.streams, f_now.batch.front());
    }

    // ---- charge the one-off graph build to the graph route's host time ----------------
    // Building the graph (dependency analysis + cuGraphAddKernelNode + instantiate) is
    // an expensive one-off host cost that every replay depends on - cuGraphInstantiate
    // alone costs milliseconds on some driver versions. Reporting the replay path
    // without it would flatter the graph route, so group E's host submit carries its
    // share of the build on top of the per-replay cuGraphLaunch: build / rounds for
    // every measured batch. That is what a caller who amortises one graph over
    // `rounds` batches actually pays. The un-charged replay cost is kept aside for the
    // "replays to pay back the build" line below so that comparison stays like-for-like.
    auto e_replay_submit = e.submit;
    if (have_graph && !e.submit.empty()) {
        auto build_per_replay = graph_build_ms / static_cast<double>(e.submit.size());
        for (auto &&sample : e.submit) { sample += build_per_replay; }
    }

    // ---- validation ------------------------------------------------------------------
    // B and C run on the same imported module as A, so bit-exactness is required;
    // it is also what proves the merged layers did not race.
    //
    // Every route of this benchmark produces the same values, so comparing against
    // whatever the previous route left in the destination buffer would pass even if a
    // route executed nothing at all. The destination is therefore cleared before
    // every validation batch: a route that does not run now leaves zeros behind and
    // fails its check instead of silently passing on someone else's output.
    luisa::vector<float4> zero_dst(dst_size, float4{0.f, 0.f, 0.f, 0.f});
    auto clear_vk_dst = [&] { stream << vk_dst.copy_from(luisa::span{zero_dst}) << synchronize(); };
    auto clear_cuda_dst = [&] { cuda_stream << cuda_dst.copy_from(luisa::span{zero_dst}) << synchronize(); };
    reorder_ext->set_command_reorder_enabled(true);
    clear_vk_dst();
    stream << build_vk_cuda_batch().commit() << synchronize();
    auto vk_cuda_reordered = download(stream, vk_dst);
    reorder_ext->set_command_reorder_enabled(false);
    clear_vk_dst();
    stream << build_vk_cuda_batch().commit() << synchronize();
    auto vk_cuda_serialized = download(stream, vk_dst);
    reorder_ext->set_command_reorder_enabled(true);
    clear_vk_dst();
    stream << build_vk_native_batch().commit() << synchronize();
    auto vk_native = download(stream, vk_dst);
    reorder_ext->set_command_reorder_enabled(true);

    auto reference_span = luisa::span<const float4>{reference};
    // Group F: the multi-stream batch must reproduce the same reference bit for bit
    // wherever the batch really is independent. In mode 1 the rotating writes are
    // genuine write-after-write chains between dispatches, and separate CUDA streams
    // give no ordering between them at all, so the "batch" is a race by construction
    // there: the group is timed, but the check is skipped and said so out loud
    // instead of being papered over.
    auto multistream_check_applies = (opt.mode & kOutputRotate) == 0u;
    luisa::vector<float4> multistream_result;
    auto multistream_matches = true;
    {
        clear_cuda_dst();
        auto lists = build_cuda_multi_batch();
        for (auto i = size_t{0}; i < multi_stream_ptrs.size(); i++) {
            *multi_stream_ptrs[i] << lists[i].commit();
        }
        for (auto *s : multi_stream_ptrs) { *s << synchronize(); }
        multistream_result = download(cuda_stream, cuda_dst);
        multistream_matches = multistream_check_applies &&
                              bitwise_equal(luisa::span<const float4>{multistream_result}, reference_span);
    }
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
      // The graph replays group A's exact commands from the exact same buffers,
      // so its output must be bit-identical to the reference as well.
      luisa::vector<float4> graph_result;
      auto graph_matches = true;
      if (have_graph) {
          clear_cuda_dst();
          launch_graph();
          cuda_stream << synchronize();
          graph_result = download(cuda_stream, cuda_dst);
          graph_matches = bitwise_equal(luisa::span<const float4>{graph_result}, reference_span);
      }


  // ---- report ------------------------------------------------------------------------
  auto a_batch = summarize(a.batch);
  auto b_batch = summarize(b.batch);
  auto c_batch = summarize(c.batch);
  auto d_batch = summarize(d.batch);
  auto a_submit = summarize(a.submit);
  auto f_batch = summarize(f.batch);
  auto f_submit = summarize(f.submit);
  LUISA_INFO("===========================================================================");
  LUISA_INFO("{:<38}{:>10}{:>10}{:>10}{:>10} |{:>10}{:>10}{:>10}{:>10}",
             "batch (ms) / host submit (ms)", "min", "median", "mean", "max",
             "min", "median", "mean", "max");
  print_row({"A cuda backend (no reorder)", a_batch, summarize(a.submit)});
  print_row({"B vk+cuda kernel, reorder OFF", b_batch, summarize(b.submit)});
  print_row({"C vk+cuda kernel, reorder ON", c_batch, summarize(c.submit)});
  print_row({"D vk native kernel, reorder ON", d_batch, summarize(d.submit)});
  if (have_graph) {
      print_row({"E cuda graph (host incl. build)", summarize(e.batch), summarize(e.submit)});
  }
  // Group F is the only route whose name depends on the run: name it after the
  // stream count it was given so a sweep of `streams` values stays readable.
  auto f_name = luisa::format("F {} cuda streams (no reorder)", opt.streams);
  print_row({f_name, f_batch, f_submit});
  LUISA_INFO("===========================================================================");
  LUISA_INFO("Reorder effect inside the vk+CUDA route (B/C): {:.2f}x", b_batch.median / c_batch.median);
  LUISA_INFO("vs. the cuda backend (A/C) : {:.2f}x", a_batch.median / c_batch.median);
  LUISA_INFO("vk+CUDA route vs native vk route (D/C) : {:.2f}x", d_batch.median / c_batch.median);
  // Group F is the multi-stream answer to the same problem: it is the same batch as
  // A, so A/F reads "what splitting the batch by hand buys over one stream", and
  // C/F and E/F read "how much of the reorder/graph win is left to the caller's own
  // bookkeeping". Its submit column is the cost of recording and queueing the same
  // number of commands, just into K lists instead of one.
  LUISA_INFO("Multi-stream cuda, {} streams vs. single stream (A/F) : {:.2f}x  "
             "(A {:.3f} ms vs F {:.3f} ms)",
             opt.streams, a_batch.median / f_batch.median, a_batch.median, f_batch.median);
  LUISA_INFO("Multi-stream cuda, {} streams vs. reordered vk+CUDA (C/F) : {:.2f}x  "
             "(C {:.3f} ms vs F {:.3f} ms)",
             opt.streams, c_batch.median / f_batch.median, c_batch.median, f_batch.median);
  LUISA_INFO("Multi-stream cuda host submit vs. one stream (A/F) : {:.2f}x  "
             "({:.3f} ms vs {:.3f} ms for the same {} commands).",
             a_submit.median / f_submit.median, a_submit.median, f_submit.median,
             opt.dispatches);
  if (have_graph) {
      auto e_batch = summarize(e.batch);
      auto e_submit = summarize(e.submit);
      auto e_replay_submit_stats = summarize(e_replay_submit);
      auto e_replay_total = std::accumulate(e_replay_submit.begin(), e_replay_submit.end(), 0.0);
      LUISA_INFO("CUDA graph route (E): one-off host build (dependency analysis + instantiate) "
                 "= {:.3f} ms, charged to the submit column above as {:.3f} ms per replay "
                 "({} measured replays).",
                 graph_build_ms, graph_build_ms / static_cast<double>(e_replay_submit.size()),
                 e_replay_submit.size());
      LUISA_INFO("Graph route total host time over the measured batch: {:.3f} ms "
                 "(build {:.3f} + {} replays {:.3f}) = {:.3f} ms per replay incl. build.",
                 graph_build_ms + e_replay_total, graph_build_ms, e_replay_submit.size(),
                 e_replay_total,
                 (graph_build_ms + e_replay_total) / static_cast<double>(e_replay_submit.size()));
      LUISA_INFO("Graph replay vs. naive cuda stream, batch wall time (A/E) : {:.2f}x",
                 a_batch.median / e_batch.median);
      LUISA_INFO("Graph replay vs. naive cuda stream, host submit incl. build (A/E): {:.2f}x  "
                 "({:.3f} ms vs {:.3f} ms)",
                 a_submit.median / e_submit.median, a_submit.median, e_submit.median);
      LUISA_INFO("Graph replay vs. naive cuda stream, host submit of the replay alone (A/E): {:.2f}x",
                 a_submit.median / e_replay_submit_stats.median);
      LUISA_INFO("Graph replay vs. reordered vk+CUDA (C/E)                  : {:.2f}x",
                 c_batch.median / e_batch.median);
      LUISA_INFO("Graph replay vs. multi-stream cuda (E/F) over {} streams     : {:.2f}x  "
                 "(E {:.3f} ms vs F {:.3f} ms)",
                 opt.streams, e_batch.median / f_batch.median, e_batch.median, f_batch.median);
      // The graph's win is on the host: it replaces the per-dispatch submit cost
      // of route A with one cuGraphLaunch. Amortise the one-off build against
      // that per-replay host saving, measured WITHOUT the build charge so both
      // sides are like-for-like; when the batch is device-bound the wall times
      // match (A/E ~ 1.0x above) and only the host time is saved.
      auto host_saving = a_submit.median - e_replay_submit_stats.median;
      LUISA_INFO("Graph build amortised against the replay's host saving: {:.3f} ms one-off vs "
                 "{:.3f} ms saved per replay vs A (~{:.0f} replays to pay back the build).",
                 graph_build_ms, host_saving,
                 host_saving > 1e-6
                     ? graph_build_ms / host_saving
                     : std::numeric_limits<double>::infinity());
  }
    LUISA_INFO("Checks: vk+CUDA reordered == cuda reference = {}, strictly ordered == "
               "reference = {}, outputs finite = {}, one distinct sub-range per dispatch "
               "= {}, vk native max relative deviation = {:.3e} (< {:.1e} required).",
               reordered_matches, serialized_matches, all_finite, distinct_ranges,
               native_deviation, kNativeTolerance);
    if (have_graph) {
        LUISA_INFO("CUDA graph (E) replay == cuda reference (bit-exact): {}.", graph_matches);
    }
    if (multistream_check_applies) {
        LUISA_INFO("CUDA multi-stream (F) batch over {} streams == cuda reference "
                   "(bit-exact): {}.",
                   opt.streams, multistream_matches);
    } else {
        LUISA_INFO("CUDA multi-stream (F) batch output NOT checked: mode {} contains genuine "
                   "write-after-write chains between dispatches, and separate CUDA streams "
                   "impose no order between them - the multi-stream batch is a race there, "
                   "which is exactly the point this group is here to show.",
                   opt.mode);
    }

    auto failed = false;
    if (!reordered_matches || !serialized_matches || !all_finite || !distinct_ranges ||
        native_deviation > kNativeTolerance || !graph_matches ||
        (multistream_check_applies && !multistream_matches)) {
        LUISA_ERROR_WITH_LOCATION(
            "Comparison validation FAILED: both Vulkan CUDA-launch groups must reproduce "
            "the strictly ordered cuda-backend result exactly, produce finite outputs and "
            "write one distinct sub-range per dispatch; the Vulkan-compiled group must stay "
            "within the numerical tolerance; the CUDA-graph replay (if built) must match the "
            "reference bit for bit; and the multi-stream batch must match it too wherever "
            "the batch is hazard-free (modes 0/2).");
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
                "in the same strictly serialized regime as A. A declared READ is a "
                "read-only contract that must NOT serialize the {} dispatches over the "
                "shared input of mode {} - check that command reordering is actually "
                "enabled (verbose run: look for 'N commands -> M layers'), and that the "
                "launches declare their usages (read()/write() or CudaArg) instead of "
                "defaulting every argument to READ_WRITE. Rerun with mode {} (private "
                "input ranges, same device work) to isolate the effect.",
                b_batch.median, c_batch.median, b_batch.median / c_batch.median,
                opt.dispatches, opt.mode, opt.mode ^ kPrivateInput);
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
    // Where the multi-stream route stands, independent of the A/C verdict: it is a
    // caller-side answer to the same problem, so its distance from the reorder route
    // is the number that matters (both run the identical batch, with no barrier
    // structure in between - only the stream split differs).
    LUISA_INFO("Multi-stream summary ({} streams over {} dispatches): F {:.3f} ms/batch = "
               "{:.3f} ms per dispatch, {:.2f}x the reorder route (C {:.3f} ms/batch = "
               "{:.3f} ms per dispatch) and A {:.3f} ms per dispatch.",
               opt.streams, opt.dispatches, f_batch.median,
               f_batch.median / static_cast<double>(opt.dispatches),
               f_batch.median / c_batch.median,
               c_batch.median, c_batch.median / static_cast<double>(opt.dispatches),
               a_batch.median / static_cast<double>(opt.dispatches));
    // The verdict is printed above; the resources go through their normal
    // destructors, which is also what keeps this benchmark honest about the
    // teardown paths (Context/device/stream destruction plus the imported CUDA
    // module and its Vulkan-side wrapper) that the vk + CUDA route uses.
    return failed ? 1 : 0;
}
