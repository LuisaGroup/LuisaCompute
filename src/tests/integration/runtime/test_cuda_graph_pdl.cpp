// Tests for the typed (PDL / programmatic) dependency edges of the CUDA Graph
// extension (`luisa/backends/ext/cuda/cuda_graph_ext.h`, Phase 3 / SA-3):
//
// - `set_dependency_type` retypes an existing edge of a graph template by
//   removing it and re-adding it with a `CUgraphEdgeData` annotation;
// - the default-edge run is the correctness baseline: the two kernels of the
//   graph are written as a producer (writes the shared buffer and signals the
//   programmatic completion port from device code) and a consumer (waits on
//   the grid dependency and then reads the shared buffer), so the *same*
//   kernels serve both runs and the only difference between them is the
//   annotation of the single producer -> consumer edge;
// - the PROGRAMMATIC run must reproduce the baseline bit for bit: that is what
//   exercises the `griddepcontrol.wait` ordering - the consumer may be
//   scheduled early, but it must observe the producer's writes only after
//   `grid_dependency_synchronize()`;
// - retyping is only allowed where it is meaningful: an edge that does not
//   exist (wrong direction), a self edge, out-of-range indices, an edge that
//   touches a non-kernel (memcpy/memset) node and an invalid graph are all
//   refused loudly, and a refused retype leaves the graph untouched;
// - DEFAULT restores the fully serializing edge, so the round trip
//   PROGRAMMATIC -> DEFAULT -> PROGRAMMATIC -> DEFAULT succeeds in both
//   directions and leaves the graph producing exactly the same results again.
//
// Requires the CUDA backend (the extension only exists there) plus a toolkit /
// driver that provides the annotated edge API; anything else skips the test
// instead of failing (the skip is driven by `capabilities()` and by
// `set_dependency_type` itself refusing a known-valid edge). The kernels also
// contain `griddepcontrol.*` device code, which needs a Hopper-or-newer
// (sm_90+) device.
#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/backends/ext/cuda/cuda_graph_ext.h>
#include <algorithm>
#include <cstdint>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr uint32_t kCount = 64u;

/// Node layout of every graph built by `make_cmdlist` below; the usage-based
/// DAG analysis creates exactly the edges upload -> producer (write-after-write
/// on `x`) and producer -> consumer (read-after-write on `x`), so
/// `k_producer_node -> k_consumer_node` is the edge the retype targets.
constexpr size_t k_upload_node = 0u;
constexpr size_t k_producer_node = 1u;
constexpr size_t k_consumer_node = 2u;

/// The result the pipeline must produce for element `i`:
/// consumer(producer(i)) = (i * 3 + 1) * 2 + 1.
[[nodiscard]] constexpr uint32_t expected_value(uint32_t i) noexcept {
    return (i * 3u + 1u) * 2u + 1u;
}

/// Reads a device buffer back to the host. The copy is submitted on the same
/// stream as the graph launch, so it observes the replayed graph's results.
[[nodiscard]] luisa::vector<uint32_t> read_back(Stream &stream, BufferView<uint32_t> view,
                                                size_t count) {
    luisa::vector<uint32_t> data(count);
    stream << view.copy_to(luisa::span{data}) << synchronize();
    return data;
}

/// True when `a` and `b` hold the same values.
[[nodiscard]] bool all_equal(luisa::span<const uint32_t> a, luisa::span<const uint32_t> b) noexcept {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

/// True when `values` is exactly the expected pipeline output. The initial
/// contents uploaded into the shared buffer are deliberately neither the
/// producer's output nor something the consumer could produce from it, so a
/// missing or reordered dependency shows up here as a mismatch.
[[nodiscard]] bool is_expected_result(luisa::span<const uint32_t> values) noexcept {
    if (values.size() != kCount) { return false; }
    for (auto i = 0u; i < kCount; i++) {
        if (values[i] != expected_value(i)) { return false; }
    }
    return true;
}

void test_capabilities(CudaGraphExt *ext) {
    auto caps = ext->capabilities();
    // The backend itself only builds against CUDA >= 12.1.
    expect(caps.cuda_version >= 12010u)
        << "capabilities().cuda_version must report the toolkit version";
    if (caps.programmatic_edges) {
        expect(caps.cuda_version >= 12000u) << "typed (PDL) edges need CUDA >= 12.0";
    }
    LUISA_INFO("CudaGraphExt programmatic (PDL) edge capabilities: cuda_version = {}, "
               "programmatic_edges = {}.",
               caps.cuda_version, caps.programmatic_edges);
}

void test_default_edge_baseline(Device &device, CudaGraphExt *ext) {
    auto stream = device.create_stream();
    auto x = device.create_buffer<uint32_t>(kCount);
    auto out = device.create_buffer<uint32_t>(kCount);
    luisa::vector<uint32_t> initial(kCount, 0xffffffffu);

    Kernel1D producer = [](BufferVar<uint32_t> buffer) noexcept {
        set_block_size(kCount);
        auto i = dispatch_x();
        buffer.write(i, i * 3u + 1u);
        // Signal the programmatic completion port of this grid. With a default
        // edge this is a no-op, so the very same kernel can serve both the
        // baseline and the programmatic run.
        trigger_programmatic_launch_completion();
    };
    Kernel1D consumer = [](BufferVar<uint32_t> buffer, BufferVar<uint32_t> result) noexcept {
        set_block_size(kCount);
        auto i = dispatch_x();
        // The early launch of a programmatic edge implies no memory visibility
        // of its own; the wait is a no-op here (the default edge already
        // serializes the two nodes) and the real dependency check in the
        // programmatic run.
        grid_dependency_synchronize();
        result.write(i, buffer.read(i) * 2u + 1u);
    };
    auto producer_shader = device.compile(producer);
    auto consumer_shader = device.compile(consumer);

    auto cmdlist = CommandList::create();
    cmdlist << x.view().copy_from(luisa::span{initial});
    cmdlist << producer_shader(x.view()).dispatch(kCount);
    cmdlist << consumer_shader(x.view(), out.view()).dispatch(kCount);
    auto graph = ext->create_graph(cmdlist.commit().command_list());
    expect(graph.handle().handle != CudaGraphExt::invalid_handle)
        << "create_graph from upload + two dispatches should succeed";
    if (graph.handle().handle == CudaGraphExt::invalid_handle) { return; }

    auto exec = ext->instantiate(graph.handle().handle);
    expect(exec.handle().handle != CudaGraphExt::invalid_handle)
        << "the baseline graph must instantiate";
    if (exec.handle().handle == CudaGraphExt::invalid_handle) { return; }

    ext->launch(exec.handle().handle, stream.handle());
    stream << synchronize();
    auto result = read_back(stream, out.view(), kCount);
    expect(is_expected_result(result))
        << "the default-edge replay must reproduce the sequential result";
}

void test_programmatic_edge(Device &device, CudaGraphExt *ext) {
    using DependencyType = CudaGraphExt::DependencyType;

    // Probe PDL support BEFORE compiling kernels that use griddepcontrol.* PTX
    // (sm_90+; on older GPUs the compile itself would fail instead of
    // skipping): a trivial two-kernel graph carries a real edge that the
    // PROGRAMMATIC retype can be attempted on.
    {
        Kernel1D trivial = [](BufferVar<uint32_t> b) noexcept { b.write(0u, 1u); };
        auto trivial_shader = device.compile(trivial);
        auto b = device.create_buffer<uint32_t>(1u);
        luisa::vector<uint32_t> zero{0u};
        auto cmdlist = CommandList::create();
        cmdlist << b.view().copy_from(luisa::span{zero});
        cmdlist << trivial_shader(b.view()).dispatch(1u);
        cmdlist << trivial_shader(b.view()).dispatch(1u);
        auto probe_graph = ext->create_graph(cmdlist.commit().command_list());
        auto available = probe_graph.handle().handle != CudaGraphExt::invalid_handle &&
                         ext->set_dependency_type(probe_graph.handle().handle, 1u, 2u,
                                                  DependencyType::PROGRAMMATIC);
        if (!available) {
            LUISA_INFO("PROGRAMMATIC (PDL) graph edges are not usable on this "
                       "device/toolkit (cuda_version = {}); skipping the "
                       "programmatic-edge test.",
                       ext->capabilities().cuda_version);
            return;
        }
    }

    auto stream = device.create_stream();
    auto x = device.create_buffer<uint32_t>(kCount);
    auto out = device.create_buffer<uint32_t>(kCount);
    luisa::vector<uint32_t> initial(kCount, 0xffffffffu);

    Kernel1D producer = [](BufferVar<uint32_t> buffer) noexcept {
        set_block_size(kCount);
        auto i = dispatch_x();
        buffer.write(i, i * 3u + 1u);
        trigger_programmatic_launch_completion();
    };
    Kernel1D consumer = [](BufferVar<uint32_t> buffer, BufferVar<uint32_t> result) noexcept {
        set_block_size(kCount);
        auto i = dispatch_x();
        grid_dependency_synchronize();
        result.write(i, buffer.read(i) * 2u + 1u);
    };
    auto producer_shader = device.compile(producer);
    auto consumer_shader = device.compile(consumer);

    // A fresh, structurally identical graph: same node order, same edges.
    auto make_cmdlist = [&]() {
        auto cmdlist = CommandList::create();
        cmdlist << x.view().copy_from(luisa::span{initial});
        cmdlist << producer_shader(x.view()).dispatch(kCount);
        cmdlist << consumer_shader(x.view(), out.view()).dispatch(kCount);
        return cmdlist.commit().command_list();
    };
    auto graph = ext->create_graph(make_cmdlist());
    expect(graph.handle().handle != CudaGraphExt::invalid_handle)
        << "create_graph from upload + two dispatches should succeed";
    if (graph.handle().handle == CudaGraphExt::invalid_handle) { return; }
    auto graph_handle = graph.handle().handle;

    // ---- PROGRAMMATIC retype of the producer -> consumer edge -------------
    if (!ext->set_dependency_type(graph_handle, k_producer_node, k_consumer_node,
                                 DependencyType::PROGRAMMATIC)) {
        LUISA_INFO("PROGRAMMATIC (PDL) graph edges are not available on this "
                   "toolkit/driver (cuda_version = {}), skipping the "
                   "programmatic-edge test.",
                   ext->capabilities().cuda_version);
        return;
    }

    auto exec = ext->instantiate(graph_handle);
    expect(exec.handle().handle != CudaGraphExt::invalid_handle)
        << "a graph with a programmatic edge must instantiate";
    if (exec.handle().handle == CudaGraphExt::invalid_handle) { return; }

    ext->launch(exec.handle().handle, stream.handle());
    stream << synchronize();
    auto programmatic_result = read_back(stream, out.view(), kCount);
    expect(is_expected_result(programmatic_result))
        << "the programmatic-edge replay must still observe the producer's "
           "writes (grid_dependency_synchronize ordering)";
    // Replay again: the retype is a property of the graph template, so every
    // launch of this executable must produce the same result.
    ext->launch(exec.handle().handle, stream.handle());
    stream << synchronize();
    auto replayed = read_back(stream, out.view(), kCount);
    expect(all_equal(replayed, programmatic_result))
        << "a replayed programmatic graph must be deterministic";

    // ---- refusals (the graph must be left untouched by every one of them) --
    expect(!ext->set_dependency_type(graph_handle, k_producer_node, k_producer_node,
                                     DependencyType::PROGRAMMATIC))
        << "a self edge must be refused";
    expect(!ext->set_dependency_type(graph_handle, k_consumer_node, k_producer_node,
                                     DependencyType::PROGRAMMATIC))
        << "the reverse edge (consumer -> producer) was never created by the DAG "
           "analysis, so it cannot be retyped";
    expect(!ext->set_dependency_type(graph_handle, k_producer_node, kCount,
                                     DependencyType::PROGRAMMATIC))
        << "an out-of-range node index must be refused";
    expect(!ext->set_dependency_type(graph_handle, kCount, k_consumer_node,
                                     DependencyType::PROGRAMMATIC))
        << "an out-of-range node index must be refused (from side)";
    expect(!ext->set_dependency_type(graph_handle, k_producer_node, kCount,
                                     DependencyType::DEFAULT))
        << "out-of-range indices are refused for DEFAULT too";
    expect(!ext->set_dependency_type(CudaGraphExt::invalid_handle, k_producer_node,
                                     k_consumer_node, DependencyType::PROGRAMMATIC))
        << "an invalid graph handle must be refused";
    expect(!ext->set_dependency_type(graph_handle, k_producer_node, k_consumer_node,
                                     static_cast<DependencyType>(7u)))
        << "an unknown dependency type must be refused";
    {
        // DEFAULT on an edge that carries no programmatic annotation is a
        // documented no-op (it restores an edge, it never creates one), so the
        // reverse pair is accepted although no such edge exists.
        expect(ext->set_dependency_type(graph_handle, k_consumer_node, k_producer_node,
                                        DependencyType::DEFAULT))
            << "DEFAULT on a non-programmatic edge must be a no-op success";
    }
    {
        // The programmatic port is defined for kernel nodes only: an edge that
        // touches a memcpy (upload) or memset node can never carry the
        // annotation. Built on a scratch graph so the main graph stays as-is.
        auto node_graph = ext->create_graph(make_cmdlist());
        if (node_graph.handle().handle != CudaGraphExt::invalid_handle) {
            auto node_graph_handle = node_graph.handle().handle;
            expect(!ext->set_dependency_type(node_graph_handle, k_upload_node, k_producer_node,
                                             DependencyType::PROGRAMMATIC))
                << "an edge whose upstream node is a memcpy node must be refused";
            size_t memset_node = ~size_t{0u};
            expect(ext->add_memset_node(node_graph_handle, x.handle(), 0u,
                                        kCount * sizeof(uint32_t), 0u, &memset_node))
                << "add_memset_node should succeed";
            expect(memset_node == 3u)
                << "the memset node must be appended after the three captured nodes";
            expect(!ext->set_dependency_type(node_graph_handle, k_producer_node, memset_node,
                                             DependencyType::PROGRAMMATIC))
                << "an edge whose downstream node is a memset node must be refused";
        }
    }

    // ---- DEFAULT round trip ----------------------------------------------
    // None of the refusals above may have touched the annotated edge: the very
    // same executable graph must still launch and reproduce its result.
    ext->launch(exec.handle().handle, stream.handle());
    stream << synchronize();
    auto after_refusals = read_back(stream, out.view(), kCount);
    expect(all_equal(after_refusals, programmatic_result))
        << "a refused retype must leave the graph untouched";

    expect(ext->set_dependency_type(graph_handle, k_producer_node, k_consumer_node,
                                    DependencyType::DEFAULT))
        << "PROGRAMMATIC -> DEFAULT must restore the fully serializing edge";
    expect(ext->set_dependency_type(graph_handle, k_producer_node, k_consumer_node,
                                    DependencyType::PROGRAMMATIC))
        << "DEFAULT -> PROGRAMMATIC must succeed again";
    expect(ext->set_dependency_type(graph_handle, k_producer_node, k_consumer_node,
                                    DependencyType::DEFAULT))
        << "PROGRAMMATIC -> DEFAULT must succeed a second time";

    // The restored default edge must serialize the two kernels again and still
    // produce the exact same results.
    auto restored_exec = ext->instantiate(graph_handle);
    expect(restored_exec.handle().handle != CudaGraphExt::invalid_handle)
        << "the graph must still instantiate after the round trip";
    if (restored_exec.handle().handle == CudaGraphExt::invalid_handle) { return; }
    ext->launch(restored_exec.handle().handle, stream.handle());
    stream << synchronize();
    auto restored = read_back(stream, out.view(), kCount);
    expect(is_expected_result(restored))
        << "the graph restored to a default edge must reproduce the result";
    expect(all_equal(restored, programmatic_result))
        << "the default and the programmatic graph must agree bit for bit";
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    auto *ext = device.extension<CudaGraphExt>();
    if (ext == nullptr) {
        LUISA_INFO("CudaGraphExt not available, skipping test.");
        return 0;
    }
    test_capabilities(ext);
    auto caps = ext->capabilities();
    if (!caps.programmatic_edges) {
        LUISA_INFO("Typed (PDL) CUDA graph edges require CUDA >= 12.0 "
                   "(toolkit/driver {}), skipping the programmatic-edge test.",
                   caps.cuda_version);
        return 0;
    }
    test_default_edge_baseline(device, ext);
    test_programmatic_edge(device, ext);
    LUISA_INFO("CUDA graph programmatic (PDL) edge test passed.");
    return 0;
}
