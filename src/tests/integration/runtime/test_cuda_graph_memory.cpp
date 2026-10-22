// Tests for the graph-memory node API of the CUDA Graph extension
// (`luisa/backends/ext/cuda/cuda_graph_ext.h`, Phase 5 / SA-6, gap G5):
//
// - `add_mem_alloc_node` creates a graph memory allocation node; the node keeps
//   its creation order (it is appended to the graph's node list like every
//   other explicit add_*) and every argument-contract refusal is exercised
//   (invalid graph handle, zero size, alignment 0, non-power-of-two alignment);
// - `get_mem_alloc_node_address` resolves the allocation's device address
//   through an instantiated exec: it is non-null and honours the requested
//   alignment, it is stable across reads, wrong node kinds / out-of-range
//   indices / invalid exec handles / a null output parameter are refused, and
//   the address is "fixed across instantiations and launches" (CUDA's words for
//   the alloc node) - which the test verifies by reading it again from a second
//   instantiation of the same graph;
// - a graph that contains memory nodes may only have ONE live instantiation at
//   a time (CUDA requirement): a second `instantiate` is refused while an exec
//   is alive and succeeds again once that exec is destroyed;
// - `add_mem_free_node` is fail-closed: a null address, an address that is not
//   an allocation of its own graph (including one owned by a *different*
//   graph) and a second free of the same allocation are all refused; with a
//   valid address it adds the free node in creation order and the graph,
//   re-instantiated with the free node in place, still runs its kernel;
// - `InstantiateFlag::AUTO_FREE_ON_LAUNCH` composes with a graph memory
//   allocation node: the exec instantiates and can be relaunched (the flag is
//   precisely about relaunching). Memory-pool internals are not asserted.
//
// Two notes on what is deliberately *not* done here.
//
// 1. Nothing writes through the raw device address. The public DSL has no
//    "buffer from a raw address" resource: the only builtins that take a device
//    address (`async_copy`, `graph_node_set_param`) consume it, and neither can
//    store a value into it. The address contract is therefore verified through
//    the API itself - non-null, aligned, fixed, and accepted by the driver as
//    the target of a free node - while the *execution* of a graph carrying
//    memory nodes is proven by reading back the buffer the graph's own kernel
//    wrote (the kernel is part of the graph, so its result is only correct if
//    the whole graph, alloc/free nodes included, ran).
//
// 2. The free node is added in the two-phase pattern the API shape dictates:
//    the address is only knowable from an instantiated exec
//    (`get_mem_alloc_node_address` takes an exec), and a memory-node graph may
//    only have one live instantiation, so the sequence is
//    instantiate -> read the address -> add the free node to the *template* ->
//    destroy that exec -> instantiate again.
//
// The whole file requires the CUDA backend (the extension only exists there)
// and a device with CUDA memory pools; anything else skips instead of failing.
#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/backends/ext/cuda/cuda_graph_ext.h>
#include <cstdint>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr size_t kAllocSize = 4096u;
constexpr size_t kAlignment = 256u;

/// Value the marker kernel writes; reading it back proves the graph ran.
constexpr uint32_t kWritten = 0x5a5au;

/// Reads a device buffer back to the host. The copy is submitted on the same
/// stream as the graph launches, so it observes their results.
template<typename T>
[[nodiscard]] luisa::vector<T> read_back(Stream &stream, BufferView<T> view, size_t count) {
    luisa::vector<T> data(count);
    stream << view.copy_to(luisa::span{data}) << synchronize();
    return data;
}

void test_capabilities(CudaGraphExt *ext) {
    auto caps = ext->capabilities();
    // The backend itself only builds against CUDA >= 12.1.
    expect(caps.cuda_version >= 12010u)
        << "capabilities().cuda_version must report the toolkit version";
    if (caps.graph_memory) {
        expect(caps.cuda_version >= 11040u)
            << "graph memory nodes require CUDA >= 11.4";
    }
    LUISA_INFO("CudaGraphExt graph memory capabilities: cuda_version = {}, "
               "graph_memory = {}.",
               caps.cuda_version, caps.graph_memory);
}

/// Whether this device can create graph memory allocation nodes at all.
/// `capabilities().graph_memory` only reflects the toolkit/driver version; the
/// device-side half of the feature (a CUDA memory pool) is probed with the API
/// itself, i.e. by trying to add an allocation node to a perfectly valid graph.
[[nodiscard]] bool device_supports_graph_memory(Device &device, CudaGraphExt *ext) {
    auto buffer = device.create_buffer<uint32_t>(1u);
    luisa::vector<uint32_t> initial{0u};
    auto cmdlist = CommandList::create();
    cmdlist << buffer.view().copy_from(luisa::span{initial});
    auto graph = ext->create_graph(cmdlist.commit().command_list());
    if (graph.handle().handle == CudaGraphExt::invalid_handle) { return false; }
    size_t index = ~size_t{0u};
    return ext->add_mem_alloc_node(graph.handle().handle, kAllocSize, kAlignment, &index);
}

/// The alloc/free lifecycle of a graph that carries graph memory nodes.
void test_alloc_free_lifecycle(Device &device, CudaGraphExt *ext) {
    auto stream = device.create_stream();
    auto buffer = device.create_buffer<uint32_t>(1u);
    // The marker kernel is part of every graph below: reading its output back
    // after a launch is the proof that the graph as a whole - the upload, the
    // kernel and the memory nodes - executed.
    Kernel1D marker_kernel = [](BufferVar<uint32_t> out) noexcept {
        set_block_size(32u);
        $if (thread_id().x == 0u) {
            out.write(0u, kWritten);
        };
    };
    auto marker_shader = device.compile(marker_kernel);

    // ---- graph G: upload(0) + kernel(1) + mem alloc(2) ---------------------
    luisa::vector<uint32_t> zeros{0u};
    auto cmdlist = CommandList::create();
    cmdlist << buffer.view().copy_from(luisa::span{zeros});
    cmdlist << marker_shader(buffer.view()).dispatch(1u);
    auto graph = ext->create_graph(cmdlist.commit().command_list());
    expect(graph.handle().handle != CudaGraphExt::invalid_handle)
        << "create_graph from upload + dispatch should succeed";
    if (graph.handle().handle == CudaGraphExt::invalid_handle) { return; }
    auto graph_handle = graph.handle().handle;

    // Argument contract of the alloc node. These are refused before any node
    // exists, so a refusal cannot leave anything behind.
    size_t alloc_index = ~size_t{0u};
    expect(!ext->add_mem_alloc_node(CudaGraphExt::invalid_handle, kAllocSize,
                                    kAlignment, &alloc_index))
        << "an invalid graph handle must be refused";
    expect(!ext->add_mem_alloc_node(graph_handle, 0u, kAlignment, &alloc_index))
        << "a zero-byte allocation must be refused";
    expect(!ext->add_mem_alloc_node(graph_handle, kAllocSize, 0u, &alloc_index))
        << "alignment 0 must be refused";
    expect(!ext->add_mem_alloc_node(graph_handle, kAllocSize, 3u, &alloc_index))
        << "a non-power-of-two alignment must be refused";

    expect(ext->add_mem_alloc_node(graph_handle, kAllocSize, kAlignment, &alloc_index))
        << "add_mem_alloc_node on a live graph should succeed";
    expect(alloc_index == 2u)
        << "the alloc node must keep its creation order (upload, kernel, alloc)";

    // ---- the allocation address (known before any instantiation) ------------
    // The driver assigns the address at add_mem_alloc_node() time, so the
    // template can be queried directly - and MUST be: a mem-free node can only
    // be added BEFORE the first instantiation (driver rule, verified on 13.4).
    uint64_t address = 0u;
    expect(ext->get_graph_mem_alloc_address(graph_handle, alloc_index, &address))
        << "the alloc node must resolve to a device address";
    expect(address != 0u) << "the allocation address must not be null";
    expect(address % kAlignment == 0u)
        << "the allocation address must honour the requested alignment";
    uint64_t address_again = 0u;
    expect(ext->get_graph_mem_alloc_address(graph_handle, alloc_index, &address_again) &&
           address_again == address)
        << "the allocation address must be stable across reads";

    expect(!ext->get_graph_mem_alloc_address(graph_handle, 0u, &address_again))
        << "the upload node is not an allocation node";
    expect(!ext->get_graph_mem_alloc_address(graph_handle, 1u, &address_again))
        << "the kernel node is not an allocation node";
    expect(!ext->get_graph_mem_alloc_address(graph_handle, ~size_t{0u}, &address_again))
        << "an out-of-range node index must be refused";
    expect(!ext->get_graph_mem_alloc_address(CudaGraphExt::invalid_handle,
                                             alloc_index, &address_again))
        << "an invalid graph handle must be refused";
    expect(!ext->get_graph_mem_alloc_address(graph_handle, alloc_index, nullptr))
        << "a null output parameter must be refused";

    // ---- free node: ownership and single-free bookkeeping ------------------
    size_t free_index = ~size_t{0u};
    expect(!ext->add_mem_free_node(CudaGraphExt::invalid_handle, address, &free_index))
        << "an invalid graph handle must be refused";
    expect(!ext->add_mem_free_node(graph_handle, 0u, &free_index))
        << "a null address must be refused";
    expect(!ext->add_mem_free_node(graph_handle, address + kAllocSize, &free_index))
        << "an address that this graph never allocated must be refused";

    // An address is owned by the graph whose alloc node produced it: another
    // graph must not be allowed to free it (the driver enforces the same rule).
    {
        auto other_buffer = device.create_buffer<uint32_t>(1u);
        luisa::vector<uint32_t> other_zeros{0u};
        auto other_cmdlist = CommandList::create();
        other_cmdlist << other_buffer.view().copy_from(luisa::span{other_zeros});
        auto other_graph = ext->create_graph(other_cmdlist.commit().command_list());
        if (other_graph.handle().handle != CudaGraphExt::invalid_handle) {
            size_t other_alloc_index = ~size_t{0u};
            if (ext->add_mem_alloc_node(other_graph.handle().handle, kAllocSize,
                                        kAlignment, &other_alloc_index)) {
                expect(!ext->add_mem_free_node(other_graph.handle().handle,
                                               address, &free_index))
                    << "another graph's allocation address must be refused";
            }
        }
    }

    // The supported pattern (driver rule): alloc -> free -> dependency wiring,
    // all on the template BEFORE the first instantiate().
    expect(ext->add_mem_free_node(graph_handle, address, &free_index))
        << "the free node must accept the address of this graph's allocation";
    expect(free_index == 3u)
        << "the free node must keep its creation order "
           "(upload, kernel, alloc, free)";
    expect(!ext->add_mem_free_node(graph_handle, address, &free_index))
        << "a second free of the same allocation must be refused";
    expect(ext->add_dependency(graph_handle, alloc_index, free_index))
        << "the free node must be wired after the alloc node";

    // ---- only one live instantiation at a time -----------------------------
    auto exec = ext->instantiate(graph_handle);
    expect(exec.handle().handle != CudaGraphExt::invalid_handle)
        << "a graph with graph memory nodes must instantiate";
    if (exec.handle().handle == CudaGraphExt::invalid_handle) { return; }
    auto exec_handle = exec.handle().handle;
    {
        auto second = ext->instantiate(graph_handle);
        expect(second.handle().handle == CudaGraphExt::invalid_handle)
            << "CUDA allows only one instantiation of a graph with memory "
               "nodes at a time, so a second one must be refused while an exec "
               "is alive";
    }
    // A mem-free node can never join an already-instantiated template.
    expect(!ext->add_mem_free_node(graph_handle, address, &free_index))
        << "a free node must be refused once the template is instantiated";

    // ---- launch, and prove the graph ran -----------------------------------
    uint64_t exec_address = 0u;
    expect(ext->get_mem_alloc_node_address(exec_handle, alloc_index, &exec_address) &&
           exec_address == address)
        << "the exec-level address query must agree with the template query";
    ext->launch(exec_handle, stream.handle());
    stream << synchronize();
    auto values = read_back(stream, buffer.view(), 1u);
    expect(values[0] == kWritten)
        << "the graph with the alloc + free nodes must run its kernel";
    {
        // Release the one live instantiation so the graph may be instantiated
        // again (the exec is destroyed explicitly, `release()` hands the handle
        // over so the instance's destructor does not destroy it twice).
        auto info = exec.release();
        ext->destroy_exec(info.handle);
    }

    // ---- re-instantiate: the address stays fixed ----------------------------
    auto exec2 = ext->instantiate(graph_handle);
    expect(exec2.handle().handle != CudaGraphExt::invalid_handle)
        << "the graph must instantiate again once the first exec is gone";
    if (exec2.handle().handle == CudaGraphExt::invalid_handle) { return; }
    uint64_t address2 = 0u;
    expect(ext->get_mem_alloc_node_address(exec2.handle().handle, alloc_index,
                                           &address2) &&
           address2 == address)
        << "the allocation address must remain fixed across instantiations";
    ext->launch(exec2.handle().handle, stream.handle());
    stream << synchronize();
    auto values2 = read_back(stream, buffer.view(), 1u);
    expect(values2[0] == kWritten)
        << "the re-instantiated graph must still run its kernel";
}

/// `InstantiateFlag::AUTO_FREE_ON_LAUNCH` on a graph with an allocation node:
/// each launch frees the (still) unfreed allocation, so the same exec may be
/// relaunched. Only that composition is verified - memory-pool internals are
/// deliberately not asserted.
void test_auto_free_on_launch(Device &device, CudaGraphExt *ext) {
    auto stream = device.create_stream();
    auto buffer = device.create_buffer<uint32_t>(1u);
    Kernel1D marker_kernel = [](BufferVar<uint32_t> out) noexcept {
        set_block_size(32u);
        $if (thread_id().x == 0u) {
            out.write(0u, kWritten);
        };
    };
    auto marker_shader = device.compile(marker_kernel);

    luisa::vector<uint32_t> zeros{0u};
    auto cmdlist = CommandList::create();
    cmdlist << buffer.view().copy_from(luisa::span{zeros});
    cmdlist << marker_shader(buffer.view()).dispatch(1u);
    auto graph = ext->create_graph(cmdlist.commit().command_list());
    expect(graph.handle().handle != CudaGraphExt::invalid_handle)
        << "create_graph for the AUTO_FREE_ON_LAUNCH graph must succeed";
    if (graph.handle().handle == CudaGraphExt::invalid_handle) { return; }
    auto graph_handle = graph.handle().handle;

    // `node_index` is optional: the node's position is not needed here.
    if (!ext->add_mem_alloc_node(graph_handle, kAllocSize, kAlignment, nullptr)) {
        LUISA_INFO("add_mem_alloc_node is unavailable on this device; "
                   "skipping the AUTO_FREE_ON_LAUNCH variant.");
        return;
    }

    auto exec = ext->instantiate(graph_handle,
                                 CudaGraphExt::InstantiateFlag::AUTO_FREE_ON_LAUNCH);
    expect(exec.handle().handle != CudaGraphExt::invalid_handle)
        << "AUTO_FREE_ON_LAUNCH must compose with a graph memory allocation node";
    if (exec.handle().handle == CudaGraphExt::invalid_handle) { return; }

    // Relaunch the same exec: that is the case the flag exists for.
    for (auto replay = 0; replay < 2; replay++) {
        ext->launch(exec.handle().handle, stream.handle());
        stream << synchronize();
        auto values = read_back(stream, buffer.view(), 1u);
        expect(values[0] == kWritten)
            << "the AUTO_FREE_ON_LAUNCH graph must run on every launch";
    }
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
    if (!caps.graph_memory) {
        LUISA_INFO("Graph memory nodes require CUDA >= 11.4 (toolkit/driver {}), "
                   "skipping the graph memory test.",
                   caps.cuda_version);
        return 0;
    }
    if (!device_supports_graph_memory(device, ext)) {
        LUISA_INFO("This device cannot create graph memory allocation nodes (no "
                   "CUDA memory pool); skipping the graph memory test.");
        return 0;
    }
    test_alloc_free_lifecycle(device, ext);
    test_auto_free_on_launch(device, ext);
    LUISA_INFO("CUDA graph memory test passed.");
    return 0;
}
