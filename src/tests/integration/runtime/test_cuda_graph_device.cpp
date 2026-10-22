// Tests for the device-side half of the CUDA Graph extension
// (`luisa/backends/ext/cuda/cuda_graph_ext.h`, Phase 4 / SA-4):
//
// G3 - device-side graph launch:
//   * an exec instantiated with `InstantiateFlag::DEVICE_LAUNCH` hands out a
//     device launch token through `get_device_launch_token`; an exec without
//     the flag (and an invalid handle) does not;
//   * a kernel running inside a host-launched graph can tail-launch that exec
//     from device code (`graph_device_launch`), and the launched child graph
//     really runs: the child writes its result plus a completion flag, which
//     the host polls with a bounded loop (a tail launch is applied to the
//     launching graph's environment, so the host-side completion of the parent
//     does not by itself tell us when the child's stores are visible);
//   * `current_graph_exec()` reports a non-zero exec handle inside a kernel
//     launched as part of a graph.
//
// G4 - device-updatable kernel nodes:
//   * `mark_node_device_updatable` opts a kernel node of a *graph template*
//     into device updatability and `get_device_node_handle` hands out the
//     driver's device node handle for the instantiated exec;
//   * a kernel in another (host-launched) graph patches the parameters of that
//     node from the device (`graph_node_set_param`), and the target graph,
//     launched afterwards from the host, observes the patched parameter;
//   * the restrictions of research doc 3.3 are enforced loudly: only kernel
//     nodes of a not-yet-instantiated template can opt in, whole-graph
//     `update()` on such a graph fails with UNSUPPORTED, and the graph cannot
//     be instantiated twice.
//
// The parameter that is patched is the *first* argument of the target kernel.
// A DSL kernel takes one by-value `struct alignas(16) Params` argument whose
// members are the kernel arguments in declaration order, each in a
// 16-byte-aligned slot (see `cuda_codegen_ast.cpp` and
// `GraphDagVisitor::pack_kernel_arguments`), so byte offset 0 of that parameter
// block is the first argument slot - a 4-byte `float` uniform here, making
// `offset = 0, size = 4` the exact patch.
//
// Everything requires a CUDA backend, a toolkit/driver that reports the
// corresponding `capabilities()` flags, and (for device launch) a
// Hopper-or-newer device; anything else skips instead of failing.
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
#include <thread>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr uint32_t kCount = 64u;

/// Completion magic the device-launched child graph writes once its result is
/// in place. The host polls the flag instead of assuming a timing.
constexpr uint32_t kDone = 0x5aa5a55au;

/// Upper bound of every host-side poll: the child graph of a device launch is
/// asynchronous with respect to the parent's host-visible completion.
constexpr uint32_t kPollIterations = 1000u;

/// Reads a device buffer back to the host. The copy is submitted on the same
/// stream as the graph launches, so it observes their results.
template<typename T>
[[nodiscard]] luisa::vector<T> read_back(Stream &stream, BufferView<T> view, size_t count) {
    luisa::vector<T> data(count);
    stream << view.copy_to(luisa::span{data}) << synchronize();
    return data;
}

/// Fills a device buffer (e.g. with an opaque handle) from the host.
template<typename T>
void write_host(Stream &stream, BufferView<T> view, const luisa::vector<T> &data) {
    stream << view.copy_from(luisa::span{data}) << synchronize();
}

/// True when the n-th element of `values` is the kernel's `scale`-scaled index.
[[nodiscard]] bool is_scaled(luisa::span<const uint32_t> values, uint32_t scale) noexcept {
    if (values.size() != kCount) { return false; }
    for (auto i = 0u; i < kCount; i++) {
        if (values[i] != i * scale) { return false; }
    }
    return true;
}

void test_capabilities(CudaGraphExt *ext) {
    auto caps = ext->capabilities();
    // The backend itself only builds against CUDA >= 12.1.
    expect(caps.cuda_version >= 12010u)
        << "capabilities().cuda_version must report the toolkit version";
    if (caps.device_launch) {
        expect(caps.cuda_version >= 12000u) << "device launch needs CUDA >= 12.0";
    }
    if (caps.device_updatable_nodes) {
        expect(caps.cuda_version >= 12040u) << "device-updatable nodes need CUDA >= 12.4";
    }
    LUISA_INFO("CudaGraphExt device capabilities: cuda_version = {}, "
               "device_launch = {}, device_updatable_nodes = {}.",
               caps.cuda_version, caps.device_launch, caps.device_updatable_nodes);
}

/// G4: a kernel node of a graph template is opted into device updatability; a
/// kernel running in a second, host-launched graph then patches the first
/// argument of that node from the device, and the target graph - uploaded
/// beforehand, launched after the patch - runs with the patched value.
///
/// CUDA requires a graph that carries device-updatable nodes to be uploaded
/// before it is launched, and the device-side update to be applied before that
/// launch, so the order is fixed: instantiate + upload A, run B (the patch),
/// then launch A.
void test_device_updatable_node(Device &device, CudaGraphExt *ext) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<uint32_t>(kCount);
    auto new_scale = device.create_buffer<float>(1u);

    // Target kernel: `scale` is the FIRST argument, so it lives in the first
    // slot of the packed parameter block (byte offset 0).
    Kernel1D scale_kernel = [](Var<float> scale, BufferVar<uint32_t> out) noexcept {
        set_block_size(kCount);
        auto i = dispatch_x();
        out.write(i, (i.cast<float>() * scale).cast<uint>());
    };
    auto scale_shader = device.compile(scale_kernel);

    // Patching kernel: overwrites `size` bytes at `offset` of the target
    // node's parameter block with the contents of the device buffer it is
    // handed. `node`, `offset` and `size` are opaque values computed on the
    // host (the node handle comes from get_device_node_handle).
    Kernel1D patch_kernel = [](Var<uint64_t> node, BufferVar<float> source,
                               Var<uint64_t> offset, Var<uint64_t> size) noexcept {
        set_block_size(32u);

        $if (thread_id().x == 0u) {

            graph_node_set_param(node, offset, source.device_address(), size);
        };

    };
    auto patch_shader = device.compile(patch_kernel);

    // Compile-level coverage of the other two device-side node mutators (and of
    // the enabled/disabled path): this kernel is lowered - including the
    // cudadevrt link - but never dispatched, so the test does not depend on
    // node enable/grid-dim update semantics.
    Kernel1D other_mutators = [](Var<uint64_t> node, Var<uint> enabled,
                                 Var<uint> grid_x) noexcept {
        set_block_size(32u);

        $if (thread_id().x == 0u) {

            graph_node_set_enabled(node, enabled != 0u);
            graph_node_set_grid_dim(node, grid_x, grid_x, grid_x);
        };

    };
    auto other_mutators_shader = device.compile(other_mutators);
    (void)other_mutators_shader;

    // ---- graph A: upload(out <- 0) + the scaling kernel --------------------
    auto cmdlist = CommandList::create();
    luisa::vector<uint32_t> zeros(kCount, 0u);
    cmdlist << out.view().copy_from(luisa::span{zeros});
    cmdlist << scale_shader(1.0f, out.view()).dispatch(kCount);
    auto graph = ext->create_graph(cmdlist.commit().command_list());
    expect(graph.handle().handle != CudaGraphExt::invalid_handle)
        << "create_graph from upload + dispatch should succeed";
    if (graph.handle().handle == CudaGraphExt::invalid_handle) { return; }
    auto graph_handle = graph.handle().handle;

    // Node 0 is the upload memcpy, node 1 the scaling kernel.
    expect(!ext->mark_node_device_updatable(graph_handle, 0u))
        << "a memcpy node cannot be opted into device updatability";
    expect(!ext->mark_node_device_updatable(graph_handle, ~size_t{0u}))
        << "an out-of-range node index must be refused";
    expect(!ext->mark_node_device_updatable(CudaGraphExt::invalid_handle, 1u))
        << "an invalid graph handle must be refused";
    expect(ext->mark_node_device_updatable(graph_handle, 1u))
        << "the kernel node must be opted into device updatability";
    // Idempotent while the template is still updatable: the attribute cannot be
    // cleared once set, so a repeated request is not an error.
    expect(ext->mark_node_device_updatable(graph_handle, 1u))
        << "a repeated opt-in for the same node must succeed";

    auto exec = ext->instantiate(graph_handle);
    expect(exec.handle().handle != CudaGraphExt::invalid_handle)
        << "a graph with a device-updatable node must instantiate";
    if (exec.handle().handle == CudaGraphExt::invalid_handle) { return; }
    auto exec_handle = exec.handle().handle;

    // Restriction: the opt-in is a property of the graph template, valid only
    // before its first instantiation.
    expect(!ext->mark_node_device_updatable(graph_handle, 1u))
        << "the opt-in must be refused once the graph has been instantiated";

    // Restriction: a graph containing device-updatable nodes allows exactly one
    // instantiation (the device node handles are bound to that exec).
    {
        auto second = ext->instantiate(graph_handle);
        expect(second.handle().handle == CudaGraphExt::invalid_handle)
            << "a graph with device-updatable nodes must not instantiate twice";
    }

    // Restriction: neither the graph nor its exec may be passed to
    // cuGraphExecUpdate.
    {
        auto update_cmdlist = CommandList::create();
        update_cmdlist << out.view().copy_from(luisa::span{zeros});
        update_cmdlist << scale_shader(1.0f, out.view()).dispatch(kCount);
        CudaGraphExt::GraphUpdateError error{};
        expect(!ext->update(exec_handle, update_cmdlist.commit().command_list(), &error))
            << "whole-graph update() must be refused on a graph with "
               "device-updatable nodes";
        expect(error.type == CudaGraphExt::GraphUpdateErrorType::UNSUPPORTED)
            << "the refusal must be reported as UNSUPPORTED";
    }

    // A graph with device-updatable nodes must be uploaded before it is
    // launched, so that the launch has a device-resident parameter block the
    // device-side update can patch.
    ext->upload(exec_handle, stream.handle());

    auto node_handle = ext->get_device_node_handle(exec_handle, 1u);
    expect(node_handle != 0u)
        << "a device-updatable kernel node must hand out a device node handle";
    expect(ext->get_device_node_handle(exec_handle, 0u) == 0u)
        << "the upload node is not device-updatable, so it has no device node handle";
    expect(ext->get_device_node_handle(CudaGraphExt::invalid_handle, 1u) == 0u)
        << "an invalid exec handle must be refused";
    if (node_handle == 0u) { return; }

    // Baseline: the uploaded graph runs with the parameter block it was
    // instantiated with (scale = 1.0), which both proves the graph itself works
    // and makes the later assertion differential: the patched result below can
    // only come from the device-side update.
    ext->launch(exec_handle, stream.handle());
    stream << synchronize();
    auto baseline = read_back(stream, out.view(), kCount);
    expect(is_scaled(baseline, 1u))
        << "the uploaded graph must first run with its instantiation-time "
           "parameter (expected out[i] = i)";

    // ---- graph B: upload(new_scale <- 3.0f) + the patching kernel ----------
    auto patch_cmdlist = CommandList::create();
    luisa::vector<float> scale_value{3.0f};
    uint64_t patch_offset = 0u;  // first slot of the parameter block
    uint64_t patch_size = sizeof(float);
    patch_cmdlist << new_scale.view().copy_from(luisa::span{scale_value});
    patch_cmdlist << patch_shader(node_handle, new_scale.view(),
                                 patch_offset, patch_size)
                         .dispatch(1u);
    auto patch_graph = ext->create_graph(patch_cmdlist.commit().command_list());
    expect(patch_graph.handle().handle != CudaGraphExt::invalid_handle)
        << "create_graph from upload + device-side patch dispatch should succeed";
    if (patch_graph.handle().handle == CudaGraphExt::invalid_handle) { return; }
    auto patch_exec = ext->instantiate(patch_graph.handle().handle);
    expect(patch_exec.handle().handle != CudaGraphExt::invalid_handle)
        << "the patching graph must instantiate";
    if (patch_exec.handle().handle == CudaGraphExt::invalid_handle) { return; }

    ext->launch(patch_exec.handle().handle, stream.handle());
    stream << synchronize();

    // ---- A runs with the patched parameter ---------------------------------
    ext->launch(exec_handle, stream.handle());
    stream << synchronize();
    auto values = read_back(stream, out.view(), kCount);
    expect(is_scaled(values, 3u))
        << "the host-launched graph must observe the parameter patched from "
           "device code by the other graph (expected out[i] = 3 * i)";
}

/// G3: a host-launched graph whose kernel launches a DEVICE_LAUNCH exec from
/// device code. The child's completion is detected by a completion flag it
/// writes, polled by the host with a bounded loop.
void test_device_launch(Device &device, CudaGraphExt *ext) {
    using InstantiateFlag = CudaGraphExt::InstantiateFlag;
    auto stream = device.create_stream();
    auto child_out = device.create_buffer<uint32_t>(1u);
    auto child_done = device.create_buffer<uint32_t>(1u);
    auto parent_marker = device.create_buffer<uint32_t>(1u);

    // The child's kernel writes its result and then the completion flag, so the
    // host can tell "the device-launched graph ran" from its side effects. It
    // also records current_graph_exec(): the child is the DEVICE-launched graph
    // here, and %current_graph_exec is documented to be non-zero only inside a
    // device graph (a host-launched graph does NOT set it - see
    // test_current_graph_exec).
    auto child_exec_rec = device.create_buffer<uint64_t>(1u);
    Kernel1D child_kernel = [](BufferVar<uint32_t> out, BufferVar<uint32_t> done,
                               BufferVar<uint64_t> exec_rec) noexcept {
        set_block_size(32u);

        $if (thread_id().x == 0u) {

            out.write(0u, 7u);
            done.write(0u, kDone);
            exec_rec.write(0u, current_graph_exec());
        };

    };
    auto child_shader = device.compile(child_kernel);

    // ---- child graph C: upload(0) + upload(0) + the result kernel ----------
    luisa::vector<uint32_t> child_initial{0u};
    luisa::vector<uint64_t> child_zero{0u};
    auto child_cmdlist = CommandList::create();
    child_cmdlist << child_out.view().copy_from(luisa::span{child_initial});
    child_cmdlist << child_done.view().copy_from(luisa::span{child_initial});
    child_cmdlist << child_exec_rec.view().copy_from(luisa::span{child_zero});
    child_cmdlist << child_shader(child_out.view(), child_done.view(),
                                  child_exec_rec.view()).dispatch(1u);
    auto child_graph = ext->create_graph(child_cmdlist.commit().command_list());
    expect(child_graph.handle().handle != CudaGraphExt::invalid_handle)
        << "the child graph must be created";
    if (child_graph.handle().handle == CudaGraphExt::invalid_handle) { return; }

    auto child_exec = ext->instantiate(child_graph.handle().handle,
                                       InstantiateFlag::DEVICE_LAUNCH);
    expect(child_exec.handle().handle != CudaGraphExt::invalid_handle)
        << "the child graph must instantiate with DEVICE_LAUNCH";
    if (child_exec.handle().handle == CudaGraphExt::invalid_handle) { return; }

    auto token = ext->get_device_launch_token(child_exec.handle().handle);
    expect(token != 0u)
        << "an exec instantiated with DEVICE_LAUNCH must hand out a launch token";
    if (token == 0u) { return; }
    // Device launch requires the graph to have been uploaded.
    ext->upload(child_exec.handle().handle, stream.handle());

    // ---- parent graph P: upload + the kernel that tail-launches C ----------
    Kernel1D launch_kernel = [](Var<uint64_t> token, BufferVar<uint32_t> marker) noexcept {
        set_block_size(32u);

        $if (thread_id().x == 0u) {

            marker.write(0u, 0x1234u);
            graph_device_launch(token, GraphLaunchMode::TAIL);
        };

    };
    auto launch_shader = device.compile(launch_kernel);

    // Compile-level coverage of the remaining launch modes and of the raw
    // `uint` overload: the kernel is lowered but never dispatched, so the test
    // does not depend on fire-and-forget/sibling timing.
    Kernel1D other_modes = [](Var<uint64_t> token, BufferVar<uint32_t> marker,
                              Var<uint> mode) noexcept {
        set_block_size(32u);

        $if (thread_id().x == 0u) {

            marker.write(0u, 0x4321u);
            graph_device_launch(token, GraphLaunchMode::FIRE_AND_FORGET);
            graph_device_launch(token, GraphLaunchMode::SIBLING);
            graph_device_launch(token, mode);
        };

    };
    auto other_modes_shader = device.compile(other_modes);
    (void)other_modes_shader;

    luisa::vector<uint32_t> marker_initial{0u};
    auto parent_cmdlist = CommandList::create();
    parent_cmdlist << parent_marker.view().copy_from(luisa::span{marker_initial});
    parent_cmdlist << launch_shader(token, parent_marker.view()).dispatch(1u);
    auto parent_graph = ext->create_graph(parent_cmdlist.commit().command_list());
    expect(parent_graph.handle().handle != CudaGraphExt::invalid_handle)
        << "the parent graph must be created";
    if (parent_graph.handle().handle == CudaGraphExt::invalid_handle) { return; }

    // Negative: the parent exec was instantiated without DEVICE_LAUNCH, so it
    // has no device launch token (and neither has an invalid handle).
    expect(ext->get_device_launch_token(CudaGraphExt::invalid_handle) == 0u)
        << "an invalid exec handle has no device launch token";
    auto parent_exec = ext->instantiate(parent_graph.handle().handle);
    expect(parent_exec.handle().handle != CudaGraphExt::invalid_handle)
        << "the parent graph must instantiate";
    if (parent_exec.handle().handle == CudaGraphExt::invalid_handle) { return; }
    expect(ext->get_device_launch_token(parent_exec.handle().handle) == 0u)
        << "an exec without DEVICE_LAUNCH has no device launch token";

    ext->launch(parent_exec.handle().handle, stream.handle());
    stream << synchronize();
    auto marker = read_back(stream, parent_marker.view(), 1u);
    expect(marker[0] == 0x1234u) << "the parent graph itself must have run";

    // The tail-launched child is applied to the launching graph's environment;
    // poll its completion flag with a bounded loop before reading the result.
    auto child_ran = false;
    for (auto attempt = 0u; attempt < kPollIterations; attempt++) {
        auto flag = read_back(stream, child_done.view(), 1u);
        if (flag[0] == kDone) {
            child_ran = true;
            break;
        }
        std::this_thread::yield();
    }
    expect(child_ran)
        << "the graph launched from device code must run (its completion flag "
           "never appeared)";
    auto result = read_back(stream, child_out.view(), 1u);
    expect(result[0] == 7u)
        << "the device-launched child graph must have written its result";
    // %current_graph_exec is non-zero only inside a device graph: the child was
    // launched from device code, so its recording must be non-zero. (The value
    // is the child's own exec handle; the host-side exec of the child graphs
    // equals it - assert non-zero only, per the launch-mode timing caveat.)
    auto child_exec_value = read_back(stream, child_exec_rec.view(), 1u);
    expect(child_exec_value[0] != 0u)
        << "current_graph_exec() must be non-zero inside a device-launched graph";
}

/// G3: `current_graph_exec()` semantics: per the CUDA documentation
/// (%current_graph_exec / cudaGetCurrentGraphExec), the register is non-zero
/// only inside a DEVICE graph - a kernel running as part of a HOST-launched
/// graph still reads 0. The non-zero case is asserted in test_device_launch
/// (the child graph there is device-launched).
void test_current_graph_exec(Device &device, CudaGraphExt *ext) {
    auto stream = device.create_stream();
    auto exec_buffer = device.create_buffer<uint64_t>(1u);

    Kernel1D record_kernel = [](BufferVar<uint64_t> exec_out) noexcept {
        set_block_size(32u);

        $if (thread_id().x == 0u) {

            exec_out.write(0u, current_graph_exec());
        };

    };
    auto record_shader = device.compile(record_kernel);

    luisa::vector<uint64_t> zeros{0u};
    auto cmdlist = CommandList::create();
    cmdlist << exec_buffer.view().copy_from(luisa::span{zeros});
    cmdlist << record_shader(exec_buffer.view()).dispatch(1u);
    auto graph = ext->create_graph(cmdlist.commit().command_list());
    expect(graph.handle().handle != CudaGraphExt::invalid_handle)
        << "create_graph for the current_graph_exec() kernel must succeed";
    if (graph.handle().handle == CudaGraphExt::invalid_handle) { return; }

    auto exec = ext->instantiate(graph.handle().handle);
    expect(exec.handle().handle != CudaGraphExt::invalid_handle)
        << "the recording graph must instantiate";
    if (exec.handle().handle == CudaGraphExt::invalid_handle) { return; }

    ext->launch(exec.handle().handle, stream.handle());
    stream << synchronize();
    auto values = read_back(stream, exec_buffer.view(), 1u);
    expect(values[0] == 0u)
        << "current_graph_exec() must read 0 in a host-launched graph "
           "(it is non-zero only inside a device graph)";
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
    if (caps.device_updatable_nodes) {
        test_device_updatable_node(device, ext);
    } else {
        LUISA_INFO("Device-updatable CUDA graph kernel nodes require CUDA >= 12.4 "
                   "(toolkit/driver {}), skipping that test.",
                   caps.cuda_version);
    }
    if (caps.device_launch) {
        test_device_launch(device, ext);
        test_current_graph_exec(device, ext);
    } else {
        LUISA_INFO("Device-side CUDA graph launch requires CUDA >= 12.0 "
                   "(toolkit/driver {}), skipping that test.",
                   caps.cuda_version);
    }
    LUISA_INFO("CUDA graph device-side test passed.");
    return 0;
}
