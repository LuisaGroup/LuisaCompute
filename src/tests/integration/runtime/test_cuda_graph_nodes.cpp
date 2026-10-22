// Tests for the explicit node API of the CUDA Graph extension
// (`luisa/backends/ext/cuda/cuda_graph_ext.h`).
//
// - capabilities() reports the compile-time toolkit version and the
//   version-gated feature set (never a feature the toolkit lacks);
// - G6 node types: memset / host / event record+wait / child graph, each
//   appended to the graph with its creation index and wired explicitly with
//   add_dependency;
// - texture commands (upload / download / copy / buffer<->texture) are
//   captured as CUDA array memcpy nodes instead of being dropped, and the
//   replay reproduces the source pixels;
// - content without a graph-node equivalent (bindless array update) is
//   refused loudly: create_graph fails instead of silently dropping it;
// - not-yet-implemented features (conditional nodes, device launch,
//   device-updatable nodes, graph memory) fail gracefully with an invalid
//   handle / false instead of pretending to work;
// - G7: the 3-arg update() reports why an update failed.
//
// Requires the CUDA backend (the extension only exists there); any other
// backend silently skips the test.
#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/core/dynamic_module.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/stream.h>
#include <luisa/backends/ext/cuda/cuda_graph_ext.h>
#include <algorithm>
#include <atomic>
#include <cstdint>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr uint32_t kCount = 64u;

/// The extension refuses empty command lists, so every graph built here starts
/// from a host -> device upload of `data` into `view`.
[[nodiscard]] CudaGraphInstance make_upload_graph(CudaGraphExt *ext, BufferView<uint32_t> view,
                                                  const luisa::vector<uint32_t> &data) {
    auto cmdlist = CommandList::create();
    cmdlist << view.copy_from(luisa::span{data});
    return ext->create_graph(cmdlist.commit().command_list());
}

[[nodiscard]] bool all_equal(luisa::span<const uint32_t> a, luisa::span<const uint32_t> b) noexcept {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

void test_capabilities(CudaGraphExt *ext) {
    auto caps = ext->capabilities();
    // The backend itself only builds against CUDA >= 12.1.
    expect(caps.cuda_version >= 12010u)
        << "capabilities().cuda_version must report the toolkit version";
    // Each flag is gated on min(toolkit, driver), so a reported feature is a
    // subset of what the reported toolkit can provide.
    if (caps.conditional_nodes) {
        expect(caps.cuda_version >= 12030u) << "conditional nodes need CUDA >= 12.3";
    }
    if (caps.switch_conditional) {
        expect(caps.cuda_version >= 12080u) << "switch conditions need CUDA >= 12.8";
    }
    if (caps.node_priority) {
        expect(caps.cuda_version >= 12080u) << "node priority needs CUDA >= 12.8";
    }
    if (caps.device_updatable_nodes) {
        expect(caps.cuda_version >= 12040u) << "device-updatable nodes need CUDA >= 12.4";
    }
    if (caps.graph_memory) {
        expect(caps.cuda_version >= 11040u) << "graph memory needs CUDA >= 11.4";
    }
    if (caps.programmatic_edges || caps.device_launch) {
        expect(caps.cuda_version >= 12000u) << "PDL / device launch need CUDA >= 12.0";
    }
    LUISA_INFO("CudaGraphExt capabilities: cuda_version = {}, programmatic_edges = {}, "
               "device_launch = {}, conditional_nodes = {}, device_updatable_nodes = {}, "
               "graph_memory = {}, node_priority = {}, switch_conditional = {}.",
               caps.cuda_version, caps.programmatic_edges, caps.device_launch,
               caps.conditional_nodes, caps.device_updatable_nodes, caps.graph_memory,
               caps.node_priority, caps.switch_conditional);
}

void test_memset_node(Device &device, CudaGraphExt *ext) {
    auto stream = device.create_stream();
    auto buffer = device.create_buffer<uint32_t>(kCount);
    luisa::vector<uint32_t> upload(kCount, 0xdeadbeefu);
    auto graph = make_upload_graph(ext, buffer.view(), upload);
    expect(graph.handle().handle != CudaGraphExt::invalid_handle)
        << "create_graph from an upload should succeed";
    if (graph.handle().handle == CudaGraphExt::invalid_handle) { return; }
    auto graph_handle = graph.handle().handle;

    // The upload is node 0; the memset node is appended as node 1 and wired
    // to run after the upload.
    size_t memset_index = ~size_t{0u};
    expect(ext->add_memset_node(graph_handle, buffer.handle(), 0u,
                                kCount * sizeof(uint32_t), 0u, &memset_index))
        << "add_memset_node should succeed";
    expect(memset_index == 1u) << "the memset node must be the second node of the graph";
    expect(ext->add_dependency(graph_handle, 0u, memset_index))
        << "wiring the memset node after the upload should succeed";

    // Out-of-range fills are refused, never clamped.
    size_t rejected_index = ~size_t{0u};
    expect(!ext->add_memset_node(graph_handle, buffer.handle(), 0u,
                                 kCount * sizeof(uint32_t) + 1u, 0u, &rejected_index))
        << "an out-of-range memset node must be refused";

    auto exec = ext->instantiate(graph_handle);
    expect(exec.handle().handle != CudaGraphExt::invalid_handle) << "instantiate should succeed";
    if (exec.handle().handle == CudaGraphExt::invalid_handle) { return; }
    ext->launch(exec.handle().handle, stream.handle());

    luisa::vector<uint32_t> downloaded(kCount, 0xffffffffu);
    stream << buffer.view().copy_to(luisa::span{downloaded}) << synchronize();
    expect(std::all_of(downloaded.begin(), downloaded.end(),
                       [](uint32_t v) noexcept { return v == 0u; }))
        << "the memset node must have zeroed the whole buffer after the upload";
}

void host_node_callback(void *user_data) noexcept {
    static_cast<std::atomic_uint *>(user_data)->fetch_add(1u, std::memory_order_relaxed);
}

void test_host_node(Device &device, CudaGraphExt *ext) {
    auto stream = device.create_stream();
    auto buffer = device.create_buffer<uint32_t>(kCount);
    luisa::vector<uint32_t> upload(kCount, 7u);
    auto graph = make_upload_graph(ext, buffer.view(), upload);
    expect(graph.handle().handle != CudaGraphExt::invalid_handle);
    if (graph.handle().handle == CudaGraphExt::invalid_handle) { return; }
    auto graph_handle = graph.handle().handle;

    std::atomic_uint counter{0u};
    size_t host_index = ~size_t{0u};
    expect(ext->add_host_node(graph_handle, host_node_callback, &counter, &host_index))
        << "add_host_node should succeed";
    expect(host_index == 1u) << "the host node must be the second node of the graph";
    expect(ext->add_dependency(graph_handle, 0u, host_index))
        << "the host node must be wired after the upload";
    expect(!ext->add_host_node(graph_handle, nullptr, nullptr, nullptr))
        << "a null host callback must be refused";

    auto exec = ext->instantiate(graph_handle);
    expect(exec.handle().handle != CudaGraphExt::invalid_handle);
    if (exec.handle().handle == CudaGraphExt::invalid_handle) { return; }
    ext->launch(exec.handle().handle, stream.handle());
    stream << synchronize();
    expect(counter.load() == 1u) << "the host node must run once per launch";
    ext->launch(exec.handle().handle, stream.handle());
    stream << synchronize();
    expect(counter.load() == 2u) << "the host node must run on every launch";
}

/// The subset of the CUDA driver event API the event-node test needs. The
/// driver is reached through the module the runtime already loaded, so the
/// test binary needs no link-time dependency on libcuda (the CUDA backend
/// itself is loaded dynamically).
struct CudaEventApi {
    luisa::DynamicModule module;
    int (*create)(void **event, unsigned int flags){nullptr};
    int (*destroy)(void *event){nullptr};
    int (*query)(void *event){nullptr};

    [[nodiscard]] bool valid() const noexcept {
        return static_cast<bool>(module) && create != nullptr &&
               destroy != nullptr && query != nullptr;
    }
};

[[nodiscard]] CudaEventApi load_cuda_event_api() noexcept {
    CudaEventApi api{};
    auto module = luisa::DynamicModule::load_exact("libcuda.so.1");
    if (!module) { module = luisa::DynamicModule::load("cuda"); }
    if (!module) { module = luisa::DynamicModule::load_exact("nvcuda.dll"); }
    if (!module) { return api; }
    api.create = module.function<int(void **, unsigned int)>("cuEventCreate");
    api.destroy = module.function<int(void *)>("cuEventDestroy");
    api.query = module.function<int(void *)>("cuEventQuery");
    api.module = std::move(module);
    return api;
}

void test_event_nodes(Device &device, CudaGraphExt *ext, const CudaEventApi &api) {
    if (!api.valid()) {
        LUISA_INFO("CUDA driver event API unavailable, skipping the event-node test.");
        return;
    }
    constexpr auto cuda_event_disable_timing = 0x2u;
    void *event = nullptr;
    if (api.create(&event, cuda_event_disable_timing) != 0 /* CUDA_SUCCESS */) {
        LUISA_WARNING_WITH_LOCATION("cuEventCreate failed, skipping the event-node test.");
        return;
    }
    auto event_handle = reinterpret_cast<uint64_t>(event);

    auto stream = device.create_stream();
    auto buffer = device.create_buffer<uint32_t>(kCount);
    luisa::vector<uint32_t> upload(kCount, 3u);
    auto graph = make_upload_graph(ext, buffer.view(), upload);
    if (graph.handle().handle == CudaGraphExt::invalid_handle) {
        api.destroy(event);
        return;
    }
    auto graph_handle = graph.handle().handle;

    size_t record_index = ~size_t{0u};
    size_t wait_index = ~size_t{0u};
    expect(ext->add_event_record_node(graph_handle, event_handle, &record_index))
        << "add_event_record_node should succeed";
    expect(ext->add_event_wait_node(graph_handle, event_handle, &wait_index))
        << "add_event_wait_node should succeed";
    expect(record_index == 1u && wait_index == 2u)
        << "the event nodes keep their creation order";
    expect(ext->add_dependency(graph_handle, 0u, record_index))
        << "the record node must run after the upload";
    expect(ext->add_dependency(graph_handle, record_index, wait_index))
        << "the wait node must run after the record node";
    expect(!ext->add_dependency(graph_handle, 0u, 0u))
        << "a self dependency must be refused";
    expect(!ext->add_dependency(graph_handle, 0u, 99u))
        << "an out-of-range dependency must be refused";
    expect(!ext->add_event_record_node(graph_handle, 0u, nullptr))
        << "a null event must be refused";
    // The `wait` node needs a recorded event before it: an event that was
    // never recorded is not usable inside a graph, so the instantiate below is
    // only expected to succeed because the record node precedes the wait.

    auto exec = ext->instantiate(graph_handle);
    expect(exec.handle().handle != CudaGraphExt::invalid_handle)
        << "instantiate with event nodes should succeed";
    if (exec.handle().handle != CudaGraphExt::invalid_handle) {
        ext->launch(exec.handle().handle, stream.handle());
        stream << synchronize();
        expect(api.query(event) == 0 /* CUDA_SUCCESS */)
            << "the record node must have recorded the event during the launch";
        // A second launch must work as well (the nodes are reusable).
        ext->launch(exec.handle().handle, stream.handle());
        stream << synchronize();
        expect(api.query(event) == 0);
    }
    api.destroy(event);
}

void test_child_graph(Device &device, CudaGraphExt *ext) {
    auto stream = device.create_stream();
    auto buffer = device.create_buffer<uint32_t>(kCount);
    luisa::vector<uint32_t> child_data(kCount);
    for (auto i = 0u; i < kCount; i++) { child_data[i] = i + 1u; }
    luisa::vector<uint32_t> parent_data(kCount, 0u);

    // The child must be declared before - and therefore outlive - the parent.
    auto child = make_upload_graph(ext, buffer.view(), child_data);
    expect(child.handle().handle != CudaGraphExt::invalid_handle);
    auto parent = make_upload_graph(ext, buffer.view(), parent_data);
    expect(parent.handle().handle != CudaGraphExt::invalid_handle);
    if (child.handle().handle == CudaGraphExt::invalid_handle ||
        parent.handle().handle == CudaGraphExt::invalid_handle) {
        return;
    }
    auto parent_handle = parent.handle().handle;

    size_t child_index = ~size_t{0u};
    expect(ext->add_child_graph_node(parent_handle, child.handle().handle, &child_index))
        << "add_child_graph_node should succeed";
    expect(child_index == 1u) << "the child graph node keeps its creation order";
    expect(ext->add_dependency(parent_handle, 0u, child_index))
        << "the child graph must run after the parent's own upload";
    expect(!ext->add_child_graph_node(parent_handle, CudaGraphExt::invalid_handle, nullptr))
        << "an invalid child graph must be refused";
    expect(!ext->add_child_graph_node(parent_handle, parent_handle, nullptr))
        << "a self-referencing child graph must be refused";

    auto exec = ext->instantiate(parent_handle);
    expect(exec.handle().handle != CudaGraphExt::invalid_handle);
    if (exec.handle().handle == CudaGraphExt::invalid_handle) { return; }
    ext->launch(exec.handle().handle, stream.handle());

    luisa::vector<uint32_t> downloaded(kCount, 0xffffffffu);
    stream << buffer.view().copy_to(luisa::span{downloaded}) << synchronize();
    expect(all_equal(downloaded, child_data))
        << "the inlined child graph must have written its data after the parent's upload";
}

/// Phase 0: the texture family is captured as CUDA array memcpy nodes.
void test_texture_capture(Device &device, CudaGraphExt *ext) {
    auto stream = device.create_stream();
    constexpr uint32_t kSize = 8u;
    auto src = device.create_image<float>(PixelStorage::FLOAT4, make_uint2(kSize, kSize));
    auto dst = device.create_image<float>(PixelStorage::FLOAT4, make_uint2(kSize, kSize));
    luisa::vector<float> pixels(kSize * kSize * 4u);
    for (auto i = 0u; i < pixels.size(); i++) { pixels[i] = static_cast<float>(i) + 0.5f; }

    // 1) host -> texture upload, captured into a graph, replayed.
    {
        auto cmdlist = CommandList::create();
        cmdlist << src.view(0u).copy_from(luisa::span{pixels});
        auto graph = ext->create_graph(cmdlist.commit().command_list());
        expect(graph.handle().handle != CudaGraphExt::invalid_handle)
            << "a TextureUploadCommand must be capturable";
        if (graph.handle().handle == CudaGraphExt::invalid_handle) { return; }
        auto exec = ext->instantiate(graph.handle().handle);
        expect(exec.handle().handle != CudaGraphExt::invalid_handle);
        if (exec.handle().handle == CudaGraphExt::invalid_handle) { return; }
        ext->launch(exec.handle().handle, stream.handle());
        luisa::vector<float> downloaded(pixels.size(), 0.0f);
        stream << src.view(0u).copy_to(luisa::span{downloaded}) << synchronize();
        expect(downloaded == pixels) << "the captured texture upload must match the source data";
    }

    // 2) texture -> texture copy, captured into a graph, replayed.
    {
        auto cmdlist = CommandList::create();
        cmdlist << dst.view(0u).copy_from(src.view(0u));
        auto graph = ext->create_graph(cmdlist.commit().command_list());
        expect(graph.handle().handle != CudaGraphExt::invalid_handle)
            << "a TextureCopyCommand must be capturable";
        if (graph.handle().handle == CudaGraphExt::invalid_handle) { return; }
        auto exec = ext->instantiate(graph.handle().handle);
        if (exec.handle().handle == CudaGraphExt::invalid_handle) { return; }
        ext->launch(exec.handle().handle, stream.handle());
        luisa::vector<float> downloaded(pixels.size(), 0.0f);
        stream << dst.view(0u).copy_to(luisa::span{downloaded}) << synchronize();
        expect(downloaded == pixels) << "the captured texture copy must reproduce the source";
    }

    // 3) texture -> buffer and buffer -> texture copies (a graph with both
    //    directions, so the buffer read/write hazards are exercised too).
    {
        auto buffer = device.create_buffer<float>(kSize * kSize * 4u);
        auto cmdlist = CommandList::create();
        cmdlist << src.view(0u).copy_to(buffer.view());// TextureToBufferCopyCommand
        auto graph = ext->create_graph(cmdlist.commit().command_list());
        expect(graph.handle().handle != CudaGraphExt::invalid_handle)
            << "a TextureToBufferCopyCommand must be capturable";
        if (graph.handle().handle != CudaGraphExt::invalid_handle) {
            auto exec = ext->instantiate(graph.handle().handle);
            if (exec.handle().handle != CudaGraphExt::invalid_handle) {
                ext->launch(exec.handle().handle, stream.handle());
                luisa::vector<float> downloaded(pixels.size(), 0.0f);
                stream << buffer.view().copy_to(luisa::span{downloaded}) << synchronize();
                expect(downloaded == pixels)
                    << "the captured texture -> buffer copy must reproduce the source";
            }
        }
        auto dst2 = device.create_image<float>(PixelStorage::FLOAT4, make_uint2(kSize, kSize));
        auto cmdlist2 = CommandList::create();
        cmdlist2 << dst2.view(0u).copy_from(buffer.view());// BufferToTextureCopyCommand
        auto graph2 = ext->create_graph(cmdlist2.commit().command_list());
        expect(graph2.handle().handle != CudaGraphExt::invalid_handle)
            << "a BufferToTextureCopyCommand must be capturable";
        if (graph2.handle().handle != CudaGraphExt::invalid_handle) {
            auto exec2 = ext->instantiate(graph2.handle().handle);
            if (exec2.handle().handle != CudaGraphExt::invalid_handle) {
                ext->launch(exec2.handle().handle, stream.handle());
                luisa::vector<float> downloaded(pixels.size(), 0.0f);
                stream << dst2.view(0u).copy_to(luisa::span{downloaded}) << synchronize();
                expect(downloaded == pixels)
                    << "the captured buffer -> texture copy must reproduce the source";
            }
        }
    }
}

/// Content with no graph-node equivalent must fail the capture loudly.
void test_refusals(Device &device, CudaGraphExt *ext) {
    auto buffer = device.create_buffer<uint32_t>(kCount);
    auto bindless = device.create_bindless_array(16u);
    bindless.emplace_on_update(0u, buffer);
    auto cmdlist = CommandList::create();
    cmdlist << bindless.update();
    auto graph = ext->create_graph(cmdlist.commit().command_list());
    expect(graph.handle().handle == CudaGraphExt::invalid_handle)
        << "a BindlessArrayUpdateCommand must be refused instead of being dropped";
}

/// Every phase feature must refuse gracefully on obviously-invalid arguments.
/// (Phase 2 conditional nodes are implemented now — covered by
/// test_cuda_graph_conditional.cpp.)
void test_not_implemented_stubs(Device &device, CudaGraphExt *ext) {
    auto buffer = device.create_buffer<uint32_t>(kCount);
    luisa::vector<uint32_t> upload(kCount, 1u);
    auto graph = make_upload_graph(ext, buffer.view(), upload);
    if (graph.handle().handle == CudaGraphExt::invalid_handle) { return; }
    auto graph_handle = graph.handle().handle;
    auto exec = ext->instantiate(graph_handle);
    auto exec_handle = exec.handle().handle;
    expect(exec_handle != CudaGraphExt::invalid_handle);

    // G1 conditional nodes are implemented (SA-2); here only verify that
    // obviously-invalid arguments are still refused.
    expect(ext->create_conditional_handle(CudaGraphExt::invalid_handle, 0u, true) == 0u)
        << "conditional handles require a valid graph";
    luisa::vector<CudaGraphExt::GraphHandle> bodies(1u, CudaGraphExt::invalid_handle);
    size_t conditional_index = ~size_t{0u};
    expect(!ext->add_conditional_node(graph_handle, CudaGraphExt::ConditionalType::IF, 0u,
                                      1u, luisa::span{bodies}, &conditional_index))
        << "conditional nodes reject a null handle";

    // G2 typed (PDL) edges.
    expect(!ext->set_dependency_type(graph_handle, 0u, 0u, CudaGraphExt::DependencyType::PROGRAMMATIC))
        << "a self edge can never be retyped";
    expect(!ext->set_dependency_type(graph_handle, 0u, 99u, CudaGraphExt::DependencyType::PROGRAMMATIC))
        << "an out-of-range edge can never be retyped";

    // G3 device-side launch: this exec was instantiated without DEVICE_LAUNCH,
    // so no token is handed out (the SA-4 positive path lives in
    // test_cuda_graph_device.cpp).
    expect(ext->get_device_launch_token(exec_handle) == 0u)
        << "an exec without DEVICE_LAUNCH must not hand out a token";

    // G4 device-updatable kernel nodes (implemented; invalid arguments only).
    // Node 0 is a memcpy node, so marking it must fail.
    expect(!ext->mark_node_device_updatable(graph_handle, 0u))
        << "device-updatable opt-in rejects non-kernel nodes";

    // G5 graph memory (implemented; invalid arguments only).
    size_t mem_index = ~size_t{0u};
    expect(!ext->add_mem_alloc_node(graph_handle, 0u, 256u, &mem_index))
        << "mem-alloc rejects a zero size";
    expect(!ext->add_mem_alloc_node(graph_handle, 4096u, 24u, &mem_index))
        << "mem-alloc rejects a non-power-of-two alignment";
    expect(!ext->add_mem_free_node(graph_handle, 0u, &mem_index))
        << "mem-free rejects a null address";
    uint64_t address = 0u;
    expect(!ext->get_mem_alloc_node_address(exec_handle, 0u, &address))
        << "node 0 is not a mem-alloc node";
}

/// G7: update() diagnostics.
void test_update_diagnostics(Device &device, CudaGraphExt *ext) {
    auto stream = device.create_stream();
    auto buffer = device.create_buffer<uint32_t>(kCount);
    luisa::vector<uint32_t> first(kCount, 1u);
    luisa::vector<uint32_t> sink(kCount, 0u);
    auto cmdlist = CommandList::create();
    cmdlist << buffer.view().copy_from(luisa::span{first});
    cmdlist << buffer.view().copy_to(luisa::span{sink});
    auto graph = ext->create_graph(cmdlist.commit().command_list());
    expect(graph.handle().handle != CudaGraphExt::invalid_handle)
        << "create_graph from upload + download should succeed";
    if (graph.handle().handle == CudaGraphExt::invalid_handle) { return; }
    auto exec = ext->instantiate(graph.handle().handle);
    expect(exec.handle().handle != CudaGraphExt::invalid_handle);
    if (exec.handle().handle == CudaGraphExt::invalid_handle) { return; }
    auto exec_handle = exec.handle().handle;

    // A structurally different command list (one node instead of two) can only
    // fail, and the failure must be reported as a topology change.
    {
        auto changed = CommandList::create();
        changed << buffer.view().copy_from(luisa::span{first});
        CudaGraphExt::GraphUpdateError error{};
        auto updated = ext->update(exec_handle, changed.commit().command_list(), &error);
        expect(!updated) << "an update with a different node count must fail";
        expect(error.type == CudaGraphExt::GraphUpdateErrorType::TOPOLOGY_CHANGED)
            << "the failure must be reported as TOPOLOGY_CHANGED";
        expect(!error.detail.empty()) << "the failure must carry a human-readable detail";
    }

    // The matching structure updates in place, and the diagnostics stay clean.
    {
        luisa::vector<uint32_t> second(kCount, 5u);
        luisa::vector<uint32_t> sink2(kCount, 0u);
        auto same = CommandList::create();
        same << buffer.view().copy_from(luisa::span{second});
        same << buffer.view().copy_to(luisa::span{sink2});
        CudaGraphExt::GraphUpdateError error{};
        auto updated = ext->update(exec_handle, same.commit().command_list(), &error);
        expect(updated) << "a structurally identical update must succeed";
        if (updated) {
            expect(error.type == CudaGraphExt::GraphUpdateErrorType::NONE)
                << "a successful update must not report an error";
            expect(error.detail.empty());
            ext->launch(exec_handle, stream.handle());
            stream << synchronize();
            expect(all_equal(sink2, second))
                << "the updated executable graph must use the new host buffers";
        }
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
    test_memset_node(device, ext);
    test_host_node(device, ext);
    test_event_nodes(device, ext, load_cuda_event_api());
    test_child_graph(device, ext);
    test_texture_capture(device, ext);
    test_refusals(device, ext);
    test_not_implemented_stubs(device, ext);
    test_update_diagnostics(device, ext);
    LUISA_INFO("CUDA graph node test passed.");
    return 0;
}
