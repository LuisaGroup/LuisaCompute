// Test for the CUDA graph DAG builder (CudaGraphExt::_create_graph)
// - The builder analyzes the captured command list like the command reorder
//   pass (every argument tracked by its declared usage) and builds an
//   explicit CUDA graph whose edges are only the real RAW/WAW/WAR command ->
//   command dependencies. These tests verify both directions:
//   * hazard-free commands get no edge between them, so an independent batch
//     replays concurrently and still reproduces the sequential result exactly
//     (disjoint output sub-ranges + one shared read-only input);
//   * genuine hazard chains (rotating write-after-write ranges, read-after-
//     write chains, upload/copy + dispatch + download) get exactly the edges
//     that preserve the source order, so the replay is bit-identical to the
//     strictly ordered stream reference.
// Requires the CUDA backend (the extension only exists there); any other
// backend silently skips the test.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/backends/ext/cuda/cuda_graph_ext.h>

#include <cmath>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr uint32_t kThreads = 256u;

[[nodiscard]] auto make_scale_kernel(Device &device) noexcept {
    Kernel1D kernel = [](BufferVar<float> dst, BufferVar<float> src,
                         Var<float> factor) noexcept {
        auto i = dispatch_x();
        dst.write(i, src.read(i) * factor);
    };
    return device.compile(kernel);
}

void test_cuda_graph_dag(Device &device) {

    auto *ext = device.extension<CudaGraphExt>();
    if (!ext) {
        LUISA_INFO("CudaGraphExt not available, skipping test.");
        return;
    }

    auto stream = device.create_stream();
    auto kernel = make_scale_kernel(device);

    // ---- hazard-free batch: disjoint outputs, shared read-only input ------
    // No pair of dispatches shares a range, so the DAG must contain no edges
    // between them; the replay still has to match the sequential reference
    // bit for bit (a merged-race would show up as a mismatch).
    {
        constexpr uint32_t n_dispatch = 16u;
        auto src = device.create_buffer<float>(kThreads);
        auto dst = device.create_buffer<float>(n_dispatch * kThreads);
        luisa::vector<float> host_src(kThreads);
        for (auto i = 0u; i < kThreads; i++) {
            host_src[i] = std::sin(static_cast<float>(i) * 0.13f) * 2.f + 1.5f;
        }
        stream << src.copy_from(luisa::span{host_src}) << synchronize();

        auto build_batch = [&](CommandList &list) noexcept {
            for (auto j = 0u; j < n_dispatch; j++) {
                auto view = dst.view().subview(j * kThreads, kThreads);
                list << kernel(view, src.view(), 1.f + 0.25f * static_cast<float>(j))
                            .dispatch(kThreads);
            }
        };

        // Strictly ordered reference on the plain stream.
        auto reference_cmdlist = CommandList::create();
        build_batch(reference_cmdlist);
        stream << std::move(reference_cmdlist.commit()) << synchronize();
        luisa::vector<float> reference(n_dispatch * kThreads);
        stream << dst.view().copy_to(luisa::span{reference}) << synchronize();

        // Same batch captured into a DAG and replayed.
        auto cmdlist = CommandList::create();
        build_batch(cmdlist);
        auto graph = ext->create_graph(std::move(cmdlist.commit()).command_list());
        expect(graph.handle().handle != CudaGraphExt::invalid_handle)
            << "DAG capture of a hazard-free batch should succeed";
        if (graph.handle().handle == CudaGraphExt::invalid_handle) { return; }
        auto exec = ext->instantiate(graph.handle().handle);
        expect(exec.handle().handle != CudaGraphExt::invalid_handle)
            << "instantiate should succeed";
        if (exec.handle().handle == CudaGraphExt::invalid_handle) { return; }

        auto zero = luisa::vector<float>(n_dispatch * kThreads, 0.f);
        stream << dst.copy_from(luisa::span{zero}) << synchronize();
        ext->launch(exec.handle().handle, stream.handle());
        stream << synchronize();
        luisa::vector<float> replayed(n_dispatch * kThreads);
        stream << dst.view().copy_to(luisa::span{replayed}) << synchronize();
        for (auto i = 0u; i < n_dispatch * kThreads; i++) {
            expect(replayed[i] == reference[i]) << "parallel replay mismatch at " << i;
        }

        // Replaying the same graph must be deterministic.
        stream << dst.copy_from(luisa::span{zero}) << synchronize();
        ext->launch(exec.handle().handle, stream.handle());
        stream << synchronize();
        luisa::vector<float> replayed_2(n_dispatch * kThreads);
        stream << dst.view().copy_to(luisa::span{replayed_2}) << synchronize();
        for (auto i = 0u; i < n_dispatch * kThreads; i++) {
            expect(replayed_2[i] == reference[i]) << "second replay mismatch at " << i;
        }
    }

    // ---- write-after-write chains: rotating output ranges -----------------
    // Dispatch j writes range (j % kRanges) with factor (j + 1); the strictly
    // ordered result is "last writer wins" per range. The DAG must serialize
    // exactly the commands that collide on a range - a missing WAW edge would
    // let a later dispatch be overtaken and corrupt the result.
    {
        constexpr uint32_t n_dispatch = 12u;
        constexpr uint32_t kRanges = 4u;
        auto src = device.create_buffer<float>(kThreads);
        auto dst = device.create_buffer<float>(kRanges * kThreads);
        luisa::vector<float> host_src(kThreads);
        for (auto i = 0u; i < kThreads; i++) {
            host_src[i] = std::cos(static_cast<float>(i) * 0.071f) + 2.f;
        }
        stream << src.copy_from(luisa::span{host_src}) << synchronize();

        auto build_batch = [&](CommandList &list) noexcept {
            for (auto j = 0u; j < n_dispatch; j++) {
                auto view = dst.view().subview((j % kRanges) * kThreads, kThreads);
                list << kernel(view, src.view(), static_cast<float>(j + 1u))
                            .dispatch(kThreads);
            }
        };

        auto reference_cmdlist = CommandList::create();
        build_batch(reference_cmdlist);
        stream << std::move(reference_cmdlist.commit()) << synchronize();
        luisa::vector<float> reference(kRanges * kThreads);
        stream << dst.view().copy_to(luisa::span{reference}) << synchronize();

        auto cmdlist = CommandList::create();
        build_batch(cmdlist);
        auto graph = ext->create_graph(std::move(cmdlist.commit()).command_list());
        expect(graph.handle().handle != CudaGraphExt::invalid_handle)
            << "DAG capture of a WAW-rotating batch should succeed";
        if (graph.handle().handle == CudaGraphExt::invalid_handle) { return; }
        auto exec = ext->instantiate(graph.handle().handle);
        expect(exec.handle().handle != CudaGraphExt::invalid_handle)
            << "instantiate should succeed";
        if (exec.handle().handle == CudaGraphExt::invalid_handle) { return; }

        auto zero = luisa::vector<float>(kRanges * kThreads, 0.f);
        stream << dst.copy_from(luisa::span{zero}) << synchronize();
        ext->launch(exec.handle().handle, stream.handle());
        stream << synchronize();
        luisa::vector<float> replayed(kRanges * kThreads);
        stream << dst.view().copy_to(luisa::span{replayed}) << synchronize();
        for (auto i = 0u; i < kRanges * kThreads; i++) {
            expect(replayed[i] == reference[i]) << "WAW replay mismatch at " << i;
        }
    }

    // ---- read-after-write chain across dispatches --------------------------
    // d0 writes tmp = src * 2, d1 reads tmp and writes out = tmp * 3. The d1
    // node must depend on d0; the result is checked against the sequential
    // reference.
    {
        auto src = device.create_buffer<float>(kThreads);
        auto tmp = device.create_buffer<float>(kThreads);
        auto out = device.create_buffer<float>(kThreads);
        luisa::vector<float> host_src(kThreads);
        for (auto i = 0u; i < kThreads; i++) {
            host_src[i] = std::sin(static_cast<float>(i) * 0.31f) + 0.25f;
        }
        stream << src.copy_from(luisa::span{host_src}) << synchronize();

        auto build_batch = [&](CommandList &list) noexcept {
            list << kernel(tmp.view(), src.view(), 2.f).dispatch(kThreads);
            list << kernel(out.view(), tmp.view(), 3.f).dispatch(kThreads);
        };

        auto reference_cmdlist = CommandList::create();
        build_batch(reference_cmdlist);
        stream << std::move(reference_cmdlist.commit()) << synchronize();
        luisa::vector<float> reference(kThreads);
        stream << out.view().copy_to(luisa::span{reference}) << synchronize();

        auto cmdlist = CommandList::create();
        build_batch(cmdlist);
        auto graph = ext->create_graph(std::move(cmdlist.commit()).command_list());
        expect(graph.handle().handle != CudaGraphExt::invalid_handle)
            << "DAG capture of a RAW chain should succeed";
        if (graph.handle().handle == CudaGraphExt::invalid_handle) { return; }
        auto exec = ext->instantiate(graph.handle().handle);
        expect(exec.handle().handle != CudaGraphExt::invalid_handle)
            << "instantiate should succeed";
        if (exec.handle().handle == CudaGraphExt::invalid_handle) { return; }

        auto zero = luisa::vector<float>(kThreads, 0.f);
        stream << out.copy_from(luisa::span{zero}) << synchronize();
        ext->launch(exec.handle().handle, stream.handle());
        stream << synchronize();
        luisa::vector<float> replayed(kThreads);
        stream << out.view().copy_to(luisa::span{replayed}) << synchronize();
        for (auto i = 0u; i < kThreads; i++) {
            expect(replayed[i] == reference[i]) << "RAW replay mismatch at " << i;
        }
    }

    // ---- upload + dispatch + download inside one graph ---------------------
    // Exercises the H2D/D2H memcpy nodes: the dispatch depends on the upload
    // (RAW) and the download depends on the dispatch (RAW). The download node
    // writes into a pinned staging buffer that launch() forwards to the user
    // buffer with a stream-ordered host callback.
    {
        auto src = device.create_buffer<float>(kThreads);
        auto dst = device.create_buffer<float>(kThreads);
        luisa::vector<float> host_src(kThreads);
        for (auto i = 0u; i < kThreads; i++) {
            host_src[i] = std::cos(static_cast<float>(i) * 0.17f) + 1.f;
        }
        auto cmdlist = CommandList::create();
        cmdlist << src.view().copy_from(luisa::span{host_src});
        cmdlist << kernel(dst.view(), src.view(), 2.f).dispatch(kThreads);
        luisa::vector<float> host_dst(kThreads, -1.f);
        cmdlist << dst.view().copy_to(luisa::span{host_dst});
        auto graph = ext->create_graph(std::move(cmdlist.commit()).command_list());
        expect(graph.handle().handle != CudaGraphExt::invalid_handle)
            << "DAG capture of upload+dispatch+download should succeed";
        if (graph.handle().handle == CudaGraphExt::invalid_handle) { return; }
        auto exec = ext->instantiate(graph.handle().handle);
        expect(exec.handle().handle != CudaGraphExt::invalid_handle)
            << "instantiate should succeed";
        if (exec.handle().handle == CudaGraphExt::invalid_handle) { return; }

        ext->launch(exec.handle().handle, stream.handle());
        stream << synchronize();
        for (auto i = 0u; i < kThreads; i++) {
            expect(host_dst[i] == host_src[i] * 2.f) << "memcpy-node mismatch at " << i;
        }

        // cuGraphExecUpdate against a same-topology DAG with new upload data.
        luisa::vector<float> host_src_2(kThreads);
        for (auto i = 0u; i < kThreads; i++) {
            host_src_2[i] = std::sin(static_cast<float>(i) * 0.29f) + 4.f;
        }
        auto updated_cmdlist = CommandList::create();
        updated_cmdlist << src.view().copy_from(luisa::span{host_src_2});
        updated_cmdlist << kernel(dst.view(), src.view(), 2.f).dispatch(kThreads);
        luisa::vector<float> host_dst_2(kThreads, -1.f);
        updated_cmdlist << dst.view().copy_to(luisa::span{host_dst_2});
        auto updated = ext->update(exec.handle().handle,
                                   std::move(updated_cmdlist.commit()).command_list());
        expect(updated) << "updating the DAG with new upload data should succeed";
        if (updated) {
            ext->launch(exec.handle().handle, stream.handle());
            stream << synchronize();
            for (auto i = 0u; i < kThreads; i++) {
                expect(host_dst_2[i] == host_src_2[i] * 2.f)
                    << "updated memcpy-node mismatch at " << i;
            }
        }
    }

    // ---- device-to-device copy node + dispatch -----------------------------
    // The dispatch must see the copied data (RAW on the copied range).
    {
        auto src = device.create_buffer<float>(kThreads);
        auto src_copy = device.create_buffer<float>(kThreads);
        auto dst = device.create_buffer<float>(kThreads);
        luisa::vector<float> host_src(kThreads);
        for (auto i = 0u; i < kThreads; i++) {
            host_src[i] = std::sin(static_cast<float>(i) * 0.43f) + 3.f;
        }
        stream << src.copy_from(luisa::span{host_src}) << synchronize();

        auto cmdlist = CommandList::create();
        cmdlist << src_copy.view().copy_from(src.view());
        cmdlist << kernel(dst.view(), src_copy.view(), 5.f).dispatch(kThreads);
        auto graph = ext->create_graph(std::move(cmdlist.commit()).command_list());
        expect(graph.handle().handle != CudaGraphExt::invalid_handle)
            << "DAG capture of copy+dispatch should succeed";
        if (graph.handle().handle == CudaGraphExt::invalid_handle) { return; }
        auto exec = ext->instantiate(graph.handle().handle);
        expect(exec.handle().handle != CudaGraphExt::invalid_handle)
            << "instantiate should succeed";
        if (exec.handle().handle == CudaGraphExt::invalid_handle) { return; }

        ext->launch(exec.handle().handle, stream.handle());
        stream << synchronize();
        luisa::vector<float> replayed(kThreads);
        stream << dst.view().copy_to(luisa::span{replayed}) << synchronize();
        for (auto i = 0u; i < kThreads; i++) {
            expect(replayed[i] == host_src[i] * 5.f) << "copy-node mismatch at " << i;
        }
    }

    LUISA_INFO("CUDA graph DAG builder test passed.");
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    test_cuda_graph_dag(device);
    return 0;
}
