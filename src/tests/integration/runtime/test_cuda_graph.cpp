// Test for CUDA Graph extension
// - Captures upload + download commands into a CUDA graph
// - Instantiates and launches the graph
// - Verifies downloaded data matches uploaded data

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/core/logging.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/backends/ext/cuda/cuda_graph_ext.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_cuda_graph(Device &device) {

    // Only available on CUDA backend
    auto *ext = device.extension<CudaGraphExt>();
    if (!ext) {
        LUISA_INFO("CudaGraphExt not available, skipping test.");
        return;
    }

    static constexpr uint n = 1024u;

    // Create device buffer
    Buffer<float> buffer = device.create_buffer<float>(n);

    // Prepare host data: upload known values, download into zeroed buffer
    luisa::vector<float> upload_data(n);
    luisa::vector<float> download_data(n, -1.0f);
    for (uint i = 0u; i < n; i++) {
        upload_data[i] = static_cast<float>(i);
    }

    // Build a CommandList with upload + download
    auto cmdlist = CommandList::create();
    cmdlist << buffer.view().copy_from(luisa::span{upload_data});
    cmdlist << buffer.view().copy_to(luisa::span{download_data});

    // Create CUDA graph from the command list
    auto graph = ext->create_graph(std::move(cmdlist.commit()).command_list());
    expect(graph.handle().handle != CudaGraphExt::invalid_handle)
        << "create_graph should succeed";
    if (graph.handle().handle == CudaGraphExt::invalid_handle) {
        return;
    }

    // Instantiate the graph
    auto exec = ext->instantiate(graph.handle().handle);
    expect(exec.handle().handle != CudaGraphExt::invalid_handle)
        << "instantiate should succeed";
    if (exec.handle().handle == CudaGraphExt::invalid_handle) {
        return;
    }

    // Launch the graph on a stream
    Stream stream = device.create_stream();
    ext->launch(exec.handle().handle, stream.handle());
    stream << synchronize();

    // Verify: download_data should now match upload_data
    for (uint i = 0u; i < n; i++) {
        expect(static_cast<bool>(download_data[i] == upload_data[i]))
            << "mismatch at index " << i;
    }

    // Update the executable graph with different upload/download host buffers and run it again.
    luisa::vector<float> upload_data_2(n);
    luisa::vector<float> download_data_2(n, -2.0f);
    for (uint i = 0u; i < n; i++) {
        upload_data_2[i] = static_cast<float>(n + i);
    }

    auto updated_cmdlist = CommandList::create();
    updated_cmdlist << buffer.view().copy_from(luisa::span{upload_data_2});
    updated_cmdlist << buffer.view().copy_to(luisa::span{download_data_2});

    auto updated = ext->update(exec.handle().handle, std::move(updated_cmdlist.commit()).command_list());
    expect(updated) << "updating graph with new upload/download buffers should succeed";
    if (updated) {
        ext->launch(exec.handle().handle, stream.handle());
        stream << synchronize();

        bool result_changed = false;
        for (uint i = 0u; i < n; i++) {
            expect(static_cast<bool>(download_data_2[i] == upload_data_2[i]))
                << "second launch mismatch at index " << i;
            result_changed = result_changed || download_data_2[i] != download_data[i];
        }
        expect(result_changed) << "second launch should return updated data";
    }

    LUISA_INFO("CUDA graph upload/download test passed.");

    // RAII cleanup via destructors
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    test_cuda_graph(device);
    return 0;
}
