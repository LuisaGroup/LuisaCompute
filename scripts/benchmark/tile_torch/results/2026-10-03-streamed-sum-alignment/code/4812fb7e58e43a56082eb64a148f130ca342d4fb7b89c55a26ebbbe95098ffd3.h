#pragma once
#include <luisa/core/logging.h>
#include <luisa/runtime/device.h>
#include <luisa/tile/runtime.h>
#include <luisa/backends/ext/cuda/cuda_graph_ext.h>
#include <cuda.h>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace sum_alignment_diagnostic {
struct ScopedAlignment {
    std::string previous;
    bool existed{};
    explicit ScopedAlignment(bool enabled) {
        if (auto value = std::getenv("LUISA_CUDA_TILE_IR_ALIGNED16")) {
            previous = value;
            existed = true;
        }
        LUISA_ASSERT(_putenv_s("LUISA_CUDA_TILE_IR_ALIGNED16", enabled ? "1" : "0") == 0,
                     "Cannot set diagnostic compilation policy.");
    }
    ~ScopedAlignment() {
        LUISA_ASSERT(_putenv_s("LUISA_CUDA_TILE_IR_ALIGNED16", existed ? previous.c_str() : "") == 0,
                     "Cannot restore diagnostic compilation policy.");
    }
};

inline bool numeric_marker(luisa::string_view text, luisa::string_view key, uint32_t &value) {
    auto begin = text.find(key);
    if (begin == luisa::string_view::npos || text.find(key, begin + key.size()) != luisa::string_view::npos) { return false; }
    begin += key.size();
    auto end = text.find(';', begin);
    if (end == luisa::string_view::npos) { end = text.size(); }
    if (end == begin) { return false; }
    value = 0u;
    for (auto i = begin; i < end; i++) {
        auto digit = text[i];
        if (digit < '0' || digit > '9' || value > (std::numeric_limits<uint32_t>::max() - static_cast<uint32_t>(digit - '0')) / 10u) { return false; }
        value = value * 10u + static_cast<uint32_t>(digit - '0');
    }
    return true;
}

// Inspect the actual graph that will be timed. This is an untimed Driver
// observation, not a device trace or an observation of ordinary live launches.
inline bool observe(luisa::compute::Device &device, luisa::compute::tile::Shader &shader,
                    const luisa::compute::CudaGraphInstance &graph,
                    const std::array<uint64_t, 4u> &pointers,
                    uint32_t rows, uint32_t nodes_expected, bool requested,
                    const std::filesystem::path &directory) {
    using luisa::string_view;
    auto text = string_view{shader.metadata().realization};
    uint32_t mask{}, loads{};
    if (requested) {
        if (!numeric_marker(text, "aligned16-buffer-mask=", mask) ||
            !numeric_marker(text, "aligned16-partition-loads=", loads) || mask != 1u || loads != 1u ||
            text.find("host-selected-dual-entry-aligned16-v1") == string_view::npos) { return false; }
    } else if (text.find("aligned16-requested") != string_view::npos) { return false; }
    if (cuCtxPushCurrent(static_cast<CUcontext>(device.native_handle())) != CUDA_SUCCESS) { return false; }
    struct Pop {
        ~Pop() {
            CUcontext previous{};
            LUISA_ASSERT(cuCtxPopCurrent(&previous) == CUDA_SUCCESS, "Diagnostic context restoration failed.");
        }
    } pop;
    auto main = static_cast<CUfunction>(shader.native_handle());
    CUmodule module{};
    CUfunction main_lookup{}, aligned{};
    if (cuFuncGetModule(&module, main) != CUDA_SUCCESS ||
        cuModuleGetFunction(&main_lookup, module, "luisa_tile_main") != CUDA_SUCCESS || main_lookup != main) { return false; }
    if (mask != 0u && cuModuleGetFunction(&aligned, module, "luisa_tile_aligned16") != CUDA_SUCCESS) { return false; }
    uint64_t alignment_bits{};
    for (size_t slot = 0; slot < pointers.size(); slot++) {
        if ((mask & (1u << slot)) != 0u) { alignment_bits |= pointers[slot]; }
    }
    auto selected_aligned = aligned != nullptr && (alignment_bits & 15u) == 0u;
    auto expected = selected_aligned ? aligned : main;
    auto expected_name = selected_aligned ? "luisa_tile_aligned16" : "luisa_tile_main";
    auto handle = static_cast<CUgraph>(graph.handle().native_handle);
    size_t count{};
    if (cuGraphGetNodes(handle, nullptr, &count) != CUDA_SUCCESS) { return false; }
    std::vector<CUgraphNode> nodes(count);
    if (cuGraphGetNodes(handle, nodes.data(), &count) != CUDA_SUCCESS) { return false; }
    std::ofstream csv{directory / "diagnostic-graph.csv"};
    csv << "function,entry,grid_x,grid_y,grid_z,block_x,block_y,block_z,shared_bytes,alignment_mask,arg0,arg1,arg2,arg3\n";
    uint32_t kernels{};
    for (auto node : nodes) {
        CUgraphNodeType type{};
        if (cuGraphNodeGetType(node, &type) != CUDA_SUCCESS || type != CU_GRAPH_NODE_TYPE_KERNEL) { return false; }
        CUDA_KERNEL_NODE_PARAMS p{};
        if (cuGraphKernelNodeGetParams(node, &p) != CUDA_SUCCESS || p.func != expected ||
            p.gridDimX != rows || p.gridDimY != 1u || p.gridDimZ != 1u ||
            p.blockDimX != 1u || p.blockDimY != 1u || p.blockDimZ != 1u ||
            p.sharedMemBytes != 0u || p.kernelParams == nullptr || p.extra != nullptr) { return false; }
        for (size_t slot = 0; slot < pointers.size(); slot++) {
            if (p.kernelParams[slot] == nullptr) { return false; }
            CUdeviceptr pointer{};
            std::memcpy(&pointer, p.kernelParams[slot], sizeof(pointer));
            if (pointer != pointers[slot]) { return false; }
        }
        const char *name{};
        if (cuFuncGetName(&name, p.func) != CUDA_SUCCESS || name == nullptr || std::strcmp(name, expected_name) != 0) { return false; }
        csv << p.func << ',' << name << ',' << p.gridDimX << ",1,1,1,1,1,0," << mask;
        for (auto pointer : pointers) { csv << ",0x" << std::hex << pointer << std::dec; }
        csv << '\n';
        kernels++;
    }
    if (kernels != nodes_expected || !csv) { return false; }
    // These attributes are exact loaded-entry observations. They are not
    // physical thread counts, occupancy, spill traffic or a performance cause.
    std::ofstream resources{directory / "diagnostic-resources.csv"};
    resources << "entry,module,function,attribute,value,known,cuda_status\n";
    constexpr std::array attributes{CU_FUNC_ATTRIBUTE_NUM_REGS, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK};
    constexpr std::array names{"registers", "static_shared_bytes", "local_bytes", "max_threads"};
    for (auto is_aligned : {false, true}) {
        auto function = is_aligned ? aligned : main;
        auto name = is_aligned ? "luisa_tile_aligned16" : "luisa_tile_main";
        if (function == nullptr) { resources << name << ',' << module << ",none,all,unknown,0,not-queried\n"; continue; }
        for (size_t i = 0; i < attributes.size(); i++) {
            int value = -1;
            auto status = cuFuncGetAttribute(&value, attributes[i], function);
            auto known = status == CUDA_SUCCESS && value >= 0;
            resources << name << ',' << module << ',' << function << ',' << names[i] << ',';
            if (known) { resources << value; } else { resources << "unknown"; }
            resources << ',' << known << ',' << static_cast<int>(status) << '\n';
        }
    }
    std::ofstream out{directory / "diagnostic-alignment.json"};
    out << "{\"schema\":1,\"compile_mode\":" << (requested ? 1 : 0)
        << ",\"buffer_mask\":" << mask << ",\"partition_loads\":" << loads
        << ",\"selected_entry\":\"" << expected_name << "\",\"actual_graph_nodes_checked\":" << kernels
        << ",\"same_module_function_identity_checked\":true,\"actual_final_pointers_checked\":true"
        << ",\"grid\":[" << rows << ",1,1],\"block\":[1,1,1],\"shared_bytes\":0,\"final_argument_mod16\":[";
    for (size_t slot = 0; slot < pointers.size(); slot++) { if (slot != 0u) { out << ','; } out << (pointers[slot] & 15u); }
    out << "]}";
    return static_cast<bool>(out) && static_cast<bool>(resources);
}
}// namespace sum_alignment_diagnostic
