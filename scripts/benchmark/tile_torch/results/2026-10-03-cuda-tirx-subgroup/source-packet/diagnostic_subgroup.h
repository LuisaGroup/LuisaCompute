#pragma once
#include <luisa/core/logging.h>
#include <luisa/runtime/device.h>
#include <luisa/tile/runtime.h>
#include <luisa/backends/ext/cuda/cuda_graph_ext.h>
#include <cuda.h>
#include <array>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace cuda_subgroup_diagnostic {

inline bool field(std::string_view text, std::string_view key, std::string_view &value) {
    auto position = text.find(key);
    if (position == std::string_view::npos || text.find(key, position + key.size()) != std::string_view::npos) { return false; }
    auto first = position + key.size();
    auto end = text.find(';', first);
    value = text.substr(first, end == std::string_view::npos ? text.size() - first : end - first);
    return !value.empty();
}

inline bool observe(luisa::compute::Device &device, luisa::compute::tile::Shader &shader,
                    const luisa::compute::CudaGraphInstance &graph,
                    const std::array<uint64_t, 4u> &pointers,
                    uint32_t rows, bool rms, bool fast,
                    const std::filesystem::path &directory) {
    auto &realization = shader.metadata().realization;
    auto text = std::string_view{realization.data(), realization.size()};
    for (auto marker : {"aligned16-requested", "worker-warps-hint=", "program-partition-",
                        "partition-cost-", "streaming-scan-", "cub-scan-requested",
                        "independent-axis-extent=", "scan-chunk=", "collective-cost-profile="}) {
        if (text.find(marker) != std::string_view::npos) { return false; }
    }
    std::string_view entry, plan, count, arguments, math;
    if (!field(text, "; cuda-subgroup-entry=", entry) ||
        !field(text, "; cuda-subgroup-plan=", plan) ||
        !field(text, "; cuda-subgroup-plans=", count) ||
        !field(text, "; cuda-subgroup-bindings=", arguments) ||
        !field(text, "; cuda-subgroup-math=", math) || count != "1" ||
        plan != (rms ? "threads128:programs2:warps2:lane-elements8:reductions1" : "threads128:programs2:warps2:lane-elements8:reductions2") ||
        math != (fast ? "fast-elements-preserved-reductions-v1" : "strict-no-contract-v1")) { return false; }
    for (auto c : entry) {
        if (!(c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))) { return false; }
    }
    std::array<bool, 4u> seen{};
    std::vector<uint32_t> bindings;
    while (!arguments.empty()) {
        auto end = arguments.find(',');
        auto token = arguments.substr(0u, end);
        uint32_t value{};
        auto parsed = std::from_chars(token.data(), token.data() + token.size(), value);
        if (token.empty() || parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() ||
            value >= seen.size() || seen[value]) { return false; }
        seen[value] = true;
        bindings.emplace_back(value);
        if (end == std::string_view::npos) { break; }
        arguments.remove_prefix(end + 1u);
        if (arguments.empty()) { return false; }
    }
    if (!seen[0] || !seen[3] || (rms && !seen[1])) { return false; }
    auto block = shader.block_size();
    auto dispatch = shader.metadata().dispatch_size;
    auto programs = (rows + 1u) / 2u;
    if (block.x != 128u || block.y != 1u || block.z != 1u ||
        dispatch.x % block.x != 0u || dispatch.x / block.x != programs ||
        dispatch.y != 1u || dispatch.z != 1u) { return false; }
    if (cuCtxPushCurrent(static_cast<CUcontext>(device.native_handle())) != CUDA_SUCCESS) { return false; }
    struct Pop {
        ~Pop() {
            CUcontext previous{};
            LUISA_ASSERT(cuCtxPopCurrent(&previous) == CUDA_SUCCESS, "Diagnostic context restoration failed.");
        }
    } pop;
    auto function = static_cast<CUfunction>(shader.native_handle());
    CUmodule module{};
    CUfunction lookup{};
    auto name_string = std::string{entry};
    const char *actual_name{};
    if (cuFuncGetModule(&module, function) != CUDA_SUCCESS ||
        cuModuleGetFunction(&lookup, module, name_string.c_str()) != CUDA_SUCCESS || lookup != function ||
        cuFuncGetName(&actual_name, function) != CUDA_SUCCESS || actual_name == nullptr || entry != actual_name) { return false; }
    auto handle = static_cast<CUgraph>(graph.handle().native_handle);
    size_t count_nodes{};
    if (cuGraphGetNodes(handle, nullptr, &count_nodes) != CUDA_SUCCESS) { return false; }
    std::vector<CUgraphNode> nodes(count_nodes);
    if (cuGraphGetNodes(handle, nodes.data(), &count_nodes) != CUDA_SUCCESS || count_nodes != 100u) { return false; }
    std::ofstream csv{directory / "diagnostic-graph.csv"};
    csv << "function,entry,grid_x,grid_y,grid_z,block_x,block_y,block_z,dynamic_shared_bytes";
    for (auto slot : bindings) { csv << ",host_arg" << slot; }
    csv << '\n';
    for (auto node : nodes) {
        CUgraphNodeType type{};
        if (cuGraphNodeGetType(node, &type) != CUDA_SUCCESS || type != CU_GRAPH_NODE_TYPE_KERNEL) { return false; }
        CUDA_KERNEL_NODE_PARAMS p{};
        if (cuGraphKernelNodeGetParams(node, &p) != CUDA_SUCCESS || p.func != function ||
            p.gridDimX != programs || p.gridDimY != 1u || p.gridDimZ != 1u ||
            p.blockDimX != 128u || p.blockDimY != 1u || p.blockDimZ != 1u ||
            p.sharedMemBytes != 0u || p.kernelParams == nullptr || p.extra != nullptr) { return false; }
        csv << p.func << ',' << entry << ',' << programs << ",1,1,128,1,1,0";
        for (auto slot = size_t{0u}; slot < bindings.size(); slot++) {
            if (p.kernelParams[slot] == nullptr) { return false; }
            CUdeviceptr pointer{};
            std::memcpy(&pointer, p.kernelParams[slot], sizeof(pointer));
            if (pointer != pointers[bindings[slot]]) { return false; }
            csv << ",0x" << std::hex << pointer << std::dec;
        }
        csv << '\n';
    }
    if (!csv) { return false; }
    std::ofstream resources{directory / "diagnostic-resources.csv"};
    resources << "entry,module,function,attribute,value,known,cuda_status\n";
    constexpr std::array attributes{CU_FUNC_ATTRIBUTE_NUM_REGS, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK};
    constexpr std::array names{"registers", "static_shared_bytes", "local_bytes", "max_threads"};
    for (auto i = size_t{0u}; i < attributes.size(); i++) {
        int value = -1;
        auto status = cuFuncGetAttribute(&value, attributes[i], function);
        auto known = status == CUDA_SUCCESS && value >= 0;
        resources << entry << ',' << module << ',' << function << ',' << names[i] << ',';
        if (known) { resources << value; } else { resources << "unknown"; }
        resources << ',' << known << ',' << static_cast<int>(status) << '\n';
    }
    std::ofstream out{directory / "diagnostic-observation.json"};
    out << "{\"schema\":1,\"route\":\"cuda_tirx_subgroup\",\"actual_capture_br\":1,\"programs_per_group\":2,"
        << "\"threads_per_group\":128,\"warps_per_program\":2,\"lane_elements\":8,\"cache_reduction_inputs\":false,"
        << "\"reductions\":" << (rms ? 1 : 2) << ",\"fast_math\":" << (fast ? "true" : "false")
        << ",\"math_policy\":\"" << math << "\",\"actual_graph_nodes_checked\":100,"
        << "\"same_module_function_identity_checked\":true,\"actual_final_pointers_checked\":true,"
        << "\"entry\":\"" << entry << "\",\"grid\":[" << programs << ",1,1],\"block\":[128,1,1],"
        << "\"dynamic_shared_bytes\":0,\"device_slot_to_host_index\":[";
    for (auto i = size_t{0u}; i < bindings.size(); i++) { if (i != 0u) { out << ','; } out << bindings[i]; }
    out << "]}";
    return static_cast<bool>(out) && static_cast<bool>(resources);
}
}// namespace cuda_subgroup_diagnostic
