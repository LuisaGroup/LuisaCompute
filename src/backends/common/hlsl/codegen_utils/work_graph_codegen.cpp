// Work Graph Code Generation Helpers

#include <algorithm>
#include "../hlsl_codegen.h"
#include "../codegen_stack_data.h"
#include "../register_indexer.h"
#include <luisa/vstl/string_utility.h>
#include <luisa/ast/type_registry.h>
#include <luisa/core/logging.h>
#include "../constant_printer.h"
#include "../shader_property.h"

namespace lc::hlsl {

using namespace luisa;
using namespace luisa::compute;

// Helper to generate [MaxRecords(n)] attribute for NodeOutput
void CodegenUtility::GenerateMaxRecordsAttribute(uint max_records, vstd::StringBuilder &result) {
    result << "[MaxRecords("sv;
    vstd::to_string(max_records, result);
    result << ")]"sv;
}

// Helper to generate NodeOutput(Array)<T> declaration
void CodegenUtility::GenerateNodeOutputDecl(
    const luisa::compute::WorkGraph& work_graph,
    const luisa::compute::detail::WorkGraphEdge& edge,
    vstd::StringBuilder &result
) {
    const auto& dest = work_graph.nodes().at(edge.dest);
    LUISA_ASSERT(edge.dest_array == ~0u || dest.array == edge.dest_array);

    uint max_records = edge.max_records;
    uint output_index = edge.source_output_index;
    const Type* record_type = dest.input_record_type;

    // Generate [MaxRecords(n)] attribute
    GenerateMaxRecordsAttribute(max_records, result);
    result << ' ';

    // Generating [NodeID(id)] or [NodeID(array, index)] for output
    if (dest.array != ~0u) {
        const auto& dest_array = work_graph.node_arrays().at(dest.array);
        uint index = edge.dest_array == ~0u ? edge.dest - dest_array.start : 0;
        result << "[NodeID(\""sv << dest_array.array_name << "\", "sv; vstd::to_string(index, result); result << ")] "sv;
    }
    else {
        result << "[NodeID(\""sv << dest.name << "\")] "sv;
    }

    if (edge.dest_array != ~0u) {
        const auto& dest_array = work_graph.node_arrays().at(edge.dest_array);
        result << "[NodeArraySize("sv; vstd::to_string(dest_array.count, result); result << ")] "sv;
    }

    // Generate NodeOutput(Array)<T> or EmptyNodeOutput
    if (record_type != nullptr) {
        if (edge.dest_array != ~0u) {
            result << "NodeOutputArray<"sv;
        }
        else {
            result << "NodeOutput<"sv;
        }
        GetTypeName(*record_type, result, Usage::READ);
        result << "> "sv;
    } else {
        if (edge.dest_array != ~0u) {
            result << "EmptyNodeOutputArray "sv;
        }
        else {
            result << "EmptyNodeOutput "sv;
        }
    }

    result << "_work_graph_output_"sv;
    vstd::to_string(static_cast<int64_t>(output_index), result);
}

// Helper to generate node input parameter
void CodegenUtility::GenerateNodeInputDecl(
    const luisa::compute::detail::WorkGraphNode& node,
    luisa::string_view var_name,
    bool more_arguments,
    vstd::StringBuilder &result) {

    result << "    "sv;

    bool has_input_record = node.input_record_type != nullptr;
    if (!has_input_record) {
        result << "// empty input record\n"sv;
        return;
    }

    switch (node.node_type) {
        case WorkGraphLaunchType::BROADCASTING: {
            result << "DispatchNodeInputRecord<"sv;
            GetTypeName(*node.input_record_type, result, Usage::READ);
            result << "> "sv << var_name;
        } break;
        case WorkGraphLaunchType::THREAD: {
            result << "ThreadNodeInputRecord<"sv;
            GetTypeName(*node.input_record_type, result, Usage::READ);
            result << "> "sv << var_name;
        } break;
    }

    if (more_arguments) {
        result << ",\n"sv;
    }
}

// Helper to generate node shader attributes
void CodegenUtility::GenerateNodeShaderAttributes(
    const luisa::compute::detail::WorkGraphNode &node,
    luisa::span<const luisa::compute::detail::WorkGraphNodeArray> node_arrays,
    vstd::StringBuilder &result) {

    // [Shader("node")] attribute (required for all work graph nodes)
    result << "[Shader(\"node\")]\n"sv;

    // use NodeId annotation to group nodes into array
    if (node.array != ~0u) {
        const auto& array = node_arrays[node.array];
        result << "[NodeID(\""sv
               << array.array_name
               << "\", "sv;
        vstd::to_string(node.index - array.start, result);;
        result << ")]\n"sv;
    }

    switch (node.node_type) {
        case WorkGraphLaunchType::BROADCASTING: {
            result << "[NodeLaunch(\"broadcasting\")]\n"sv;

            luisa::string threadgroup_dim = luisa::format(
                "[NumThreads({}, {}, {})]",
                node.threadgroup_dim.x, node.threadgroup_dim.y, node.threadgroup_dim.z
            );
            result << threadgroup_dim << '\n';

            luisa::string dispatch_properties;
            if (node.input_record_has_dispatch_grid) {
                dispatch_properties = luisa::format(
                    "[NodeMaxDispatchGrid({}, {}, {})]",
                    node.dispatch_dim.x, node.dispatch_dim.y, node.dispatch_dim.z
                );
            }
            else {
                dispatch_properties = luisa::format(
                    "[NodeDispatchGrid({}, {}, {})]",
                    node.dispatch_dim.x, node.dispatch_dim.y, node.dispatch_dim.z
                );
            }
            result << dispatch_properties << '\n';
        } break;
        case WorkGraphLaunchType::THREAD: {
            result << "[NodeLaunch(\"thread\")]\n"sv;
        } break;
    }

    // Add comment with node name for debugging
    if (!node.name.empty()) {
        result << "// node name: "sv << node.name << "\n"sv;
    }

}

// Helper to generate work graph node function signature
void CodegenUtility::GenerateNodeFunctionSignature(
    Function node_func,
    const luisa::compute::WorkGraph &work_graph,
    const luisa::compute::detail::WorkGraphNode &node,
    vstd::StringBuilder &result) {

    using luisa::compute::detail::WorkGraphEdge;

    // Generate system values for broadcasting nodes
    if (node.node_type == WorkGraphLaunchType::BROADCASTING) {
        result << "    "sv << "uint3 thdId : SV_GroupThreadID,\n"sv;
        result << "    "sv << "uint3 grpId : SV_GroupID,\n"sv;
        result << "    "sv << "uint3 dspId : SV_DispatchThreadID,\n"sv;
    }

    // Generate NodeInput parameter for input record
    bool more_arguments = !node.out_edges.empty();
    GenerateNodeInputDecl(node, "_work_graph_input"sv, more_arguments, result);

    // Collect and generate NodeOutput parameters
    for (size_t i = 0; i < node.out_edges.size(); ++i) {
        const auto &edge = node.out_edges[i];

        result << "    "sv;
        GenerateNodeOutputDecl(work_graph, edge, result);

        bool is_last_output = (i == node.out_edges.size() - 1);
        if (!is_last_output) {
            result << ",\n"sv;
        }
    }
}

// Helper to generate the _work_graph_output call for emitting records
void CodegenUtility::GenerateWorkGraphOutputCall(
    int output_index,
    luisa::string_view record_var_name,
    vstd::StringBuilder &result) {

    result << "_work_graph_output(_work_graph_output_"sv;
    vstd::to_string(static_cast<int64_t>(output_index), result);
    result << ", "sv << record_var_name << ", true)"sv;
}

// Helper to generate record struct definition with proper alignment
void CodegenUtility::GenerateRecordStructDef(
    const Type *record_type,
    vstd::StringBuilder &result) {

    if (!record_type || !record_type->is_structure()) {
        return;
    }

    // Register the struct type for code generation
    RegistStructType(record_type);
}

// Helper to generate node dispatch grid setup (for entry point nodes)
void CodegenUtility::GenerateNodeDispatchGrid(
    const uint3 &grid_size,
    vstd::StringBuilder &result) {

    // HLSL work graphs use DispatchNodeInputRecordIndex() for grid setup
    // This is typically handled implicitly by the runtime, but we can
    // provide grid size information via constants if needed
    result << "    // Dispatch grid: ("sv;
    vstd::to_string(grid_size.x, result);
    result << ", "sv;
    vstd::to_string(grid_size.y, result);
    result << ", "sv;
    vstd::to_string(grid_size.z, result);
    result << ")\n"sv;
}

// Helper to generate the complete node function body
void CodegenUtility::GenerateNodeFunctionBody(
    Function node_func,
    const luisa::compute::detail::WorkGraphNode& node,
    vstd::StringBuilder &result) {

    result << " {\n"sv;

    // extract input record
    if (node.input_record_type != nullptr) {
        auto &first_arg = node_func.arguments()[0];
        GetTypeName(*node.input_record_type, result, Usage::READ);
        result << " "sv << "l"sv;
        vstd::to_string(first_arg.uid(), result);
        result << " = _work_graph_input.Get();\n"sv;
    }

    // Generate function body using StringStateVisitor
    opt->funcType = CodegenStackData::FuncType::WorkGraphNode;

#ifdef LUISA_ENABLE_IR
    vstd::unordered_set<Variable> grad_vars;
#endif
    {
        StringStateVisitor vis(node_func, result, this);
        vis.sharedVariables = &opt->sharedVariable;
        vis.VisitFunction(
#ifdef LUISA_ENABLE_IR
            grad_vars,
#endif
            node_func);
    }

    result << "}\n\n"sv;
}

void CodegenUtility::CodegenWorkGraphNode(
    const WorkGraph &work_graph,
    size_t node_index,
    vstd::StringBuilder &result,
    vstd::unordered_set<uint64_t> &callableMap,
    const vstd::unordered_map<uint64_t, uint32_t>& handle_to_canonical_uid
) {
    using luisa::compute::detail::WorkGraphEdge;
    const auto &nodes = work_graph.nodes();
    const auto &node = nodes[node_index];

    auto codegenOneFunc = [&](Function func) {
        auto constants = func.constants();
        for (auto &&i : constants) {
            vstd::StringBuilder constValueName;
            if (!GetConstName(i.hash(), i, constValueName)) continue;
            result << "static const "sv;
            GetTypeName(*i.type(), result, Usage::READ);
            result << ' ' << constValueName << " = "sv;
            CodegenConstantPrinter printer{*this, result};
            i.decode(printer);
            result << ";\n"sv;
        }
#ifdef LUISA_ENABLE_IR
        vstd::unordered_set<Variable> grad_vars;
        // glob_variables_with_grad(func, grad_vars);
#endif

        opt->funcType = CodegenStackData::FuncType::Callable;
        GetFunctionDecl(func, result);
        result << "{\n"sv;
        {

            StringStateVisitor vis(func, result, this);
            vis.sharedVariables = &opt->sharedVariable;
            vis.VisitFunction(
#ifdef LUISA_ENABLE_IR
                grad_vars,
#endif
                func);
        }
        result << "}\n"sv;
    };

    auto callable = [&](auto &&callable, Function func) -> void {
        for (auto &&i : func.custom_callables()) {
            if (callableMap.emplace(i->hash()).second) {
                callable(callable, i->function());
            }
        }

        // Remap this callable's captured variables before generating its body
        auto c_args = func.arguments();
        auto c_bindings = func.bound_arguments();
        auto old_remap = std::move(opt->uid_remap);
        opt->uid_remap = {};

        for (size_t j = 0; j < c_bindings.size(); j++) {
            luisa::visit([&]<typename T>(T const &b) noexcept {
                if constexpr (std::is_same_v<T, Function::BufferBinding> ||
                              std::is_same_v<T, Function::TextureBinding> ||
                              std::is_same_v<T, Function::AccelBinding> ||
                              std::is_same_v<T, Function::BindlessArrayBinding>) {
                    auto it = handle_to_canonical_uid.find(b.handle);
                    LUISA_ASSERT(it != handle_to_canonical_uid.end(), "all bound arguments should be canonicalized");

                    uint32_t canonical_uid = it->second;
                    uint32_t current_uid = c_args[j].uid();
                    opt->uid_remap[current_uid] = canonical_uid;
                }
            }, c_bindings[j]);
        }

        codegenOneFunc(func);
        opt->uid_remap = std::move(old_remap);
    };

    auto node_func = node.fn_builder->function();
    for (auto &&i : node_func.custom_callables()) {
        if (callableMap.emplace(i->hash()).second) {
            callable(callable, i->function());
        }
    }

    // Emit static const declarations for the node function's own constants
    auto node_constants = node_func.constants();
    for (auto &&i : node_constants) {
        vstd::StringBuilder constValueName;
        if (!GetConstName(i.hash(), i, constValueName)) continue;
        result << "static const "sv;
        GetTypeName(*i.type(), result, Usage::READ);
        result << ' ' << constValueName << " = "sv;
        CodegenConstantPrinter printer{*this, result};
        i.decode(printer);
        result << ";\n"sv;
    }

    // Generate node shader attributes using helper
    GenerateNodeShaderAttributes(node, work_graph.node_arrays(), result);

    // Generate node function signature
    // use actual name from frontend here, rather than custom_<i>, since node names are meaningful
    result << "void "sv << node.name << "(\n"sv;

    // Generate node function parameters using helper
    GenerateNodeFunctionSignature(node_func, work_graph, node, result);

    result << "\n)"sv;

    // Generate function body using helper
    GenerateNodeFunctionBody(node_func, node, result);
}

vstd::vector<CodegenUtility::WorkGraphCapturedBinding> CodegenUtility::CollectWorkGraphBindings(
    const WorkGraph &work_graph,
    CodegenResult::Properties &out_properties,
    vstd::unordered_map<uint64_t, uint32_t> &out_uid_map,
    uint &out_bind_count,
    uint &out_preamble_count) {
    out_preamble_count = 0;

    // Maps for merging usage across nodes (keyed by handle)
    vstd::unordered_map<uint64_t, Usage> handle_to_usage;
    vstd::unordered_map<uint64_t, Function::Binding> handle_to_binding;
    vstd::unordered_map<uint64_t, const Type *> handle_to_type;
    uint32_t uid_counter = 0;

    // First-encounter traversal: assign UIDs in node/binding order, merge usage for duplicates
    for (const auto &node : work_graph.nodes()) {
        auto func = node.fn_builder->function();
        auto args = func.arguments();
        auto bindings = func.bound_arguments();
        LUISA_ASSERT(args.size() == bindings.size(), "`args` and `bindings` of AST function should be parallel, same size");

        for (size_t j = 0; j < bindings.size(); j++) {
            luisa::visit([&]<typename T>(T const &binding) noexcept {
                uint64_t handle;
                if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                    handle = binding.handle;
                } else if constexpr (std::is_same_v<T, Function::TextureBinding>) {
                    handle = binding.handle;
                } else if constexpr (std::is_same_v<T, Function::AccelBinding>) {
                    handle = binding.handle;
                } else if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>) {
                    handle = binding.handle;
                } else if constexpr (std::is_same_v<T, luisa::monostate>) {
                    return;
                } else {
                    LUISA_ERROR("unsupported captured binding type in work graph");
                }

                auto &var = args[j];
                auto *type = var.type();
                auto usage = func.variable_usage(var.uid());

                if (out_uid_map.find(handle) != out_uid_map.end()) {
                    // Already seen: validate compatibility and merge usage
                    if (Function::Binding(binding) != handle_to_binding[handle])
                        LUISA_ERROR("aliasing different views of buffer / different mip levels of texture not supported");
                    if (type != handle_to_type[handle])
                        LUISA_ERROR("aliasing different types of buffer / texture not supported");
                    handle_to_usage[handle] = Usage(luisa::to_underlying(handle_to_usage[handle]) | luisa::to_underlying(usage));
                    return;
                }

                out_uid_map.emplace(handle, uid_counter++);
                handle_to_type.emplace(handle, type);
                handle_to_binding.emplace(handle, binding);
                handle_to_usage.emplace(handle, usage);
            }, bindings[j]);
        }
    }

    // Sort handles into UID order (= first-encounter order) so that out_properties[preamble + prop_offset(i)]
    // and the returned captured[i] correspond to the same resource.
    vstd::vector<std::pair<uint32_t, uint64_t>> sorted; // (uid, handle)
    sorted.reserve(out_uid_map.size());
    for (auto &[handle, uid] : out_uid_map) {
        sorted.emplace_back(uid, handle);
    }
    std::sort(sorted.begin(), sorted.end());

    // Check whether any bindless array was captured so we can emit the preamble properties.
    bool has_bindless = false;
    for (auto &[uid, handle] : sorted) {
        luisa::visit([&]<typename T>(T const &) noexcept {
            if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>)
                has_bindless = true;
        }, handle_to_binding[handle]);
        if (has_bindless) break;
    }

    // Preamble: sampler heap (space 1) + three bindless heaps (spaces 2/3/4).
    // Added conservatively when any bindless array is captured; unused heap slots
    // in the root signature are harmless.
    if (has_bindless) {
        out_properties.emplace_back(Property{ShaderVariableType::SamplerHeap,    1u, 0u, 16u});
        out_properties.emplace_back(Property{ShaderVariableType::SRVBufferHeap,  2u, 0u, 1u});
        out_properties.emplace_back(Property{ShaderVariableType::SRVTextureHeap, 3u, 0u, 1u});
        out_properties.emplace_back(Property{ShaderVariableType::SRVTextureHeap, 4u, 0u, 1u});
        out_preamble_count = 4u;
    }

    // Index into DXILRegisterIndexer: 0=CBV(b), 1=UAV(u), 2=SRV(t)
    constexpr uint kUAV = 1u;
    constexpr uint kSRV = 2u;
    // Default-construct without calling init(): work graphs have no dispatch constant,
    // so all register counters start at 0 (unlike compute kernels where b0 is reserved).
    DXILRegisterIndexer registers{};

    vstd::vector<WorkGraphCapturedBinding> captured;
    captured.reserve(sorted.size());

    for (auto &[uid, handle] : sorted) {
        auto usage = handle_to_usage[handle];
        auto *type = handle_to_type[handle];
        bool writable = (to_underlying(usage) & to_underlying(Usage::WRITE)) != 0;

        luisa::visit([&]<typename T>(T const &binding) noexcept {
            if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                auto reg_type = writable ? kUAV : kSRV;
                auto stype = writable ? ShaderVariableType::RWStructuredBuffer : ShaderVariableType::StructuredBuffer;
                out_properties.emplace_back(Property{stype, 0u, registers.get(reg_type)++, 1u});
                out_bind_count += 2;

                Argument arg{};
                arg.tag = Argument::Tag::BUFFER;
                arg.buffer = binding;
                captured.emplace_back(WorkGraphCapturedBinding{arg, usage, type});
            } else if constexpr (std::is_same_v<T, Function::TextureBinding>) {
                auto reg_type = writable ? kUAV : kSRV;
                auto stype = writable ? ShaderVariableType::UAVTextureHeap : ShaderVariableType::SRVTextureHeap;
                out_properties.emplace_back(Property{stype, 0u, registers.get(reg_type)++, 1u});
                out_bind_count += 1;

                Argument arg{};
                arg.tag = Argument::Tag::TEXTURE;
                arg.texture = binding;
                captured.emplace_back(WorkGraphCapturedBinding{arg, usage, type});
            } else if constexpr (std::is_same_v<T, Function::AccelBinding>) {
                // Accel (read-only): two SRV root descriptors (TLAS buffer + instance data buffer).
                LUISA_ASSERT(!writable, "only support read-only acceleration structure for now");
                out_properties.emplace_back(Property{ShaderVariableType::StructuredBuffer, 0u, registers.get(kSRV)++, 1u});
                out_properties.emplace_back(Property{ShaderVariableType::StructuredBuffer, 0u, registers.get(kSRV)++, 1u});
                out_bind_count += 4;

                Argument arg{};
                arg.tag = Argument::Tag::ACCEL;
                arg.accel = binding;
                captured.emplace_back(WorkGraphCapturedBinding{arg, usage, type});
            } else if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>) {
                // Indirection buffer for this specific bindless array (space 0).
                out_properties.emplace_back(Property{ShaderVariableType::StructuredBuffer, 0u, registers.get(kSRV)++, 1u});
                out_bind_count += 2;

                Argument arg{};
                arg.tag = Argument::Tag::BINDLESS_ARRAY;
                arg.bindless_array = binding;
                captured.emplace_back(WorkGraphCapturedBinding{arg, usage, type});
            } else {
                LUISA_ERROR("this should not happen");
            }
        }, handle_to_binding[handle]);
    }

    return captured;
}

}// namespace lc::hlsl
