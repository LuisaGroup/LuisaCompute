// Work Graph Code Generation Helpers

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

// Helper to generate NodeOutput<T> declaration
void CodegenUtility::GenerateNodeOutputDecl(
    const luisa::compute::detail::WorkGraphNode& dest,
    uint max_records,
    luisa::string_view var_name_prefix,
    int output_index,
    vstd::StringBuilder &result) {

    const Type* record_type = dest.input_record_type;

    // Generate [MaxRecords(n)] attribute
    GenerateMaxRecordsAttribute(max_records, result);
    result << ' ';

    // Generating [NodeId(id)] for output
    result << "[NodeId(\""sv << dest.name << "\")] "sv;

    // Generate NodeOutput<T> or EmptyNodeOutput
    if (record_type != nullptr) {
        result << "NodeOutput<"sv;
        GetTypeName(*record_type, result, Usage::READ);
        result << "> "sv;
    } else {
        result << "EmptyNodeOutput "sv;
    }

    result  << var_name_prefix;
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
    vstd::StringBuilder &result) {

    // [Shader("node")] attribute (required for all work graph nodes)
    result << "[Shader(\"node\")]\n"sv;

    switch (node.node_type) {
        case WorkGraphLaunchType::BROADCASTING: {
            result << "[NodeLaunch(\"broadcasting\")]\n"sv;

            luisa::string threadgroup_dim = luisa::format(
                "[NumThreads({}, {}, {})]",
                node.threadgroup_dim.x, node.threadgroup_dim.y, node.threadgroup_dim.z
            );
            result << threadgroup_dim << '\n';

            luisa::string dispatch_properties;
            if (node.dispatch_grid_member) {
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
    const luisa::compute::detail::WorkGraphNode &node,
    const luisa::vector<luisa::compute::detail::WorkGraphNode> &all_nodes,
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
        const auto &dest = all_nodes[edge.dest];

        uint max_records = edge.max_records;

        result << "    "sv;
        GenerateNodeOutputDecl(dest, max_records, "_work_graph_output_"sv,
                               static_cast<int>(i), result);

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
                              std::is_same_v<T, Function::TextureBinding>) {
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

    // Generate node shader attributes using helper
    GenerateNodeShaderAttributes(node, result);

    // Generate node function signature
    // use actual name from frontend here, rather than custom_<i>, since node names are meaningful
    LUISA_ASSERT(!node_func.name().empty(), "work graph node's FunctionBuilder has invalid name");
    result << "void "sv << node_func.name() << "(\n"sv;

    // Generate node function parameters using helper
    GenerateNodeFunctionSignature(node_func, node, nodes, result);

    result << "\n)"sv;

    // Generate function body using helper
    GenerateNodeFunctionBody(node_func, node, result);
}

vstd::unordered_map<uint64_t, uint32_t> CodegenUtility::CodegenWorkGraphProperties(
    CodegenResult::Properties &properties,
    vstd::StringBuilder &varData,
    const WorkGraph &work_graph,
    RegisterIndexer &registerCount,
    uint &bind_count) {

    vstd::unordered_map<uint64_t, uint32_t> handle_to_canonical_uid;
    vstd::unordered_map<uint64_t, Usage> handle_to_usage;
    vstd::unordered_map<uint64_t, Function::Binding> handle_to_binding;
    vstd::unordered_map<uint64_t, const Type*> handle_to_type;

    uint32_t uid_counter = 0;

    Function primary_function;
    for (const auto &node : work_graph.nodes()) {
        auto func = node.fn_builder->function();
        if (primary_function.builder() == nullptr) { primary_function = func; }

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
                } else if constexpr (std::is_same_v<T, luisa::monostate>) {
                    return;
                } else {
                    LUISA_ERROR("only capturing buffers / textures is supported for now");
                }

                auto &var = args[j];
                auto *type = var.type();
                auto usage = func.variable_usage(var.uid());

                if (handle_to_canonical_uid.find(handle) != handle_to_canonical_uid.end()) {
                    auto [_, existing_binding] = *handle_to_binding.find(handle);
                    auto [_1, existing_type] = *handle_to_type.find(handle);
                    if (Function::Binding(binding) != existing_binding) {
                        LUISA_ERROR("aliasing different views of buffer / different mip levels of texture not supported");
                    }

                    if (type != existing_type) {
                        LUISA_ERROR("aliasing different type of buffer / texture not supported");
                    }

                    auto existing_usage = handle_to_usage[handle];
                    handle_to_usage[handle] = Usage(luisa::to_underlying(existing_usage) | luisa::to_underlying(usage));
                    return;
                }

                handle_to_canonical_uid.emplace(handle, uid_counter);
                handle_to_type.emplace(handle, type);
                handle_to_binding.emplace(handle, binding);
                handle_to_usage.emplace(handle, usage);
                uid_counter += 1;
            }, bindings[j]);
        }
    }

    // Index into registerCount: 0=CBV(b), 1=UAV(u), 2=SRV(t)
    constexpr uint kUAV = 1u;
    constexpr uint kSRV = 2u;

    auto emit_global = [&](const Type* type, Variable::Tag tag, uint32_t uid, Usage usage, uint reg_type, char reg_char) {
        GetTypeName(*type, varData, usage);
        varData << ' ';
        if (tag == Variable::Tag::BUFFER) {
            varData << "_b"sv;
        }
        else if (tag == Variable::Tag::TEXTURE) {
            varData << "_t"sv;
        }
        else {
            LUISA_ERROR("only buffer / texture supported for now");
        }
        vstd::to_string(uid, varData);

        if (!opt->noRegister) {
            auto &r = registerCount.get(reg_type);
            varData << " : register(" << reg_char;
            vstd::to_string(r, varData);
            varData << ");\n";
            r++;
        } else {
            varData << ";\n";
        }
    };

    for (const auto [handle, uid] : handle_to_canonical_uid) {
        auto binding = handle_to_binding[handle];
        luisa::visit([&]<typename T>(T const& binding) {
            if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                auto usage_union = handle_to_usage[binding.handle];
                bool writable = (to_underlying(usage_union) & to_underlying(Usage::WRITE)) != 0;
                auto reg_type = writable ? kUAV : kSRV;
                auto reg_char = writable ? 'u' : 't';
                auto stype = writable ? ShaderVariableType::RWStructuredBuffer : ShaderVariableType::StructuredBuffer;
                auto &r = registerCount.get(reg_type);
                properties.emplace_back(Property{stype, 0, r, 1});

                auto var_type = handle_to_type[handle];
                emit_global(var_type, Variable::Tag::BUFFER, uid, usage_union, reg_type, reg_char);
                bind_count += 2;
            } else if constexpr (std::is_same_v<T, Function::TextureBinding>) {
                auto usage_union = handle_to_usage[binding.handle];
                bool writable = (to_underlying(usage_union) & to_underlying(Usage::WRITE)) != 0;
                auto reg_type = writable ? kUAV : kSRV;
                auto reg_char = writable ? 'u' : 't';
                auto stype = writable ? ShaderVariableType::UAVTextureHeap : ShaderVariableType::SRVTextureHeap;
                auto &r = registerCount.get(reg_type);
                properties.emplace_back(Property{stype, 0, r, 1});

                auto var_type = handle_to_type[handle];
                emit_global(var_type, Variable::Tag::TEXTURE, uid, usage_union, reg_type, reg_char);
                bind_count += 1;
            }
            else {
                LUISA_ERROR("this should not happen");
            }
        }, binding);
    }

    return handle_to_canonical_uid;
}

}// namespace lc::hlsl
