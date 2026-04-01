// Work Graph Code Generation Helpers

#include "../hlsl_codegen.h"
#include "../codegen_stack_data.h"
#include <luisa/vstl/string_utility.h>
#include <luisa/ast/type_registry.h>
#include <luisa/core/logging.h>
#include "../constant_printer.h"

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
    const Type *record_type,
    uint max_records,
    luisa::string_view var_name_prefix,
    int output_index,
    vstd::StringBuilder &result) {

    // Generate [MaxRecords(n)] attribute
    GenerateMaxRecordsAttribute(max_records, result);
    result << ' ';

    // Generate NodeOutput<T>
    result << "NodeOutput<"sv;
    if (record_type != nullptr) {
        GetTypeName(*record_type, result, Usage::READ);
    } else {
        result << "uint"sv;// Fallback for empty records
    }
    result << "> "sv << var_name_prefix;
    vstd::to_string(static_cast<int64_t>(output_index), result);
}

// Helper to generate NodeInput<T> declaration
void CodegenUtility::GenerateNodeInputDecl(
    const Type *record_type,
    luisa::string_view var_name,
    vstd::StringBuilder &result) {

    result << "NodeInput<"sv;
    if (record_type != nullptr) {
        GetTypeName(*record_type, result, Usage::READ);
    } else {
        result << "uint"sv;// Fallback
    }
    result << "> "sv << var_name;
}

// Helper to generate node shader attributes
void CodegenUtility::GenerateNodeShaderAttributes(
    bool is_entry_point,
    luisa::string_view node_name,
    vstd::StringBuilder &result) {

    // [Shader("node")] attribute (required for all work graph nodes)
    result << "[Shader(\"node\")]\n"sv;

    // Entry point nodes need [NodeLaunch("broadcasting")]
    if (is_entry_point) {
        result << "[NodeLaunch(\"broadcasting\")]\n"sv;
    }

    // Add comment with node name for debugging
    if (!node_name.empty()) {
        result << "// node name: "sv << node_name << "\n"sv;
    }
}

// Helper to collect output edges grouped by source_output_index
luisa::vector<luisa::vector<const luisa::compute::detail::WorkGraphEdge *>>
CodegenUtility::CollectOutputEdgesByIndex(const luisa::compute::detail::WorkGraphNode &node) {
    using luisa::compute::detail::WorkGraphEdge;

    luisa::vector<luisa::vector<const WorkGraphEdge *>> output_edges_by_index;
    for (const auto &edge : node.out_edges) {
        if (edge.source_output_index >= output_edges_by_index.size()) {
            output_edges_by_index.resize(edge.source_output_index + 1);
        }
        output_edges_by_index[edge.source_output_index].push_back(&edge);
    }
    return output_edges_by_index;
}

// Helper to get the output record type for a set of edges
const Type *CodegenUtility::GetOutputRecordType(
    const luisa::vector<const luisa::compute::detail::WorkGraphEdge *> &edges,
    const luisa::vector<luisa::compute::detail::WorkGraphNode> &nodes) {

    for (const auto *edge : edges) {
        if (edge->dest < nodes.size()) {
            return nodes[edge->dest].input_record_type;
        }
    }
    return nullptr;
}

// Helper to generate work graph node function signature
void CodegenUtility::GenerateNodeFunctionSignature(
    Function node_func,
    const luisa::compute::detail::WorkGraphNode &node,
    const luisa::vector<luisa::compute::detail::WorkGraphNode> &all_nodes,
    vstd::StringBuilder &result) {

    using luisa::compute::detail::WorkGraphEdge;

    // Track arguments for mapping
    opt->arguments.clear();
    uint arg_index = 0;

    // Generate NodeInput parameter for input record
    bool has_input_record = node.input_record_type != nullptr;
    if (has_input_record) {
        result << "    "sv;
        GenerateNodeInputDecl(node.input_record_type, "_work_graph_input"sv, result);
        opt->arguments.emplace(0, arg_index++);

        if (!node.out_edges.empty() || node_func.arguments().size() > 1) {
            result << ",\n"sv;
        }
    }

    // Collect and generate NodeOutput parameters
    auto output_edges_by_index = CollectOutputEdgesByIndex(node);

    for (size_t i = 0; i < output_edges_by_index.size(); ++i) {
        const auto &edges = output_edges_by_index[i];
        if (edges.empty()) continue;

        uint max_records = edges[0]->max_records;
        const Type *output_record_type = GetOutputRecordType(edges, all_nodes);

        result << "    "sv;
        GenerateNodeOutputDecl(output_record_type, max_records, "_work_graph_output_"sv,
                               static_cast<int>(i), result);

        bool is_last_output = (i == output_edges_by_index.size() - 1);
        bool has_more_args = node_func.arguments().size() > (has_input_record ? 1 : 0);
        if (!is_last_output || has_more_args) {
            result << ",\n"sv;
        }
    }

    // Generate remaining arguments (buffers, textures, etc.)
    auto args = node_func.arguments();
    size_t start_idx = has_input_record ? 1 : 0;
    for (size_t i = start_idx; i < args.size(); ++i) {
        auto &arg = args[i];
        result << "    "sv;
        GetTypeName(*arg.type(), result, Usage::READ);
        vstd::StringBuilder var_name;
        GetVariableName(node_func, arg, var_name);
        result << ' ' << var_name;
        opt->arguments.emplace(arg.uid(), arg_index++);

        if (i < args.size() - 1) {
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
    vstd::StringBuilder &result) {

    result << " {\n"sv;

    // Generate function body using StringStateVisitor
    opt->funcType = CodegenStackData::FuncType::Kernel;

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

void CodegenUtility::CodegenWorkGraphNode(const WorkGraph &work_graph, size_t node_index, bool is_entry_point, vstd::StringBuilder &result, vstd::unordered_set<uint64_t> &callableMap, bool cbufferNonEmpty) {
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
        codegenOneFunc(func);
    };

    auto node_func = node.fn_builder->function();
    for (auto &&i : node_func.custom_callables()) {
        if (callableMap.emplace(i->hash()).second) {
            callable(callable, i->function());
        }
    }

    // Generate node shader attributes using helper
    GenerateNodeShaderAttributes(is_entry_point, node.name, result);

    // Generate node function signature
    result << "void "sv << node_func.name() << "(\n"sv;

    // Generate node function parameters using helper
    GenerateNodeFunctionSignature(node_func, node, nodes, result);

    // Generate function body using helper
    GenerateNodeFunctionBody(node_func, result);
}

}// namespace lc::hlsl
