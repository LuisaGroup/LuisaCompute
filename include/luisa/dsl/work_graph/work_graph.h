#pragma once

#include "work_graph_kernel.h"
#include "work_graph_output.h"

#include <luisa/core/stl/vector.h>

namespace luisa::compute {

class WorkGraphBuilder;

class WorkGraphNode {
public:
    WorkGraphNode(WorkGraphBuilder* builder, luisa::string name, const Type* input_record_type, uint index) noexcept :
        _builder(builder), _name(std::move(name)), _input_record_type(input_record_type), _index(index) {}

    [[nodiscard]] luisa::string_view name() const noexcept { return _name; }

    template<typename T>
    [[nodiscard]] WorkGraphOutput<T> output() noexcept {
        auto output_index = _output_record_types.size();
        _output_record_types.push_back(Type::of<T>());
        return WorkGraphOutput<T>(output_index);
    }

    template<typename Kernel>
    void define(Kernel&& kernel) noexcept {
        // yoink the function builder, make sure type matches what we were declared with
        _fn_builder = kernel.function_builder();
        if (_input_record_type != nullptr) {
            LUISA_ASSERT(_fn_builder->arguments().size() > 0 && _fn_builder->arguments().front().type() == _input_record_type,
                "type mismatch between work graph node and its implementation");
        }
    }

private:
    WorkGraphBuilder* _builder;
    luisa::shared_ptr<const detail::FunctionBuilder> _fn_builder;

    luisa::string _name;
    const Type* _input_record_type;
    luisa::vector<const Type*> _output_record_types {};
    uint _index;
    bool _defined = false;
};

class WorkGraphBuilder {
public:

    template<typename InputRecord>
    WorkGraphNode* add_node(luisa::string_view name) noexcept {
        const Type* input_record_type;
        if constexpr (std::is_same_v<InputRecord, WorkGraphEmptyRecord>) {
            input_record_type = nullptr;
        }
        else {
            input_record_type = Type::of<InputRecord>();
        }

        _nodes.emplace_back(
            this,
            luisa::string(name),
            input_record_type,
            _nodes.size()
        );

        WorkGraphNode* node = &_nodes.back();
        return node;
    }

    void build() noexcept;

private:
    luisa::vector<WorkGraphNode> _nodes;
};

} // namespace luisa::compute