#pragma once
#include <luisa/core/logging.h>
#include <luisa/vstl/vector.h>
#include <luisa/vstl/unique_ptr.h>
#include <luisa/runtime/graph/utils.h>

namespace luisa::compute::graph {
class GraphBuilder;
template<typename T>
class InputSubVarCollection {
    template<typename T>
    using U = luisa::unique_ptr<T>;
public:
    size_t _input_var_count = 0;
    vector<U<T>> _sub_vars;

    auto input_vars() noexcept { return span{_sub_vars}.subspan(0, _input_var_count); }
    auto input_vars() const noexcept { return span{_sub_vars}.subspan(0, _input_var_count); }
    auto sub_vars() noexcept { return span{_sub_vars}.subspan(_input_var_count); }
    auto sub_vars() const noexcept { return span{_sub_vars}.subspan(_input_var_count); }

    InputSubVarCollection() noexcept = default;
    InputSubVarCollection(const InputSubVarCollection &rhs) noexcept;
    InputSubVarCollection &operator=(const InputSubVarCollection &) noexcept = delete;
    InputSubVarCollection(InputSubVarCollection &&) noexcept = default;
    InputSubVarCollection &operator=(InputSubVarCollection &&) noexcept = delete;

    void def_input_var(U<T> &&input_var) noexcept;

    void def_sub_var(U<T> &&sub_var) noexcept;
};
}// namespace luisa::compute::graph

#include <luisa/runtime/graph/graph_var.h>
namespace luisa::compute::graph {
template<typename T>
void InputSubVarCollection<T>::def_input_var(InputSubVarCollection<T>::U<T> &&input_var) noexcept {
    //LUISA_ASSERT(input_var->is_input_var(), "this function can only be invoked when it is a input var");
    //LUISA_ASSERT(_input_var_count == _sub_vars.size(), "input var must be defined before sub var");

    auto ptr = input_var.get();
    _sub_vars.emplace_back(std::move(input_var));
    ++_input_var_count;
}

template<typename T>
void InputSubVarCollection<T>::def_sub_var(InputSubVarCollection<T>::U<T> &&sub_var) noexcept {
    _sub_vars.emplace_back(std::move(sub_var));
}
template<typename T>
InputSubVarCollection<T>::InputSubVarCollection(const InputSubVarCollection &rhs) noexcept
    : _input_var_count{rhs._input_var_count} {
    _sub_vars.resize(rhs._sub_vars.size());
    for (size_t i = 0; i < rhs._sub_vars.size(); i++) {
        _sub_vars[i] = std::move(rhs._sub_vars[i]->clone());
    }
}
}// namespace luisa::compute::graph