#include <luisa/ast/function.h>
#include <luisa/ast/function_builder.h>

namespace luisa::compute {

uint64_t Function::BufferBinding::hash() const noexcept {
    using namespace std::string_view_literals;
    static auto seed = hash_value("__hash_buffer_binding"sv);
    return hash_combine({handle, static_cast<uint64_t>(offset)}, seed);
}

uint64_t Function::TextureBinding::hash() const noexcept {
    using namespace std::string_view_literals;
    static auto seed = hash_value("__hash_texture_binding"sv);
    return hash_combine({handle, static_cast<uint64_t>(level)}, seed);
}

uint64_t Function::AccelBinding::hash() const noexcept {
    using namespace std::string_view_literals;
    static auto seed = hash_value("__hash_accel_binding"sv);
    return hash_value(handle, seed);
}

uint64_t Function::BindlessArrayBinding::hash() const noexcept {
    using namespace std::string_view_literals;
    static auto seed = hash_value("__hash_bindless_array_binding"sv);
    return hash_value(handle, seed);
}

luisa::span<const Variable> Function::builtin_variables() const noexcept {
    return _builder->builtin_variables();
}

luisa::span<const Function::Constant> Function::constants() const noexcept {
    return _builder->constants();
}

luisa::span<const Variable> Function::arguments() const noexcept {
    return _builder->arguments();
}

Function::Tag Function::tag() const noexcept {
    return _builder->tag();
}

const ScopeStmt *Function::body() const noexcept {
    return _builder->body();
}

luisa::span<const luisa::shared_ptr<const detail::FunctionBuilder>>
Function::custom_callables() const noexcept {
    return _builder->custom_callables();
}

luisa::span<const luisa::shared_ptr<const ExternalFunction>>
Function::external_callables() const noexcept {
    return _builder->external_callables();
}

CallOpSet Function::direct_builtin_callables() const noexcept {
    return _builder->direct_builtin_callables();
}

CallOpSet Function::propagated_builtin_callables() const noexcept {
    return _builder->propagated_builtin_callables();
}

const Type *Function::return_type() const noexcept {
    return _builder->return_type();
}

Usage Function::variable_usage(uint32_t uid) const noexcept {
    return _builder->variable_usage(uid);
}

uint3 Function::block_size() const noexcept {
    return _builder->block_size();
}

bool Function::hash_computed() const noexcept {
    return _builder->hash_computed();
}

uint64_t Function::hash() const noexcept {
    return _builder->hash();
}

bool Function::requires_atomic() const noexcept {
    return _builder->requires_atomic();
}

bool Function::requires_atomic_float() const noexcept {
    return _builder->requires_atomic_float();
}

bool Function::requires_raytracing() const noexcept {
    return _builder->requires_raytracing();
}
bool Function::requires_motion_blur() const noexcept {
    return _builder->requires_motion_blur();
}

luisa::shared_ptr<const detail::FunctionBuilder> Function::shared_builder() const noexcept {
    return _builder->shared_from_this();
}

luisa::span<const Variable> Function::local_variables() const noexcept {
    return _builder->local_variables();
}

luisa::span<const Variable> Function::shared_variables() const noexcept {
    return _builder->shared_variables();
}

luisa::span<const Function::Binding> Function::bound_arguments() const noexcept {
    return _builder->bound_arguments();
}

luisa::span<const Variable> Function::unbound_arguments() const noexcept {
    return _builder->unbound_arguments();
}

bool Function::requires_autodiff() const noexcept {
    return _builder->requires_autodiff();
}

bool Function::requires_printing() const noexcept {
    return _builder->requires_printing();
}

CurveBasisSet Function::required_curve_bases() const noexcept {
    return _builder->required_curve_bases();
}

}// namespace luisa::compute
