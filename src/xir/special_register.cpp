#include <luisa/core/logging.h>
#include <luisa/ast/type_registry.h>
#include <luisa/xir/module.h>
#include <luisa/xir/special_register.h>

namespace luisa::compute::xir {

namespace detail {
const Type *special_register_type_uint() noexcept { return Type::of<uint32_t>(); }
const Type *special_register_type_uint3() noexcept { return Type::of<uint3>(); }
const Type *special_register_type_float3() noexcept { return Type::of<float3>(); }
const Type *special_register_type_bool() noexcept { return Type::of<bool>(); }
}// namespace detail

SentinelSpecialRegister::SentinelSpecialRegister(Module *module) noexcept
    : SpecialRegister{module, nullptr} {}

DerivedSpecialRegisterTag SentinelSpecialRegister::derived_special_register_tag() const noexcept {
    LUISA_ERROR_WITH_LOCATION("Sentinel special register should not be used.");
}

}// namespace luisa::compute::xir
