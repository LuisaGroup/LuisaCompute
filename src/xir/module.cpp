#include <luisa/core/logging.h>
#include <luisa/ast/type.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>

#include <algorithm>
#include <cstring>

namespace luisa::compute::xir {

KernelFunction *Module::create_kernel() noexcept {
    auto f = luisa::make_managed<KernelFunction>(this);
    return static_cast<KernelFunction *>(_function_list.push_back(std::move(f)));
}

CallableFunction *Module::create_callable(const Type *ret_type) noexcept {
    auto f = luisa::make_managed<CallableFunction>(this, ret_type);
    return static_cast<CallableFunction *>(_function_list.push_back(std::move(f)));
}

RasterStageFunction *Module::create_raster_stage(
    const Type *ret_type, RasterStage stage) noexcept {
    auto f = luisa::make_managed<RasterStageFunction>(this, ret_type, stage);
    return static_cast<RasterStageFunction *>(_function_list.push_back(std::move(f)));
}
ExternalFunction *Module::create_external_function(const Type *ret_type) noexcept {
    auto f = luisa::make_managed<ExternalFunction>(this, ret_type);
    return static_cast<ExternalFunction *>(_function_list.push_back(std::move(f)));
}

Constant *Module::_get_or_create_constant(const Constant &temp) noexcept {
    auto iter = _hash_to_constants.try_emplace(temp.hash()).first;
    for (auto *constant : iter->second) {
        if (constant->type() == temp.type() &&
            std::memcmp(constant->data(), temp.data(),
                        temp.type()->size()) == 0) {
            return constant;
        }
    }
    auto pooled = luisa::make_managed<Constant>(
        this, temp.type(), temp.data(), temp.hash());
    auto *constant = static_cast<Constant *>(
        _constant_list.push_back(std::move(pooled)));
    iter->second.emplace_back(constant);
    return constant;
}

Module::Module() noexcept
    : _function_list{this},
      _constant_list{this},
      _undefined_list{this},
      _special_register_list{this} {}

Constant *Module::create_constant(const Type *type, const void *data) noexcept {
    Constant temp{this, type, data};
    return _get_or_create_constant(temp);
}

Constant *Module::create_constant_zero(const Type *type) noexcept {
    Constant temp{this, type, Constant::ctor_tag_zero{}};
    return _get_or_create_constant(temp);
}

Constant *Module::create_constant_one(const Type *type) noexcept {
    Constant temp{this, type, Constant::ctor_tag_one{}};
    return _get_or_create_constant(temp);
}

bool Module::remove_constant_if_unused(Constant *constant) noexcept {
    if (constant == nullptr || constant->parent_module() != this ||
        !constant->is_linked() || !constant->use_list().empty()) {
        return false;
    }
    auto bucket = _hash_to_constants.find(constant->hash());
    LUISA_ASSERT(bucket != _hash_to_constants.end(),
                 "Interned constant is missing from its hash bucket.");
    auto &constants = bucket->second;
    auto iter = std::find(constants.begin(), constants.end(), constant);
    LUISA_ASSERT(iter != constants.end(),
                 "Interned constant is missing from its hash bucket.");
    constants.erase(iter);
    if (constants.empty()) { _hash_to_constants.erase(bucket); }
    constant->remove_self();
    return true;
}

Undefined *Module::create_undefined(const Type *type) noexcept {
    auto [iter, success] = _type_to_undefined.try_emplace(type, nullptr);
    if (success) {
        auto undef = luisa::make_managed<Undefined>(this, type);
        iter->second = _undefined_list.push_back(std::move(undef));
    }
    return iter->second;
}

SpecialRegister *Module::create_special_register(DerivedSpecialRegisterTag tag) noexcept {
    auto [iter, success] = _tag_to_special_register.try_emplace(tag, nullptr);
    if (success) {
        auto sreg = [tag, this]() noexcept -> ManagedPtr<SpecialRegister> {
            switch (tag) {
                case DerivedSpecialRegisterTag::THREAD_ID: return luisa::make_managed<SPR_ThreadID>(this);
                case DerivedSpecialRegisterTag::BLOCK_ID: return luisa::make_managed<SPR_BlockID>(this);
                case DerivedSpecialRegisterTag::WARP_LANE_ID: return luisa::make_managed<SPR_WarpLaneID>(this);
                case DerivedSpecialRegisterTag::DISPATCH_ID: return luisa::make_managed<SPR_DispatchID>(this);
                case DerivedSpecialRegisterTag::KERNEL_ID: return luisa::make_managed<SPR_KernelID>(this);
                case DerivedSpecialRegisterTag::RASTER_OBJECT_ID: return luisa::make_managed<SPR_ObjectID>(this);
                case DerivedSpecialRegisterTag::RASTER_BARYCENTRICS: return luisa::make_managed<SPR_Barycentrics>(this);
                case DerivedSpecialRegisterTag::BLOCK_SIZE: return luisa::make_managed<SPR_BlockSize>(this);
                case DerivedSpecialRegisterTag::WARP_SIZE: return luisa::make_managed<SPR_WarpSize>(this);
                case DerivedSpecialRegisterTag::DISPATCH_SIZE: return luisa::make_managed<SPR_DispatchSize>(this);
                case DerivedSpecialRegisterTag::RASTER_FRONT_FACING: return luisa::make_managed<SPR_FrontFacing>(this);
                default: break;
            }
            LUISA_ERROR_WITH_LOCATION("Unsupported special register tag.");
        }();
        iter->second = _special_register_list.push_back(std::move(sreg));
    }
    return iter->second;
}

SPR_ThreadID *Module::create_thread_id() noexcept {
    auto sreg = create_special_register(DerivedSpecialRegisterTag::THREAD_ID);
    LUISA_DEBUG_ASSERT(sreg->isa<SPR_ThreadID>(), "Invalid special register type.");
    return static_cast<SPR_ThreadID *>(sreg);
}

SPR_BlockID *Module::create_block_id() noexcept {
    auto sreg = create_special_register(DerivedSpecialRegisterTag::BLOCK_ID);
    LUISA_DEBUG_ASSERT(sreg->isa<SPR_BlockID>(), "Invalid special register type.");
    return static_cast<SPR_BlockID *>(sreg);
}

SPR_WarpLaneID *Module::create_warp_lane_id() noexcept {
    auto sreg = create_special_register(DerivedSpecialRegisterTag::WARP_LANE_ID);
    LUISA_DEBUG_ASSERT(sreg->isa<SPR_WarpLaneID>(), "Invalid special register type.");
    return static_cast<SPR_WarpLaneID *>(sreg);
}

SPR_DispatchID *Module::create_dispatch_id() noexcept {
    auto sreg = create_special_register(DerivedSpecialRegisterTag::DISPATCH_ID);
    LUISA_DEBUG_ASSERT(sreg->isa<SPR_DispatchID>(), "Invalid special register type.");
    return static_cast<SPR_DispatchID *>(sreg);
}

SPR_KernelID *Module::create_kernel_id() noexcept {
    auto sreg = create_special_register(DerivedSpecialRegisterTag::KERNEL_ID);
    LUISA_DEBUG_ASSERT(sreg->isa<SPR_KernelID>(), "Invalid special register type.");
    return static_cast<SPR_KernelID *>(sreg);
}

SPR_ObjectID *Module::create_object_id() noexcept {
    auto sreg = create_special_register(DerivedSpecialRegisterTag::RASTER_OBJECT_ID);
    LUISA_DEBUG_ASSERT(sreg->isa<SPR_ObjectID>(), "Invalid special register type.");
    return static_cast<SPR_ObjectID *>(sreg);
}

SPR_Barycentrics *Module::create_bary_centrics() noexcept {
    auto sreg = create_special_register(DerivedSpecialRegisterTag::RASTER_BARYCENTRICS);
    LUISA_DEBUG_ASSERT(sreg->isa<SPR_Barycentrics>(), "Invalid special register type.");
    return static_cast<SPR_Barycentrics *>(sreg);
}

SPR_FrontFacing *Module::create_front_facing() noexcept {
    auto sreg = create_special_register(DerivedSpecialRegisterTag::RASTER_FRONT_FACING);
    LUISA_DEBUG_ASSERT(sreg->isa<SPR_FrontFacing>(), "Invalid special register type.");
    return static_cast<SPR_FrontFacing *>(sreg);
}

SPR_BlockSize *Module::create_block_size() noexcept {
    auto sreg = create_special_register(DerivedSpecialRegisterTag::BLOCK_SIZE);
    LUISA_DEBUG_ASSERT(sreg->isa<SPR_BlockSize>(), "Invalid special register type.");
    return static_cast<SPR_BlockSize *>(sreg);
}

SPR_WarpSize *Module::create_warp_size() noexcept {
    auto sreg = create_special_register(DerivedSpecialRegisterTag::WARP_SIZE);
    LUISA_DEBUG_ASSERT(sreg->isa<SPR_WarpSize>(), "Invalid special register type.");
    return static_cast<SPR_WarpSize *>(sreg);
}

SPR_DispatchSize *Module::create_dispatch_size() noexcept {
    auto sreg = create_special_register(DerivedSpecialRegisterTag::DISPATCH_SIZE);
    LUISA_DEBUG_ASSERT(sreg->isa<SPR_DispatchSize>(), "Invalid special register type.");
    return static_cast<SPR_DispatchSize *>(sreg);
}

}// namespace luisa::compute::xir
