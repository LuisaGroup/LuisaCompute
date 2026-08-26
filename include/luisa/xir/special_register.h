#pragma once

#include <luisa/xir/value.h>

namespace luisa::compute::xir {

enum struct DerivedSpecialRegisterTag {
    THREAD_ID,
    BLOCK_ID,
    WARP_LANE_ID,
    DISPATCH_ID,
    KERNEL_ID,
    RASTER_OBJECT_ID,
    RASTER_BARYCENTRICS,
    BLOCK_SIZE,
    WARP_SIZE,
    DISPATCH_SIZE,
};

[[nodiscard]] constexpr luisa::string_view to_string(DerivedSpecialRegisterTag tag) noexcept {
    using namespace std::string_view_literals;
    switch (tag) {
        case DerivedSpecialRegisterTag::THREAD_ID: return "thread_id"sv;
        case DerivedSpecialRegisterTag::BLOCK_ID: return "block_id"sv;
        case DerivedSpecialRegisterTag::WARP_LANE_ID: return "warp_lane_id"sv;
        case DerivedSpecialRegisterTag::DISPATCH_ID: return "dispatch_id"sv;
        case DerivedSpecialRegisterTag::KERNEL_ID: return "kernel_id"sv;
        case DerivedSpecialRegisterTag::RASTER_OBJECT_ID: return "object_id"sv;
        case DerivedSpecialRegisterTag::RASTER_BARYCENTRICS: return "barycentrics"sv;
        case DerivedSpecialRegisterTag::BLOCK_SIZE: return "block_size"sv;
        case DerivedSpecialRegisterTag::WARP_SIZE: return "warp_size"sv;
        case DerivedSpecialRegisterTag::DISPATCH_SIZE: return "dispatch_size"sv;
    }
    return "unknown"sv;
}

class LUISA_XIR_API SpecialRegister : public DerivedGlobalValue<SpecialRegister, DerivedValueTag::SPECIAL_REGISTER> {
public:
    SpecialRegister(Module *module, const Type *type) noexcept : Super{module, type} {}
    [[nodiscard]] virtual DerivedSpecialRegisterTag derived_special_register_tag() const noexcept = 0;
    LUISA_XIR_DEFINED_ISA_METHOD(SpecialRegister, special_register)
};

class LUISA_XIR_API SentinelSpecialRegister final : public SpecialRegister {
public:
    explicit SentinelSpecialRegister(Module *module) noexcept;
    [[nodiscard]] DerivedSpecialRegisterTag derived_special_register_tag() const noexcept override;
};

using SpecialRegisterList = ManagedIntrusiveList<SpecialRegister, SentinelSpecialRegister>;

namespace detail {

[[nodiscard]] LUISA_XIR_API const Type *special_register_type_uint() noexcept;
[[nodiscard]] LUISA_XIR_API const Type *special_register_type_uint3() noexcept;
[[nodiscard]] LUISA_XIR_API const Type *special_register_type_float3() noexcept;

template<typename T>
[[nodiscard]] auto get_special_register_type() noexcept {
    if constexpr (std::is_same_v<T, uint32_t>) {
        return special_register_type_uint();
    } else if constexpr (std::is_same_v<T, uint3>) {
        return special_register_type_uint3();
    } else if constexpr (std::is_same_v<T, float3>) {
        return special_register_type_float3();
    } else {
        static_assert(always_false_v<T>, "Unsupported special register type.");
    }
}

}// namespace detail

template<typename T, DerivedSpecialRegisterTag Tag>
class DerivedSpecialRegister final : public SpecialRegister {
public:
    using derived_special_register_type = DerivedSpecialRegister;

    explicit DerivedSpecialRegister(Module *module) noexcept
        : SpecialRegister{module, detail::get_special_register_type<T>()} {}

    [[nodiscard]] static constexpr auto
    static_derived_special_register_tag() noexcept { return Tag; }

    [[nodiscard]] DerivedSpecialRegisterTag
    derived_special_register_tag() const noexcept override {
        return static_derived_special_register_tag();
    }
};

// special registers
// note that we add the `SPR` prefix to avoid potential name conflicts with macros
using SPR_ThreadID = DerivedSpecialRegister<uint3, DerivedSpecialRegisterTag::THREAD_ID>;
using SPR_BlockID = DerivedSpecialRegister<uint3, DerivedSpecialRegisterTag::BLOCK_ID>;
using SPR_WarpLaneID = DerivedSpecialRegister<uint32_t, DerivedSpecialRegisterTag::WARP_LANE_ID>;
using SPR_DispatchID = DerivedSpecialRegister<uint3, DerivedSpecialRegisterTag::DISPATCH_ID>;
using SPR_KernelID = DerivedSpecialRegister<uint32_t, DerivedSpecialRegisterTag::KERNEL_ID>;
using SPR_ObjectID = DerivedSpecialRegister<uint32_t, DerivedSpecialRegisterTag::RASTER_OBJECT_ID>;
using SPR_Barycentrics = DerivedSpecialRegister<float3, DerivedSpecialRegisterTag::RASTER_BARYCENTRICS>;
using SPR_BlockSize = DerivedSpecialRegister<uint3, DerivedSpecialRegisterTag::BLOCK_SIZE>;
using SPR_WarpSize = DerivedSpecialRegister<uint32_t, DerivedSpecialRegisterTag::WARP_SIZE>;
using SPR_DispatchSize = DerivedSpecialRegister<uint3, DerivedSpecialRegisterTag::DISPATCH_SIZE>;

}// namespace luisa::compute::xir
