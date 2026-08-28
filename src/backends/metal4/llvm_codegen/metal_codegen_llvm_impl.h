#pragma once

#include "metal_codegen_llvm.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <utility>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include <luisa/ast/type.h>
#include <luisa/core/logging.h>
#include <luisa/core/mathematics.h>
#include <luisa/dsl/raster/raster_interpolation.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/metadata/curve_basis.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/dom_tree.h>

namespace luisa::compute::metal::detail {

static constexpr auto air_address_space_generic = 0u;
static constexpr auto air_address_space_device = 1u;
static constexpr auto air_address_space_constant = 2u;
static constexpr auto air_address_space_threadgroup = 3u;
static constexpr auto air_texture_access_read = 1u;
static constexpr auto air_texture_access_write = 2u;
static constexpr auto air_texture_access_read_write = 3u;
static constexpr auto air_texture_access_sample = 4u;
static constexpr auto kernel_argument_alignment = 16u;
static constexpr auto indirect_dispatch_buffer_type_name =
    luisa::string_view{"LC_IndirectDispatchBuffer"};
static constexpr auto indirect_dispatch_buffer_air_type_name =
    luisa::string_view{"LCIndirectDispatchBuffer"};
static constexpr auto accel_air_type_name = luisa::string_view{"LCAccel"};
static constexpr auto accel_instance_air_type_name = luisa::string_view{"LCInstance"};
static constexpr auto accel_handle_air_type_name =
    luisa::string_view{"acceleration_structure<instancing>"};
static constexpr auto ray_query_all_type_name =
    luisa::string_view{"LC_RayQueryAll"};
static constexpr auto ray_query_any_type_name =
    luisa::string_view{"LC_RayQueryAny"};
static constexpr auto ray_query_triangle_intrinsic_suffix =
    luisa::string_view{".instancing.triangle_data"};
static constexpr auto ray_query_curve_intrinsic_suffix =
    luisa::string_view{".instancing.triangle_data.curve_data"};
static constexpr auto ray_query_triangle_extended_intrinsic_suffix =
    luisa::string_view{".instancing.triangle_data.extended_limits"};
static constexpr auto ray_query_curve_extended_intrinsic_suffix =
    luisa::string_view{
        ".instancing.triangle_data.curve_data.extended_limits"};
static constexpr auto ray_trace_motion_intrinsic_suffix =
    luisa::string_view{
        ".instancing.triangle_data.primitive_motion.instance_motion"};
static constexpr auto ray_trace_curve_motion_intrinsic_suffix =
    luisa::string_view{
        ".instancing.triangle_data.curve_data.primitive_motion.instance_motion"};
static constexpr auto ray_trace_motion_extended_intrinsic_suffix =
    luisa::string_view{
        ".instancing.triangle_data.primitive_motion.instance_motion.extended_limits"};
static constexpr auto ray_trace_curve_motion_extended_intrinsic_suffix =
    luisa::string_view{
        ".instancing.triangle_data.curve_data.primitive_motion.instance_motion.extended_limits"};
static constexpr auto accel_instance_transform_field = 0u;
static constexpr auto accel_instance_options_field = 1u;
static constexpr auto accel_instance_mask_field = 2u;
static constexpr auto accel_instance_user_id_field = 3u;
static constexpr auto accel_instance_mesh_index_field = 4u;
static constexpr auto accel_instance_resource_id_field = 5u;
static constexpr auto shader_log_subsystem =
    luisa::string_view{"org.luisa.compute"};
static constexpr auto shader_log_category =
    luisa::string_view{"shader"};
static constexpr auto shader_log_bool_prefix =
    luisa::string_view{"__luisa_metal_bool_"};
static constexpr auto shader_log_bool_suffix =
    luisa::string_view{"__"};
static constexpr auto air_data_layout =
    "e-p:64:64:64"
    "-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"
    "-f32:32:32-f64:64:64"
    "-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
    "-n8:16:32";

[[nodiscard]] inline bool is_indirect_dispatch_buffer_type(const Type *type) noexcept {
    return type != nullptr && type->is_custom() &&
           type->description() == indirect_dispatch_buffer_type_name;
}

[[nodiscard]] inline bool is_ray_query_type(const Type *type) noexcept {
    if (type == nullptr || !type->is_custom()) { return false; }
    auto description = type->description();
    return description == ray_query_all_type_name ||
           description == ray_query_any_type_name;
}

[[nodiscard]] inline bool is_ray_query_any_type(const Type *type) noexcept {
    return is_ray_query_type(type) &&
           type->description() == ray_query_any_type_name;
}

struct AIRRayTracingConfig {
    bool curves{false};
    bool motion{false};
    bool extended_limits{false};
    uint32_t curve_basis{0u};
    uint32_t curve_control_point_count{0u};

    [[nodiscard]] constexpr uint32_t geometry_type(
        bool procedural = false) const noexcept {
        return 1u | (procedural ? 2u : 0u) |
               (curves ? 4u : 0u);
    }

    [[nodiscard]] constexpr luisa::string_view intrinsic_suffix() const noexcept {
        if (motion) {
            if (curves) {
                return extended_limits ?
                           ray_trace_curve_motion_extended_intrinsic_suffix :
                           ray_trace_curve_motion_intrinsic_suffix;
            }
            return extended_limits ?
                       ray_trace_motion_extended_intrinsic_suffix :
                       ray_trace_motion_intrinsic_suffix;
        }
        if (curves) {
            return extended_limits ?
                       ray_query_curve_extended_intrinsic_suffix :
                       ray_query_curve_intrinsic_suffix;
        }
        return extended_limits ?
                   ray_query_triangle_extended_intrinsic_suffix :
                   ray_query_triangle_intrinsic_suffix;
    }
};

enum class AIRRasterDepthMode : uint8_t {
    NONE,
    ANY,
    GREATER_EQUAL,
    LESS_EQUAL,
};

[[nodiscard]] constexpr AIRRasterDepthMode air_raster_depth_mode(
    xir::ThreadGroupOp op) noexcept {
    switch (op) {
        case xir::ThreadGroupOp::RASTER_SET_Z_DEPTH:
            return AIRRasterDepthMode::ANY;
        case xir::ThreadGroupOp::RASTER_SET_Z_DEPTH_GREATER_EQUAL:
            return AIRRasterDepthMode::GREATER_EQUAL;
        case xir::ThreadGroupOp::RASTER_SET_Z_DEPTH_LESS_EQUAL:
            return AIRRasterDepthMode::LESS_EQUAL;
        default: return AIRRasterDepthMode::NONE;
    }
}

[[nodiscard]] bool resolve_raster_interpolation(
    const Type *payload_type,
    size_t member_index,
    RasterInterpolation &interpolation,
    luisa::string &reason) noexcept;

[[nodiscard]] bool validate_raster_interpolation(
    const Type *payload_type,
    luisa::string &reason) noexcept;

[[nodiscard]] constexpr bool is_direct_texture_sample(xir::ResourceQueryOp op) noexcept {
    switch (op) {
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE: [[fallthrough]];
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD: [[fallthrough]];
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE: [[fallthrough]];
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD: [[fallthrough]];
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL: return true;
        default: return false;
    }
}

[[nodiscard]] constexpr bool is_bindless_texture_sample(xir::ResourceQueryOp op) noexcept {
    switch (op) {
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER: return true;
        default: return false;
    }
}

[[nodiscard]] constexpr size_t bindless_texture_sample_operand_count(
    xir::ResourceQueryOp op) noexcept {
    switch (op) {
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE: return 3u;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: return 4u;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: return 5u;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL: return 6u;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER: return 5u;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER: return 6u;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER: return 7u;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER: return 8u;
        default: return 0u;
    }
}

[[nodiscard]] static std::string version_string(MetalAIRVersion version) noexcept {
    return std::to_string(version.major) + "." +
           std::to_string(version.minor) + "." +
           std::to_string(version.patch);
}

[[nodiscard]] static std::string air_target_triple(const MetalCodegenLLVMConfig &config) noexcept {
    auto architecture = "air64_v" + std::to_string(config.air_version.major) +
                        std::to_string(config.air_version.minor);
    auto operating_system = config.platform == MetalAIRPlatform::MACOS ?
                                "macosx" :
                                "ios";
    return architecture + "-apple-" + operating_system +
           version_string(config.platform_version);
}

[[nodiscard]] static llvm::Metadata *md_i32(llvm::LLVMContext &context, uint32_t value) noexcept {
    return llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), value));
}

[[nodiscard]] static llvm::Metadata *md_string(llvm::LLVMContext &context, luisa::string_view value) noexcept {
    return llvm::MDString::get(context, llvm::StringRef{value.data(), value.size()});
}

class MetalCodegenLLVMImpl {

private:
    struct LLVMTypeInfo {
        llvm::Type *mem_type;
        llvm::Type *reg_type;
        luisa::vector<size_t> member_indices;
        luisa::vector<size_t> member_offsets;
    };

    struct KernelArguments {
        luisa::vector<size_t> offsets;
        luisa::vector<unsigned> member_indices;
        luisa::vector<size_t> sampled_texture_offsets;
        luisa::vector<unsigned> sampled_texture_member_indices;
        llvm::StructType *type;
        size_t size;
    };

    struct PrintFormat {
        luisa::string format;
        luisa::string native_format;
        const Type *record_type;
        llvm::Constant *native_format_pointer{nullptr};
    };

    struct RasterVertexInput {
        llvm::Type *type;
        luisa::string air_type_name;
        uint32_t dimension;
        bool signed_integer;
    };

    struct FunctionContext {
        struct RayQueryAllocation {
            llvm::Value *value;
            AIRRayTracingConfig config;
        };

        llvm::Function *function;
        llvm::BasicBlock *alloca_block;
        llvm::BasicBlock *entry_block;
        llvm::Value *dispatch_size{nullptr};
        llvm::Value *kernel_id{nullptr};
        llvm::Value *thread_id{nullptr};
        llvm::Value *block_id{nullptr};
        llvm::Value *dispatch_id{nullptr};
        llvm::Value *block_size{nullptr};
        llvm::Value *warp_size{nullptr};
        llvm::Value *warp_lane_id{nullptr};
        llvm::Value *raster_object_id{nullptr};
        llvm::Value *raster_barycentrics{nullptr};
        llvm::Value *raster_front_facing{nullptr};
        llvm::Value *raster_base_instance{nullptr};
        llvm::Value *raster_depth{nullptr};
        llvm::DenseMap<const xir::Value *, llvm::Value *> values;
        llvm::DenseMap<const xir::Value *, llvm::Value *> sampled_textures;
        llvm::DenseMap<const xir::BasicBlock *, llvm::BasicBlock *> block_exits;
        llvm::DenseMap<llvm::Value *, llvm::Value *> last_query_proceed;
        luisa::vector<RayQueryAllocation> ray_query_allocations;
        luisa::vector<const xir::PhiInst *> pending_phi_nodes;

        explicit FunctionContext(llvm::Function *f) noexcept
            : function{f},
              alloca_block{llvm::BasicBlock::Create(f->getContext(), "alloca", f)},
              entry_block{llvm::BasicBlock::Create(f->getContext(), "entry", f)} {
            llvm::IRBuilder<> builder{alloca_block};
            builder.CreateBr(entry_block);
        }

        template<typename T = llvm::Value>
            requires std::derived_from<T, llvm::Value>
        [[nodiscard]] T *value(const xir::Value *xir_value) const noexcept {
            auto iter = values.find(xir_value);
            LUISA_ASSERT(iter != values.end() && llvm::isa<T>(iter->second),
                         "Metal LLVM codegen could not resolve XIR value.");
            return llvm::cast<T>(iter->second);
        }

        [[nodiscard]] llvm::BasicBlock *block_exit(
            const xir::BasicBlock *block) const noexcept {
            auto iter = block_exits.find(block);
            LUISA_ASSERT(iter != block_exits.end(),
                         "Metal LLVM codegen could not resolve XIR block exit.");
            return iter->second;
        }

        [[nodiscard]] llvm::Value *sampled_texture(
            const xir::Value *texture) const noexcept {
            if (auto iter = sampled_textures.find(texture);
                iter != sampled_textures.end()) {
                return iter->second;
            }
            return value(texture);
        }
    };

    using IB = llvm::IRBuilder<>;

private:
    MetalCodegenLLVMConfig _config;
    MetalCodegenLLVMResult _result;
    llvm::LLVMContext &_context;
    llvm::Module &_module;
    llvm::DataLayout _data_layout;
    llvm::DenseMap<const Type *, std::unique_ptr<LLVMTypeInfo>> _types;
    llvm::DenseMap<const xir::Function *, llvm::Function *> _functions;
    llvm::DenseMap<const xir::Constant *, llvm::Constant *> _constants;
    KernelArguments _root_argument_layout_cache{};
    bool _root_argument_layout_initialized{false};
    luisa::vector<const xir::Argument *> _root_arguments;
    llvm::DenseMap<const Type *, llvm::StructType *> _buffer_types;
    llvm::StructType *_indirect_dispatch_buffer_type{nullptr};
    llvm::StructType *_indirect_dispatch_slot_type{nullptr};
    llvm::StructType *_bindless_item_type{nullptr};
    llvm::StructType *_bindless_array_type{nullptr};
    std::array<llvm::StructType *, 4u> _air_texture_handle_types{};
    std::array<llvm::StructType *, 4u> _air_texture_wrapper_types{};
    llvm::StructType *_air_sampler_handle_type{nullptr};
    llvm::StructType *_air_sampler_wrapper_type{nullptr};
    llvm::StructType *_air_accel_handle_type{nullptr};
    llvm::StructType *_air_accel_wrapper_type{nullptr};
    llvm::StructType *_accel_instance_type{nullptr};
    llvm::StructType *_accel_type{nullptr};
    llvm::StructType *_air_intersection_function_table_type{nullptr};
    llvm::StructType *_air_intersection_result_type{nullptr};
    llvm::StructType *_air_curve_intersection_result_type{nullptr};
    llvm::StructType *_air_intersection_query_type{nullptr};
    llvm::Function *_shader_log_helper{nullptr};
    const xir::KernelFunction *_kernel{nullptr};
    const xir::RasterStageFunction *_raster_stage{nullptr};
    AIRRasterDepthMode _raster_depth_mode{AIRRasterDepthMode::NONE};
    llvm::GlobalVariable *_sampler_table{nullptr};
    luisa::vector<PrintFormat> _print_formats;
    llvm::DenseMap<const xir::PrintInst *, uint32_t> _print_tokens;

public:
    explicit MetalCodegenLLVMImpl(MetalCodegenLLVMConfig config) noexcept
        : _config{std::move(config)},
          _result{},
          _context{*(_result.context = std::make_unique<llvm::LLVMContext>())},
          _module{*(_result.module = std::make_unique<llvm::Module>("luisa.metal.air", _context))},
          _data_layout{air_data_layout} {
        _module.setDataLayout(_data_layout);
        _module.setTargetTriple(llvm::Triple{air_target_triple(_config)});
        _module.setSourceFileName(std::string_view{_config.source_file});
    }

    [[nodiscard]] MetalCodegenLLVMResult generate(const xir::Module &xir_module) noexcept;

private:
    [[noreturn]] static void _unsupported_instruction(const xir::Instruction *inst) noexcept;
    [[noreturn]] static void _unsupported_type(const Type *type) noexcept;
    [[nodiscard]] static size_t _type_alignment(const Type *type) noexcept;
    [[nodiscard]] const LLVMTypeInfo *_type(const Type *type) noexcept;
    [[nodiscard]] llvm::StructType *_buffer(const Type *element) noexcept;
    [[nodiscard]] llvm::StructType *_indirect_dispatch_buffer() noexcept;
    [[nodiscard]] llvm::StructType *_indirect_dispatch_slot() noexcept;
    [[nodiscard]] llvm::StructType *_bindless_item() noexcept;
    [[nodiscard]] llvm::StructType *_bindless_array() noexcept;
    [[nodiscard]] llvm::StructType *_air_texture_handle(unsigned dimension) noexcept;
    [[nodiscard]] llvm::StructType *_air_texture_wrapper(unsigned dimension) noexcept;
    [[nodiscard]] llvm::StructType *_air_sampler_handle() noexcept;
    [[nodiscard]] llvm::StructType *_air_accel_handle() noexcept;
    [[nodiscard]] llvm::StructType *_air_accel_wrapper() noexcept;
    [[nodiscard]] llvm::StructType *_accel_instance() noexcept;
    [[nodiscard]] llvm::StructType *_accel() noexcept;
    [[nodiscard]] llvm::StructType *_air_intersection_function_table() noexcept;
    [[nodiscard]] llvm::StructType *_air_intersection_result(
        bool curves = false) noexcept;
    [[nodiscard]] llvm::StructType *_air_intersection_query() noexcept;
    [[nodiscard]] llvm::Constant *_constant_string(
        luisa::string_view value, luisa::string_view name) noexcept;
    [[nodiscard]] llvm::Function *_shader_log() noexcept;
    void _append_shader_log_type(
        luisa::string &format, const Type *type) const noexcept;
    [[nodiscard]] luisa::string _shader_log_format(
        luisa::string_view format,
        luisa::span<const Type *const> arguments) const noexcept;
    void _append_shader_log_arguments(
        IB &builder, llvm::Value *value, const Type *type,
        llvm::SmallVectorImpl<llvm::Value *> &arguments,
        size_t &argument_size) noexcept;
    void _set_air_pointer_element_types(
        llvm::Function *function,
        llvm::ArrayRef<std::pair<unsigned, llvm::Type *>> arguments,
        llvm::Type *return_element = nullptr) noexcept;
    void _set_struct_pointer_element_type(
        llvm::StructType *structure, unsigned field,
        llvm::Type *element) noexcept;
    [[nodiscard]] const KernelArguments &_root_argument_layout() noexcept;
    [[nodiscard]] RasterVertexInput _raster_vertex_input(PixelFormat format) noexcept;
    [[nodiscard]] llvm::Value *_load_root_argument(
        IB &builder, llvm::Value *root, const xir::Argument *argument,
        size_t root_index, bool sampled_texture = false) noexcept;

    [[nodiscard]] llvm::Value *_reg_to_mem(IB &builder, llvm::Value *value, const Type *type) noexcept;
    [[nodiscard]] llvm::Value *_mem_to_reg(IB &builder, llvm::Value *value, const Type *type) noexcept;
    [[nodiscard]] llvm::Value *_load(IB &builder, llvm::Value *pointer, const Type *type, bool is_volatile = false) noexcept;
    void _store(IB &builder, llvm::Value *pointer, llvm::Value *value, const Type *type, bool is_volatile = false) noexcept;
    [[nodiscard]] llvm::Value *_temporary(const FunctionContext &function, llvm::Type *type, size_t alignment) noexcept;

    [[nodiscard]] llvm::Value *_literal(IB &builder, const Type *type, const void *data) noexcept;
    [[nodiscard]] llvm::Value *_constant(IB &builder, const xir::Constant *constant) noexcept;
    [[nodiscard]] llvm::Value *_value(IB &builder, const FunctionContext &function, const xir::Value *value) noexcept;
    [[nodiscard]] llvm::Value *_special_register(const FunctionContext &function, xir::DerivedSpecialRegisterTag tag) noexcept;

    void _set_float_control_attributes(llvm::Function *function) const noexcept;
    [[nodiscard]] llvm::Function *_function(const xir::Function *function) noexcept;
    [[nodiscard]] llvm::Function *_declare_kernel(const xir::KernelFunction *function) noexcept;
    [[nodiscard]] llvm::Function *_declare_raster_stage(const xir::RasterStageFunction *function) noexcept;
    [[nodiscard]] llvm::Function *_declare_callable(const xir::CallableFunction *function) noexcept;
    [[nodiscard]] llvm::Function *_declare_external(const xir::ExternalFunction *function) noexcept;
    [[nodiscard]] llvm::Function *_translate_kernel(const xir::KernelFunction *function) noexcept;
    [[nodiscard]] llvm::Function *_translate_raster_stage(const xir::RasterStageFunction *function) noexcept;
    [[nodiscard]] llvm::Function *_translate_callable(const xir::CallableFunction *function) noexcept;
    [[nodiscard]] llvm::BasicBlock *_translate_function(FunctionContext &context, const xir::FunctionDefinition *function) noexcept;
    void _bind_state_parameters(FunctionContext &context, llvm::Function::arg_iterator iterator) noexcept;
    void _append_state_arguments(const FunctionContext &context, llvm::SmallVectorImpl<llvm::Value *> &arguments) noexcept;
    void _emit_kernel_entry(const xir::KernelFunction *kernel, llvm::Function *implementation, bool indirect) noexcept;
    void _emit_raster_vertex_entry(const xir::RasterStageFunction *stage, llvm::Function *implementation) noexcept;
    void _emit_raster_fragment_entry(const xir::RasterStageFunction *stage, llvm::Function *implementation) noexcept;

    void _translate_instruction(IB &builder, FunctionContext &function, const xir::Instruction *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_arithmetic(IB &builder, FunctionContext &function, const xir::ArithmeticInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_cast(IB &builder, FunctionContext &function, const xir::CastInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_gep(IB &builder, FunctionContext &function, const xir::GEPInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_resource_query(IB &builder, FunctionContext &function, const xir::ResourceQueryInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_ray_query_object_read(IB &builder, FunctionContext &function, const xir::RayQueryObjectReadInst *inst) noexcept;
    void _translate_ray_query_object_write(IB &builder, FunctionContext &function, const xir::RayQueryObjectWriteInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_resource_read(IB &builder, FunctionContext &function, const xir::ResourceReadInst *inst) noexcept;
    void _translate_resource_write(IB &builder, FunctionContext &function, const xir::ResourceWriteInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_atomic(IB &builder, FunctionContext &function, const xir::AtomicInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_thread_group(IB &builder, FunctionContext &function, const xir::ThreadGroupInst *inst) noexcept;
    void _translate_print(IB &builder, FunctionContext &function, const xir::PrintInst *inst) noexcept;

    [[nodiscard]] std::pair<llvm::Value *, const Type *> _access_chain(
        IB &builder, FunctionContext &function, llvm::Value *pointer,
        const Type *type, luisa::span<const xir::Use *const> indices) noexcept;
    [[nodiscard]] llvm::Value *_buffer_pointer(IB &builder, llvm::Value *buffer, llvm::Value *index, size_t stride) noexcept;
    [[nodiscard]] llvm::Value *_bindless_slot(IB &builder, llvm::Value *array, llvm::Value *index) noexcept;
    [[nodiscard]] llvm::Value *_bindless_slot_field(IB &builder, llvm::Value *slot, unsigned field) noexcept;
    [[nodiscard]] llvm::Value *_bindless_buffer_size(IB &builder, llvm::Value *slot) noexcept;
    [[nodiscard]] llvm::Value *_bindless_texture(IB &builder, llvm::Value *slot, unsigned dimension) noexcept;
    [[nodiscard]] llvm::Value *_bindless_sampler_code(IB &builder, llvm::Value *slot, unsigned dimension) noexcept;
    [[nodiscard]] llvm::Value *_device_pointer_offset(IB &builder, llvm::Value *pointer, llvm::Value *offset, size_t stride) noexcept;
    [[nodiscard]] llvm::Value *_accel_instance_pointer(IB &builder, llvm::Value *accel, llvm::Value *index) noexcept;
    [[nodiscard]] AIRRayTracingConfig _air_ray_tracing_config(
        const xir::Value *value, bool motion = false) const noexcept;
    [[nodiscard]] llvm::Value *_air_trace(
        IB &builder, llvm::Value *accel, llvm::Value *ray,
        llvm::Value *mask, llvm::Value *time,
        AIRRayTracingConfig config, bool accept_any) noexcept;
    [[nodiscard]] llvm::CallInst *_air_ray_query_call(
        IB &builder, luisa::string_view operation,
        llvm::Type *return_type,
        llvm::ArrayRef<llvm::Value *> arguments,
        AIRRayTracingConfig config,
        bool read_only = false,
        llvm::ArrayRef<std::pair<unsigned, llvm::Type *>> extra_pointer_types = {}) noexcept;
    void _deallocate_ray_queries(IB &builder, const FunctionContext &function) noexcept;

    [[nodiscard]] llvm::Value *_static_cast(IB &builder, llvm::Value *value, const Type *source, const Type *target) noexcept;
    [[nodiscard]] llvm::Value *_bitwise_cast(IB &builder, FunctionContext &function, llvm::Value *value, const Type *source, const Type *target) noexcept;
    [[nodiscard]] llvm::Value *_air_unary(IB &builder, llvm::StringRef name, llvm::Value *value) noexcept;
    [[nodiscard]] llvm::Value *_air_binary(IB &builder, llvm::StringRef name, llvm::Value *lhs, llvm::Value *rhs) noexcept;
    [[nodiscard]] llvm::Value *_air_ternary(IB &builder, llvm::StringRef name, llvm::Value *a, llvm::Value *b, llvm::Value *c) noexcept;
    [[nodiscard]] llvm::Value *_air_scalar_call(IB &builder, llvm::StringRef name, llvm::ArrayRef<llvm::Value *> arguments) noexcept;
    void _air_atomic_fence(IB &builder, uint32_t memory_flags) noexcept;
    [[nodiscard]] llvm::Value *_air_sampler(IB &builder, llvm::Value *filter, llvm::Value *address) noexcept;
    [[nodiscard]] llvm::Value *_air_sampler_code(IB &builder, llvm::Value *code) noexcept;
    [[nodiscard]] llvm::Value *_air_integer_call(IB &builder, llvm::StringRef name, llvm::Value *value, bool zero_is_undefined) noexcept;
    [[nodiscard]] llvm::Value *_air_simd_call(IB &builder, llvm::StringRef name, llvm::Value *value,
                                              bool signed_integer, llvm::ArrayRef<llvm::Value *> extra_arguments = {}) noexcept;

    [[nodiscard]] luisa::string _air_type_name(const Type *type) const noexcept;
    [[nodiscard]] luisa::string _air_texture_type_name(const Type *type, uint32_t access) const noexcept;
    [[nodiscard]] uint32_t _texture_access(const xir::Value *texture) const noexcept;
    [[nodiscard]] bool _texture_needs_sampled_split(
        const xir::Value *texture) const noexcept;
    [[nodiscard]] llvm::MDNode *_air_struct_type_info(const Type *type) noexcept;
    [[nodiscard]] size_t _air_indirect_location_count(const Type *type) const noexcept;
    [[nodiscard]] llvm::MDNode *_air_indirect_struct_type_info(const Type *type) noexcept;
    void _add_kernel_metadata(llvm::Function *function, size_t argument_struct_size, bool indirect) noexcept;
    [[nodiscard]] llvm::MDNode *_root_argument_metadata(
        size_t argument_struct_size, uint32_t argument_index) noexcept;
    void _add_raster_vertex_metadata(llvm::Function *function, llvm::ArrayRef<llvm::Type *> outputs) noexcept;
    void _add_raster_fragment_metadata(llvm::Function *function, llvm::ArrayRef<llvm::Type *> outputs) noexcept;
    void _add_module_metadata() noexcept;
    void _collect_print_formats(const xir::Module &module) noexcept;
    void _link_native_include() noexcept;
};

}// namespace luisa::compute::metal::detail
