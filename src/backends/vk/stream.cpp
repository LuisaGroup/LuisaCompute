#include "stream.h"
#include "device.h"
#include "replay_graph.h"
#include "submit_profile.h"
#include "compute_shader.h"
#include "bindless_array.h"
#include <luisa/core/logging.h>
#include <vulkan/vk_enum_string_helper.h>
#include "log.h"
#include <cstdlib>
#include "blas.h"
#include "cluster_blas.h"
#include "tlas.h"
#include "motion_instance.h"
#include "rt_shader.h"
#include "swapchain.h"
#include "sparse_buffer.h"
#include "sparse_binding_plan.h"
#include "sparse_heap.h"
#include "timeline_semaphore_plan.h"
#include "in_flight_depth.h"
#include "queue_submit_plan.h"
#include "device_feature_plan.h"
#include "command_buffer_sync.h"
#include "indirect_buffer.h"
#include "copy_command_plan.h"
#include "device_address_command_plan.h"
#include "indirect_copy_command_buffer.h"
#include "descriptor_interface_plan.h"
#include "queue_family_contract.h"
#include "resource_barrier_contract.h"
#include <luisa/runtime/swapchain.h>
#include <luisa/backends/ext/vk_custom_cmd.h>
#include <luisa/backends/ext/vk_cuda_interop.h>
#include <luisa/backends/ext/batch_copy_cmd.h>
#include <luisa/core/stl/memory.h>
#ifdef LUISA_VULKAN_ENABLE_CUDA_INTEROP
#include "vk_cuda_interop_ext.h"
#endif
#include "../common/argument_block_layout.h"
#include "../common/shader_print_formatter.h"
#include "raster_shader.h"
#include "native_shader_ext.h"
#include "native_raster_shader.h"
#include "../../common/env_flag.h"
#include <algorithm>
#include <bit>
#include <limits>
namespace lc::vk {

namespace {
// Phase 3B (opt C): converts a microsecond budget into a `LUISA_INTRIN_PAUSE` loop
// bound. A `PAUSE` retires in roughly 10 ns on this class of CPU, i.e. ~100 per
// microsecond; the value only has to be in the right order of magnitude because it
// decides nothing but how long a thread polls before it blocks.
[[nodiscard]] uint32_t pause_iterations_from_us(char const *env_name,
                                                uint64_t default_us) noexcept {
    return detail::parse_spin_pause_iterations(std::getenv(env_name),
                                               default_us);
}
}// namespace
struct PresentCommand {
    luisa::fixed_vector<VkFence, 1> submit_fences;
    luisa::fixed_vector<VkSemaphore, 1> submit_wait_semaphores;
    luisa::fixed_vector<VkSemaphore, 1> signal_semaphores;
    luisa::fixed_vector<VkPipelineStageFlags, 1> wait_stages;
    luisa::fixed_vector<VkSemaphore, 1> present_wait_semaphores;
    luisa::fixed_vector<uint, 1> image_indices;
};
template<typename Visitor>
void decode_cmd(vstd::span<const Argument> args, Visitor &&visitor) {
    using Tag = Argument::Tag;
    for (auto &&i : args) {
        switch (i.tag) {
            case Tag::BUFFER: {
                visitor(i.buffer);
            } break;
            case Tag::TEXTURE: {
                visitor(i.texture);
            } break;
            case Tag::UNIFORM: {
                visitor(i.uniform);
            } break;
            case Tag::BINDLESS_ARRAY: {
                visitor(i.bindless_array);
            } break;
            case Tag::ACCEL: {
                visitor(i.accel);
            } break;
            default: {
                LUISA_ASSUME(false);
            } break;
        }
    }
}
ResourceBarrier::ResourceView get_resource_view(VKCustomCmd::ResourceHandle const &res) {
    return luisa::visit(
        [&]<typename T>(T const &t) -> ResourceBarrier::ResourceView {
            if constexpr (std::is_same_v<T, Argument::Buffer>) {
                auto buffer = reinterpret_cast<Buffer const *>(t.handle);
                return BufferView(buffer, t.offset, t.size);
            } else if constexpr (std::is_same_v<T, Argument::Texture>) {
                auto tex = reinterpret_cast<Texture const *>(t.handle);
                return TexView(tex, t.level);
            } else {
                LUISA_ERROR(
                    "Vulkan bindless state must be expanded through the "
                    "structured ResourceBarrier bindless APIs.");
            }
        },
        res);
}

void record_custom_resource_usage(
    ResourceBarrier *barrier,
    VKCustomCmd::ResourceUsage const &usage) {
    LUISA_ASSERT(barrier != nullptr,
                 "Vulkan custom-command preprocessing requires a resource barrier.");
    if (auto bindless = luisa::get_if<Argument::BindlessArray>(
            &usage.resource)) {
        LUISA_ASSERT(bindless->handle != 0u,
                     "Vulkan custom command contains a null bindless-array handle.");
        auto array = reinterpret_cast<BindlessArray const *>(bindless->handle);
        // The bindless declaration names the index object and reaches the
        // encoded descriptor members through the same declared native Vulkan
        // scope. Record both sides so native barriers match the reorder snapshot.
        barrier->record_bindless(
            array, usage.stage, usage.access, usage.texture_layout);
        return;
    }
    barrier->record(
        get_resource_view(usage.resource),
        usage.stage, usage.access, usage.texture_layout);
}

void set_config_resource_before_state(
    ResourceBarrier *barrier,
    VKCustomCmd::ResourceUsage const &usage) {
    LUISA_ASSERT(barrier != nullptr,
                 "Vulkan config before-state preprocessing requires a "
                 "resource barrier.");
    if (auto bindless = luisa::get_if<Argument::BindlessArray>(
            &usage.resource)) {
        LUISA_ASSERT(bindless->handle != 0u,
                     "Vulkan config before-state contains a null "
                     "bindless-array handle.");
        barrier->set_bindless_before_state(
            reinterpret_cast<BindlessArray const *>(bindless->handle),
            usage.stage, usage.access, usage.texture_layout);
        return;
    }
    barrier->set_res(
        get_resource_view(usage.resource),
        usage.stage, usage.access, usage.texture_layout);
}

void set_config_resource_restore_state(
    ResourceBarrier *barrier,
    VKCustomCmd::ResourceUsage const &usage) {
    LUISA_ASSERT(barrier != nullptr,
                 "Vulkan config restore-state preprocessing requires a "
                 "resource barrier.");
    if (auto bindless = luisa::get_if<Argument::BindlessArray>(
            &usage.resource)) {
        LUISA_ASSERT(bindless->handle != 0u,
                     "Vulkan config restore-state contains a null "
                     "bindless-array handle.");
        barrier->set_bindless_restore_state(
            reinterpret_cast<BindlessArray const *>(bindless->handle),
            usage.stage, usage.access, usage.texture_layout);
        return;
    }
    barrier->set_restore_state(
        get_resource_view(usage.resource),
        usage.stage, usage.access, usage.texture_layout);
}

uint64_t ReorderFuncTable::canonical_buffer_handle(
    uint64_t handle) const noexcept {
    auto buffer = reinterpret_cast<Buffer const *>(handle);
    return luisa::bit_cast<uint64_t>(buffer->vk_buffer());
}

uint64_t ReorderFuncTable::canonical_texture_handle(
    uint64_t handle) const noexcept {
    auto texture = reinterpret_cast<Texture const *>(handle);
    return luisa::bit_cast<uint64_t>(texture->vk_image());
}

void ReorderFuncTable::traverse_bindless_resources(
    uint64_t bindless_handle,
    ReorderBindlessResourceVisitor visitor) const noexcept {
    auto bindless = reinterpret_cast<BindlessArray *>(bindless_handle);
    std::lock_guard lock{bindless->mtx};
    bindless->traverse_pending_resources(
        [&](uint64_t resource_handle) noexcept {
            auto resource = reinterpret_cast<Resource const *>(
                resource_handle);
            LUISA_ASSERT(
                resource != nullptr &&
                    (resource->tag() == Resource::Tag::kBuffer ||
                     resource->tag() == Resource::Tag::kTexture),
                "Vulkan bindless reorder snapshot contains an invalid resource.");
            visitor(
                resource_handle,
                resource->tag() == Resource::Tag::kBuffer);
        });
}
void ReorderFuncTable::update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::Modification> modifications) const noexcept {
    reinterpret_cast<BindlessArray *>(handle)->bind(modifications);
}
void ReorderFuncTable::update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::BufferModification> modifications) const noexcept {
    reinterpret_cast<BindlessArray *>(handle)->bind(modifications);
}
void ReorderFuncTable::update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::Texture2DModification> modifications) const noexcept {
    reinterpret_cast<BindlessArray *>(handle)->bind(modifications);
}
void ReorderFuncTable::update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::Texture3DModification> modifications) const noexcept {
    reinterpret_cast<BindlessArray *>(handle)->bind(modifications);
}
struct ResourceAccess {
    bool reads;
    bool writes;
};
[[nodiscard]] static ResourceAccess resource_access(Usage usage) noexcept {
    auto bits = luisa::to_underlying(usage);
    return {
        .reads = (bits & luisa::to_underlying(Usage::READ)) != 0u ||
                 usage == Usage::NONE,
        .writes = (bits & luisa::to_underlying(Usage::WRITE)) != 0u};
}

struct TextureDescriptorBindings {
    static constexpr auto invalid =
        std::numeric_limits<uint32_t>::max();
    uint32_t sampled{invalid};
    uint32_t storage{invalid};
};

[[nodiscard]] static TextureDescriptorBindings
consume_texture_descriptor_bindings(
    const Shader *shader,
    vstd::span<const hlsl::Property> bindings,
    uint32_t &descriptor_index, Usage usage,
    const char *phase) noexcept {
    auto descriptor_roles = detail::texture_descriptor_roles(usage);
    auto result = TextureDescriptorBindings{};
    auto consume = [&](hlsl::ShaderVariableType expected,
                       uint32_t &role_binding,
                       const char *role_name) noexcept {
        auto index = descriptor_index;
        auto *property = shader != nullptr ?
                             shader->local_descriptor_property(index) :
                             detail::find_local_descriptor_property(
                                 bindings, index);
        LUISA_ASSERT(
            property != nullptr,
            "Vulkan {} found no {} descriptor for texture argument at "
            "binding {}.",
            phase, role_name, index);
        LUISA_ASSERT(
            property->type == expected,
            "Vulkan {} expected a {} descriptor for texture argument at "
            "binding {}, but found property type {}.",
            phase, role_name, index,
            static_cast<uint32_t>(property->type));
        role_binding = index;
        ++descriptor_index;
    };
    if (descriptor_roles.sampled) {
        consume(hlsl::ShaderVariableType::SRVTextureHeap,
                result.sampled, "sampled-image");
    }
    if (descriptor_roles.storage) {
        consume(hlsl::ShaderVariableType::UAVTextureHeap,
                result.storage, "storage-image");
    }
    return result;
}

struct ValidatedIndirectDispatch {
    const Buffer *source;
    IndirectDispatchPlan plan;
};

[[nodiscard]] static ValidatedIndirectDispatch
validate_indirect_dispatch_source(
    const ShaderDispatchCommand *command) noexcept {
    LUISA_ASSERT(command != nullptr && command->is_indirect(),
                 "Vulkan indirect-dispatch validation requires an indirect command.");
    auto argument = command->indirect_dispatch();
    LUISA_ASSERT(argument.handle != invalid_resource_handle &&
                     argument.handle != 0u,
                 "Vulkan indirect dispatch has an invalid source buffer handle.");
    auto source = reinterpret_cast<const Buffer *>(argument.handle);
    LUISA_ASSERT(
        source->is_indirect_dispatch_buffer(),
        "Vulkan indirect dispatch source is not a backend-owned "
        "IndirectDispatchBuffer.");
    auto plan = plan_indirect_dispatch(
        source->indirect_dispatch_capacity(), argument.offset,
        argument.max_dispatch_size);
    LUISA_ASSERT(
        static_cast<bool>(plan),
        "Invalid Vulkan indirect-dispatch range: capacity {}, offset {}, "
        "maximum count {}, planner error {}.",
        source->indirect_dispatch_capacity(), argument.offset,
        argument.max_dispatch_size, static_cast<uint32_t>(plan.error));
    size_t expected_size = 0u;
    LUISA_ASSERT(
        IndirectDispatchLayout::try_total_size(
            source->indirect_dispatch_capacity(), expected_size) &&
            source->byte_size() == expected_size,
        "Vulkan indirect-dispatch buffer has an invalid physical layout: "
        "capacity {}, expected {} bytes, got {} bytes.",
        source->indirect_dispatch_capacity(), expected_size,
        source->byte_size());
    return {source, plan.plan};
}

static void validate_indirect_dispatch_target(
    const ShaderDispatchCommand *command, const Shader *shader,
    const ValidatedIndirectDispatch &indirect,
    bool require_initialized_source) noexcept {
    LUISA_ASSERT(
        command != nullptr && command->is_indirect() &&
            indirect.source != nullptr && indirect.plan.command_count != 0u,
        "Vulkan indirect target validation requires a nonempty source plan.");
    LUISA_ASSERT(shader != nullptr &&
                     shader->shader_tag() == Shader::ShaderTag::kComputeShader,
                 "Vulkan indirect dispatch is supported only for compute "
                 "pipelines (ray-query kernels are compute pipelines).");
    LUISA_ASSERT(
        shader->uses_indirect_dispatch(),
        "Vulkan indirect dispatch requires the native XIR-to-SPIR-V logical "
        "metadata ABI. This shader was compiled through an incompatible path.");
    auto argument = command->indirect_dispatch();
    if (require_initialized_source) {
        // This check belongs to command execution, not preprocessing. An
        // authoring dispatch earlier in the same command list claims and
        // initializes the header only when its descriptors are bound during
        // execution; preprocessing necessarily visits the later consumer
        // before that has happened.
        LUISA_ASSERT(
            indirect.source->indirect_header_initialization_claimed(),
            "Vulkan indirect dispatch source has never been submitted to a GPU "
            "authoring shader. Set its dispatch count or records before consuming it.");
    }
    auto saved_arguments = shader->saved_arguments();
    auto saved_argument_index = size_t{0u};
    auto reject_writable_source_alias = [&](auto args) noexcept {
        for (auto &&arg : args) {
            LUISA_ASSERT(
                saved_argument_index < saved_arguments.size(),
                "Vulkan indirect-dispatch argument table is shorter than "
                "the encoded shader arguments.");
            auto &&saved = saved_arguments[saved_argument_index++];
            if (arg.tag == Argument::Tag::BINDLESS_ARRAY &&
                resource_access(saved.var_usage).writes) {
                auto *bindless = reinterpret_cast<const BindlessArray *>(
                    arg.bindless_array.handle);
                if (bindless != nullptr &&
                    bindless->contains_buffer_alias(indirect.source)) {
                    LUISA_ERROR_WITH_LOCATION(
                        "Vulkan indirect-dispatch target binds its "
                        "GPU-authored source through a writable bindless "
                        "array. Logical metadata loads would race with "
                        "target writes; use a separate array/buffer.");
                }
            }
            if (arg.tag == Argument::Tag::BUFFER &&
                resource_access(saved.var_usage).writes) {
                auto *target_buffer = reinterpret_cast<const Buffer *>(
                    arg.buffer.handle);
                // Luisa handles are not an alias boundary: importing the same
                // native VkBuffer creates a distinct wrapper. Compare the
                // actual descriptor resource as well as the fast-path handle.
                if (arg.buffer.handle == argument.handle ||
                    (target_buffer != nullptr &&
                     target_buffer->vk_buffer() ==
                         indirect.source->vk_buffer())) {
                    LUISA_ERROR_WITH_LOCATION(
                        "Vulkan indirect-dispatch target aliases its "
                        "GPU-authored source through a writable shader "
                        "argument. Logical metadata loads would race with "
                        "target writes; use a separate destination buffer.");
                }
            }
        }
    };
    reject_writable_source_alias(shader->captured());
    reject_writable_source_alias(command->arguments());
    LUISA_ASSERT(
        saved_argument_index == saved_arguments.size(),
        "Vulkan indirect-dispatch argument table has {} unbound entries.",
        saved_arguments.size() - saved_argument_index);
}

static void ensure_indirect_header_initialized(
    CommandBuffer *command_buffer, const Buffer *buffer,
    ResourceBarrier::Usage next_usage) noexcept {
    LUISA_ASSERT(command_buffer != nullptr && buffer != nullptr &&
                     buffer->is_indirect_dispatch_buffer(),
                 "Vulkan indirect initialization requires an indirect buffer.");
    if (!buffer->claim_indirect_header_initialization()) { return; }
    auto whole_buffer = BufferView{buffer, 0u, buffer->byte_size()};
    command_buffer->resource_barrier->record(
        whole_buffer, ResourceBarrier::Usage::kCopyDest);
    command_buffer->resource_barrier->update_states(
        command_buffer->cmdbuffer());
    vkCmdFillBuffer(
        command_buffer->cmdbuffer(), buffer->vk_buffer(), 0u,
        IndirectDispatchLayout::header_size, 0u);
    command_buffer->resource_barrier->record(
        whole_buffer, next_usage);
    command_buffer->resource_barrier->update_states(
        command_buffer->cmdbuffer());
}

// `detail::PlannedArgumentBlock` (the per-dispatch argument block of one
// command) lives in argument_plan_scratch.h, because the `Stream` owns the table
// that keeps one plan per command of the current batch (plan P1.4).

class SavedArgumentCursor {
private:
    luisa::span<const SavedArgument> _arguments;
    size_t _index{};

    [[nodiscard]] const SavedArgument &_next(
        const char *runtime_kind) noexcept {
        LUISA_ASSERT(
            _index < _arguments.size(),
            "Vulkan dispatch contains more {} arguments than the shader ABI "
            "table (consumed {}, available {}).",
            runtime_kind, _index, _arguments.size());
        return _arguments[_index++];
    }

public:
    SavedArgumentCursor() noexcept = default;
    explicit SavedArgumentCursor(
        luisa::span<const SavedArgument> arguments) noexcept
        : _arguments{arguments} {}

    [[nodiscard]] const SavedArgument &next_buffer() noexcept {
        auto &argument = _next("buffer");
        LUISA_ASSERT(
            argument.tag == Type::Tag::BUFFER ||
                argument.tag == Type::Tag::CUSTOM,
            "Vulkan dispatch buffer argument {} does not match saved ABI tag {}.",
            _index - 1u, luisa::to_underlying(argument.tag));
        return argument;
    }

    [[nodiscard]] const SavedArgument &next_texture() noexcept {
        auto &argument = _next("texture");
        LUISA_ASSERT(
            argument.tag == Type::Tag::TEXTURE,
            "Vulkan dispatch texture argument {} does not match saved ABI tag {}.",
            _index - 1u, luisa::to_underlying(argument.tag));
        return argument;
    }

    [[nodiscard]] const SavedArgument &next_bindless_array() noexcept {
        auto &argument = _next("bindless-array");
        LUISA_ASSERT(
            argument.tag == Type::Tag::BINDLESS_ARRAY,
            "Vulkan dispatch bindless-array argument {} does not match saved ABI tag {}.",
            _index - 1u, luisa::to_underlying(argument.tag));
        return argument;
    }

    [[nodiscard]] const SavedArgument &next_accel() noexcept {
        auto &argument = _next("acceleration-structure");
        LUISA_ASSERT(
            argument.tag == Type::Tag::ACCEL,
            "Vulkan dispatch accel argument {} does not match saved ABI tag {}.",
            _index - 1u, luisa::to_underlying(argument.tag));
        return argument;
    }

    [[nodiscard]] const SavedArgument &next_uniform(
        size_t encoded_size) noexcept {
        auto &argument = _next("uniform");
        auto resource_tag =
            argument.tag == Type::Tag::BUFFER ||
            argument.tag == Type::Tag::TEXTURE ||
            argument.tag == Type::Tag::BINDLESS_ARRAY ||
            argument.tag == Type::Tag::ACCEL ||
            argument.tag == Type::Tag::CUSTOM;
        LUISA_ASSERT(
            !resource_tag && argument.struct_size == encoded_size,
            "Vulkan dispatch uniform argument {} has saved tag {} and size {}, "
            "but the command encodes {} bytes.",
            _index - 1u, luisa::to_underlying(argument.tag),
            argument.struct_size, encoded_size);
        return argument;
    }

    void finish(const char *phase) const noexcept {
        LUISA_ASSERT(
            _index == _arguments.size(),
            "Vulkan {} consumed {} saved shader arguments, but the ABI table "
            "contains {}.",
            phase, _index, _arguments.size());
    }
};

static void validate_saved_argument_count(
    const Shader *shader,
    const ShaderDispatchCommandBase *command) noexcept {
    auto captured_count = shader->captured().size();
    auto runtime_count = command->arguments().size();
    LUISA_ASSERT(
        captured_count <=
            std::numeric_limits<size_t>::max() - runtime_count,
        "Vulkan dispatch argument count overflowed.");
    auto encoded_count = captured_count + runtime_count;
    LUISA_ASSERT(
        encoded_count == shader->saved_arguments().size(),
        "Vulkan dispatch encodes {} captured plus {} runtime arguments, but "
        "the shader ABI table contains {} entries.",
        captured_count, runtime_count,
        shader->saved_arguments().size());
}

[[nodiscard]] static size_t argument_block_metadata_count(
    const Shader *shader) noexcept {
    return shader->buffer_metadata_count();
}

[[nodiscard]] static ArgumentBlockTrailerLayout argument_block_trailer_layout(
    const Shader *shader, size_t buffer_metadata_count) noexcept {
    return ArgumentBlockTrailerLayout{
        .metadata_count = buffer_metadata_count,
        .metadata_stride = sizeof(StorageBufferMetadata),
        .metadata_alignment = alignof(StorageBufferMetadata),
        .validation_count = shader->validation_count(),
        .validation_stride = sizeof(uint32_t),
        .validation_alignment = alignof(uint32_t),
        .word_alignment = sizeof(uint32_t)};
}

[[nodiscard]] static detail::PlannedArgumentBlock plan_argument_block(
    const Shader *shader, const ShaderDispatchCommandBase *command,
    size_t descriptor_range_limit) noexcept {
    // Counter-only: one call per command that needs an argument block, i.e. one
    // per recorded dispatch (F4). Before P1.4 the sizing pass and the
    // preprocessing pass each planned every dispatch, so this counter read
    // 2 x dispatches.
    submit_profile::record_argument_plan();
    detail::PlannedArgumentBlock plan{
        .layout = ArgumentBlockLayout{descriptor_range_limit},
        .buffer_metadata_count =
            argument_block_metadata_count(shader)};
    validate_saved_argument_count(shader, command);
    auto append_uniforms = [&](auto arguments) noexcept {
        for (auto &argument : arguments) {
            if (argument.tag != Argument::Tag::UNIFORM) { continue; }
            auto data = command->uniform(argument.uniform);
            size_t offset = 0u;
            if (!plan.layout.append(
                    data.size_bytes(), argument.uniform.alignment,
                    offset)) {
                return false;
            }
        }
        return true;
    };
    if (!append_uniforms(shader->captured()) ||
        !append_uniforms(command->arguments())) {
        return plan;
    }
    static_cast<void>(plan.layout.append_trailers(
        argument_block_trailer_layout(
            shader, plan.buffer_metadata_count),
        plan.trailer));
    return plan;
}

static void validate_argument_block_plan(
    const detail::PlannedArgumentBlock &plan) noexcept {
    LUISA_ASSERT(
        static_cast<bool>(plan.layout),
        "Vulkan per-dispatch argument block is invalid for the device's "
        "maxStorageBufferRange ({} bytes): {}.",
        plan.layout.limit(),
        argument_block_layout_status_name(plan.layout.status()));
}

// ---------------------------------------------------------------------------
// Software (fallback) acceleration structures in the command stream.
//
// A device that answers ray tracing with the fallback has no
// VkAccelerationStructureKHR at all: the acceleration-structure argument of a
// shader is bound to the fallback's shared acceleration buffer and to its
// instance buffer, i.e. to two ordinary storage buffers
// (src/backends/common/rtx/fallback_rtx_layout.h).  Everything such an
// argument contributes to the command stream is therefore the plain
// storage-buffer contract, and the fallback's own build commands are ordinary
// dispatches, spliced in by `CommandBuffer::execute`.
// ---------------------------------------------------------------------------

// A `uint4` is the element type of both fallback buffers, so it is also the
// stride a storage-buffer descriptor of one of them has to respect.
inline constexpr size_t fallback_accel_element_stride = 16u;

// The push-constant block of a compute shader starts with the fixed dispatch
// block the HLSL codegen always declares; a device that runs the software
// fallback appends one 32-bit word per fallback `accel` argument right after it
// (the tree's region base, `accelBase` in fallback_rtx_header.bytes), and
// `Device::_create_shader_hlsl` sizes the block accordingly.
inline constexpr uint32_t fallback_accel_base_push_offset =
    static_cast<uint32_t>(sizeof(IndirectDispatchPushConstants));
static_assert(sizeof(IndirectDispatchPushConstants) == 32u);

// The bindless heap of a software (fallback) acceleration structure plus the
// two buffers the fallback's storage keeps, resolved from
// `FallbackRtxDevice::binding()` at every use and never cached: the fallback
// storage grows by appending a larger buffer and retiring the old one, and the
// TLAS' heap is replaced when it outgrows its slot count (the library documents
// both).
//
// The heap is what the shader-visible ABI is built on: a heap entry is a view
// that starts at the region it names, so a traversal resolves a BLAS by its slot
// and indexes every region relatively - there is no absolute-offset descriptor
// and no backwards descriptor indexing any more
// (src/backends/common/rtx/fallback_rtx_layout.h).
//
//  * `heap` is the TLAS' bindless array.  Its slot table is the `accel`
//    argument's first descriptor, and its slots (the backend's global bindless
//    heap holds the buffers) are the regions the traversal resolves.
//  * `accel` / `accel_region_offset` are the *shared acceleration buffer* and the
//    byte offset of this tree's region inside it.  The GPU path never names them
//    (the heap does), but the barrier pass still has to put the buffer the heap
//    will read into its state, and the (debug) validators download it.
//  * `instances` is the instance buffer bound at this tree's slice, so element 0
//    is the tree's first instance record and `instance_u4 * instance` indexes it.
struct FallbackAccelBuffers {
    BindlessArray const *heap{};
    Buffer const *accel{};
    Buffer const *instances{};
    size_t instance_offset{};
    size_t accel_region_offset{};
    // The heap slot of the tree's own region (`heap_tlas_slot`), which is the
    // `accelBase` the generated shader is handed.
    uint32_t region_slot{};
};

[[nodiscard]] static FallbackAccelBuffers fallback_accel_buffers(
    Device *device, uint64_t handle, const char *phase) noexcept {
    auto binding = device->fallback_rtx()->binding(handle);
    LUISA_ASSERT(
        binding.valid(),
        "Vulkan {} was handed the software fallback acceleration structure "
        "of handle {}, which has not been built yet. Build the acceleration "
        "structure before dispatching a shader that traces through it.",
        phase, handle);
    auto *heap = reinterpret_cast<BindlessArray const *>(binding.accel_heap);
    auto *accel = reinterpret_cast<Buffer const *>(binding.accel_buffer);
    auto *instances = reinterpret_cast<Buffer const *>(binding.instance_buffer);
    LUISA_ASSERT(
        heap != nullptr && accel != nullptr && instances != nullptr &&
            binding.accel_offset_bytes <= accel->byte_size() &&
            binding.instance_offset_bytes <= instances->byte_size(),
        "Vulkan {} found an uninitialized software fallback acceleration "
        "structure for handle {}.",
        phase, handle);
    return FallbackAccelBuffers{
        heap, accel, instances, binding.instance_offset_bytes,
        binding.accel_offset_bytes, binding.accel_slot};
}

// The region's `accelBase`: the heap slot of the tree's own region.  It is the
// only thing that still distinguishes one fallback `accel` argument from
// another in the push-constant block.
[[nodiscard]] static uint32_t fallback_accel_region_base(
    const FallbackAccelBuffers &buffers) noexcept {
    return buffers.region_slot;
}

// The instance view spans the tree's instance records to the end of the
// instance buffer; the acceleration view spans the whole buffer (the tree's
// offsets are absolute, see above).
[[nodiscard]] static size_t fallback_accel_descriptor_range(
    const Buffer *buffer, size_t offset) noexcept {
    LUISA_ASSERT(offset <= buffer->byte_size(),
                 "Vulkan software fallback buffer offset {} exceeds the "
                 "buffer size {}.",
                 offset, buffer->byte_size());
    return buffer->byte_size() - offset;
}

[[nodiscard]] static void validate_fallback_accel_descriptor(
    Device *device, const Buffer *buffer, size_t offset,
    const char *name) noexcept {
    auto &limits = device->properties().limits;
    auto status = detail::validate_direct_storage_buffer_descriptor(
        offset, fallback_accel_descriptor_range(buffer, offset),
        buffer->byte_size(), fallback_accel_element_stride,
        std::max<VkDeviceSize>(1u, limits.minStorageBufferOffsetAlignment),
        limits.maxStorageBufferRange);
    LUISA_ASSERT(
        status == detail::DirectStorageBufferDescriptorStatus::SUCCESS,
        "The Vulkan software ray-tracing fallback cannot describe its {} "
        "buffer as a storage-buffer descriptor at offset {} of a {} byte "
        "buffer: {}. The fallback's offsets are multiples of {} bytes; this "
        "device requires descriptor offsets to be multiples of {}. Disable "
        "the fallback (VulkanDeviceConfigExt::use_fallback_rtx()) on this "
        "device.",
        name, offset, buffer->byte_size(),
        detail::direct_storage_buffer_descriptor_status_name(status),
        fallback_accel_element_stride,
        limits.minStorageBufferOffsetAlignment);
}

// The descriptor slots an acceleration-structure argument occupies.  The HLSL
// codegen declares the two (the acceleration view and the instance view) for
// every `accel` argument of a fallback shader, which is the two-descriptor ABI
// the native argument already uses; the roles are read from the compiled
// property list so the barrier pass and the bind pass consume exactly what the
// shader declares.
struct FallbackAccelRoles {
    static constexpr auto invalid = std::numeric_limits<uint32_t>::max();
    uint32_t accel{invalid};
    uint32_t instance{invalid};
    bool instance_writable{false};
};

// The HLSL codegen declares the fallback's two views as ordinary storage
// buffers (property.cpp): the acceleration view (`StructuredBuffer<uint4>`,
// the whole acceleration buffer, an absolute handle space) and the instance
// view (`StructuredBuffer`/`RWStructuredBuffer<uint4>`, bound at this tree's
// slice).  A write-only `accel` argument declares the instance view alone, and
// its type is then the RW one - which is what tells the two cases apart.
[[nodiscard]] static FallbackAccelRoles consume_fallback_accel_roles(
    const Shader *shader,
    vstd::span<const hlsl::Property> bindings,
    uint32_t &descriptor_index) noexcept {
    using hlsl::ShaderVariableType;
    FallbackAccelRoles roles;
    auto lookup = [&](uint32_t index) noexcept {
        return shader != nullptr ?
                   shader->local_descriptor_property(index) :
                   detail::find_local_descriptor_property(bindings, index);
    };
    auto *property = lookup(descriptor_index);
    if (property != nullptr &&
        property->type == ShaderVariableType::StructuredBuffer) {
        roles.accel = descriptor_index++;
        property = lookup(descriptor_index);
    }
    if (property != nullptr &&
        (property->type == ShaderVariableType::StructuredBuffer ||
         property->type == ShaderVariableType::RWStructuredBuffer)) {
        roles.instance = descriptor_index++;
        roles.instance_writable =
            property->type == ShaderVariableType::RWStructuredBuffer;
    }

    LUISA_ASSERT(
        roles.accel != FallbackAccelRoles::invalid ||
            roles.instance != FallbackAccelRoles::invalid,
        "Vulkan found a software fallback acceleration-structure argument "
        "without a descriptor at binding {}, but the shader declares one.",
        descriptor_index);
    return roles;
}

struct ResourceBarrierVisitor {
    ResourceBarrier *barrier;
    SavedArgumentCursor arguments;
    vstd::vector<std::byte> *arg_buffer;
    ArgumentBlockLayout *argument_layout;
    size_t argument_block_offset;
    vstd::vector<StorageBufferMetadata> *buffer_metadata;
    vstd::vector<uint32_t> *validation_values;
    ShaderDispatchCommandBase const &cmd;
    ResourceBarrier::Usage uav_usage;
    ResourceBarrier::Usage read_usage;
    ResourceBarrier::Usage accel_read_usage;
    vstd::span<const hlsl::Property> bindings;
    const Shader *shader{nullptr};
    // The device the buffered commands belong to.  Only the fallback path
    // needs it (to resolve its acceleration-structure binding); it is set by
    // the caller right after construction.
    Device *device{nullptr};
    uint32_t descriptor_index{};
    detail::ShaderCodegenDialect codegen_dialect{
        detail::ShaderCodegenDialect::HLSL_SPIRV};
    [[nodiscard]] const hlsl::Property *binding_at(
        uint32_t index) const noexcept {
        return shader != nullptr ?
                   shader->local_descriptor_property(index) :
                   detail::find_local_descriptor_property(bindings, index);
    }
    [[nodiscard]] const hlsl::Property *consume_binding() noexcept {
        auto *property = binding_at(descriptor_index);
        if (property != nullptr) { ++descriptor_index; }
        return property;
    }
    void emplace_data(
        const void *data, size_t byte_size,
        size_t alignment) {
        size_t relative_offset = 0u;
        if (!argument_layout->append(
                byte_size, alignment, relative_offset)) {
            LUISA_ERROR_WITH_LOCATION(
                "Vulkan argument preprocessing diverged from its checked "
                "layout plan: {}.",
                argument_block_layout_status_name(
                    argument_layout->status()));
        }
        LUISA_ASSERT(
            argument_block_offset <=
                std::numeric_limits<size_t>::max() -
                    argument_layout->size(),
            "Vulkan cumulative argument-buffer size overflowed during "
            "preprocessing.");
        auto required_size =
            argument_block_offset + argument_layout->size();
        luisa::vector_resize(*arg_buffer, required_size);
        if (byte_size != 0u) {
            std::memcpy(
                arg_buffer->data() + argument_block_offset + relative_offset,
                data, byte_size);
        }
    }
    ResourceBarrierVisitor(
        ResourceBarrier *barrier,
        luisa::span<const SavedArgument> saved_arguments,
        vstd::vector<std::byte> *arg_buffer,
        ArgumentBlockLayout *argument_layout,
        size_t argument_block_offset,
        vstd::vector<StorageBufferMetadata> *buffer_metadata,
        vstd::vector<uint32_t> *validation_values,
        ShaderDispatchCommandBase const &cmd,
        bool is_raster,
        vstd::span<const hlsl::Property> bindings,
        uint32_t resource_binding_offset,
        detail::ShaderCodegenDialect codegen_dialect)
        : barrier(barrier), arguments(saved_arguments),
          arg_buffer(arg_buffer),
          argument_layout(argument_layout),
          argument_block_offset(argument_block_offset),
          buffer_metadata(buffer_metadata),
          validation_values(validation_values), cmd(cmd),
          bindings(bindings), descriptor_index(resource_binding_offset),
          codegen_dialect(codegen_dialect) {
        if (is_raster) {
            uav_usage = ResourceBarrier::Usage::kRasterUAV;
            read_usage = ResourceBarrier::Usage::kRasterRead;
            accel_read_usage = ResourceBarrier::Usage::kRasterAccelRead;
        } else {
            uav_usage = ResourceBarrier::Usage::kComputeUAV;
            read_usage = ResourceBarrier::Usage::kComputeRead;
            accel_read_usage = ResourceBarrier::Usage::kComputeAccelRead;
        }
    }
    void operator()(Argument::Buffer const &bf) {
        auto &argument = arguments.next_buffer();
        auto property_index = descriptor_index;
        auto *property = consume_binding();
        LUISA_ASSERT(
            property != nullptr &&
                (property->type ==
                     hlsl::ShaderVariableType::StructuredBuffer ||
                 property->type ==
                     hlsl::ShaderVariableType::RWStructuredBuffer),
            "Vulkan barrier preprocessing found no buffer descriptor at binding {}.",
            property_index);
        LUISA_ASSERT(bf.handle != 0u,
                     "Vulkan dispatch contains a null buffer handle.");
        auto res = reinterpret_cast<Buffer const *>(bf.handle);
        LUISA_ASSERT(
            (argument.tag == Type::Tag::CUSTOM) ==
                res->is_indirect_dispatch_buffer(),
            "Vulkan dispatch buffer kind does not match saved ABI tag {}.",
            luisa::to_underlying(argument.tag));
        auto view = BufferView{res, bf.offset, bf.size};
        if (res->is_indirect_dispatch_buffer()) {
            LUISA_ASSERT(
                !argument.has_buffer_metadata() && bf.offset == 0u &&
                    (bf.size == res->indirect_dispatch_capacity() ||
                     bf.size == res->byte_size()),
                "Vulkan indirect-dispatch shader arguments must encode the "
                "whole record range (offset 0, capacity {} or byte size {}), "
                "got offset {} and encoded size {}.",
                res->indirect_dispatch_capacity(), res->byte_size(),
                bf.offset, bf.size);
            view = BufferView{res, 0u, res->byte_size()};
        } else {
            LUISA_ASSERT(
                bf.offset <= res->byte_size() &&
                    bf.size <= res->byte_size() - bf.offset,
                "Vulkan buffer argument offset {} and size {} exceed the "
                "backing buffer size {}.",
                bf.offset, bf.size, res->byte_size());
        }
        if (argument.has_buffer_metadata()) {
            LUISA_ASSERT(buffer_metadata != nullptr &&
                             argument.buffer_metadata_index() < buffer_metadata->size(),
                         "Missing Vulkan XIR/SPIR-V buffer metadata slot {}.",
                         argument.buffer_metadata_index());
            (*buffer_metadata)[argument.buffer_metadata_index()] =
                storage_buffer_descriptor_range(
                    res, bf.offset, bf.size, argument.struct_size,
                    argument.native_buffer_uses_device_address())
                    .metadata;
        }
        if (validation_values != nullptr) {
            uint32_t value = 0u;
            LUISA_ASSERT(
                argument_block_validation_value(
                    bf.size, argument.struct_size, value),
                "Vulkan buffer validation size {} bytes is not an exact, "
                "32-bit element count for stride {}.",
                bf.size, argument.struct_size);
            validation_values->emplace_back(value);
        }
        if (((uint)argument.var_usage & (uint)Usage::WRITE) != 0) {
            // LUISA_ASSERT(is_device_buffer(res), "Unordered access buffer can not be host-buffer.");
            barrier->record(
                view,
                uav_usage);
        } else {
            barrier->record(
                view,
                read_usage);
        }
    }
    void operator()(Argument::Texture const &bf) {
        auto &argument = arguments.next_texture();
        static_cast<void>(consume_texture_descriptor_bindings(
            shader, bindings, descriptor_index, argument.var_usage,
            "barrier preprocessing"));
        LUISA_ASSERT(bf.handle != 0u,
                     "Vulkan dispatch contains a null texture handle.");
        auto rt = reinterpret_cast<Texture *>(bf.handle);
        LUISA_ASSERT(bf.level < rt->mip(),
                     "Texture argument base mip {} is outside {} mip levels.",
                     bf.level, rt->mip());
        auto descriptor_roles =
            detail::texture_descriptor_roles(argument.var_usage);
        if (descriptor_roles.storage) {
            if (!rt->allow_uav()) {
                LUISA_ERROR("Texture not allowed for Unordered-Access.");
            }
        }
        barrier->record_texture_descriptor(
            rt, bf.level,
            descriptor_roles.sampled,
            descriptor_roles.storage,
            read_usage, uav_usage);
    }
    void operator()(Argument::BindlessArray const &bf) {
        auto &argument = arguments.next_bindless_array();
        auto property_index = descriptor_index;
        auto *property = consume_binding();
        LUISA_ASSERT(
            property != nullptr && property->type ==
                                       hlsl::ShaderVariableType::StructuredBuffer,
            "Vulkan barrier preprocessing found no bindless index descriptor at binding {}.",
            property_index);
        auto uses_metadata = false;
        if (auto *metadata = binding_at(descriptor_index);
            metadata != nullptr && metadata->type ==
                                       hlsl::ShaderVariableType::SPIRVBindlessBufferMetadata) {
            ++descriptor_index;
            uses_metadata = true;
        }
        LUISA_ASSERT(bf.handle != 0u,
                     "Vulkan dispatch contains a null bindless-array handle.");
        auto bdls = reinterpret_cast<BindlessArray *>(bf.handle);
        if (codegen_dialect ==
                detail::ShaderCodegenDialect::XIR_SPIRV &&
            argument.native_bindless_uses_device_address()) {
            LUISA_ASSERT(
                bdls->encoded_buffers_support_device_address(),
                "Native Vulkan bindless buffer device-address queries "
                "require every buffer in the encoded descriptor snapshot "
                "to carry an address-capable creation attestation.");
        }
        auto &buffer = bdls->indices_buffer();
        barrier->record(
            BufferView(&buffer, 0, buffer.byte_size()),
            read_usage);
        if (uses_metadata) {
            auto metadata = bdls->buffer_metadata();
            LUISA_ASSERT(
                metadata != nullptr,
                "A shader uses bindless buffer metadata, but the bound "
                "bindless array has no metadata storage.");
            barrier->record(
                BufferView(metadata, 0, metadata->byte_size()),
                read_usage);
        }
        auto access = resource_access(argument.var_usage);
        barrier->process_bindless(
            bdls,
            access.writes ? uav_usage : read_usage,
            read_usage);
        if (validation_values != nullptr) {
            uint32_t value = 0u;
            LUISA_ASSERT(
                argument_block_validation_value(
                    bdls->size(), 0u, value),
                "Vulkan bindless-array capacity {} is not representable by "
                "the HLSL validation ABI.",
                bdls->size());
            validation_values->emplace_back(value);
        }
    }
    void operator()(Argument::Uniform const &a) {
        static_cast<void>(arguments.next_uniform(a.size));
        auto bf = cmd.uniform(a);
        emplace_data(bf.data(), bf.size_bytes(), a.alignment);
    }
    void operator()(Argument::Accel const &bf) {
        auto &argument = arguments.next_accel();
        LUISA_ASSERT(bf.handle != 0u,
                     "Vulkan dispatch contains a null accel handle.");
        if (validation_values != nullptr) {
            // One HLSL validation word per accel argument, in kernel-argument
            // order: the instance count the shader's `accel.size()` query reads
            // from the host-injected validation slot. An acceleration structure
            // that was never built has no instance count and reports zero.
            uint32_t value = 0u;
            if (device->owns_fallback_accel(bf.handle)) {
                auto binding = device->fallback_rtx()->binding(bf.handle);
                value = binding.valid() ? binding.instance_count : 0u;
            } else {
                value = reinterpret_cast<Tlas *>(bf.handle)
                                ->last_instance_count();
            }
            validation_values->emplace_back(value);
        }
        // ---- software (fallback) acceleration structure ----------------------
        // Its descriptors are storage buffers, so the argument contributes the
        // storage-buffer read/write contract and nothing else: there is no
        // VkAccelerationStructureKHR to transition and no acceleration-
        // structure build to order against (the fallback builds are ordinary
        // dispatches, spliced into this command buffer at the position of the
        // build command).
        if (device->owns_fallback_accel(bf.handle)) {
            auto roles = consume_fallback_accel_roles(
                shader, bindings, descriptor_index);
            auto buffers = fallback_accel_buffers(
                device, bf.handle, "barrier preprocessing");
            if (roles.accel != FallbackAccelRoles::invalid) {
                barrier->record(
                    BufferView{buffers.accel, 0u, buffers.accel->byte_size()},
                    read_usage);
                // The slot table the traversal indexes is a storage buffer of
                // its own, and the descriptors it resolves are written by the
                // bindless-array update command - so the table is read here.
                auto &index_buffer = buffers.heap->indices_buffer();
                barrier->record(
                    BufferView{&index_buffer, 0u, index_buffer.byte_size()},
                    read_usage);
            }
            if (roles.instance != FallbackAccelRoles::invalid) {
                barrier->record(
                    BufferView{buffers.instances, 0u,
                               buffers.instances->byte_size()},
                    roles.instance_writable ? uav_usage : read_usage);
            }
            return;
        }
        auto tlas = reinterpret_cast<Tlas *>(bf.handle);
        auto [reads, writes] = resource_access(argument.var_usage);
        auto native = codegen_dialect ==
                      detail::ShaderCodegenDialect::XIR_SPIRV;
        auto traversal = false;
        auto instance_read = false;
        auto instance_write = false;
        if (native) {
            LUISA_ASSERT(
                argument.has_explicit_native_accel_roles(),
                "Native Vulkan accel argument has no persisted exact-role mask.");
            traversal = argument.native_accel_uses_traversal();
            auto instance =
                argument.native_accel_uses_instance_buffer();
            if (argument.var_usage == Usage::NONE) {
                LUISA_ASSERT(!traversal && !instance,
                             "Unused native Vulkan accel has nonempty descriptor roles.");
                return;
            }
            if (traversal) {
                auto *property = binding_at(descriptor_index);
                LUISA_ASSERT(
                    property != nullptr &&
                        property->type ==
                            hlsl::ShaderVariableType::SPIRVAccel,
                    "Native Vulkan accel traversal role has no descriptor at binding {}.",
                    descriptor_index);
                ++descriptor_index;
            }
            if (instance) {
                auto *property = binding_at(descriptor_index);
                auto expected = writes ?
                                    hlsl::ShaderVariableType::SPIRVAccelInstanceRW :
                                    hlsl::ShaderVariableType::SPIRVAccelInstance;
                LUISA_ASSERT(
                    property != nullptr && property->type == expected,
                    "Native Vulkan accel instance role has no matching descriptor at binding {}.",
                    descriptor_index);
                ++descriptor_index;
                instance_read = !writes;
                instance_write = writes;
            }
        } else {
            auto *property = binding_at(descriptor_index);
            traversal = property != nullptr &&
                        property->type ==
                            hlsl::ShaderVariableType::SPIRVAccel;
            if (traversal) {
                ++descriptor_index;
                property = binding_at(descriptor_index);
            }
            instance_read = property != nullptr &&
                            property->type ==
                                hlsl::ShaderVariableType::SPIRVAccelInstance;
            instance_write = property != nullptr &&
                             property->type ==
                                 hlsl::ShaderVariableType::SPIRVAccelInstanceRW;
            if (instance_read || instance_write) { ++descriptor_index; }
        }
        auto instance = instance_read || instance_write;
        LUISA_ASSERT(
            native ? (traversal || instance) : traversal == reads,
            "Vulkan accel barrier contract disagrees with the {} descriptor dialect.",
            native ? "native XIR" : "legacy HLSL/LLVM");
        LUISA_ASSERT(
            !instance || instance_write == writes,
            "Vulkan accel instance descriptor writability disagrees with saved usage.");
        if (instance) {
            LUISA_ASSERT(
                tlas->instance_buffer() != nullptr,
                "Cannot access an uninitialized Vulkan accel instance buffer.");
            barrier->record(
                BufferView(tlas->instance_buffer()),
                instance_write ? uav_usage : read_usage);
        }
        if (traversal) {
            if (!tlas->accel_buffer()) [[unlikely]] {
                LUISA_ERROR("Accel not initialized.");
            }
            barrier->record(
                BufferView(tlas->accel_buffer()),
                accel_read_usage);
        }
    }
};
struct BindPropVisitor {
    // Each sets
    CommandBuffer *cmdbuffer;
    VkDescriptorSet desc_set;
    CommandBufferState::CachedDescriptorSetEntry *cached_desc_entry{nullptr};
    uint desc_index;
    vstd::vector<VkImageView> *img_views;
    SavedArgumentCursor arguments;
    vstd::span<const hlsl::Property> bindings;
    const Shader *shader{nullptr};
    // The device the descriptors belong to; only the fallback path needs it
    // (to resolve its acceleration-structure binding).  Set by the caller.
    Device *device{nullptr};
    // The region base of every software (fallback) `accel` argument, in
    // argument order: the command encoder writes them into the shader's
    // push-constant block, after the fixed dispatch block
    // (`fallback_accel_base_push_offset`).
    vstd::vector<uint32_t> fallback_accel_bases;
    detail::ShaderCodegenDialect codegen_dialect{
        detail::ShaderCodegenDialect::HLSL_SPIRV};
    ResourceBarrier::Usage uav_usage;
    [[nodiscard]] const hlsl::Property *binding_at(uint index) const noexcept {
        return shader != nullptr ?
                   shader->local_descriptor_property(index) :
                   detail::find_local_descriptor_property(bindings, index);
    }
    void operator()(Argument::Buffer const &bf) {
        auto &argument = arguments.next_buffer();
        LUISA_ASSERT(bf.handle != 0u,
                     "Vulkan dispatch contains a null buffer handle.");
        auto *binding = binding_at(desc_index);
        LUISA_ASSERT(
            binding != nullptr &&
                (binding->type == hlsl::ShaderVariableType::StructuredBuffer ||
                 binding->type == hlsl::ShaderVariableType::RWStructuredBuffer),
            "Buffer argument at binding {} has no explicit storage-buffer descriptor.",
            desc_index);
        auto idx = desc_index++;
        auto buffer = reinterpret_cast<Buffer const *>(bf.handle);
        LUISA_ASSERT(
            (argument.tag == Type::Tag::CUSTOM) ==
                buffer->is_indirect_dispatch_buffer(),
            "Vulkan dispatch buffer kind does not match saved ABI tag {}.",
            luisa::to_underlying(argument.tag));
        auto buffer_descs = cmdbuffer->temp_desc->allocate_memory<VkDescriptorBufferInfo>();
        auto validate_direct_descriptor = [&](size_t offset, size_t size,
                                              size_t element_stride) noexcept {
            auto &limits = buffer->device()->properties().limits;
            auto status = detail::validate_direct_storage_buffer_descriptor(
                offset, size, buffer->byte_size(), element_stride,
                std::max<VkDeviceSize>(
                    1u, limits.minStorageBufferOffsetAlignment),
                limits.maxStorageBufferRange);
            LUISA_ASSERT(
                status == detail::DirectStorageBufferDescriptorStatus::SUCCESS,
                "Vulkan HLSL storage-buffer descriptor for view [{}, {}) is "
                "invalid: {}. The legacy HLSL ABI cannot represent a "
                "descriptor-relative subview bias; use an aligned view or "
                "the native XIR/SPIR-V path.",
                offset, offset + size,
                detail::direct_storage_buffer_descriptor_status_name(status));
        };
        if (buffer->is_indirect_dispatch_buffer()) {
            LUISA_ASSERT(
                !argument.has_buffer_metadata() && bf.offset == 0u &&
                    (bf.size == buffer->indirect_dispatch_capacity() ||
                     bf.size == buffer->byte_size()),
                "Vulkan indirect-dispatch descriptor must cover the whole "
                "record buffer.");
            auto writes =
                (luisa::to_underlying(argument.var_usage) &
                 luisa::to_underlying(Usage::WRITE)) != 0u;
            if (writes) {
                // Initialization belongs to the first authoring pass. A
                // reader cannot safely claim it while a writer is merely
                // recorded (but not yet submitted) on another stream.
                ensure_indirect_header_initialized(
                    cmdbuffer, buffer, uav_usage);
            }
            validate_direct_descriptor(
                0u, buffer->byte_size(), argument.struct_size);
            *buffer_descs = VkDescriptorBufferInfo{
                buffer->vk_buffer(), 0u, buffer->byte_size()};
        } else if (argument.has_buffer_metadata()) {
            auto descriptor = storage_buffer_descriptor_range(
                buffer, bf.offset, bf.size, argument.struct_size);
            *buffer_descs = VkDescriptorBufferInfo{
                buffer->vk_buffer(), descriptor.offset, descriptor.range};
        } else {
            validate_direct_descriptor(
                bf.offset, bf.size, argument.struct_size);
            *buffer_descs = VkDescriptorBufferInfo{
                buffer->vk_buffer(), bf.offset, bf.size};
        }
        cmdbuffer->write_desc_sets->emplace_back(VkWriteDescriptorSet{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            desc_set,
            idx,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            buffer_descs,
            nullptr});
    }
    void operator()(Argument::Texture const &bf) {
        auto &argument = arguments.next_texture();
        LUISA_ASSERT(bf.handle != 0u,
                     "Vulkan dispatch contains a null texture handle.");
        auto tex = reinterpret_cast<Texture const *>(bf.handle);
        LUISA_ASSERT(bf.level < tex->mip(),
                     "Texture argument base mip {} is outside {} mip levels.",
                     bf.level, tex->mip());
        auto descriptor_bindings = consume_texture_descriptor_bindings(
            shader, bindings, desc_index, argument.var_usage,
            "descriptor binding");
        auto bind_view = [&](uint32_t idx, bool writable) {
            VkImageViewCreateInfo imgview_create_info{
                VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                nullptr,
                0,
                tex->vk_image(),
                VkImageViewType(tex->dimension() - 1),
                Texture::to_vk_format(tex->format()),
                VkComponentMapping{VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY},
                VkImageSubresourceRange{
                    tex->get_aspect(),
                    bf.level,
                    writable ? 1u : tex->mip() - bf.level,
                    0,
                    1}};
            VkImageView img_view;
            VK_CHECK_RESULT(vkCreateImageView(cmdbuffer->device()->logic_device(), &imgview_create_info, Device::alloc_callbacks(), &img_view));
            img_views->emplace_back(img_view);
            auto image_descs = cmdbuffer->temp_desc->allocate_memory<VkDescriptorImageInfo>();
            auto level_count = writable ? 1u : tex->mip() - bf.level;
            *image_descs = VkDescriptorImageInfo{
                VkSampler{nullptr},
                img_view,
                cmdbuffer->resource_barrier->get_texture_descriptor_layout(
                    tex, bf.level, level_count)};
            cmdbuffer->write_desc_sets->emplace_back(VkWriteDescriptorSet{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                desc_set,
                idx,
                0,
                1,
                writable ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE :
                           VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                image_descs,
                nullptr,
                nullptr});
        };
        if (descriptor_bindings.sampled !=
            TextureDescriptorBindings::invalid) {
            bind_view(descriptor_bindings.sampled, false);
        }
        if (descriptor_bindings.storage !=
            TextureDescriptorBindings::invalid) {
            bind_view(descriptor_bindings.storage, true);
        }
    }
    void operator()(Argument::Uniform const &a) {
        static_cast<void>(arguments.next_uniform(a.size));
    }
    void operator()(Argument::BindlessArray const &bf) {
        static_cast<void>(arguments.next_bindless_array());
        LUISA_ASSERT(bf.handle != 0u,
                     "Vulkan dispatch contains a null bindless-array handle.");
        auto bindless = reinterpret_cast<BindlessArray const *>(bf.handle);
        auto &buffer = bindless->indices_buffer();
        auto index_binding = binding_at(desc_index);
        LUISA_ASSERT(index_binding != nullptr &&
                         index_binding->type ==
                             hlsl::ShaderVariableType::StructuredBuffer,
                     "Bindless array at binding {} has no explicit index-buffer descriptor.",
                     desc_index);
        auto idx = desc_index++;
        auto buffer_descs = cmdbuffer->temp_desc->allocate_memory<VkDescriptorBufferInfo>();
        *buffer_descs = VkDescriptorBufferInfo{
            buffer.vk_buffer(),
            0,
            buffer.byte_size()};
        cmdbuffer->write_desc_sets->emplace_back(VkWriteDescriptorSet{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            desc_set,
            idx,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            buffer_descs,
            nullptr});
        if (auto metadata_binding = binding_at(desc_index);
            metadata_binding != nullptr &&
            metadata_binding->type ==
                hlsl::ShaderVariableType::SPIRVBindlessBufferMetadata) {
            auto metadata = bindless->buffer_metadata();
            LUISA_ASSERT(metadata != nullptr,
                         "A shader uses bindless buffers, but the bound "
                         "bindless array has no buffer metadata storage.");
            auto metadata_idx = desc_index++;
            auto metadata_desc =
                cmdbuffer->temp_desc->allocate_memory<VkDescriptorBufferInfo>();
            *metadata_desc = VkDescriptorBufferInfo{
                metadata->vk_buffer(), 0, metadata->byte_size()};
            cmdbuffer->write_desc_sets->emplace_back(VkWriteDescriptorSet{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                desc_set,
                metadata_idx,
                0,
                1,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                nullptr,
                metadata_desc,
                nullptr});
        }
    }
    void operator()(Argument::Accel const &bf) {
        auto &argument = arguments.next_accel();
        LUISA_ASSERT(bf.handle != 0u,
                     "Vulkan dispatch contains a null accel handle.");
        // ---- software (fallback) acceleration structure ----------------------
        // The two slots the shader declared are bound to the fallback's shared
        // acceleration buffer and to its instance buffer, both ordinary storage
        // buffers.  The acceleration view is the *whole* buffer (a handle is an
        // absolute `uint4` index), and the tree's own region reaches the shader
        // as the `accelBase` push-constant word.
        if (device->owns_fallback_accel(bf.handle)) {
            auto roles = consume_fallback_accel_roles(
                shader, bindings, desc_index);
            auto buffers = fallback_accel_buffers(
                device, bf.handle, "descriptor binding");
            auto bind_buffer = [&](uint32_t index, const Buffer *buffer,
                                   size_t offset, const char *name) noexcept {
                validate_fallback_accel_descriptor(
                    device, buffer, offset, name);
                auto buffer_desc =
                    cmdbuffer->temp_desc->allocate_memory<VkDescriptorBufferInfo>();
                *buffer_desc = VkDescriptorBufferInfo{
                    buffer->vk_buffer(), offset,
                    fallback_accel_descriptor_range(buffer, offset)};
                cmdbuffer->write_desc_sets->emplace_back(VkWriteDescriptorSet{
                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    nullptr,
                    desc_set,
                    index,
                    0,
                    1,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    nullptr,
                    buffer_desc,
                    nullptr});
            };
            if (roles.accel != FallbackAccelRoles::invalid) {
                // The `accel` argument's first descriptor is the TLAS' bindless
                // heap: its *slot table* is a stream of `uint` handles, exactly
                // the shape a `BindlessArray` argument's index buffer has (the
                // buffers the slots name live in the device's global bindless
                // heap, which the shader metadata binds).
                auto &index_buffer = buffers.heap->indices_buffer();
                auto index_desc =
                    cmdbuffer->temp_desc->allocate_memory<VkDescriptorBufferInfo>();
                *index_desc = VkDescriptorBufferInfo{
                    index_buffer.vk_buffer(), 0u, index_buffer.byte_size()};
                cmdbuffer->write_desc_sets->emplace_back(VkWriteDescriptorSet{
                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    nullptr,
                    desc_set,
                    roles.accel,
                    0,
                    1,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    nullptr,
                    index_desc,
                    nullptr});
            }
            if (roles.instance != FallbackAccelRoles::invalid) {
                bind_buffer(roles.instance, buffers.instances,
                            buffers.instance_offset, "instance");
            }
            // Every fallback `accel` argument owns one region-base word of the
            // shader's push-constant block (`entry_points.cpp` appends one
            // `ConstantValue` marker per argument), in argument order - the
            // order this pass visits them in.
            fallback_accel_bases.emplace_back(
                fallback_accel_region_base(buffers));
            return;
        }
        auto tlas = reinterpret_cast<Tlas *>(bf.handle);
        auto [reads, writes] = resource_access(argument.var_usage);
        auto native = codegen_dialect ==
                      detail::ShaderCodegenDialect::XIR_SPIRV;
        auto bind_traversal = [&] {
            auto *binding = binding_at(desc_index);
            LUISA_ASSERT(
                binding != nullptr &&
                    binding->type ==
                        hlsl::ShaderVariableType::SPIRVAccel,
                "Missing Vulkan acceleration-structure descriptor at binding {}.",
                desc_index);
            LUISA_ASSERT(reads,
                         "Acceleration-structure descriptor at binding {} has no traversal/query read usage.",
                         desc_index);
            LUISA_ASSERT(tlas->accel_buffer() != nullptr &&
                             tlas->accel() != VK_NULL_HANDLE,
                         "Cannot bind an unbuilt Vulkan acceleration structure.");
            auto idx = desc_index++;
            auto accel_info = cmdbuffer->temp_desc->allocate_memory<VkWriteDescriptorSetAccelerationStructureKHR>();
            *accel_info = VkWriteDescriptorSetAccelerationStructureKHR{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
                .pNext = nullptr,
                .accelerationStructureCount = 1u,
                .pAccelerationStructures = &tlas->accel()};
            cmdbuffer->write_desc_sets->emplace_back(VkWriteDescriptorSet{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                accel_info,
                desc_set,
                idx,
                0,
                1,
                VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                nullptr,
                nullptr,
                nullptr});
        };
        auto bind_instance = [&](bool writable) {
            auto *binding = binding_at(desc_index);
            auto expected = writable ?
                                hlsl::ShaderVariableType::SPIRVAccelInstanceRW :
                                hlsl::ShaderVariableType::SPIRVAccelInstance;
            LUISA_ASSERT(
                binding != nullptr && binding->type == expected,
                "Missing Vulkan accel instance descriptor at binding {} "
                "(expected type {}, got {}).",
                desc_index, static_cast<uint32_t>(expected),
                binding == nullptr ?
                    std::numeric_limits<uint32_t>::max() :
                    static_cast<uint32_t>(binding->type));
            LUISA_ASSERT(tlas->instance_buffer() != nullptr,
                         "Cannot bind an uninitialized Vulkan accel instance buffer.");
            auto idx = desc_index++;
            auto buffer_descs = cmdbuffer->temp_desc->allocate_memory<VkDescriptorBufferInfo>();
            *buffer_descs = VkDescriptorBufferInfo{
                tlas->instance_buffer()->vk_buffer(),
                0,
                tlas->instance_buffer()->byte_size()};
            cmdbuffer->write_desc_sets->emplace_back(VkWriteDescriptorSet{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                desc_set,
                idx,
                0,
                1,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                nullptr,
                buffer_descs,
                nullptr});
        };
        if (native) {
            LUISA_ASSERT(
                argument.has_explicit_native_accel_roles(),
                "Native Vulkan accel argument has no persisted exact-role mask.");
            auto traversal =
                argument.native_accel_uses_traversal();
            auto instance =
                argument.native_accel_uses_instance_buffer();
            if (argument.var_usage == Usage::NONE) {
                LUISA_ASSERT(!traversal && !instance,
                             "Unused native Vulkan accel has nonempty descriptor roles.");
                return;
            }
            if (traversal) { bind_traversal(); }
            if (instance) { bind_instance(writes); }
            LUISA_ASSERT(
                traversal || instance,
                "Used native Vulkan accel has an empty descriptor-role mask.");
            return;
        }

        // Legacy HLSL/LLVM artifacts have no explicit role mask. Their stable
        // ABI makes every read a traversal descriptor and may append one
        // instance buffer before the next argument's mandatory descriptor.
        if (reads) { bind_traversal(); }
        auto *next = binding_at(desc_index);
        auto has_instance =
            next != nullptr &&
            (next->type ==
                 hlsl::ShaderVariableType::SPIRVAccelInstance ||
             next->type ==
                 hlsl::ShaderVariableType::SPIRVAccelInstanceRW);
        if (has_instance) {
            bind_instance(writes);
        } else {
            LUISA_ASSERT(!writes,
                         "Legacy Vulkan accel write has no writable instance descriptor.");
        }
    }
};
namespace temp_buffer {
uint64 DefaultBufferDeferredVisitor::allocate(uint64 size) {
    auto bf = new DefaultBuffer(device, size);
    buffers.try_emplace(
        reinterpret_cast<uint64_t>(bf),
        bf);
    return reinterpret_cast<uint64>(bf);
}
void DefaultBufferDeferredVisitor::deallocate(uint64 handle) {
    if (buffers.empty()) return;
    auto iter = buffers.find(handle);
    LUISA_ASSERT(iter != buffers.end());
    cmdbuffer->states()->dispose_after_flush(std::move(iter->second));
    buffers.erase(iter);
}
template<typename Pack>
uint64 Visitor<Pack>::allocate(uint64 size) {
    return reinterpret_cast<uint64_t>(new Pack(device, size));
}
template<typename Pack>
void Visitor<Pack>::deallocate(uint64 handle) {
    auto *ptr = reinterpret_cast<Pack *>(handle);
    if (deferred_frees != nullptr) {
        deferred_frees->emplace_back(ptr);
        return;
    }
    delete ptr;
}
template<typename Pack>
auto Visitor<Pack>::create(uint64 size) -> Pack * {
    return new Pack{device, size};
}
[[nodiscard]] size_t round_up_large_buffer_size(size_t size) noexcept {
    constexpr size_t kMinSize = 128ull * 1024ull;
    constexpr size_t kOneMiB = 1024ull * 1024ull;
    constexpr size_t kEightMiB = 8ull * 1024ull * 1024ull;
    constexpr size_t kFourMiB = 4ull * 1024ull * 1024ull;
    if (size <= kOneMiB) {
        return std::max(kMinSize, std::bit_ceil(size));
    }
    size_t step = (size <= kEightMiB) ? kOneMiB : kFourMiB;
    if (size > std::numeric_limits<size_t>::max() - (step - 1u)) {
        return size;
    }
    return ((size + step - 1u) / step) * step;
}
template<typename T>
void BufferAllocator<T>::clear(vstd::vector<vstd::unique_ptr<T>> *deferred_frees) {
    // Soft clear: reset positions without freeing buffers
    // to keep staging buffers warm across dispatches.
    static constexpr size_t kMaxWarmCapacity = 8ull * 1024ull * 1024ull;
    size_t total_size = 0;
    for (auto &buf : alloc.allocated_buffer()) {
        total_size += buf.fullSize;
    }
    if (total_size <= kMaxWarmCapacity) {
        alloc.soft_clear();
    } else {
        visitor.deferred_frees = deferred_frees;
        auto reset_visitor = vstd::scope_exit([&]() {
            visitor.deferred_frees = nullptr;
        });
        alloc.dispose();
    }
    ++clear_epoch;
    for (auto &lb : active_large_buffers) {
        lb.used_offset = 0u;
        lb.last_used_epoch = clear_epoch;
        free_large_buffers.emplace_back(std::move(lb));
    }
    active_large_buffers.clear();

    static constexpr size_t kMaxWarmLargeBytes = 64ull * 1024ull * 1024ull;
    static constexpr size_t kMaxSingleWarmLargeBytes = 256ull * 1024ull * 1024ull;
    static constexpr size_t kHardMaxLargeBytes = 256ull * 1024ull * 1024ull;
    static constexpr size_t kMaxFreeLargeBuffers = 16u;
    static constexpr uint64_t kSoftEvictIdleEpochs = 8u;
    static constexpr uint64_t kSingleLargeIdleEpochs = 64u;
    static constexpr uint64_t kIdleEvictEpochs = 256u;

    size_t total_free_bytes = 0u;
    for (auto const &lb : free_large_buffers) {
        total_free_bytes += lb.capacity;
    }
    auto evict_at = [&](size_t idx) {
        total_free_bytes -= free_large_buffers[idx].capacity;
        if (deferred_frees != nullptr) {
            deferred_frees->emplace_back(std::move(free_large_buffers[idx].buffer));
        }
        if (idx + 1u < free_large_buffers.size()) {
            free_large_buffers[idx] = std::move(free_large_buffers.back());
        }
        free_large_buffers.pop_back();
    };

    while (free_large_buffers.size() > 1u && total_free_bytes > kHardMaxLargeBytes) {
        size_t victim = 0u;
        for (size_t i = 1u; i < free_large_buffers.size(); ++i) {
            if (free_large_buffers[i].last_used_epoch < free_large_buffers[victim].last_used_epoch ||
                (free_large_buffers[i].last_used_epoch == free_large_buffers[victim].last_used_epoch &&
                 free_large_buffers[i].capacity > free_large_buffers[victim].capacity)) {
                victim = i;
            }
        }
        evict_at(victim);
    }

    if (free_large_buffers.size() == 1u) {
        uint64_t idle = clear_epoch - free_large_buffers[0].last_used_epoch;
        if ((free_large_buffers[0].capacity > kMaxSingleWarmLargeBytes && idle >= 4u) ||
            (free_large_buffers[0].capacity > kMaxWarmLargeBytes && idle >= kSingleLargeIdleEpochs) ||
            idle >= kIdleEvictEpochs) {
            evict_at(0u);
        }
    } else if (free_large_buffers.size() > 1u) {
        bool over_soft = total_free_bytes > kMaxWarmLargeBytes || free_large_buffers.size() > kMaxFreeLargeBuffers;
        uint64_t min_idle = over_soft ? kSoftEvictIdleEpochs : kIdleEvictEpochs;
        size_t victim = free_large_buffers.size();
        for (size_t i = 0u; i < free_large_buffers.size(); ++i) {
            uint64_t idle = clear_epoch - free_large_buffers[i].last_used_epoch;
            if (idle >= min_idle) {
                if (victim == free_large_buffers.size() ||
                    free_large_buffers[i].last_used_epoch < free_large_buffers[victim].last_used_epoch ||
                    (free_large_buffers[i].last_used_epoch == free_large_buffers[victim].last_used_epoch &&
                     free_large_buffers[i].capacity > free_large_buffers[victim].capacity)) {
                    victim = i;
                }
            }
        }
        if (victim < free_large_buffers.size()) {
            evict_at(victim);
        }
    }
}
template<typename T>
BufferAllocator<T>::BufferAllocator(size_t init_capacity)
    : alloc(init_capacity, &visitor) {
}
template<typename T>
BufferAllocator<T>::~BufferAllocator() {
}
template<typename T>
void BufferAllocator<T>::flush_host() {
    for (auto &b : alloc.allocated_buffer()) {
        if (b.position > 0u) {
            reinterpret_cast<T *>(b.handle)->flush_host();
        }
    }
    for (auto &lb : active_large_buffers) {
        lb.buffer->flush_host();
    }
}
template<typename T>
BufferView BufferAllocator<T>::_allocate_large(size_t size, size_t align) {
    align = std::max<size_t>(align, 16u);
    auto align_up = [](size_t offset, size_t alignment, size_t &out) noexcept -> bool {
        if (alignment <= 1u) {
            out = offset;
            return true;
        }
        auto rem = offset % alignment;
        if (rem == 0u) {
            out = offset;
            return true;
        }
        auto add = alignment - rem;
        if (offset > std::numeric_limits<size_t>::max() - add) {
            return false;
        }
        out = offset + add;
        return true;
    };

    // Pass 1: sub-allocate from an already active large buffer.
    size_t best_active = active_large_buffers.size();
    size_t best_offset = 0u;
    size_t best_slack = std::numeric_limits<size_t>::max();
    for (size_t i = 0u; i < active_large_buffers.size(); ++i) {
        auto const &entry = active_large_buffers[i];
        size_t aligned_offset = 0u;
        if (!align_up(entry.used_offset, align, aligned_offset)) {
            continue;
        }
        if (aligned_offset <= entry.capacity &&
            size <= entry.capacity - aligned_offset) {
            size_t slack = entry.capacity - (aligned_offset + size);
            if (slack < best_slack) {
                best_slack = slack;
                best_active = i;
                best_offset = aligned_offset;
            }
        }
    }
    if (best_active < active_large_buffers.size()) {
        auto &entry = active_large_buffers[best_active];
        entry.used_offset = best_offset + size;
        entry.last_used_epoch = clear_epoch;
        return BufferView(entry.buffer.get(), best_offset, size);
    }

    // Pass 2: best-fit reuse from free_large_buffers.
    size_t best_free = free_large_buffers.size();
    size_t best_capacity = std::numeric_limits<size_t>::max();
    for (size_t i = 0u; i < free_large_buffers.size(); ++i) {
        auto const &entry = free_large_buffers[i];
        if (entry.capacity >= size && entry.capacity < best_capacity) {
            best_capacity = entry.capacity;
            best_free = i;
        }
    }
    if (best_free < free_large_buffers.size()) {
        auto reused = std::move(free_large_buffers[best_free]);
        if (best_free + 1u < free_large_buffers.size()) {
            free_large_buffers[best_free] = std::move(free_large_buffers.back());
        }
        free_large_buffers.pop_back();
        reused.used_offset = size;
        reused.last_used_epoch = clear_epoch;
        auto *raw_ptr = reused.buffer.get();
        active_large_buffers.emplace_back(std::move(reused));
        return BufferView(raw_ptr, 0u, size);
    }

    // Pass 3: allocate a new rounded large buffer.
    size_t alloc_size = round_up_large_buffer_size(size);
    vstd::unique_ptr<T> new_buf{visitor.create(alloc_size)};
    auto *raw_ptr = new_buf.get();
    active_large_buffers.emplace_back(LargeBuffer{
        .buffer = std::move(new_buf),
        .capacity = alloc_size,
        .used_offset = size,
        .last_used_epoch = clear_epoch});
    return BufferView(raw_ptr, 0u, size);
}
template<typename T>
BufferView BufferAllocator<T>::allocate(size_t size) {
    if (size <= kLargeBufferSize) [[likely]] {
        auto chunk = alloc.allocate(size);
        return BufferView(reinterpret_cast<T const *>(chunk.handle), chunk.offset, size);
    } else {
        return _allocate_large(size, 16u);
    }
}

template<typename T>
BufferView BufferAllocator<T>::allocate(size_t size, size_t align) {
    if (size <= kLargeBufferSize) [[likely]] {
        auto chunk = alloc.allocate(size, align);
        return BufferView(reinterpret_cast<T const *>(chunk.handle), chunk.offset, size);
    } else {
        return _allocate_large(size, align);
    }
}
}// namespace temp_buffer

namespace stream_state_detail {

[[nodiscard]] VkDescriptorPool create_descriptor_pool_with_sizes(
    Device &device,
    uint32_t requested_max_sets,
    uint32_t requested_per_type) {
    auto const &limits = device.properties().limits;
    auto pool_plan = detail::plan_descriptor_pool_sizes(
        limits.maxPerStageDescriptorStorageBuffers,
        limits.maxPerStageDescriptorStorageImages,
        limits.maxPerStageDescriptorSampledImages,
        limits.maxPerStageDescriptorSamplers,
        limits.maxPerStageDescriptorUniformBuffers,
        limits.maxDescriptorSetStorageBuffers,
        limits.maxDescriptorSetStorageImages,
        limits.maxDescriptorSetSampledImages,
        limits.maxDescriptorSetSamplers,
        limits.maxDescriptorSetUniformBuffers,
        device.enable_raytracing(),
        requested_max_sets,
        requested_per_type);
    VkDescriptorPoolSize pool_sizes[6];
    pool_sizes[0].descriptorCount = pool_plan.storage_buffers;
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[1].descriptorCount = pool_plan.storage_images;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[2].descriptorCount = pool_plan.sampled_images;
    pool_sizes[2].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    pool_sizes[3].descriptorCount = pool_plan.samplers;
    pool_sizes[3].type = VK_DESCRIPTOR_TYPE_SAMPLER;
    pool_sizes[4].descriptorCount = pool_plan.uniform_buffers;
    pool_sizes[4].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    auto pool_size_count = 5u;
    if (device.enable_raytracing()) {
        pool_sizes[pool_size_count].descriptorCount =
            pool_plan.acceleration_structures;
        pool_sizes[pool_size_count].type =
            VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        ++pool_size_count;
    }
    VkDescriptorPoolCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = 0,
        .maxSets = pool_plan.max_sets,
        .poolSizeCount = pool_size_count,
        .pPoolSizes = pool_sizes};
    VkDescriptorPool created{};
    VK_CHECK_RESULT(vkCreateDescriptorPool(device.logic_device(), &createInfo, Device::alloc_callbacks(), &created));
    return created;
}

}// namespace stream_state_detail

namespace vk_ext_copy_detail {

// VkQueueFlags of one queue family of the physical device. `CommandBufferState`
// reads it once to evaluate VK_KHR_copy_memory_indirect's supportedQueues
// requirement (VUID-vkCmdCopyMemoryIndirectKHR-commandBuffer-10936). An unknown
// family yields 0, which selects the pre-existing copy path.
[[nodiscard]] VkQueueFlags queue_family_flags(
    Device &device, uint32_t family_index) noexcept {
    if (family_index == VK_QUEUE_FAMILY_IGNORED) { return 0u; }
    auto physical_device = device.physical_device();
    uint32_t family_count = 0u;
    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device, &family_count, nullptr);
    if (family_index >= family_count) { return 0u; }
    luisa::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device, &family_count, families.data());
    return families[family_index].queueFlags;
}

// Device address of the byte range `[offset, offset + size)` of one buffer, or
// false when the range cannot be described: a buffer without a shader device
// address (an imported VkBuffer, or a device that does not expose buffer device
// addresses), an out-of-bounds range, or an address overflow. Every branch keeps
// the caller on the pre-existing per-copy path.
[[nodiscard]] bool buffer_view_address(
    Buffer const *buffer, uint64_t offset, uint64_t size,
    uint64_t &address) noexcept {
    if (buffer == nullptr || !buffer->device_address_capable()) {
        return false;
    }
    auto const limit = static_cast<uint64_t>(buffer->addressable_byte_size());
    if (offset > limit || size > limit - offset) { return false; }
    auto base = buffer->get_device_address();
    if (offset > std::numeric_limits<uint64_t>::max() - base) {
        return false;
    }
    address = base + offset;
    return true;
}

}// namespace vk_ext_copy_detail

static constexpr size_t kTempSize = 1024ull * 1024ull;
CommandBufferState::CommandBufferState()
    : upload_alloc(kTempSize),
      readback_alloc(kTempSize) {
}
void CommandBufferState::init(Device &device, StreamTag tag,
                              VkCommandPoolCreateFlags pool_flags) {
    this->device = &device;
    upload_alloc.visitor.device = &device;
    readback_alloc.visitor.device = &device;
    active_desc_pool_index = 0u;
    next_overflow_max_sets = 8192u;
    next_overflow_per_type = 8192u;
    // Phase 3B (opt A): the reuse cache starts out empty and is stamped with the
    // generation of the device it belongs to, so the first retirement takes the
    // conservative path (pool reset) and every later one can reuse.
    desc_set_cache.clear();
    cached_layout_generation = device.shader_layout_destroy_generation();
    cached_resource_generation = device.resource_destroy_generation();
    has_uncached_desc_allocs = false;
    orphaned_desc_sets = 0u;
    desc_pool = stream_state_detail::create_descriptor_pool_with_sizes(
        device, 4096u, 4096u);
    if (!pool) {
        VkCommandPoolCreateInfo pool_ci{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = pool_flags};
        switch (tag) {
            case StreamTag::GRAPHICS:
                pool_ci.queueFamilyIndex = device.graphics_queue_index();
                break;
            case StreamTag::COPY:
                pool_ci.queueFamilyIndex = device.copy_queue_index();
                break;
            case StreamTag::COMPUTE:
                pool_ci.queueFamilyIndex = device.compute_queue_index();
                break;
            default:
                LUISA_ERROR("Illegal stream tag.");
        }
        VK_CHECK_RESULT(vkCreateCommandPool(device.logic_device(), &pool_ci, Device::alloc_callbacks(), &pool));
        command_pool_queue_flags =
            vk_ext_copy_detail::queue_family_flags(
                device, pool_ci.queueFamilyIndex);
    }
    // VK_KHR_copy_memory_indirect records only into a command buffer whose pool
    // supports one of the queue types the implementation reports
    // (VUID-vkCmdCopyMemoryIndirectKHR-commandBuffer-10936), so both masks are
    // read once here and fed to detail::plan_buffer_copy_batch. When the
    // extension is not enabled the query is skipped and both masks stay zero,
    // which keeps the pre-existing per-copy path.
    copy_memory_indirect_supported_queues = 0u;
    if (device.optional_extension_enabled(
            detail::OptionalExtension::COPY_MEMORY_INDIRECT)) {
        VkPhysicalDeviceCopyMemoryIndirectPropertiesKHR
            copy_memory_indirect_properties{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COPY_MEMORY_INDIRECT_PROPERTIES_KHR,
                .pNext = nullptr,
                .supportedQueues = 0u};
        VkPhysicalDeviceProperties2 properties2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &copy_memory_indirect_properties,
            .properties = {}};
        vkGetPhysicalDeviceProperties2(
            device.physical_device(), &properties2);
        copy_memory_indirect_supported_queues =
            copy_memory_indirect_properties.supportedQueues;
    }
    timestamp_resettable =
        (pool_flags & VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT) != 0u;
    if (device.gpu_timeline() != nullptr) {
        ensure_timestamp_pool(device, tag, 512u);
    }
}
void CommandBufferState::ensure_timestamp_pool(
    Device &device, StreamTag tag, uint32_t min_capacity) {
    if (timestamp_pool != VK_NULL_HANDLE || !timestamp_resettable) {
        return;
    }
    uint32_t qf = 0u;
    switch (tag) {
        case StreamTag::GRAPHICS: qf = device.graphics_queue_index(); break;
        case StreamTag::COPY: qf = device.copy_queue_index(); break;
        case StreamTag::COMPUTE: qf = device.compute_queue_index(); break;
        default: break;
    }
    auto qfp = device.queue_family_properties();
    if (qf < qfp.size()) {
        timestamp_valid_bits = qfp[qf].timestampValidBits;
    }
    if (timestamp_valid_bits > 0u) {
        timestamp_pool_capacity = std::max(min_capacity, 2u);
        VkQueryPoolCreateInfo qp_ci{
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0u,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = timestamp_pool_capacity,
            .pipelineStatistics = 0u};
        VK_CHECK_RESULT(vkCreateQueryPool(
            device.logic_device(), &qp_ci, Device::alloc_callbacks(), &timestamp_pool));
    }
}
CommandBufferState::~CommandBufferState() {
    for (auto *r : retained_pipelines) {
        r->release();
    }
    retained_pipelines.clear();
    if (device != nullptr) {
        auto vk_dev = device->logic_device();
        if (timestamp_pool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(vk_dev, timestamp_pool, Device::alloc_callbacks());
            timestamp_pool = VK_NULL_HANDLE;
        }
        if (pool) {
            vkDestroyCommandPool(vk_dev, pool, Device::alloc_callbacks());
        }
        if (desc_pool) {
            vkDestroyDescriptorPool(vk_dev, desc_pool, Device::alloc_callbacks());
        }
        for (auto p : overflow_desc_pools) {
            if (p) {
                vkDestroyDescriptorPool(vk_dev, p, Device::alloc_callbacks());
            }
        }
    }
}
VkDescriptorSet CommandBufferState::allocate_descriptor_set(
    uint64_t layout_id, VkDescriptorSetLayout layout,
    uint64_t shader_id, CachedDescriptorSetEntry **out_entry) {
    if (out_entry != nullptr) {
        *out_entry = nullptr;
    }
    // Phase 3B (opt A): a set this state already owns for exactly this interned
    // layout is handed out again. It can only be taken from a pool that was not
    // reset since the set was created (see `reset`), and within one recording the
    // cursor never repeats a handle, so no set is overwritten while the command
    // buffer of the same recording still refers to it.
    if (layout_id != 0u) {
        auto iter = desc_set_cache.find(layout_id);
        if (iter) {
            auto &bucket = iter.value();
            if (bucket.cursor < bucket.sets.size()) {
                if (shader_id != 0u &&
                    bucket.sets[bucket.cursor].last_shader_id != shader_id) {
                    auto const limit = std::min(bucket.sets.size(), bucket.cursor + 8u);
                    for (size_t k = bucket.cursor + 1u; k < limit; ++k) {
                        if (bucket.sets[k].last_shader_id == shader_id) {
                            std::swap(bucket.sets[bucket.cursor], bucket.sets[k]);
                            break;
                        }
                    }
                }
                // Counter-only: a reuse is not a native allocation, so
                // `descriptor_set_allocations` keeps describing the real calls.
                auto &entry = bucket.sets[bucket.cursor++];
                if (shader_id != 0u) {
                    entry.last_shader_id = shader_id;
                }
                if (out_entry != nullptr) {
                    *out_entry = &entry;
                }
                return entry.set;
            }
        }
    } else {
        // Not an interned layout: the set belongs to no bucket, so this state must
        // reset its pools on retirement before anything can be reused.
        has_uncached_desc_allocs = true;
    }
    VkDescriptorSet set{};
    while (true) {
        VkDescriptorPool current_pool =
            active_desc_pool_index == 0u ? desc_pool : overflow_desc_pools[active_desc_pool_index - 1u];
        VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = current_pool,
            .descriptorSetCount = 1u,
            .pSetLayouts = &layout};
        VkResult res = vkAllocateDescriptorSets(device->logic_device(), &alloc_info, &set);
        if (res == VK_SUCCESS) {
            // Counter-only: one successful vkAllocateDescriptorSets (F2).
            submit_profile::record_descriptor_set_allocation();
            if (layout_id != 0u) {
                auto try_res = desc_set_cache.try_emplace(layout_id);
                auto &bucket = try_res.first.value();
                // The handle the driver returned is appended behind the ones this state
                // already owns for this layout, so `sets[i]` stays the i-th handle and the
                // cursor stays a valid "next" index.
                LUISA_ASSERT(
                    bucket.cursor == bucket.sets.size(),
                    "Vulkan descriptor-set cache cursor {} does not match its "
                    "{}-entry bucket.",
                    bucket.cursor, bucket.sets.size());
                bucket.sets.emplace_back(CachedDescriptorSetEntry{
                    .set = set,
                    .last_shader_id = shader_id});
                bucket.cursor = bucket.sets.size();
                if (out_entry != nullptr) {
                    *out_entry = &bucket.sets.back();
                }
            }
            return set;
        }
        if (res == VK_ERROR_OUT_OF_POOL_MEMORY || res == VK_ERROR_FRAGMENTED_POOL) {
            if (active_desc_pool_index >= 32u) {
                VK_CHECK_RESULT(res);
            }
            ++active_desc_pool_index;
            if (active_desc_pool_index > overflow_desc_pools.size()) {
                overflow_desc_pools.emplace_back(
                    stream_state_detail::create_descriptor_pool_with_sizes(
                        *device, next_overflow_max_sets, next_overflow_per_type));
                next_overflow_max_sets = std::min<uint32_t>(262144u, next_overflow_max_sets * 2u);
                next_overflow_per_type = std::min<uint32_t>(65536u, next_overflow_per_type * 2u);
            }
            continue;
        }
        VK_CHECK_RESULT(res);
    }
}
void CommandBufferState::retain_pipeline(PipelineRef *ref) {
    if (ref == nullptr) { return; }
    // Phase 3B (opt B): the pre-existing linear scan was O(n^2) per batch (a decode
    // batch grows `retained_pipelines` to 364 entries, i.e. ~66k pointer comparisons
    // per token). The last retained pipeline is remembered because a batch normally
    // dispatches one shader many times in a row, and the hash set keeps the general
    // case O(1). Both are cleared by `reset()`, the only place the vector is emptied.
    if (ref == _last_retained_pipeline) { return; }
    if (retained_pipeline_set.try_emplace(ref, 0u).second) {
        ref->retain();
        retained_pipelines.emplace_back(ref);
        _last_retained_pipeline = ref;
    }
}
// Phase 3B (opt C): the retirement of one batch is split in two, so
// `Stream::sync()` only has to wait for the part the host can observe.
//
// `reset_host_visible` runs what the host is allowed to see when `sync()` returns:
// the readback `memcpy`s of this batch (`callbacks`), the deferred disposals
// (`dispose_pool`, e.g. framebuffers of a raster batch) and the temporary image
// views. Destroying the image views here is what keeps an immediately following
// `destroy_texture()` from racing a `vkDestroyImageView` of a view created for
// this very submission.
void CommandBufferState::reset_host_visible(Stream *stream, Device &device) {
    for (auto &i : callbacks) {
        i();
    }
    callbacks.clear();
    for (auto &i : dispose_pool) {
        i.second(stream, this, i.first);
    }
    dispose_pool.clear();
    for (auto i : img_views) {
        vkDestroyImageView(device.logic_device(), i, Device::alloc_callbacks());
    }
    img_views.clear();
    if (has_submitted_timestamps && timestamp_pool != VK_NULL_HANDLE && device.gpu_timeline() != nullptr) {
        luisa::vector<uint64_t> raw_ts(timestamp_cursor, 0ull);
        if (timestamp_cursor > 0u) {
            VK_CHECK_RESULT(vkGetQueryPoolResults(
                device.logic_device(),
                timestamp_pool,
                0u,
                timestamp_cursor,
                timestamp_cursor * sizeof(uint64_t),
                raw_ts.data(),
                sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        }
        TimelineSubmitRecord rec{};
        rec.submit_seq = tl_submit_seq;
        rec.stream_handle = tl_stream_handle;
        rec.stream_tag = tl_stream_tag;
        rec.fence = tl_fence;
        rec.cpu_record_begin_ns = tl_cpu_record_begin_ns;
        rec.cpu_submit_begin_ns = tl_cpu_submit_begin_ns;
        rec.cpu_submit_end_ns = tl_cpu_submit_end_ns;
        rec.cpu_fence_complete_ns = tl_cpu_fence_complete_ns;
        rec.ts_bits = timestamp_valid_bits;
        rec.cmd_count = tl_cmd_count;
        rec.layer_count = tl_layer_count;
        rec.truncated_spans = tl_truncated_spans;
        if (tl_cb_q_begin < timestamp_cursor && tl_cb_q_end < timestamp_cursor) {
            rec.gpu_cb_begin_ticks = raw_ts[tl_cb_q_begin];
            rec.gpu_cb_end_ticks = raw_ts[tl_cb_q_end];
            rec.cb_valid = true;
            if (stream != nullptr && stream->online_timing_enabled()) {
                stream->record_online_cb_timing(
                    rec.gpu_cb_begin_ticks, rec.gpu_cb_end_ticks,
                    timestamp_valid_bits, device.properties().limits.timestampPeriod);
            }
        }
        rec.layers.reserve(tl_layers.size());
        for (auto &layer : tl_layers) {
            TimelineSubmitRecord::ResolvedLayer resolved{};
            resolved.layer_idx = layer.layer_idx;
            resolved.cmds = std::move(layer.cmds);
            if (layer.q_begin < timestamp_cursor && layer.q_end < timestamp_cursor) {
                resolved.gpu_begin_ticks = raw_ts[layer.q_begin];
                resolved.gpu_end_ticks = raw_ts[layer.q_end];
                resolved.valid = true;
            }
            rec.layers.emplace_back(std::move(resolved));
        }
        device.gpu_timeline()->record_submit(std::move(rec));
    } else if (has_submitted_timestamps && timestamp_pool != VK_NULL_HANDLE &&
               stream != nullptr &&
               tl_cb_q_begin < timestamp_cursor && tl_cb_q_end < timestamp_cursor &&
               tl_cb_q_begin == 0u && tl_cb_q_end == 1u) {
        uint64_t raw2[2]{0ull, 0ull};
        VK_CHECK_RESULT(vkGetQueryPoolResults(
            device.logic_device(),
            timestamp_pool,
            0u,
            2u,
            sizeof(raw2),
            raw2,
            sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        stream->record_online_cb_timing(
            raw2[0], raw2[1],
            timestamp_valid_bits, device.properties().limits.timestampPeriod);
    }
    has_submitted_timestamps = false;
    timestamp_cursor = 0u;
    tl_cb_q_begin = 0xffffffffu;
    tl_cb_q_end = 0xffffffffu;
    tl_cmd_count = 0u;
    tl_layer_count = 0u;
    tl_truncated_spans = 0u;
    tl_layers.clear();
}
// `reset_native` runs everything else, i.e. the recycling work whose effect is
// only observable by the next recording of this state: releasing the retained
// pipelines, dropping the staging allocations, resetting (or rewinding) the
// descriptor pools and forgetting the tracked bind state. Nothing here is
// visible to the host, so the retirement thread runs it *after* it notified
// `Event`, while the caller thread is already encoding the next batch.
void CommandBufferState::reset_native(
    Stream *stream,
    Device &device,
    temp_buffer::DeferredStagingFrees *deferred_frees) {
    // The command buffer that consumed the indirect copy records has completed
    // on the device (or was never submitted), so the record namespace of the
    // command array can start over.
    if (indirect_copy_commands) {
        indirect_copy_commands->reset_record_cursor();
    }
    auto *deferred_upload = deferred_frees ? &deferred_frees->upload_buffers : nullptr;
    auto *deferred_readback = deferred_frees ? &deferred_frees->readback_buffers : nullptr;
    size_t touched_overflow = std::min(active_desc_pool_index, overflow_desc_pools.size());
    for (auto *r : retained_pipelines) {
        r->release();
    }
    retained_pipelines.clear();
    // Phase 3B (opt B): the membership set and the fast-path pointer describe the
    // vector that was just emptied, so they are dropped with it.
    retained_pipeline_set.clear();
    _last_retained_pipeline = nullptr;
    upload_alloc.clear(deferred_upload);
    readback_alloc.clear(deferred_readback);
    // Phase 3B (opt A): the pools only have to be reset when something outside the
    // cache was allocated from them, when an overflow pool was used (a reset pool
    // cannot hand out its sets again before they are re-allocated, and the cache
    // does not record which pool a set came from), or when a descriptor-set layout
    // was destroyed since the cache was built (every set created from it is dead).
    // Otherwise rewinding the cursors is exactly equivalent to a pool reset for
    // every set the cache hands out, and the native call is skipped.
    auto const layout_generation = device.shader_layout_destroy_generation();
    auto need_pool_reset = detail::should_reset_descriptor_pools(
        has_uncached_desc_allocs, active_desc_pool_index, orphaned_desc_sets);
    if (!need_pool_reset && cached_layout_generation != layout_generation) {
        // Phase 3B (opt A): only the buckets whose layout really died are dropped. The
        // device logs destroyed layout ids in generation order, so this is a bounded
        // walk over the layouts destroyed since the previous recycle instead of a
        // flush of every bucket the state owns (a shader destruction used to cost the
        // whole cache, which is what kept the reuse rate at ~29% in the decode loop).
        device.collect_destroyed_layout_ids(
            cached_layout_generation, destroyed_layout_scratch);
        for (auto id : destroyed_layout_scratch) {
            // Phase 3C (defect 3): the sets of a dropped bucket are still native
            // sets of this state's pools. They can never be reused (the layout they
            // were created from is gone and the cache no longer references them),
            // so they are counted as orphans; the pool reset below is what actually
            // reclaims them, and the decision below is taken again *after* they were
            // added, so crossing the threshold in one retirement resets right away.
            if (auto iter = desc_set_cache.find(id)) {
                orphaned_desc_sets += iter.value().sets.size();
                desc_set_cache.remove(id);
            }
        }
        destroyed_layout_scratch.clear();
        cached_layout_generation = layout_generation;
        // Re-evaluate with the orphans just counted: a retirement that crosses the
        // threshold must reset now, not one retirement later.
        need_pool_reset = detail::should_reset_descriptor_pools(
            has_uncached_desc_allocs, active_desc_pool_index,
            orphaned_desc_sets);
    }
    if (!need_pool_reset) {
        auto const resource_generation = device.resource_destroy_generation();
        bool const clear_cached_bindings =
            resource_generation != cached_resource_generation;
        for (auto &entry : desc_set_cache) {
            entry.second.cursor = 0u;
            if (clear_cached_bindings) {
                for (auto &item : entry.second.sets) {
                    item.bindings.clear();
                }
            }
        }
        cached_resource_generation = resource_generation;
    } else {
        desc_set_cache.clear();
        has_uncached_desc_allocs = false;
        orphaned_desc_sets = 0u;
        cached_layout_generation = layout_generation;
        cached_resource_generation = device.resource_destroy_generation();
        VK_CHECK_RESULT(vkResetDescriptorPool(device.logic_device(), desc_pool, 0));
        for (size_t i = 0u; i < touched_overflow; ++i) {
            VK_CHECK_RESULT(vkResetDescriptorPool(device.logic_device(), overflow_desc_pools[i], 0));
        }
    }
    active_desc_pool_index = 0u;
    // The descriptor sets the tracked bindings refer to are gone with the pools
    // and the recorded command buffer is retired, so nothing may stay cached
    // (plan P1.2). `CommandBuffer::begin` resets this again for the next
    // recording; resetting here as well keeps a state that is retired but never
    // re-recorded honest.
    bind_state.reset();
}
void CommandBufferState::reset(
    Stream *stream,
    Device &device,
    temp_buffer::DeferredStagingFrees *deferred_frees) {
    // The unsplit form, kept for the paths that must retire a state synchronously
    // (a batch that was never submitted, and the replay/measurement helpers). It is
    // exactly the two halves in their original order.
    reset_host_visible(stream, device);
    reset_native(stream, device, deferred_frees);
}
bool CommandBuffer::retire_and_recycle(temp_buffer::DeferredStagingFrees *deferred_frees) {
    // A retired batch always holds both halves of its native state. The guard is
    // what the retirement thread relies on after a move: `release_native_batch()`
    // empties this object (so the graph owns the state) and a moved-from or
    // already-released buffer must be neither reset nor recycled - and must never
    // reach `*device()` / `_state->reset()`, which would be null dereferences.
    if (_cmdbuffer == VK_NULL_HANDLE || _state == nullptr) { return false; }
    // A recording is handed straight to its graph and never enters the retirement
    // FIFO, so its state is never reset here. If it ever did, `_state->reset()`
    // below would wipe the descriptor pools of a command buffer a graph resubmits.
    LUISA_ASSERT(!detail::may_release_native_batch(_ownership),
                 "Vulkan: a replay recording (PERSISTENT) must never be retired "
                 "and recycled by a stream; its native state belongs to a replay "
                 "graph.");
    auto retirement = detail::plan_command_buffer_retirement(_ownership);
    if (retirement.reset_native_buffer) {
        VK_CHECK_RESULT(vkResetCommandBuffer(_cmdbuffer, 0));
    }
    _state->reset(&_stream, *device(), deferred_frees);
    return retirement.recycle_native_buffer;
}
void CommandBuffer::retire_host_visible() {
    // Phase 3B (opt C), phase 1 of the retirement: what `Stream::sync()` must have
    // observed before it returns. Guarded exactly like `retire_and_recycle`, so a
    // moved-from or already-released buffer is a no-op.
    if (_cmdbuffer == VK_NULL_HANDLE || _state == nullptr) { return; }
    LUISA_ASSERT(!detail::may_release_native_batch(_ownership),
                 "Vulkan: a replay recording (PERSISTENT) must never be retired "
                 "and recycled by a stream; its native state belongs to a replay "
                 "graph.");
    _state->reset_host_visible(&_stream, *device());
}
bool CommandBuffer::recycle_native(
    temp_buffer::DeferredStagingFrees *deferred_frees) {
    // Phase 3B (opt C), phase 2: the part of the retirement the host cannot observe,
    // run by the retirement thread after it notified `Event`.
    if (_cmdbuffer == VK_NULL_HANDLE || _state == nullptr) { return false; }
    LUISA_ASSERT(!detail::may_release_native_batch(_ownership),
                 "Vulkan: a replay recording (PERSISTENT) must never be retired "
                 "and recycled by a stream; its native state belongs to a replay "
                 "graph.");
    auto retirement = detail::plan_command_buffer_retirement(_ownership);
    if (retirement.reset_native_buffer) {
        VK_CHECK_RESULT(vkResetCommandBuffer(_cmdbuffer, 0));
    }
    _state->reset_native(&_stream, *device(), deferred_frees);
    return retirement.recycle_native_buffer;
}

Stream::Stream(Device *device, StreamTag tag)
    : Resource{device},
      _evt(device),
      // Phase 3C (defect 1): the initializer list is in *declaration* order now
      // (`stream.h`: `_evt`, `_temp_desc`, `_scratch_buffer_alloc`, `_stream_tag`,
      // `logger`, `reorder`, `argument_scratch`, `_retire_spin_iterations`,
      // `_work_epoch`, `_thd`). That ordering is what makes the last three safe:
      // the thread function starts running inside `_thd`'s constructor and reads
      // `_retire_spin_iterations` and `_work_epoch` immediately, so both must
      // already be constructed - which they are exactly when the list follows the
      // declaration order. Before this fix `_work_epoch` was declared *after*
      // `_thd` (member-initialization-order UB) and `_retire_spin_iterations` was
      // initialized after it (the thread therefore read 0).
      _temp_desc(65536, &_temp_desc_visitor, 2),
      _scratch_buffer_alloc(kTempSize, &_scratch_buffer_alloc_visitor),
      _stream_tag(tag),
      logger([](luisa::string_view str) {
          LUISA_INFO("[DEVICE] {}", str);
      }),
      reorder({}),
      _retire_spin_iterations(
          pause_iterations_from_us("LUISA_VK_RETIRE_SPIN_US", 200u)),
      _thd([this]() {
          temp_buffer::DeferredStagingFrees deferred_staging_frees;
          // Phase 3B (opt C): command buffers whose host-visible half already ran but
          // whose native recycling is deferred until after the notification.
          vstd::vector<CommandBuffer> pending_recycle;
          auto recycle_pending = [&]() {
              for (auto &b : pending_recycle) {
                  if (b.recycle_native(&deferred_staging_frees)) {
                      _cmdbuffers.enqueue(std::move(b));
                  }
              }
              pending_recycle.clear();
              if (!deferred_staging_frees.empty()) {
                  deferred_staging_frees.clear();
              }
          };
          uint64_t last_sync_complete_ns = 0u;
          auto loop_cmd = [&]() {
              while (true) {
                  _mtx.lock();
                  auto p = _exec.dequeue();
                  _mtx.unlock();
                  if (!p) {
                      break;
                  }
                  p->visit(
                      [&]<typename T>(T &t) {
                          if constexpr (std::is_same_v<T, Callbacks>) {
                              for (auto &i : t) {
                                  i();
                              }
                          } else if constexpr (std::is_same_v<T, SyncExt>) {
                              t.evt->_host_wait(t.value);
                              if (this->device()->gpu_timeline() != nullptr) {
                                  last_sync_complete_ns = VkGpuTimeline::steady_ns();
                              }
                          } else if constexpr (std::is_same_v<T, NotifyEvt>) {
                              // Phase 3B (opt C): notify first, recycle after. The host-visible half of
                              // every batch this notification covers already ran (a `CommandBuffer`
                              // entry always precedes the `NotifyEvt` of its batch), so `Stream::sync()`
                              // may return now; the recycling that follows is invisible to it and
                              // overlaps with the caller encoding its next batch.
                              t.evt->_notify(t.value);
                              recycle_pending();
                          } else if constexpr (std::is_same_v<T, CommandBuffer>) {
                              if (this->device()->gpu_timeline() != nullptr) {
                                  t.states()->tl_cpu_fence_complete_ns = last_sync_complete_ns;
                              }
                              auto const retired = t.states()->host_retire_fence;
                              t.states()->host_retire_fence = 0u;
                              t.retire_host_visible();
                              if (retired != 0u) {
                                  auto prev = _host_retired_fence.load(std::memory_order_relaxed);
                                  while (prev < retired &&
                                         !_host_retired_fence.compare_exchange_weak(
                                             prev, retired,
                                             std::memory_order_release,
                                             std::memory_order_relaxed)) {
                                  }
                                  _host_retired_fence.notify_all();
                              }
                              pending_recycle.emplace_back(std::move(t));
                          }
                      });
              }
          };
          // Phase 3B (opt C) / phase 3C (defect 2): the idle protocol.
          //
          // The loop is driven by one number, `_work_epoch`, which every producer
          // bumps *under `_mtx`* after it enqueued into `_exec` and notifies after
          // unlocking. Consequences:
          //   * the thread never touches `_exec` outside `_mtx` (the previous
          //     `_exec.length()` polls here were an unsynchronized read racing the
          //     producers' `enqueue`), it only reads the atomic epoch;
          //   * a wakeup cannot be lost: the epoch is sampled *before* the last
          //     `loop_cmd()` of an iteration - i.e. before the point at which a
          //     producer could enqueue something this iteration does not see - and
          //     the thread only blocks while the epoch still holds that value. A
          //     producer that bumps between the sample and the wait turns the `wait`
          //     into an immediate return, and a producer that bumped *before* the
          //     sample is caught by that `loop_cmd()`.
          //   * `~Stream()` therefore cannot strand the thread: it flips `_enabled`
          //     and bumps the epoch under the same lock, and the thread re-checks
          //     `_enabled` both before and after the blocking wait.
          while (_enabled.load(std::memory_order_acquire)) {
              auto const epoch_before = _work_epoch.load(std::memory_order_acquire);
              loop_cmd();
              // Poll briefly (default LUISA_VK_RETIRE_SPIN_US = 200 us) so the
              // microsecond-scale handoff of a finished batch never pays a kernel
              // transition, then block.
              for (auto i = 0u; i < _retire_spin_iterations; ++i) {
                  if (_work_epoch.load(std::memory_order_acquire) != epoch_before ||
                      !_enabled.load(std::memory_order_acquire)) {
                      break;
                  }
                  LUISA_INTRIN_PAUSE();
              }
              if (_enabled.load(std::memory_order_acquire) &&
                  _work_epoch.load(std::memory_order_acquire) == epoch_before) {
                  _work_epoch.wait(epoch_before, std::memory_order_acquire);
              }
          }
          loop_cmd();
          recycle_pending();
          deferred_staging_frees.clear();
      }) {
    switch (tag) {
        case StreamTag::GRAPHICS:
            _queue = device->graphics_queue();
            _resource_barrier.queue_type = ResourceBarrier::QueueType::GRAPHICS;
            _resource_barrier.device = device;
            _queue_mtx = &device->graphics_queue_mtx();
            break;
        case StreamTag::COPY:
            _resource_barrier.queue_type = ResourceBarrier::QueueType::COPY;
            _resource_barrier.device = device;
            _queue = device->copy_queue();
            _queue_mtx = &device->copy_queue_mtx();
            break;
        case StreamTag::COMPUTE:
            _resource_barrier.queue_type = ResourceBarrier::QueueType::COMPUTE;
            _resource_barrier.device = device;
            _queue = device->compute_queue();
            _queue_mtx = &device->compute_queue_mtx();
            break;
        default:
            LUISA_ERROR("Illegal stream tag.");
    }
}
Stream::~Stream() {
    sync();
    {
        std::lock_guard lock{_pending_compaction_mtx};
        for (auto *b : _pending_compaction_blases) {
            if (b != nullptr) {
                b->_set_pending_compaction_stream(nullptr);
            }
        }
        _pending_compaction_blases.clear();
    }
    // Phase 3B (opt C) / phase 3C (defect 2): the retirement thread may be parked on
    // the wakeup epoch, and `_enabled = false` alone cannot release a blocked `wait`.
    // The flag flip and the epoch bump therefore happen under the same lock the
    // producers take, so the thread either sees `_enabled == false` in its own
    // re-check or observes the bumped epoch (which makes the `wait` return
    // immediately); the notification that follows the unlock is what actually wakes a
    // parked thread (and cannot be lost: the epoch already moved).
    {
        std::lock_guard lck{_mtx};
        _enabled.store(false, std::memory_order_release);
        _bump_work_epoch_locked();
    }
    _notify_retirement_thread();
    _thd.join();
    _scratch_buffer_alloc_visitor.buffers.clear();
    while (auto p = _cmdbuffers.dequeue()) {
    }
}

void Stream::register_pending_compaction(Blas *blas) noexcept {
    if (blas == nullptr) { return; }
    Stream *prev = nullptr;
    {
        std::lock_guard lock{_pending_compaction_mtx};
        prev = blas->_pending_compaction_stream;
        if (prev == this) { return; }
    }
    if (prev != nullptr) {
        prev->unregister_pending_compaction(blas);
    }
    std::lock_guard lock{_pending_compaction_mtx};
    if (blas->_pending_compaction_stream == this) { return; }
    blas->_set_pending_compaction_stream(this);
    _pending_compaction_blases.emplace_back(blas);
}

void Stream::unregister_pending_compaction(Blas *blas) noexcept {
    if (blas == nullptr) { return; }
    std::lock_guard lock{_pending_compaction_mtx};
    for (size_t i = 0u; i < _pending_compaction_blases.size();) {
        if (_pending_compaction_blases[i] == blas) {
            _pending_compaction_blases[i] = _pending_compaction_blases.back();
            _pending_compaction_blases.pop_back();
        } else {
            ++i;
        }
    }
    if (blas->_pending_compaction_stream == this) {
        blas->_set_pending_compaction_stream(nullptr);
    }
}

uint64_t Stream::execute_immediate(
    luisa::move_only_function<void(CommandBuffer &)> &&fn,
    bool wait_after) {
    uint64_t fence = 0u;
    {
        std::lock_guard lck{_dispatch_mtx};
        _flush_pending_waits();
        _temp_desc.clear();
        _throttle_in_flight();
        auto fence_plan = detail::plan_timeline_value_increment(
            _evt.last_fence(), 1u);
        LUISA_ASSERT(
            static_cast<bool>(fence_plan),
            "Vulkan stream timeline fence overflow during execute_immediate.");
        fence = fence_plan.value;
        submit_profile::record_command_batch();
        auto profile_flush = vstd::scope_exit([] {
            submit_profile::flush();
        });
        CommandBuffer cmdbuffer = _begin_batch_record();
        fn(cmdbuffer);
        cmdbuffer.states()->upload_alloc.flush_host();
        cmdbuffer.states()->readback_alloc.flush_host();
        _finish_batch_record(cmdbuffer);
        auto cb = cmdbuffer.cmdbuffer();
        auto cb_ptr = &cb;
        if (_execute_external_command_buffer(cb)) {
            cb_ptr = nullptr;
        }
        _evt._signal(*this, fence, cb_ptr);
        _mtx.lock();
        _exec.enqueue(SyncExt{
            .evt = &_evt,
            .value = fence});
        cmdbuffer.states()->host_retire_fence = fence;
        _exec.enqueue(std::move(cmdbuffer));
        _exec.enqueue(NotifyEvt{
            .evt = &_evt,
            .value = fence});
        // Phase 3B (opt C) / phase 3C (defect 2): publish the new work while still
        // holding `_mtx` (the enqueue and the epoch bump are then one indivisible step
        // for the retirement thread) and notify only after the lock is released.
        _bump_work_epoch_locked();
        _mtx.unlock();
        _notify_retirement_thread();
    }
    if (wait_after) {
        _evt.sync(fence);
    }
    return fence;
}

void Stream::remove_resource_state(Resource const *resource) noexcept {
    std::lock_guard lck{_dispatch_mtx};
    _resource_barrier.remove_resource(resource);
}

bool Stream::_execute_external_command_buffer(
    VkCommandBuffer command_buffer) noexcept {
    auto config_ext = device()->config_ext();
    if (config_ext == nullptr) { return false; }
    std::lock_guard queue_lock{*_queue_mtx};
    return config_ext->execute_command_buffer(command_buffer);
}

void Stream::_throttle_in_flight() {
    auto plan = detail::plan_in_flight_throttle(
        _evt.last_fence(), device()->max_frames_in_flight());
    if (!plan.enter) { return; }
    // Counter-only (P0.2/P1.5): every throttle decision is an entry, only the
    // decisions that really block the host are waits, and the ones the GPU
    // completion watermark proved unnecessary are skips, so
    // entries == waits + skips always holds.
    submit_profile::record_timeline_wait_entry();
    if (_evt.gpu_completed(plan.target)) {
        // Fast path (plan P1.5): the throttle exists to bound how far the GPU may
        // run ahead of the host, and the GPU having completed `target` proves the
        // bound already holds, so blocking the host would buy nothing.
        //
        // Why skipping cannot outlive anything: this wait never owned command
        // buffer or callback lifetime. The batch that produced `target` is
        // recycled by the stream's retirement thread, which walks one FIFO of
        // per-batch entries (host wait on the timeline -> command-buffer retire
        // and recycle -> notification) in submission order; a command buffer only
        // re-enters `_cmdbuffers` after its own GPU completion, and a submitter
        // that finds the pool empty allocates a fresh one, so nothing can be
        // reused early. User callbacks and the deferred staging frees run in that
        // same thread after the same completion, so skipping a host wait here
        // cannot delay or reorder them. The paths that must observe retirement -
        // public Event::sync, Stream::sync, the sparse-binding host wait and
        // ~Event - still wait for it.
        submit_profile::record_timeline_wait_skip();
        return;
    }
    submit_profile::record_timeline_wait();
    _evt.sync(plan.target);
}

void Stream::present(
    Texture const *tex,
    uint mip,
    Swapchain *swapchain,
    bool inqueue_limit) {
    std::lock_guard lck{_dispatch_mtx};
    _flush_pending_waits();
    _temp_desc.clear();
    if (inqueue_limit) {
        // Same in-flight bound as the dispatch path (plan P1.5); the engine, the
        // counters and the fast path live in `_throttle_in_flight`.
        _throttle_in_flight();
    }
    auto fence_plan = detail::plan_timeline_value_increment(
        _evt.last_fence(), 1u);
    LUISA_ASSERT(
        static_cast<bool>(fence_plan),
        "Vulkan stream timeline fence overflow during presentation.");
    auto fence = fence_plan.value;
    {
        CommandBuffer cmdbuffer = [&]() {
            auto p = _cmdbuffers.dequeue();
            if (p) return std::move(*p);
            return CommandBuffer{*this};
        }();
        _scratch_buffer_alloc_visitor.cmdbuffer = &cmdbuffer;
        _scratch_buffer_alloc_visitor.device = device();

        cmdbuffer.resource_barrier = &_resource_barrier;
        cmdbuffer.uniform_data = &_uniform_data;
        cmdbuffer.desc_sets = &_desc_sets;
        cmdbuffer.logger = logger ? &logger : nullptr;
        cmdbuffer.dispatch_offsets = &_dispatch_offsets;
        cmdbuffer.write_desc_sets = &_write_desc_sets;
        cmdbuffer.bindless_cache = &_bindless_cache;
        cmdbuffer.temp_desc = &_temp_desc;
        cmdbuffer.scratch_buffer_alloc = &_scratch_buffer_alloc;
        cmdbuffer.begin();
        PresentCommand present_cmd;
        present_cmd.submit_wait_semaphores.emplace_back();
        present_cmd.signal_semaphores.emplace_back();
        present_cmd.wait_stages.emplace_back();
        present_cmd.present_wait_semaphores.emplace_back();
        present_cmd.image_indices.emplace_back();
        VkFence vk_fence{};
        swapchain->present(
            cmdbuffer,
            present_cmd.submit_wait_semaphores.back(), present_cmd.signal_semaphores.back(),
            present_cmd.wait_stages.back(),
            present_cmd.present_wait_semaphores.back(),
            present_cmd.image_indices.back(),
            tex,
            vk_fence,
            mip);

        _resource_barrier.restore_states(cmdbuffer.cmdbuffer());
        cmdbuffer.end();

        // If fence is null, swapchain was recreated and we should skip this frame
        if (vk_fence == nullptr) {
            // Cleanup command buffer without submitting
            cmdbuffer.states()->reset(this, *device());
            // Update fence and enqueue notification to match normal completion path
            _evt._update_fence(fence);
            _mtx.lock();
            _exec.enqueue(NotifyEvt{
                .evt = &_evt,
                .value = fence});
            // Phase 3B (opt C) / phase 3C (defect 2): publish the new work while still
            // holding `_mtx` (the enqueue and the epoch bump are then one indivisible step
            // for the retirement thread) and notify only after the lock is released.
            _bump_work_epoch_locked();
            _mtx.unlock();
            _notify_retirement_thread();
            return;
        }

        {
            auto producer_wait_plan =
                detail::plan_internal_timeline_wait(
                    _evt.last_fence(),
                    _evt.last_signaled_fence());
            LUISA_ASSERT(
                static_cast<bool>(producer_wait_plan),
                "Vulkan stream GPU signal is ahead of its logical fence.");
            auto producer_wait_value = producer_wait_plan.wait_value;
            auto producer_semaphore = _evt.semaphore();
            luisa::fixed_vector<VkSemaphore, 2> wait_semaphores;
            luisa::fixed_vector<VkPipelineStageFlags, 2> wait_stages_all;
            luisa::fixed_vector<uint64_t, 2> wait_values;
            for (size_t i = 0; i < present_cmd.submit_wait_semaphores.size(); ++i) {
                wait_semaphores.emplace_back(present_cmd.submit_wait_semaphores[i]);
                wait_stages_all.emplace_back(present_cmd.wait_stages[i]);
                wait_values.emplace_back(0u);
            }
            if (producer_wait_value > 0u && producer_semaphore != VK_NULL_HANDLE) {
                wait_semaphores.emplace_back(producer_semaphore);
                wait_stages_all.emplace_back(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
                wait_values.emplace_back(producer_wait_value);
            }
            VkTimelineSemaphoreSubmitInfo timeline_info{};
            timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            timeline_info.waitSemaphoreValueCount = static_cast<uint32_t>(wait_values.size());
            timeline_info.pWaitSemaphoreValues = wait_values.data();
            timeline_info.signalSemaphoreValueCount = static_cast<uint32_t>(present_cmd.signal_semaphores.size());
            // The swapchain's signal semaphores are binary, so their timeline value
            // is zero. Sized to the count instead of a single literal: the count and
            // the array are validated against each other by the submit conversion,
            // and a one-element array cannot silently describe two semaphores.
            luisa::fixed_vector<uint64_t, 2> signal_values;
            signal_values.resize(present_cmd.signal_semaphores.size(), 0u);
            timeline_info.pSignalSemaphoreValues = signal_values.empty() ? nullptr : signal_values.data();
            auto _cmdbuffer = cmdbuffer.cmdbuffer();
            // The submission in the backend's legacy vocabulary. `detail::queue_submit`
            // issues it as vkQueueSubmit2 with per-semaphore 64-bit stage masks on a
            // sync2 device, and as the identical legacy vkQueueSubmit otherwise
            // (plan P2.1). The extra frame fences of the present batch are retired by
            // fence-only submissions (`command_buffer_count == 0`), exactly as before.
            detail::QueueSubmitRequest request{};
            if (!_execute_external_command_buffer(_cmdbuffer)) {
                request.command_buffers = &_cmdbuffer;
                request.command_buffer_count = 1u;
            }
            request.wait_semaphores = wait_semaphores.empty() ? nullptr : wait_semaphores.data();
            request.wait_stage_masks = wait_stages_all.empty() ? nullptr : wait_stages_all.data();
            request.wait_count = static_cast<uint32_t>(wait_semaphores.size());
            request.signal_semaphores = present_cmd.signal_semaphores.empty() ? nullptr : present_cmd.signal_semaphores.data();
            request.signal_count = static_cast<uint32_t>(present_cmd.signal_semaphores.size());
            request.fence = vk_fence;
            request.pnext = &timeline_info;
            _queue_mtx->lock();
            (void)detail::queue_submit(_queue, device(), request);
            _queue_mtx->unlock();
        }
        {
            VkPresentInfoKHR present_info{};
            present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            auto swp_ptr = swapchain->swapchain();
            present_info.pSwapchains = &swp_ptr;
            present_info.swapchainCount = 1u;
            present_info.waitSemaphoreCount = present_cmd.present_wait_semaphores.size();
            present_info.pWaitSemaphores = present_cmd.present_wait_semaphores.data();
            present_info.pImageIndices = present_cmd.image_indices.data();
            _queue_mtx->lock();
            auto result = vkQueuePresentKHR(_queue, &present_info);
            _queue_mtx->unlock();
            if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
                swapchain->handle_present_error();
            } else if (result != VK_SUCCESS) {
                LUISA_ERROR_WITH_LOCATION("Failed to present swapchain image: {}.", string_VkResult(result));
            }
        }
        _evt._signal(*this, fence);
        _mtx.lock();
        _exec.enqueue(SyncExt{
            .evt = &_evt,
            .value = fence});
        cmdbuffer.states()->host_retire_fence = fence;
        _exec.enqueue(std::move(cmdbuffer));
        _exec.enqueue(NotifyEvt{
            .evt = &_evt,
            .value = fence});

        // Phase 3B (opt C) / phase 3C (defect 2): publish the new work while still
        // holding `_mtx` (the enqueue and the epoch bump are then one indivisible step
        // for the retirement thread) and notify only after the lock is released.
        _bump_work_epoch_locked();
        _mtx.unlock();
        _notify_retirement_thread();
    }
}
void Stream::_wait_host_retired_fence(uint64_t target) noexcept {
    for (;;) {
        auto cur = _host_retired_fence.load(std::memory_order_acquire);
        if (cur >= target) { break; }
        _host_retired_fence.wait(cur, std::memory_order_acquire);
    }
}

void Stream::dispatch(
    vstd::span<const luisa::unique_ptr<Command>> cmds,
    luisa::vector<luisa::move_only_function<void()>> &&callbacks,
    vstd::span<const SwapchainPresent> presents,
    bool inqueue_limit) {
    std::lock_guard lck{_dispatch_mtx};
    auto wait_if_upload_overlaps = [&](vstd::span<const luisa::unique_ptr<Command>> sub_cmds) {
        auto const retired = _host_retired_fence.load(std::memory_order_acquire);
        luisa::compute::erase_host_transfer_if(
            _pending_host_downloads,
            [retired](const PendingHostDownload &entry) noexcept {
                return entry.fence <= retired;
            });
        if (_pending_host_downloads.empty() || sub_cmds.empty()) {
            return;
        }
        luisa::fixed_vector<HostByteRange, 8u> upload_ranges;
        for (auto const &c : sub_cmds) {
            if (auto const ul = command_host_upload_range(c.get()); !ul.empty()) {
                upload_ranges.push_back(ul);
            }
        }
        if (upload_ranges.empty()) {
            return;
        }
        uint64_t max_wait = 0u;
        for (auto const &entry : _pending_host_downloads) {
            for (auto const &ul : upload_ranges) {
                if (entry.range.overlaps(ul)) {
                    max_wait = std::max(max_wait, entry.fence);
                    break;
                }
            }
        }
        if (max_wait > 0u) {
            _wait_host_retired_fence(max_wait);
            auto const updated = _host_retired_fence.load(std::memory_order_acquire);
            luisa::compute::erase_host_transfer_if(
                _pending_host_downloads,
                [updated](const PendingHostDownload &entry) noexcept {
                    return entry.fence <= updated;
                });
        }
    };
    auto record_downloads = [&](vstd::span<const luisa::unique_ptr<Command>> sub_cmds, uint64_t submitted_fence) {
        if (submitted_fence == 0u || sub_cmds.empty()) {
            return;
        }
        for (auto const &c : sub_cmds) {
            if (auto const dl = command_host_download_range(c.get()); !dl.empty()) {
                _pending_host_downloads.emplace_back(PendingHostDownload{dl, submitted_fence});
            }
        }
    };
    if (cmds.empty()) {
        static_cast<void>(_dispatch_chunk(cmds, std::move(callbacks), presents, inqueue_limit));
        return;
    }
    size_t cursor = 0u;
    while (cursor < cmds.size()) {
        auto const next = find_next_host_transfer_split(
            luisa::span{cmds.data(), cmds.size()}, cursor);
        auto chunk = cmds.subspan(cursor, next - cursor);
        bool const is_last = (next == cmds.size());
        wait_if_upload_overlaps(chunk);
        auto submitted = _dispatch_chunk(
            chunk,
            is_last ? std::move(callbacks) : Callbacks{},
            is_last ? presents : vstd::span<const SwapchainPresent>{},
            inqueue_limit);
        record_downloads(chunk, submitted);
        cursor = next;
    }
}

uint64_t Stream::_dispatch_chunk(
    vstd::span<const luisa::unique_ptr<Command>> cmds,
    luisa::vector<luisa::move_only_function<void()>> &&callbacks,
    vstd::span<const SwapchainPresent> presents,
    bool inqueue_limit) {
    PresentCommand present_cmd;
    luisa::fixed_vector<VkSwapchainKHR, 1> vk_swapchains;
    _temp_desc.clear();
    luisa::fixed_vector<EventSignalEntry, 4> extra_signals;
    bool has_gpu_cmds = false;
    for (auto const &c : cmds) {
        if (c->tag() == Command::Tag::ECustomCommand) {
            auto *custom_cmd = static_cast<CustomCommand const *>(c.get());
            if (custom_cmd->custom_cmd_uuid() ==
                luisa::to_underlying(CustomCommandUUID::STREAM_SIGNAL_EVENT)) {
                auto *sc = static_cast<StreamSignalEventCommand const *>(custom_cmd);
                if (sc->event_handle() != 0ull) {
                    extra_signals.emplace_back(EventSignalEntry{
                        reinterpret_cast<Event *>(sc->event_handle()),
                        sc->fence_value()});
                }
                continue;
            }
        }
        has_gpu_cmds = true;
    }
    if (!has_gpu_cmds && extra_signals.empty() && callbacks.empty() && presents.empty()) {
        return 0u;
    }
    if (inqueue_limit) {
        // Same in-flight bound as the dispatch path (plan P1.5); the engine, the
        // counters and the fast path live in `_throttle_in_flight`.
        _throttle_in_flight();
    }
    auto fence_plan = detail::plan_timeline_value_increment(
        _evt.last_fence(), 1u);
    LUISA_ASSERT(
        static_cast<bool>(fence_plan),
        "Vulkan stream timeline fence overflow during dispatch.");
    auto fence = fence_plan.value;
    if (has_gpu_cmds || !presents.empty()) {
        if (!presents.empty() || device()->config_ext() != nullptr) {
            _flush_pending_waits();
        }
        auto *tl = device()->gpu_timeline();
        uint64_t tl_record_begin_ns =
            tl != nullptr ? VkGpuTimeline::steady_ns() : 0u;
        // Counter-only: one command batch = one command-buffer record plus (at
        // most) one submit (F6).
        submit_profile::record_command_batch();
        auto profile_flush = vstd::scope_exit([] {
            submit_profile::flush();
        });
        // Plan P4: the record prefix is shared with the replay recorder
        // (`_begin_batch_record`), so an ordinary batch and a recording cannot
        // drift apart in barriers, config-ext states, wired scratch or begin info.
        CommandBuffer cmdbuffer = _begin_batch_record();

        auto cb = cmdbuffer.cmdbuffer();
        auto cb_ptr = &cb;
        // Counter-only: host encode time of this batch, begin -> end (P0.2).
        auto const profile_encode_begin = submit_profile::timing_begin();
        if (tl != nullptr) {
            auto *st = cmdbuffer.states();
            st->has_submitted_timestamps = false;
            st->timestamp_cursor = 0u;
            st->tl_cb_q_begin = 0xffffffffu;
            st->tl_cb_q_end = 0xffffffffu;
            st->tl_cmd_count = 0u;
            st->tl_layer_count = 0u;
            st->tl_truncated_spans = 0u;
            st->tl_layers.clear();
            st->tl_cpu_record_begin_ns = tl_record_begin_ns;
            if (st->timestamp_pool != VK_NULL_HANDLE) {
                vkCmdResetQueryPool(cb, st->timestamp_pool, 0u, st->timestamp_pool_capacity);
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, st->timestamp_pool, 0u);
                st->tl_cb_q_begin = 0u;
                st->timestamp_cursor = 1u;
            }
        } else if (online_timing_enabled()) {
            auto *st = cmdbuffer.states();
            st->has_submitted_timestamps = false;
            st->timestamp_cursor = 0u;
            st->tl_cb_q_begin = 0xffffffffu;
            st->tl_cb_q_end = 0xffffffffu;
            st->ensure_timestamp_pool(*device(), _stream_tag, 2u);
            if (st->timestamp_pool != VK_NULL_HANDLE) {
                vkCmdResetQueryPool(cb, st->timestamp_pool, 0u, 2u);
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, st->timestamp_pool, 0u);
                st->tl_cb_q_begin = 0u;
                st->timestamp_cursor = 1u;
            }
        }
        cmdbuffer.execute(cmds);
        bool present_failed = false;
        for (auto &i : presents) {
            auto swapchain = reinterpret_cast<lc::vk::Swapchain *>(i.chain->handle());
            auto tex = reinterpret_cast<Texture *>(i.frame.handle());
            auto mip = i.frame.level();

            present_cmd.submit_fences.emplace_back();
            present_cmd.submit_wait_semaphores.emplace_back();
            present_cmd.signal_semaphores.emplace_back();
            present_cmd.wait_stages.emplace_back();
            present_cmd.present_wait_semaphores.emplace_back();
            present_cmd.image_indices.emplace_back();

            swapchain->present(
                cmdbuffer,
                present_cmd.submit_wait_semaphores.back(), present_cmd.signal_semaphores.back(),
                present_cmd.wait_stages.back(),
                present_cmd.present_wait_semaphores.back(),
                present_cmd.image_indices.back(),
                tex,
                present_cmd.submit_fences.back(),
                mip);

            // If fence is null, swapchain was recreated - abort present batch
            if (present_cmd.submit_fences.back() == nullptr) {
                present_failed = true;
                break;
            }

            vk_swapchains.emplace_back(swapchain->swapchain());
        }
        // Plan P4: shared suffix (config-ext after-states + trailing restore +
        // end), extracted behaviour-neutrally out of this function.
        _finish_batch_record(cmdbuffer);
        submit_profile::timing_end_encode(profile_encode_begin);

        // One present batch submission, shared by the plain and the
        // present-failed paths so the two cannot disagree (plan P2.1). The extra
        // frame fences of the batch are retired by fence-only submissions, exactly
        // as before, and `detail::queue_submit` turns each one into vkQueueSubmit2
        // on a sync2 device and into the identical legacy vkQueueSubmit otherwise.
        auto submit_present_batch = [&](VkCommandBuffer const *command_buffer,
                                        uint32_t command_buffer_count) noexcept {
            detail::QueueSubmitRequest request{};
            request.command_buffers = command_buffer;
            request.command_buffer_count = command_buffer_count;
            request.wait_semaphores = present_cmd.submit_wait_semaphores.empty() ? nullptr : present_cmd.submit_wait_semaphores.data();
            request.wait_stage_masks = present_cmd.wait_stages.empty() ? nullptr : present_cmd.wait_stages.data();
            request.wait_count = static_cast<uint32_t>(present_cmd.submit_wait_semaphores.size());
            request.signal_semaphores = present_cmd.signal_semaphores.empty() ? nullptr : present_cmd.signal_semaphores.data();
            request.signal_count = static_cast<uint32_t>(present_cmd.signal_semaphores.size());
            request.fence = present_cmd.submit_fences[0];
            (void)detail::queue_submit(_queue, device(), request);
            for (auto i : vstd::range(1u, present_cmd.submit_fences.size())) {
                detail::QueueSubmitRequest fence_only{};
                fence_only.fence = present_cmd.submit_fences[i];
                (void)detail::queue_submit(_queue, device(), fence_only);
            }
        };
        if (!presents.empty() && !present_failed) {
            auto command_buffer = cb;
            auto command_buffer_count = 1u;
            if (_execute_external_command_buffer(cb)) {
                // External execution - submit with 0 command buffers
                command_buffer_count = 0u;
            }
            _queue_mtx->lock();
            submit_present_batch(command_buffer_count == 0u ? nullptr : &command_buffer,
                                 command_buffer_count);
            _queue_mtx->unlock();

            VkPresentInfoKHR present_info{};
            present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            present_info.pSwapchains = vk_swapchains.data();
            present_info.swapchainCount = vk_swapchains.size();
            present_info.waitSemaphoreCount = present_cmd.present_wait_semaphores.size();
            present_info.pWaitSemaphores = present_cmd.present_wait_semaphores.data();
            present_info.pImageIndices = present_cmd.image_indices.data();
            _queue_mtx->lock();
            auto result = vkQueuePresentKHR(_queue, &present_info);
            _queue_mtx->unlock();
            if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
                for (auto &i : presents) {
                    auto swapchain = reinterpret_cast<lc::vk::Swapchain *>(i.chain->handle());
                    swapchain->handle_present_error();
                }
            } else if (result != VK_SUCCESS) {
                LUISA_ERROR_WITH_LOCATION("Failed to present swapchain image: {}.", string_VkResult(result));
            }
        }
        // If present failed, submit and present the successful prefix before returning
        if (present_failed) {
            // Remove the failed swapchain's entries (last element is null/invalid)
            if (!present_cmd.submit_fences.empty()) {
                present_cmd.submit_fences.pop_back();
                present_cmd.submit_wait_semaphores.pop_back();
                present_cmd.signal_semaphores.pop_back();
                present_cmd.wait_stages.pop_back();
                present_cmd.present_wait_semaphores.pop_back();
                present_cmd.image_indices.pop_back();
            }

            // If there are successful swapchains, submit and present them
            if (!present_cmd.submit_fences.empty()) {
                auto command_buffer = cb;
                auto command_buffer_count = 1u;
                if (_execute_external_command_buffer(cb)) {
                    command_buffer_count = 0u;
                }
                _queue_mtx->lock();
                submit_present_batch(command_buffer_count == 0u ? nullptr : &command_buffer,
                                     command_buffer_count);
                _queue_mtx->unlock();

                VkPresentInfoKHR present_info{};
                present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
                present_info.pSwapchains = vk_swapchains.data();
                present_info.swapchainCount = vk_swapchains.size();
                present_info.waitSemaphoreCount = present_cmd.present_wait_semaphores.size();
                present_info.pWaitSemaphores = present_cmd.present_wait_semaphores.data();
                present_info.pImageIndices = present_cmd.image_indices.data();
                _queue_mtx->lock();
                auto result = vkQueuePresentKHR(_queue, &present_info);
                _queue_mtx->unlock();
                if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
                    // Only handle present errors for the successfully presented prefix
                    for (size_t i = 0; i < vk_swapchains.size(); ++i) {
                        auto swapchain = reinterpret_cast<lc::vk::Swapchain *>(presents[i].chain->handle());
                        swapchain->handle_present_error();
                    }
                } else if (result != VK_SUCCESS) {
                    LUISA_ERROR_WITH_LOCATION("Failed to present swapchain image: {}.", string_VkResult(result));
                }

                // Use normal completion path for the submitted prefix
                cb_ptr = nullptr;
                _evt._signal(*this, fence, cb_ptr);
                _mtx.lock();
                _exec.enqueue(SyncExt{
                    .evt = &_evt,
                    .value = fence});
                cmdbuffer.states()->host_retire_fence = fence;
                _exec.enqueue(std::move(cmdbuffer));
            } else {
                // No successful swapchains - if there are commands, submit them without present
                if (has_gpu_cmds) {
                    // Check for external command buffer execution (same as normal path)
                    auto cb_ptr = &cb;
                    if (_execute_external_command_buffer(cb)) {
                        cb_ptr = nullptr;
                    }
                    _flush_pending_waits();
                    _evt._signal(*this, fence, cb_ptr);
                    _mtx.lock();
                    _exec.enqueue(SyncExt{
                        .evt = &_evt,
                        .value = fence});
                    cmdbuffer.states()->host_retire_fence = fence;
                    _exec.enqueue(std::move(cmdbuffer));
                } else {
                    // No commands and no successful presents - cleanup without submitting
                    cmdbuffer.states()->reset(this, *device());
                    _flush_pending_waits();
                    _evt._update_fence(fence);
                    _mtx.lock();
                }
                for (auto const &s : extra_signals) {
                    if (s.evt != nullptr) {
                        s.evt->_signal(*this, s.value);
                        _exec.enqueue(SyncExt{.evt = s.evt, .value = s.value});
                        _exec.enqueue(NotifyEvt{.evt = s.evt, .value = s.value});
                    }
                }
                // Enqueue callbacks and completion notification
                if (!callbacks.empty()) {
                    _exec.enqueue(std::move(callbacks));
                }
                _exec.enqueue(NotifyEvt{
                    .evt = &_evt,
                    .value = fence});
                // Phase 3B (opt C) / phase 3C (defect 2): publish the new work while still
                // holding `_mtx` (the enqueue and the epoch bump are then one indivisible step
                // for the retirement thread) and notify only after the lock is released.
                _bump_work_epoch_locked();
                _mtx.unlock();
                _notify_retirement_thread();
                return has_gpu_cmds ? fence : 0u;
            }

            for (auto const &s : extra_signals) {
                if (s.evt != nullptr) {
                    s.evt->_signal(*this, s.value);
                    _exec.enqueue(SyncExt{.evt = s.evt, .value = s.value});
                    _exec.enqueue(NotifyEvt{.evt = s.evt, .value = s.value});
                }
            }
            // Enqueue callbacks and completion notification for successful prefix
            if (!callbacks.empty()) {
                _exec.enqueue(std::move(callbacks));
            }
            _exec.enqueue(NotifyEvt{
                .evt = &_evt,
                .value = fence});

            // Phase 3B (opt C) / phase 3C (defect 2): publish the new work while still
            // holding `_mtx` (the enqueue and the epoch bump are then one indivisible step
            // for the retirement thread) and notify only after the lock is released.
            _bump_work_epoch_locked();
            _mtx.unlock();
            _notify_retirement_thread();
            return fence;
        }
        // Command buffer already submitted in present path, don't submit again
        if (!presents.empty()) {
            cb_ptr = nullptr;
        }
        if (cb_ptr && _execute_external_command_buffer(cb)) {
            cb_ptr = nullptr;
        }
        // Counter-only: host time inside the submit call of this batch (P0.2).
        auto const profile_submit_begin = submit_profile::timing_begin();
        const bool use_fused_submit =
            (device()->config_ext() == nullptr) &&
            presents.empty() &&
            (cb_ptr != nullptr);
        if (tl != nullptr) {
            auto *st = cmdbuffer.states();
            st->tl_submit_seq = tl->alloc_submit_seq();
            st->tl_stream_handle = reinterpret_cast<uint64_t>(this);
            st->tl_stream_tag = static_cast<uint32_t>(_stream_tag);
            st->tl_fence = fence;
            st->tl_cpu_record_begin_ns = tl_record_begin_ns;
            st->tl_cpu_submit_begin_ns = VkGpuTimeline::steady_ns();
            if (use_fused_submit) {
                _evt._signal_batch(*this, fence, cb_ptr, _pending_waits, extra_signals);
                _pending_waits.clear();
            } else {
                _flush_pending_waits();
                _evt._signal(*this, fence, cb_ptr);
            }
            st->tl_cpu_submit_end_ns = VkGpuTimeline::steady_ns();
            st->has_submitted_timestamps = (st->timestamp_pool != VK_NULL_HANDLE) && (cb_ptr != nullptr);
        } else {
            if (use_fused_submit) {
                _evt._signal_batch(*this, fence, cb_ptr, _pending_waits, extra_signals);
                _pending_waits.clear();
            } else {
                _flush_pending_waits();
                _evt._signal(*this, fence, cb_ptr);
            }
            if (online_timing_enabled()) {
                auto *st = cmdbuffer.states();
                st->has_submitted_timestamps =
                    (st->timestamp_pool != VK_NULL_HANDLE) && (cb_ptr != nullptr);
            }
        }
        if (!use_fused_submit) {
            for (auto const &s : extra_signals) {
                if (s.evt != nullptr) {
                    s.evt->_signal(*this, s.value);
                }
            }
        }
        submit_profile::timing_end_submit(profile_submit_begin);
        _mtx.lock();
        _exec.enqueue(SyncExt{
            .evt = &_evt,
            .value = fence});
        cmdbuffer.states()->host_retire_fence = fence;
        _exec.enqueue(std::move(cmdbuffer));
        for (auto const &s : extra_signals) {
            if (s.evt != nullptr) {
                _exec.enqueue(SyncExt{.evt = s.evt, .value = s.value});
                _exec.enqueue(NotifyEvt{.evt = s.evt, .value = s.value});
            }
        }
    } else {
        const bool need_gpu_batch =
            !extra_signals.empty() ||
            (!_pending_waits.empty() && device()->config_ext() == nullptr);
        if (need_gpu_batch) {
            if (device()->config_ext() == nullptr) {
                _evt._signal_batch(*this, fence, nullptr, _pending_waits, extra_signals);
                _pending_waits.clear();
            } else {
                _flush_pending_waits();
                _evt._signal(*this, fence, nullptr);
                for (auto const &s : extra_signals) {
                    if (s.evt != nullptr) {
                        s.evt->_signal(*this, s.value);
                    }
                }
            }
            _mtx.lock();
            _exec.enqueue(SyncExt{
                .evt = &_evt,
                .value = fence});
            for (auto const &s : extra_signals) {
                if (s.evt != nullptr) {
                    _exec.enqueue(SyncExt{.evt = s.evt, .value = s.value});
                    _exec.enqueue(NotifyEvt{.evt = s.evt, .value = s.value});
                }
            }
        } else {
            _evt._update_fence(fence);
            _mtx.lock();
        }
    }
    if (!callbacks.empty()) {
        _exec.enqueue(std::move(callbacks));
    }
    _exec.enqueue(NotifyEvt{
        .evt = &_evt,
        .value = fence});

    // Phase 3B (opt C) / phase 3C (defect 2): publish the new work while still
    // holding `_mtx` (the enqueue and the epoch bump are then one indivisible step
    // for the retirement thread) and notify only after the lock is released.
    _bump_work_epoch_locked();
    _mtx.unlock();
    _notify_retirement_thread();
    return (has_gpu_cmds || !presents.empty()) ? fence : 0u;
}
CommandBuffer Stream::_begin_batch_record(
    detail::CommandBufferOwnership ownership,
    VkCommandBufferUsageFlags flags) {
    // Extracted verbatim from `Stream::dispatch` (plan P4): a recycled command
    // buffer if the pool has one, a fresh one otherwise - unless this is a replay
    // recording, which is the one case where neither of those is allowed.
    auto const persistent =
        ownership == detail::CommandBufferOwnership::PERSISTENT;
    CommandBuffer cmdbuffer = [&]() {
        if (persistent) {
            // A recording is recorded once, submitted N times and freed by the
            // graph that owns it, so it can be neither a recycled buffer nor a
            // borrowed one:
            //   * the retirement FIFO only ever holds buffers whose state belongs to
            //     a submission that already completed (`CommandBufferState::reset`
            //     ran, descriptor pools were reset, the trailing restore was
            //     recorded) - a graph must not adopt that state;
            //   * `borrow_command_buffer()` is a one-shot contract: its buffer
            //     comes from a pool the backend does not own, so a graph that later
            //     freed it with `state->pool` would free foreign native memory (and
            //     `CommandBufferOwnership::BORROWED` deliberately never frees).
            // A fresh PERSISTENT construction gets its own pool without the reset
            // bit and its own descriptor pools - exactly the state the recorded
            // commands keep referencing until the graph is destroyed.
            return CommandBuffer{*this, detail::CommandBufferOwnership::PERSISTENT};
        }
        auto p = _cmdbuffers.dequeue();
        if (p) return std::move(*p);
        return CommandBuffer{*this, detail::CommandBufferOwnership::BACKEND};
    }();
    _scratch_buffer_alloc_visitor.cmdbuffer = &cmdbuffer;
    _scratch_buffer_alloc_visitor.device = device();
    // Every batch starts from "nothing known" (plan P3 stays NO-GO): the tracker's
    // per-frame maps are dropped here and the previous submission's trailing
    // restore is what makes the first touch of a buffer legal.
    _resource_barrier.clear_restore_states();
    if (device()->config_ext()) {
        auto before_states = device()->config_ext()->before_states(reinterpret_cast<uint64_t>(this));
        for (auto &i : before_states) {
            set_config_resource_before_state(&_resource_barrier, i);
        }
    }
    cmdbuffer.resource_barrier = &_resource_barrier;
    cmdbuffer.uniform_data = &_uniform_data;
    cmdbuffer.desc_sets = &_desc_sets;
    cmdbuffer.logger = logger ? &logger : nullptr;
    cmdbuffer.dispatch_offsets = &_dispatch_offsets;
    cmdbuffer.write_desc_sets = &_write_desc_sets;
    cmdbuffer.bindless_cache = &_bindless_cache;
    cmdbuffer.temp_desc = &_temp_desc;
    cmdbuffer.scratch_buffer_alloc = &_scratch_buffer_alloc;
    // `flags` is 0 for every ordinary batch, i.e. the flags-less begin info this
    // path always used; the replay recorder asks for SIMULTANEOUS_USE.
    cmdbuffer.begin(flags);
    return cmdbuffer;
}

void Stream::_finish_batch_record(CommandBuffer &cmdbuffer) {
    if (device()->config_ext()) {
        // Bindless updates have now advanced the encoded descriptor map. Expand
        // the after-state against this final snapshot, whereas the before-state in
        // `_begin_batch_record` deliberately used the initial snapshot.
        auto after_states = device()->config_ext()->after_states(
            reinterpret_cast<uint64_t>(this));
        for (auto &i : after_states) {
            set_config_resource_restore_state(&_resource_barrier, i);
        }
    }
    // The trailing restore is the whole cross-submission contract (plan §7 of the
    // design): it republishes every touched buffer as ALL_COMMANDS-readable and
    // every touched mip as GENERAL, so a replayed buffer is exactly as visible as
    // an ordinarily dispatched one.
    _resource_barrier.restore_states(cmdbuffer.cmdbuffer());
    if (device()->gpu_timeline() != nullptr || online_timing_enabled()) {
        auto *st = cmdbuffer.states();
        if (st->tl_cb_q_begin != 0xffffffffu &&
            st->timestamp_pool != VK_NULL_HANDLE &&
            st->timestamp_cursor < st->timestamp_pool_capacity) {
            uint32_t q_end = st->timestamp_cursor++;
            vkCmdWriteTimestamp(cmdbuffer.cmdbuffer(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, st->timestamp_pool, q_end);
            st->tl_cb_q_end = q_end;
        }
    }
    cmdbuffer.end();
}

detail::RecordedBatch Stream::record_replay_batch(
    vstd::span<const luisa::unique_ptr<Command>> cmds) {
    std::lock_guard lck{_dispatch_mtx};
    _temp_desc.clear();
    // Counter-only: a recording is a command-buffer record, exactly like a batch -
    // it just never reaches the submit below. That keeps the ordinary
    // encode/flush accounting intact while `replay_submits` counts the other half.
    submit_profile::record_command_batch();
    auto profile_flush = vstd::scope_exit([] {
        submit_profile::flush();
    });
    auto const profile_record_begin = submit_profile::timing_begin();
    // PERSISTENT: a dedicated backend-owned command buffer whose state the graph
    // adopts. Asking for PERSISTENT is what makes the acquisition refuse both the
    // stream's recycled buffers and the config ext's borrowed ones - the defect
    // this parameter exists to fix.
    auto cmdbuffer = _begin_batch_record(
        detail::CommandBufferOwnership::PERSISTENT,
        VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);
    cmdbuffer.execute(cmds);
    _finish_batch_record(cmdbuffer);
    auto const record_ns = submit_profile::timing_elapsed_ns(profile_record_begin);
    // Deliberately *not* folded into `host_encode_ns`: that counter is the
    // ordinary per-batch encode window and its us/batch ratio must keep its
    // meaning. A record has its own counter and its own denominator
    // (`replay_records`).
    submit_profile::record_replay_record(record_ns);
    return cmdbuffer.release_native_batch();
}

uint64_t Stream::submit_recorded(VkCommandBuffer command_buffer,
                                bool inqueue_limit) {
    std::lock_guard lck{_dispatch_mtx};
    _flush_pending_waits();
    _temp_desc.clear();
    if (inqueue_limit) {
        // The in-flight bound is part of the stream's contract (plan P1.5), and a
        // replay is a submission like any other: without it K replays would queue
        // K pending executions of one command buffer.
        _throttle_in_flight();
    }
    auto fence_plan = detail::plan_timeline_value_increment(
        _evt.last_fence(), 1u);
    LUISA_ASSERT(
        static_cast<bool>(fence_plan),
        "Vulkan stream timeline fence overflow during replay.");
    auto fence = fence_plan.value;
    // Exactly one native submission, through the same `Event::_signal` the
    // dispatch path uses: one reserved timeline value, one
    // vkQueueSubmit2/vkQueueSubmit, and therefore the same Event-based
    // cross-stream ordering and `Stream::sync()` behaviour as an ordinary batch.
    // Nothing is re-recorded, no argument is planned, no descriptor is touched and
    // no command buffer enters the retirement FIFO (so no state is reset).
    auto command_buffer_copy = command_buffer;
    auto const profile_submit_begin = submit_profile::timing_begin();
    _evt._signal(*this, fence, &command_buffer_copy);
    submit_profile::timing_end_submit(profile_submit_begin);
    submit_profile::record_replay_submit();
    _mtx.lock();
    _exec.enqueue(SyncExt{
        .evt = &_evt,
        .value = fence});
    _exec.enqueue(NotifyEvt{
        .evt = &_evt,
        .value = fence});
    // Phase 3B (opt C) / phase 3C (defect 2): publish the new work while still
    // holding `_mtx` (the enqueue and the epoch bump are then one indivisible step
    // for the retirement thread) and notify only after the lock is released.
    _bump_work_epoch_locked();
    _mtx.unlock();
    _notify_retirement_thread();
    // A replay is not a batch, so nothing else would fold this thread's counters.
    submit_profile::flush();
    return fence;
}

bool Stream::submission_complete(uint64_t timeline_value) const noexcept {
    return _evt.is_complete(timeline_value);
}

void Stream::update_sparse_resources(luisa::vector<SparseUpdateTile> &&textures_update) noexcept {
    std::lock_guard lck{_dispatch_mtx};
    _flush_pending_waits();
    _temp_desc.clear();
    if (textures_update.empty()) [[unlikely]]
        return;
    auto queue_family_index = device()->sparse_queue_index();
    auto sparse_queue = detail::validate_sparse_binding_queue_family(
        queue_family_index, device()->queue_family_properties());
    LUISA_ASSERT(
        static_cast<bool>(sparse_queue),
        "Cannot bind Vulkan sparse resources for the {} stream: the device's "
        "sparse queue family {} failed validation ({}, available flags "
        "0x{:x}).",
        detail::queue_family_role_name(static_cast<uint32_t>(_stream_tag)),
        queue_family_index,
        detail::sparse_binding_queue_status_name(sparse_queue.status),
        sparse_queue.available_flags);
    LUISA_ASSERT(
        device()->sparse_queue() != VK_NULL_HANDLE,
        "Vulkan sparse queue family {} has no acquired queue handle.",
        queue_family_index);
    VkBindSparseInfo info{
        .sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
    };
    auto previous_logical_fence = _evt.last_fence();
    auto previous_gpu_signal = _evt.last_signaled_fence();
    auto current_gpu_value = _evt.current_gpu_value();
    auto max_value_difference =
        device()->max_timeline_semaphore_value_difference();
    auto timeline_plan = detail::plan_sparse_submission_timeline(
        previous_logical_fence,
        previous_gpu_signal,
        current_gpu_value,
        max_value_difference);
    LUISA_ASSERT(
        static_cast<bool>(timeline_plan),
        "Cannot reserve Vulkan sparse-submission timeline values from "
        "logical fence {}, tracked GPU signal {}, current GPU value {}, and "
        "max difference {}: {}.",
        previous_logical_fence, previous_gpu_signal, current_gpu_value,
        max_value_difference,
        detail::sparse_submission_timeline_status_name(
            timeline_plan.status));
    auto bridge_fence = timeline_plan.bridge_signal_value;
    auto fence = timeline_plan.sparse_signal_value;
    // Lock the Device-wide residency state before dereferencing any sparse
    // resource or heap handles. The transaction remains uncommitted through
    // native submission and host completion.
    auto residency_transaction =
        device()->sparse_residency_registry().begin_transaction();
    struct Alloc {
        size_t size{};
        void *ptr{};
        bool is_buffer{};
    };
    vstd::unordered_map<uint64_t, Alloc> counter;
    size_t buffer_bind_count = 0;
    size_t img_bind_count = 0;
    for (auto &i : textures_update) {
        auto iter = counter.try_emplace(i.handle, Alloc{});
        auto &v = iter.first->second;
        v.size += 1;
        luisa::visit(
            [&]<typename T>(T const &op) {
                constexpr auto operation_is_buffer =
                    std::is_same_v<SparseBufferMapOperation, T> ||
                    std::is_same_v<SparseBufferUnMapOperation, T>;
                if (iter.second) {
                    v.is_buffer = operation_is_buffer;
                    if constexpr (operation_is_buffer) {
                        ++buffer_bind_count;
                    } else {
                        ++img_bind_count;
                    }
                } else {
                    LUISA_ASSERT(
                        v.is_buffer == operation_is_buffer,
                        "Sparse update handle 0x{:016x} is used as both a "
                        "buffer and an image in one Vulkan bind batch.",
                        i.handle);
                }
            },
            i.operations);
    }
    LUISA_ASSERT(
        buffer_bind_count <= std::numeric_limits<uint32_t>::max() &&
            img_bind_count <= std::numeric_limits<uint32_t>::max(),
        "Vulkan sparse update contains too many resource bind groups.");
    // Validate handles and resource kinds while the registry lock guarantees
    // their lifetime, before interpreting the opaque integers as C++ objects.
    // This turns malformed direct DeviceInterface updates into a diagnosed
    // contract failure instead of an unchecked pointer dereference.
    for (auto const &[handle, allocation] : counter) {
        auto resource_kind = allocation.is_buffer ?
                                 detail::SparseResidencyResourceKind::BUFFER :
                                 detail::SparseResidencyResourceKind::IMAGE;
        auto validation = residency_transaction.validate_resource(
            handle, resource_kind);
        LUISA_ASSERT(
            static_cast<bool>(validation),
            "Invalid Vulkan sparse resource handle 0x{:016x}: {}.",
            handle,
            detail::sparse_residency_registry_status_name(
                validation.status));
    }
    VkSparseBufferMemoryBindInfo *buffer_ptr = nullptr;
    if (buffer_bind_count != 0u) {
        auto chunk = _temp_desc.allocate(
            sizeof(VkSparseBufferMemoryBindInfo) * buffer_bind_count,
            alignof(VkSparseBufferMemoryBindInfo));
        buffer_ptr = reinterpret_cast<VkSparseBufferMemoryBindInfo *>(
            chunk.handle + chunk.offset);
    }
    VkSparseImageMemoryBindInfo *img_ptr = nullptr;
    if (img_bind_count != 0u) {
        auto chunk = _temp_desc.allocate(
            sizeof(VkSparseImageMemoryBindInfo) * img_bind_count,
            alignof(VkSparseImageMemoryBindInfo));
        img_ptr = reinterpret_cast<VkSparseImageMemoryBindInfo *>(
            chunk.handle + chunk.offset);
    }
    info.pBufferBinds = buffer_ptr;
    info.pImageBinds = img_ptr;
    info.bufferBindCount = static_cast<uint32_t>(buffer_bind_count);
    info.imageBindCount = static_cast<uint32_t>(img_bind_count);
    // Bind ptr
    for (auto &i : counter) {
        auto &a = i.second;
        LUISA_ASSERT(
            a.size <= std::numeric_limits<uint32_t>::max(),
            "Vulkan sparse resource 0x{:016x} has too many binds in one batch.",
            i.first);
        if (a.is_buffer) {
            auto chunk = _temp_desc.allocate(sizeof(VkSparseMemoryBind) * a.size, alignof(VkSparseMemoryBind));
            auto ptr = reinterpret_cast<VkSparseMemoryBind *>(chunk.handle + chunk.offset);
            a.ptr = ptr;
            buffer_ptr->buffer = reinterpret_cast<SparseBuffer *>(i.first)->vk_buffer();
            buffer_ptr->bindCount = static_cast<uint32_t>(a.size);
            buffer_ptr->pBinds = ptr;
            ++buffer_ptr;
        } else {
            auto chunk = _temp_desc.allocate(sizeof(VkSparseImageMemoryBind) * a.size, alignof(VkSparseImageMemoryBind));
            auto ptr = reinterpret_cast<VkSparseImageMemoryBind *>(chunk.handle + chunk.offset);
            a.ptr = ptr;
            img_ptr->image = reinterpret_cast<Texture *>(i.first)->vk_image();
            img_ptr->bindCount = static_cast<uint32_t>(a.size);
            img_ptr->pBinds = ptr;
            ++img_ptr;
        }
    }
    auto plan_image_binding = [](
                                  Texture const *texture,
                                  auto const &operation) noexcept {
        auto mip_extent = texture->mip_extent(operation.mip_level);
        auto const &sparse_requirements =
            texture->sparse_memory_requirements();
        auto granularity =
            sparse_requirements.formatProperties.imageGranularity;
        auto plan = detail::plan_sparse_image_binding({.mip_extent = {mip_extent.x, mip_extent.y, mip_extent.z},
                                                       .granularity = granularity,
                                                       .tile_byte_size = texture->sparse_block_size(),
                                                       .mip_level = operation.mip_level,
                                                       .mip_tail_first_lod =
                                                           sparse_requirements.imageMipTailFirstLod,
                                                       .start_tile = {operation.start_tile.x,
                                                                      operation.start_tile.y,
                                                                      operation.start_tile.z},
                                                       .tile_count = {operation.tile_count.x,
                                                                      operation.tile_count.y,
                                                                      operation.tile_count.z}});
        LUISA_ASSERT(
            static_cast<bool>(plan),
            "Invalid Vulkan sparse-image binding at mip {} (status {}). "
            "Mip-tail levels require opaque bindings and are not representable "
            "by the Luisa tile API.",
            operation.mip_level, static_cast<uint32_t>(plan.status));
        return plan;
    };
    auto plan_buffer_binding = [](
                                   SparseBuffer const *buffer,
                                   auto const &operation) noexcept {
        auto plan = detail::plan_sparse_buffer_binding({.physical_resource_size = buffer->sparse_binding_size(),
                                                        .alignment = buffer->sparse_block_size(),
                                                        .start_tile = operation.start_tile,
                                                        .tile_count = operation.tile_count});
        LUISA_ASSERT(
            static_cast<bool>(plan),
            "Invalid Vulkan sparse-buffer binding (status {}).",
            static_cast<uint32_t>(plan.status));
        return plan;
    };
    struct PlannedSparseBinding {
        bool is_buffer{};
        uint32_t mip_level{};
        detail::SparseBufferBindingPlan buffer{};
        detail::SparseImageBindingPlan image{};
    };
    luisa::vector<PlannedSparseBinding> planned_bindings;
    planned_bindings.reserve(textures_update.size());
    for (auto const &update : textures_update) {
        luisa::visit(
            [&]<typename T>(T const &operation) {
                constexpr auto operation_is_buffer =
                    std::is_same_v<SparseBufferMapOperation, T> ||
                    std::is_same_v<SparseBufferUnMapOperation, T>;
                PlannedSparseBinding planned{
                    .is_buffer = operation_is_buffer};
                if constexpr (operation_is_buffer) {
                    planned.buffer = plan_buffer_binding(
                        reinterpret_cast<SparseBuffer const *>(
                            update.handle),
                        operation);
                } else {
                    planned.mip_level = operation.mip_level;
                    planned.image = plan_image_binding(
                        reinterpret_cast<Texture const *>(
                            update.handle),
                        operation);
                }
                planned_bindings.emplace_back(planned);
            },
            update.operations);
    }
    // A VkBindSparseInfo batch may not bind any resource range more than
    // once. Validate the complete batch before acquiring heaps or emitting
    // native bind records; map/map and map/unmap overlaps are both illegal.
    for (auto index = 0u; index < textures_update.size(); ++index) {
        for (auto previous = 0u; previous < index; ++previous) {
            if (textures_update[index].handle !=
                textures_update[previous].handle) {
                continue;
            }
            auto const &lhs = planned_bindings[index];
            auto const &rhs = planned_bindings[previous];
            auto overlaps = lhs.is_buffer ?
                                detail::sparse_buffer_bindings_overlap(
                                    lhs.buffer, rhs.buffer) :
                                detail::sparse_image_bindings_overlap(
                                    lhs.image, lhs.mip_level,
                                    rhs.image, rhs.mip_level);
            LUISA_ASSERT(
                !overlaps,
                "Vulkan sparse bind batch entries {} and {} overlap for "
                "resource 0x{:016x}.",
                previous, index, textures_update[index].handle);
        }
    }
    // Apply the whole ownership transition transactionally before acquiring
    // heaps or emitting native bind records.
    for (auto index = 0u; index < textures_update.size(); ++index) {
        auto const &update = textures_update[index];
        auto const &planned = planned_bindings[index];
        auto result = luisa::visit(
            [&]<typename T>(T const &operation) {
                if constexpr (
                    std::is_same_v<SparseBufferMapOperation, T>) {
                    return residency_transaction.map_buffer(
                        update.handle, operation.allocated_heap,
                        {.offset = planned.buffer.resource_offset,
                         .size = planned.buffer.binding_size});
                } else if constexpr (
                    std::is_same_v<SparseBufferUnMapOperation, T>) {
                    return residency_transaction.unmap_buffer(
                        update.handle,
                        {.offset = planned.buffer.resource_offset,
                         .size = planned.buffer.binding_size});
                } else {
                    LUISA_ASSERT(
                        planned.image.offset.x >= 0 &&
                            planned.image.offset.y >= 0 &&
                            planned.image.offset.z >= 0,
                        "Vulkan sparse-image planner produced a negative "
                        "offset.");
                    auto box = detail::SparseImageResidencyBox{
                        .mip_level = operation.mip_level,
                        .offset = {
                            static_cast<uint64_t>(planned.image.offset.x),
                            static_cast<uint64_t>(planned.image.offset.y),
                            static_cast<uint64_t>(planned.image.offset.z)},
                        .extent = {planned.image.extent.width, planned.image.extent.height, planned.image.extent.depth}};
                    if constexpr (
                        std::is_same_v<SparseTextureMapOperation, T>) {
                        return residency_transaction.map_image(
                            update.handle, operation.allocated_heap, box);
                    } else {
                        return residency_transaction.unmap_image(
                            update.handle, box);
                    }
                }
            },
            update.operations);
        LUISA_ASSERT(
            static_cast<bool>(result),
            "Vulkan sparse residency update {} for resource 0x{:016x} and "
            "heap 0x{:016x} failed: {}. Sparse ranges must be explicitly "
            "unmapped before remapping, unmaps must cover only resident "
            "ranges, and one heap cannot back multiple live ranges.",
            index, result.resource, result.heap,
            detail::sparse_residency_registry_status_name(result.status));
    }
    // Write value
    auto planned_index = size_t{0u};
    for (auto &i : textures_update) {
        auto const &planned = planned_bindings[planned_index++];
        auto &v = counter.find(i.handle)->second;
        luisa::visit([&]<typename T>(T const &op) {
            if constexpr (std::is_same_v<SparseTextureMapOperation, T>) {
                auto ptr = reinterpret_cast<VkSparseImageMemoryBind *>(v.ptr);
                auto tex = reinterpret_cast<Texture const *>(i.handle);
                auto const &plan = planned.image;
                auto heap = reinterpret_cast<VulkanSparseHeap *>(op.allocated_heap);
                LUISA_ASSERT(heap != nullptr,
                             "Vulkan sparse-image map uses a null heap.");
                auto memory = heap->acquire(
                    tex->memory_requirements(),
                    plan.required_heap_size);
                ptr->subresource.aspectMask =
                    tex->sparse_memory_requirements()
                        .formatProperties.aspectMask;
                ptr->subresource.mipLevel = op.mip_level;
                ptr->subresource.arrayLayer = 0;
                ptr->offset = plan.offset;
                ptr->extent = plan.extent;
                ptr->memory = memory.memory;
                ptr->memoryOffset = memory.offset;
                ptr->flags = 0;
                ++ptr;
                v.ptr = ptr;
            } else if constexpr (std::is_same_v<SparseTextureUnMapOperation, T>) {
                auto ptr = reinterpret_cast<VkSparseImageMemoryBind *>(v.ptr);
                auto tex = reinterpret_cast<Texture const *>(i.handle);
                auto const &plan = planned.image;
                ptr->subresource.aspectMask =
                    tex->sparse_memory_requirements()
                        .formatProperties.aspectMask;
                ptr->subresource.mipLevel = op.mip_level;
                ptr->subresource.arrayLayer = 0;
                ptr->offset = plan.offset;
                ptr->extent = plan.extent;
                ptr->memory = VK_NULL_HANDLE;
                ptr->memoryOffset = 0u;
                ptr->flags = 0;
                ++ptr;
                v.ptr = ptr;
            } else if constexpr (std::is_same_v<SparseBufferMapOperation, T>) {
                auto ptr = reinterpret_cast<VkSparseMemoryBind *>(v.ptr);
                auto buffer = reinterpret_cast<SparseBuffer const *>(i.handle);
                auto const &plan = planned.buffer;
                auto heap = reinterpret_cast<VulkanSparseHeap *>(op.allocated_heap);
                LUISA_ASSERT(heap != nullptr,
                             "Vulkan sparse-buffer map uses a null heap.");
                auto memory = heap->acquire(
                    buffer->memory_requirements(),
                    plan.required_heap_size);
                ptr->memory = memory.memory;
                ptr->memoryOffset = memory.offset;
                ptr->resourceOffset = plan.resource_offset;
                ptr->size = plan.binding_size;
                ptr->flags = 0;
                ++ptr;
                v.ptr = ptr;
            } else if constexpr (std::is_same_v<SparseBufferUnMapOperation, T>) {
                auto ptr = reinterpret_cast<VkSparseMemoryBind *>(v.ptr);
                auto const &plan = planned.buffer;
                ptr->memory = VK_NULL_HANDLE;
                ptr->memoryOffset = 0u;
                ptr->resourceOffset = plan.resource_offset;
                ptr->size = plan.binding_size;
                ptr->flags = 0;
                ++ptr;
                v.ptr = ptr;
            }
        },
                     i.operations);
    }
    VkTimelineSemaphoreSubmitInfo timeline;
    // A preceding external-event wait/signal is real ordinary-queue work but
    // does not advance this stream's internal timeline. Submit an explicit
    // bridge signal on the ordinary queue after every prior operation, then
    // make the sparse queue wait on that exact value. Queue submission order
    // alone cannot bridge two distinct VkQueue handles.
    _evt._signal(*this, bridge_fence);
    _evt._signal_sparse(
        &bridge_fence,
        &fence, &info, &timeline);
    {
        std::lock_guard queue_lock{device()->sparse_queue_mtx()};
        VK_CHECK_RESULT(vkQueueBindSparse(
            device()->sparse_queue(),
            1,
            &info,
            VK_NULL_HANDLE));
    }
    _evt.mark_signal_fence(fence);
    // Sparse-binding submissions have no implicit ordering with command-buffer
    // submissions, even on the same queue. The bind waits for the preceding
    // stream fence above; completing it on the host before releasing the
    // dispatch mutex makes the new mapping an explicit boundary for every
    // later stream operation without serializing ordinary command batches.
    _evt._host_wait(fence);
    auto residency_commit = residency_transaction.commit();
    LUISA_ASSERT(
        static_cast<bool>(residency_commit),
        "Failed to commit Vulkan sparse residency state after successful "
        "queue submission: {}.",
        detail::sparse_residency_registry_status_name(
            residency_commit.status));
    // Logical fences can also represent callbacks or skipped presents that do
    // not signal the Vulkan semaphore. Publish sparse completion through the
    // FIFO executor so those host-only operations retain their ordering, while
    // the host wait above still makes every later GPU submission safe.
    _mtx.lock();
    _exec.enqueue(NotifyEvt{
        .evt = &_evt,
        .value = fence});
    // Phase 3B (opt C) / phase 3C (defect 2): publish the new work while still
    // holding `_mtx` (the enqueue and the epoch bump are then one indivisible step
    // for the retirement thread) and notify only after the lock is released.
    _bump_work_epoch_locked();
    _mtx.unlock();
    _notify_retirement_thread();
}
void Stream::_flush_pending_waits() {
    if (_pending_waits.empty()) {
        return;
    }
    for (auto const &w : _pending_waits) {
        if (!w.token) { continue; }
        std::lock_guard token_lock{w.token->mtx};
        if (w.token->evt != nullptr) {
            w.token->evt->_wait(*this, w.value);
        }
    }
    _pending_waits.clear();
}
void Stream::sync() {
    vstd::vector<EventWaitEntry> dangling_waits;
    {
        std::lock_guard lck{_dispatch_mtx};
        if (!_pending_waits.empty()) {
            dangling_waits = std::move(_pending_waits);
            _pending_waits.clear();
        }
    }
    _evt.sync(_evt.last_fence());
    for (auto const &w : dangling_waits) {
        if (!w.token) { continue; }
        std::lock_guard token_lock{w.token->mtx};
        if (w.token->evt != nullptr) {
            w.token->evt->_host_wait(w.value);
        }
    }
    vstd::vector<Blas *> pending;
    {
        std::lock_guard lock{_pending_compaction_mtx};
        pending = std::move(_pending_compaction_blases);
        _pending_compaction_blases.clear();
        for (auto *b : pending) {
            if (b != nullptr && b->_pending_compaction_stream == this) {
                b->_set_pending_compaction_stream(nullptr);
            }
        }
    }
    for (auto *b : pending) {
        if (b != nullptr) {
            (void)b->try_compact_in_place(*this);
        }
    }
}
CommandBuffer::CommandBuffer(Stream &stream,
                             detail::CommandBufferOwnership ownership) noexcept
    : Resource(stream.device()),
      _stream(stream),
      _state(vstd::make_unique<CommandBufferState>()),
      _ownership(ownership) {
    // A replay recording (plan P4) gets its own state whose pool has no
    // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: such a buffer is begun
    // once, submitted N times and never reset, which is the whole
    // SIMULTANEOUS_USE contract made auditable. It also never borrows a
    // config-ext command buffer - the borrow contract is one-shot and the backend
    // may not reset or free a borrowed buffer, while a recording is resubmitted
    // and later freed by the graph.
    auto const persistent =
        ownership == detail::CommandBufferOwnership::PERSISTENT;
    _state->init(*stream.device(), stream.stream_tag(),
                 persistent ?
                     VkCommandPoolCreateFlags{0u} :
                     VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    _cmdbuffer = nullptr;
    if (!persistent && device()->config_ext()) {
        _cmdbuffer = device()->config_ext()->borrow_command_buffer(stream.stream_tag());
    }
    if (_cmdbuffer) {
        _ownership = detail::CommandBufferOwnership::BORROWED;
    } else {
        _ownership = persistent ?
                         detail::CommandBufferOwnership::PERSISTENT :
                         detail::CommandBufferOwnership::BACKEND;
        VkCommandBufferAllocateInfo cb_ci{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = _state->pool,
            .commandBufferCount = 1};
        VK_CHECK_RESULT(vkAllocateCommandBuffers(device()->logic_device(), &cb_ci, &_cmdbuffer));
    }
    // VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    // VK_CHECK_RESULT(vkCreateFence(device()->logic_device(), &fence_info, Device::alloc_callbacks(), nullptr));
}
CommandBuffer::~CommandBuffer() {
    auto retirement = detail::plan_command_buffer_retirement(_ownership);
    if (_cmdbuffer && retirement.free_native_buffer) {
        vkFreeCommandBuffers(device()->logic_device(), _state->pool, 1, &_cmdbuffer);
    }
}
void CommandBuffer::begin(VkCommandBufferUsageFlags flags) {
    // Phase 3B (opt B): an ordinary batch is recorded once and submitted exactly
    // once (`Stream::dispatch` / `Stream::present` submit it and then hand it to the
    // retirement FIFO, which resets it), so the recording can say so.
    // VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT is purely a hint that lets the
    // implementation skip its resubmission bookkeeping; it changes no
    // synchronization and no lifetime rule. Two cases keep their flags: a
    // PERSISTENT recording (the replay path passes SIMULTANEOUS_USE because a graph
    // resubmits it N times) and any caller that already chose flags. A BORROWED
    // buffer is begun by this same function on the ordinary dispatch path, so it is
    // included: it is still submitted exactly once.
    if (flags == 0u &&
        _ownership != detail::CommandBufferOwnership::PERSISTENT) {
        flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    }
    // A command buffer starts its recording with no bindings of its own: whatever
    // the previous recording of this buffer (or its recycled state) bound belongs
    // to a submission that has already completed or been reset (plan P1.2).
    _state->bind_state.reset();
    VkCommandBufferBeginInfo bi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = flags,
    };
    VK_CHECK_RESULT(vkBeginCommandBuffer(_cmdbuffer, &bi));
}
void CommandBuffer::end() {
    VK_CHECK_RESULT(vkEndCommandBuffer(_cmdbuffer));
}
detail::RecordedBatch CommandBuffer::release_native_batch() noexcept {
    // Ownership moves out whole: the graph gets the native variable *and* the
    // state that keeps the recorded commands' references alive (command pool,
    // descriptor pools and sets, argument staging, retained pipelines, image
    // views). This object is empty afterwards, so its destructor neither frees the
    // buffer nor resets the state.
    //
    // The graph frees the native buffer against `state->pool`, so handing out a
    // buffer whose pool this object does not own would be a defect rather than a
    // trade-off: a BACKEND buffer is freed by the retirement FIFO / destructor
    // (second free) and a BORROWED buffer belongs to the config ext forever. Only
    // a recording (`PERSISTENT`, see `Stream::record_replay_batch`) qualifies, and
    // by construction this function has exactly that one caller.
    LUISA_ASSERT(detail::may_release_native_batch(_ownership),
                 "Vulkan: release_native_batch() is only valid for a PERSISTENT "
                 "replay recording; a BACKEND or BORROWED command buffer must not "
                 "be handed to a replay graph.");
    auto batch = detail::RecordedBatch{_cmdbuffer, _state.release()};
    _cmdbuffer = VK_NULL_HANDLE;
    return batch;
}
CommandBuffer::CommandBuffer(CommandBuffer &&rhs) noexcept
    : Resource(std::move(rhs)),
      _stream(rhs._stream),         // NOLINT(bugprone-use-after-move)
      _cmdbuffer(rhs._cmdbuffer),   // NOLINT(bugprone-use-after-move)
      _state(std::move(rhs._state)),// NOLINT(bugprone-use-after-move)
      _ownership(rhs._ownership),   // NOLINT(bugprone-use-after-move)
      // Plan P4: the record prefix is a helper that *returns* a prepared command
      // buffer, so the stream-scoped wiring has to survive the move. Leaving these
      // null and relying on copy elision would make the extraction work only by
      // luck; a move that dropped them would silently produce a command buffer
      // with no barrier tracker, no argument staging and no logger.
      logger(rhs.logger),                            // NOLINT(bugprone-use-after-move)
      desc_sets(rhs.desc_sets),                      // NOLINT(bugprone-use-after-move)
      uniform_data(rhs.uniform_data),                // NOLINT(bugprone-use-after-move)
      dispatch_offsets(rhs.dispatch_offsets),        // NOLINT(bugprone-use-after-move)
      write_desc_sets(rhs.write_desc_sets),          // NOLINT(bugprone-use-after-move)
      scratch_buffer_alloc(rhs.scratch_buffer_alloc),// NOLINT(bugprone-use-after-move)
      bindless_cache(rhs.bindless_cache),            // NOLINT(bugprone-use-after-move)
      temp_desc(rhs.temp_desc),                      // NOLINT(bugprone-use-after-move)
      resource_barrier(rhs.resource_barrier) {       // NOLINT(bugprone-use-after-move)
    // A moved-from command buffer owns nothing and points at nothing, so a later
    // destruction of it cannot free or reset the moved state.
    rhs._cmdbuffer = nullptr;
    rhs.logger = nullptr;
    rhs.desc_sets = nullptr;
    rhs.uniform_data = nullptr;
    rhs.dispatch_offsets = nullptr;
    rhs.write_desc_sets = nullptr;
    rhs.scratch_buffer_alloc = nullptr;
    rhs.bindless_cache = nullptr;
    rhs.temp_desc = nullptr;
    rhs.resource_barrier = nullptr;
}

void Stream::signal(Event *event, uint64_t value) {
    std::lock_guard lck{_dispatch_mtx};
    if (device()->config_ext() == nullptr && !_pending_waits.empty()) {
        auto fence_plan = detail::plan_timeline_value_increment(
            _evt.last_fence(), 1u);
        LUISA_ASSERT(
            static_cast<bool>(fence_plan),
            "Vulkan stream timeline fence overflow during signal.");
        auto fence = fence_plan.value;
        EventSignalEntry extra{event, value};
        _evt._signal_batch(
            *this, fence, nullptr, _pending_waits,
            vstd::span<const EventSignalEntry>{&extra, 1u});
        _pending_waits.clear();
        _mtx.lock();
        _exec.enqueue(SyncExt{&_evt, fence});
        _exec.enqueue(SyncExt{event, value});
        _exec.enqueue(NotifyEvt{event, value});
        _exec.enqueue(NotifyEvt{&_evt, fence});
        _bump_work_epoch_locked();
        _mtx.unlock();
        _notify_retirement_thread();
        return;
    }
    _flush_pending_waits();
    event->_signal(*this, value);
    _mtx.lock();
    _exec.enqueue(SyncExt{event, value});
    _exec.enqueue(NotifyEvt{event, value});
    // Phase 3B (opt C) / phase 3C (defect 2): publish the new work while still
    // holding `_mtx` (the enqueue and the epoch bump are then one indivisible step
    // for the retirement thread) and notify only after the lock is released.
    _bump_work_epoch_locked();
    _mtx.unlock();
    _notify_retirement_thread();
}
void Stream::wait(Event *event, uint64_t value) {
    std::lock_guard lck{_dispatch_mtx};
    if (device()->config_ext() != nullptr) {
        event->_wait(*this, value);
        return;
    }
    _pending_waits.emplace_back(EventWaitEntry{event->wait_token(), value});
}
// File-unique namespace: the backend is built with unity (jumbo) batches, so a
// free name at namespace scope in this file can collide with a sibling's.
namespace stream_raster_detail {

// Records the barriers of a *native* raster draw (`NativeShaderLauncher::draw()`).
//
// Such a draw has no `RasterShader` and therefore no saved-argument or binding
// table to read usages from; the launcher instead packed one `Usage` per
// non-uniform resource argument into the command itself (see
// `DrawRasterSceneCommand::set_native_shader`). That declaration *is* the
// synchronization contract, exactly like a `NativeShaderDispatchCommand`'s
// `traverse_arguments`, so the barriers are recorded from it with the raster
// stage's usages (a DSL raster draw goes through `preprocess_arguments`).
void record_native_raster_usages(
    ResourceBarrier *barrier,
    DrawRasterSceneCommand const *cmd) noexcept {
    auto arguments = cmd->arguments();
    auto usages = cmd->argument_usages();
    auto usage_index = size_t{0u};
    for (auto &&argument : arguments) {
        if (argument.tag == Argument::Tag::UNIFORM) { continue; }
        if (usage_index >= usages.size()) {
            LUISA_ERROR_WITH_LOCATION(
                "A native raster draw carries {} resource argument(s) but only "
                "{} per-argument usage declaration(s); the command was not built "
                "by NativeShaderLauncher::draw().",
                arguments.size(), usages.size());
            return;
        }
        auto usage = usages[usage_index++];
        auto barrier_usage =
            (luisa::to_underlying(usage) &
             luisa::to_underlying(Usage::WRITE)) != 0u ?
                ResourceBarrier::Usage::kRasterUAV :
                ResourceBarrier::Usage::kRasterRead;
        if (argument.tag == Argument::Tag::BUFFER) {
            LUISA_ASSERT(argument.buffer.handle != 0u,
                         "A native raster draw contains a null buffer handle.");
            auto buffer = reinterpret_cast<Buffer const *>(argument.buffer.handle);
            barrier->record(
                BufferView{buffer, argument.buffer.offset, argument.buffer.size},
                barrier_usage);
        } else {
            // Textures, bindless arrays and acceleration structures are already
            // rejected by `load_raster()`, so a draw that reaches this point was
            // hand-assembled: report it instead of recording a half barrier.
            LUISA_ERROR_WITH_LOCATION(
                "Native raster draws only support buffer arguments on the "
                "Vulkan backend (tag {}); textures, bindless arrays and "
                "acceleration structures are rejected at load_raster().",
                luisa::to_underlying(argument.tag));
            return;
        }
    }
    if (usage_index != usages.size()) {
        LUISA_ERROR_WITH_LOCATION(
            "A native raster draw carries {} per-argument usage declaration(s) "
            "for {} resource argument(s).",
            usages.size(), usage_index);
    }
}

}// namespace stream_raster_detail

void CommandBuffer::execute(vstd::span<const luisa::unique_ptr<Command>> cmds) {
    // Single argument-planning pass (plan P1.4). The batch's argument buffer is
    // sized before anything can be written into it, so every command that needs
    // an argument block is planned here, once, and the plan is retained under the
    // identity of the command object. The preprocessing pass below runs *after*
    // reordering - which permutes the commands - and looks the plan up by that
    // same identity instead of planning the command a second time. See
    // argument_plan_scratch.h for the association and lifetime contract.
    auto &argument_scratch = _stream.argument_scratch;
    LUISA_ASSERT(
        !argument_scratch.in_use,
        "Vulkan argument-planning scratch is already in use: a stream must not "
        "record a command buffer from inside the recording of another one.");
    argument_scratch.in_use = true;
    auto release_argument_scratch = vstd::scope_exit([&] {
        // The plans are keyed by pointers into this batch's commands, which die
        // with it, so nothing may survive `execute` - not even a stale key. The
        // capacity of all three containers is kept for the next batch.
        argument_scratch.plans.clear();
        argument_scratch.buffer_metadata.clear();
        argument_scratch.validation_values.clear();
        argument_scratch.in_use = false;
    });
    // collect argument buffer
    const auto argument_buffer_alignment = std::max<size_t>(
        32u, device()->properties().limits.minStorageBufferOffsetAlignment);
    const auto argument_buffer_range_limit = static_cast<size_t>(
        device()->properties().limits.maxStorageBufferRange);
    ArgumentBlockLayout planned_argument_buffer_layout;
    auto dispatch_shader = [&](Command const *identity,
                               ShaderDispatchCommandBase const *c,
                               Shader const *shader) {
        auto plan = plan_argument_block(
            shader, c, argument_buffer_range_limit);
        validate_argument_block_plan(plan);
        // Asserts that this command does not already own a plan, and fails before
        // the cumulative sizing below could absorb a duplicate.
        argument_scratch.plans.record(identity, plan);
        size_t dispatch_offset = 0u;
        if (!planned_argument_buffer_layout.append_padded(
                plan.layout.size(), argument_buffer_alignment,
                dispatch_offset)) {
            LUISA_ERROR_WITH_LOCATION(
                "Vulkan cumulative argument-buffer sizing failed: {}.",
                argument_block_layout_status_name(
                    planned_argument_buffer_layout.status()));
        }
    };
    // Reordering is decided once per batch: the visitor assigns each command to
    // a barrier-free layer, or, when the switch is off, keeps strict submission
    // order so this batch can be A/B compared against a reordered one.
    _stream.reorder.set_enabled(device()->command_reorder_enabled());
    // ---- software (fallback) acceleration-structure builds ------------------
    // A mesh/TLAS build of a device that runs the software fallback is not a
    // vkCmdBuildAccelerationStructuresKHR at all: the fallback returns the
    // dispatches of its own build (`FallbackRtxDevice::build_blas` /
    // `build_accel`) and they have to land here, *before* the reorder and the
    // argument preprocessing, so that they are reordered, barriered and
    // preprocessed exactly like the hand-written dispatches they are.  The
    // native build command is dropped: its handle belongs to the fallback, so
    // reinterpreting it as a `Blas`/`Tlas` would be meaningless.
    luisa::vector<luisa::unique_ptr<Command>> fallback_builds;
    luisa::vector<Command const *> expanded_commands;
    expanded_commands.reserve(cmds.size());
    auto is_stream_signal_cmd = [](Command const *cmd) noexcept {
        return cmd->tag() == Command::Tag::ECustomCommand &&
               static_cast<CustomCommand const *>(cmd)->custom_cmd_uuid() ==
                   to_underlying(CustomCommandUUID::STREAM_SIGNAL_EVENT);
    };
    if (device()->use_fallback_rtx()) {
        for (auto &&command : cmds) {
            if (is_stream_signal_cmd(command.get())) { continue; }
            auto expanded = false;
            switch (command->tag()) {
                case Command::Tag::EMeshBuildCommand: {
                    auto c = static_cast<MeshBuildCommand const *>(command.get());
                    if (device()->owns_fallback_blas(c->handle())) {
                        lc::fallback_rtx::FallbackRtxDevice::MeshGeometry geometry{
                            .vertex_buffer = c->vertex_buffer(),
                            .vertex_buffer_offset = c->vertex_buffer_offset(),
                            .vertex_stride = c->vertex_stride(),
                            .vertex_buffer_size = c->vertex_buffer_size(),
                            .triangle_buffer = c->triangle_buffer(),
                            .triangle_buffer_offset = c->triangle_buffer_offset(),
                            .triangle_buffer_size = c->triangle_buffer_size()};
                        auto list = device()->fallback_rtx()->build_blas(
                            c->handle(), geometry);
                        for (auto &&stolen : list.steal_commands()) {
                            expanded_commands.emplace_back(stolen.get());
                            fallback_builds.emplace_back(std::move(stolen));
                        }
                        expanded = true;
                    }
                } break;
                case Command::Tag::EAccelBuildCommand: {
                    auto c = static_cast<AccelBuildCommand const *>(command.get());
                    if (device()->owns_fallback_accel(c->handle())) {
                        auto list = device()->fallback_rtx()->build_accel(
                            c->handle(), c->instance_count(),
                            c->modifications(),
                            c->update_instance_buffer_only());
                        for (auto &&stolen : list.steal_commands()) {
                            expanded_commands.emplace_back(stolen.get());
                            fallback_builds.emplace_back(std::move(stolen));
                        }
                        expanded = true;
                    }
                } break;
                default: break;
            }
            if (!expanded) { expanded_commands.emplace_back(command.get()); }
        }
    } else {
        // The fallback is off: the batch is exactly what the caller handed in,
        // and this loop only borrows the pointers.
        for (auto &&command : cmds) {
            if (is_stream_signal_cmd(command.get())) { continue; }
            expanded_commands.emplace_back(command.get());
        }
    }
    // One plan per command is the most this batch can need. The table keeps
    // its capacity for every later batch of the same shape, so this is not a
    // per-batch allocation in steady state; the cap only keeps a batch that is
    // mostly uploads from pre-sizing a table for commands that never need a plan
    // (record() grows on demand instead).
    constexpr auto argument_plan_capacity_hint = size_t{2048u};
    argument_scratch.plans.begin_batch(
        std::min(expanded_commands.size(), argument_plan_capacity_hint));
    for (auto command : expanded_commands) {
        if (command->tag() == Command::Tag::EShaderDispatchCommand) {
            auto c = static_cast<ShaderDispatchCommand const *>(command);
            if (c->is_indirect()) {
                auto indirect = validate_indirect_dispatch_source(c);
                if (indirect.plan.command_count == 0u) {
                    // A statically empty indirect range has no target/source
                    // resource use and must not perturb reorder boundaries. It is
                    // not dispatched either, so it is planned neither here nor in
                    // the preprocessing pass.
                    continue;
                }
                auto shader = reinterpret_cast<Shader const *>(c->handle());
                validate_indirect_dispatch_target(
                    c, shader, indirect, false);
            }
        }
        command->accept(_stream.reorder);
        switch (command->tag()) {
            case Command::Tag::EShaderDispatchCommand: {
                auto c = static_cast<ShaderDispatchCommand const *>(command);
                auto shader = reinterpret_cast<Shader const *>(c->handle());
                dispatch_shader(command, c, shader);
            } break;
            case Command::Tag::ECustomCommand: {
                auto cmd = static_cast<CustomCommand const *>(command);
                if (cmd->custom_cmd_uuid() == to_underlying(CustomCommandUUID::RASTER_DRAW_SCENE)) {
                    auto c = static_cast<DrawRasterSceneCommand const *>(cmd);
                    // A native raster draw has no saved-argument ABI: there is
                    // no `Shader` (and hence no `SavedArgument` table) behind its
                    // handle, only a `NativeRasterShader` that writes its own
                    // descriptors at encode time. Planning an argument block for
                    // it would read a non-Shader handle, so it is skipped - and
                    // skipped consistently in the preprocess and execute passes
                    // below, which keeps the argument-block bookkeeping exact.
                    if (!c->is_native_shader()) {
                        auto shader = reinterpret_cast<Shader const *>(c->handle());
                        dispatch_shader(command, c, shader);
                    }
                }
            } break;
            case Command::Tag::EMotionInstanceBuildCommand: {
                // Handle motion instance build early (before reordering/preprocess)
                // to ensure child is set before TLAS build references it
                auto c = static_cast<MotionInstanceBuildCommand const *>(command);
                auto mi = reinterpret_cast<MotionInstance *>(c->handle());
                mi->set_child(reinterpret_cast<Blas *>(c->child()));
                mi->set_keyframes(const_cast<MotionInstanceBuildCommand *>(c)->steal_keyframes());
            } break;
            default: break;
        }
    }
    const auto uniform_buffer_size =
        planned_argument_buffer_layout.size();
    auto cmd_lists = _stream.reorder.command_lists();
    // Layer accounting is what decides how many barrier boundaries the batch is
    // recorded with, so make it visible when debugging reorder behaviour.
    auto clear_reorder = vstd::scope_exit([&] {
        _stream.reorder.clear();
    });
    BufferView arg_buffer;
    {
        uniform_data->clear();
        uniform_data->reserve(uniform_buffer_size);
        if (uniform_buffer_size > 0) {
            arg_buffer = _state->upload_alloc.allocate(
                uniform_buffer_size, argument_buffer_alignment);
        }
    }
    ArgumentBlockLayout preprocessed_argument_buffer_layout;
    auto preprocess_arguments = [this, argument_buffer_alignment,
                                 argument_buffer_range_limit,
                                 &argument_scratch,
                                 &preprocessed_argument_buffer_layout](
                                    Command const *identity,
                                    Shader const *shader,
                                    ShaderDispatchCommandBase const *c,
                                    bool is_raster) {
        // P1.4: the plan of this command was computed once, in the sizing pass.
        // It is looked up by command identity - never by position - because
        // reordering ran between the two passes and permuted the emission order.
        // A command without a plan is a defect, not a reason to plan it again.
        auto plan = argument_scratch.plans.consume(identity);
        LUISA_ASSERT(
            plan != nullptr,
            "Vulkan argument preprocessing reached a dispatch that the sizing "
            "pass of this batch did not plan; the argument-plan association is "
            "broken.");
        size_t block_offset = 0u;
        if (!preprocessed_argument_buffer_layout.append_padded(
                plan->layout.size(), argument_buffer_alignment,
                block_offset)) {
            LUISA_ERROR_WITH_LOCATION(
                "Vulkan cumulative argument-buffer preprocessing failed: {}.",
                argument_block_layout_status_name(
                    preprocessed_argument_buffer_layout.status()));
        }
        LUISA_ASSERT(
            uniform_data->size() == block_offset,
            "Vulkan cumulative argument-buffer layout diverged before "
            "dispatch preprocessing (emitted {}, planned {}).",
            uniform_data->size(), block_offset);

        // The trailers are emitted through stream-scoped scratch, so a dispatch
        // neither allocates a vector nor can inherit a value of the dispatch
        // before it: both containers are cleared and refilled here, and nothing
        // keeps a pointer into them past the copy below.
        auto &buffer_metadata = argument_scratch.buffer_metadata;
        auto &validation_values = argument_scratch.validation_values;
        buffer_metadata.clear();
        vstd::push_back_all(buffer_metadata, plan->buffer_metadata_count,
                            StorageBufferMetadata{});
        validation_values.clear();
        validation_values.reserve(shader->validation_count());
        ArgumentBlockLayout emitted_layout{
            argument_buffer_range_limit};
        ResourceBarrierVisitor visitor{
            resource_barrier,
            shader->saved_arguments(),
            uniform_data,
            &emitted_layout,
            block_offset,
            buffer_metadata.empty() ? nullptr : &buffer_metadata,
            shader->validation_count() == 0u ? nullptr :
                                               &validation_values,
            *c,
            is_raster,
            shader->binds(),
            shader->resource_argument_binding_offset(),
            shader->codegen_dialect()};
        visitor.shader = shader;
        // Only the fallback path reads this one (it resolves the software
        // acceleration-structure binding of an `accel` argument).
        visitor.device = device();
        decode_cmd(shader->captured(), visitor);
        decode_cmd(c->arguments(), visitor);
        visitor.arguments.finish("argument preprocessing");
        auto descriptor_tail_count =
            static_cast<uint32_t>(shader->uses_indirect_dispatch()) +
            (shader->printers().empty() ? 0u : 2u);
        auto local_descriptor_count =
            shader->local_descriptor_binding_count();
        auto expected_resource_descriptor_end =
            descriptor_tail_count <= local_descriptor_count ?
                local_descriptor_count - descriptor_tail_count :
                0u;
        LUISA_ASSERT(
            descriptor_tail_count <=
                    local_descriptor_count &&
                visitor.descriptor_index ==
                    expected_resource_descriptor_end,
            "Vulkan barrier preprocessing consumed {} local descriptor "
            "bindings before the indirect/printer tail, but the validated "
            "shader interface requires {}.",
            visitor.descriptor_index,
            expected_resource_descriptor_end);

        LUISA_ASSERT(
            validation_values.size() == shader->validation_count(),
            "Vulkan argument preprocessing produced {} HLSL validation "
            "values, but the shader ABI requires {}.",
            validation_values.size(), shader->validation_count());

        if (!buffer_metadata.empty()) {
            for (auto i = 0u; i < buffer_metadata.size(); ++i) {
                LUISA_ASSERT(buffer_metadata[i].logical_size_bytes != 0u,
                             "Vulkan XIR/SPIR-V buffer metadata slot {} was not populated.",
                             i);
            }
        }

        ArgumentBlockTrailerPlacement emitted_trailer{};
        if (!emitted_layout.append_trailers(
                argument_block_trailer_layout(
                    shader, plan->buffer_metadata_count),
                emitted_trailer)) {
            LUISA_ERROR_WITH_LOCATION(
                "Vulkan argument preprocessing could not append its "
                "checked trailer layout: {}.",
                argument_block_layout_status_name(
                    emitted_layout.status()));
        }
        LUISA_ASSERT(
            emitted_layout.size() == plan->layout.size() &&
                emitted_trailer.metadata_offset ==
                    plan->trailer.metadata_offset &&
                emitted_trailer.metadata_size ==
                    plan->trailer.metadata_size &&
                emitted_trailer.validation_offset ==
                    plan->trailer.validation_offset &&
                emitted_trailer.validation_size ==
                    plan->trailer.validation_size,
            "Vulkan argument sizing and preprocessing produced different "
            "per-dispatch layouts.");

        luisa::vector_resize(
            *uniform_data,
            preprocessed_argument_buffer_layout.size());
        auto copy_trailer = [&](size_t relative_offset, size_t byte_size,
                                const void *data,
                                size_t source_size,
                                const char *trailer_name) noexcept {
            if (byte_size == 0u) {
                LUISA_ASSERT(
                    source_size == 0u,
                    "Vulkan {} trailer has no destination but {} source bytes.",
                    trailer_name, source_size);
                return;
            }
            LUISA_ASSERT(
                block_offset <=
                    std::numeric_limits<size_t>::max() -
                        relative_offset,
                "Vulkan {} trailer destination offset overflowed.",
                trailer_name);
            auto destination_offset = block_offset + relative_offset;
            LUISA_ASSERT(
                byte_size == source_size && data != nullptr,
                "Vulkan {} trailer has {} planned bytes but {} source bytes.",
                trailer_name, byte_size, source_size);
            LUISA_ASSERT(
                destination_offset <= uniform_data->size() &&
                    byte_size <=
                        uniform_data->size() - destination_offset,
                "Vulkan {} trailer exceeds its planned argument block.",
                trailer_name);
            std::memcpy(
                uniform_data->data() + destination_offset,
                data, byte_size);
        };
        copy_trailer(
            emitted_trailer.metadata_offset,
            emitted_trailer.metadata_size,
            buffer_metadata.data(), luisa::size_bytes(buffer_metadata),
            "buffer-metadata");
        copy_trailer(
            emitted_trailer.validation_offset,
            emitted_trailer.validation_size,
            validation_values.data(), luisa::size_bytes(validation_values),
            "HLSL-validation");
        // The block the emitter will bind is recorded against the command that
        // owns it, so the encoding pass can verify it is binding this command's
        // block and not one that happens to be at the same position.
        dispatch_offsets->emplace_back(detail::DispatchArgumentBlock{
            .command = identity,
            .offset = block_offset,
            .size = emitted_layout.size()});
    };
    const bool tl_active = (device()->gpu_timeline() != nullptr) && (_state->timestamp_pool != VK_NULL_HANDLE);
    uint32_t tl_layer_idx = 0u;
    uint32_t tl_prev_q_end = _state->tl_cb_q_begin;
    // Execute
    struct BufferPair {
        VkBuffer src;
        VkBuffer dst;
        [[nodiscard]] bool operator==(BufferPair const &rhs) const noexcept {
            return src == rhs.src && dst == rhs.dst;
        }
        [[nodiscard]] uint64_t hash() const noexcept {
            return luisa::hash_combine({luisa::hash_value(src), luisa::hash_value(dst)});
        }
    };
    struct PendingCopy {
        vstd::unordered_map<BufferPair, vstd::vector<VkBufferCopy2>> copies;
    };
    PendingCopy pending_upload;
    PendingCopy pending_download;
    auto flush_pending_upload = [&]() {
        if (pending_upload.copies.empty()) { return; }
        for (auto &[buffer_pair, regions] : pending_upload.copies) {
            if (regions.empty()) continue;
            VkCopyBufferInfo2 copy_info2{
                VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                nullptr,
                buffer_pair.src,
                buffer_pair.dst,
                static_cast<uint32_t>(regions.size()),
                regions.data()};
            detail::cmd_copy_buffer(_cmdbuffer, device(), &copy_info2);
        }
        pending_upload.copies.clear();
    };
    auto flush_pending_download = [&]() {
        if (pending_download.copies.empty()) { return; }
        for (auto &[buffer_pair, regions] : pending_download.copies) {
            if (regions.empty()) continue;
            VkCopyBufferInfo2 copy_info2{
                VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                nullptr,
                buffer_pair.src,
                buffer_pair.dst,
                static_cast<uint32_t>(regions.size()),
                regions.data()};
            detail::cmd_copy_buffer(_cmdbuffer, device(), &copy_info2);
        }
        pending_download.copies.clear();
    };
    // ---- optional buffer-to-buffer copy batching -----------------------
    //
    // Eligible BufferCopyCommands of the current layer are accumulated
    // instead of being recorded one at a time; `copy_command_plan.h`
    // decides whether the batch becomes one vkCmdCopyMemoryIndirectKHR, one
    // vkCmdCopyMemoryKHR, or exactly the pre-existing sequence of per-copy
    // vkCmdCopyBuffer commands. Deferring a copy to the flush point is safe
    // inside one layer: the command reorder pass only puts commands into the
    // same layer when none of their tracked resource accesses conflict, so
    // moving a copy later inside its own layer cannot change any observed
    // result.
    struct PendingBufferCopy {
        detail::BufferCopyRange range;
        VkBuffer src;
        VkBuffer dst;
        VkDeviceSize src_offset;
        VkDeviceSize dst_offset;
    };
    // A null entry point means the loader did not resolve the command, so the
    // extension cannot be used even when it is reported enabled.
    // VK_NV_copy_memory_indirect is the alias of the KHR extension (same
    // VkCopyMemoryIndirectCommand record layout, different entry point), so
    // a device that only exposes the vendor spelling still gets the fast
    // path.
    auto const indirect_copy_khr = vkCmdCopyMemoryIndirectKHR != nullptr;
    auto const indirect_copy_nv =
        !indirect_copy_khr && vkCmdCopyMemoryIndirectNV != nullptr;
    auto const indirect_copy_enabled =
        device()->optional_extension_enabled(
            detail::OptionalExtension::COPY_MEMORY_INDIRECT) &&
        (indirect_copy_khr || indirect_copy_nv);
    auto const device_address_commands_enabled =
        device()->optional_extension_enabled(
            detail::OptionalExtension::DEVICE_ADDRESS_COMMANDS) &&
        vkCmdCopyMemoryKHR != nullptr;
    luisa::vector<PendingBufferCopy> pending_buffer_copies;
    luisa::vector<detail::BufferCopyRange> pending_buffer_copy_ranges;
    vstd::vector<VkCopyMemoryIndirectCommandKHR> indirect_copy_records;
    vstd::vector<VkDeviceMemoryCopyKHR> device_address_copy_regions;
    auto record_single_buffer_copy = [&](PendingBufferCopy const &copy) {
        VkBufferCopy2 buffer_copy{
            VK_STRUCTURE_TYPE_BUFFER_COPY_2,
            nullptr,
            copy.src_offset,
            copy.dst_offset,
            copy.range.size};
        VkCopyBufferInfo2 copy_info2{
            VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
            nullptr,
            copy.src,
            copy.dst,
            1u,
            &buffer_copy};
        detail::cmd_copy_buffer(_cmdbuffer, device(), &copy_info2);
    };
    // Lazily creates the backend command array. A device without buffer
    // device addresses cannot describe the array at all, so the indirect form
    // is unavailable and the planner selects another path.
    auto indirect_copy_command_array = [&]() -> IndirectCopyCommandBuffer * {
        if (!indirect_copy_enabled ||
            !device()->enable_device_address()) {
            return nullptr;
        }
        if (!_state->indirect_copy_commands) {
            _state->indirect_copy_commands =
                luisa::make_unique<IndirectCopyCommandBuffer>(
                    device(), detail::indirect_copy_command_capacity);
        }
        return _state->indirect_copy_commands.get();
    };
    auto flush_pending_buffer_copies = [&]() {
        if (pending_buffer_copies.empty()) { return; }
        pending_buffer_copy_ranges.clear();
        pending_buffer_copy_ranges.reserve(pending_buffer_copies.size());
        for (auto &&copy : pending_buffer_copies) {
            pending_buffer_copy_ranges.emplace_back(copy.range);
        }
        auto *command_array = indirect_copy_command_array();
        // Every batch of one command buffer owns its own record region: the
        // device reads the records when the command executes, i.e. after the
        // whole command buffer was recorded, so a later batch must never
        // overwrite the records of an earlier one. The cursor is restarted
        // only once the command buffer has retired
        // (CommandBufferState::reset).
        auto const command_array_address =
            command_array != nullptr ?
                command_array->next_record_address() :
                0u;
        auto const command_array_free_records =
            command_array != nullptr ?
                command_array->available_records() :
                size_t{0u};
        auto plan = detail::plan_buffer_copy_batch(
            {.indirect_extension_enabled = indirect_copy_enabled,
             // VUID-vkCmdCopyMemoryIndirectKHR-commandBuffer-10936: the
             // command pool must support one of the queue types the
             // implementation reports.
             .indirect_queue_supported =
                 (_state->command_pool_queue_flags &
                  _state->copy_memory_indirect_supported_queues) != 0u,
             .device_address_commands_enabled =
                 device_address_commands_enabled,
             // Only copies whose two buffers expose a shader device address
             // are accumulated, so a non-empty batch always has them.
             .device_addresses_available = true,
             .candidate_count = static_cast<uint32_t>(
                 pending_buffer_copies.size()),
             .minimum_batch_size =
                 detail::indirect_copy_minimum_batch_size,
             .record_capacity =
                 static_cast<uint32_t>(std::min<size_t>(
                     command_array_free_records,
                     std::numeric_limits<uint32_t>::max())),
             .command_array_address = command_array_address,
             .indirect_aliasing_rule_satisfied =
                 detail::indirect_copy_ranges_honor_aliasing_rule(
                     pending_buffer_copy_ranges.data(),
                     pending_buffer_copy_ranges.size()),
             .device_address_aliasing_rule_satisfied =
                 detail::device_address_copy_ranges_honor_aliasing_rule(
                     pending_buffer_copy_ranges.data(),
                     pending_buffer_copy_ranges.size())});
        switch (plan.mode) {
            case detail::BufferCopyBatchMode::INDIRECT_COMMANDS: {
                LUISA_ASSERT(
                    command_array != nullptr &&
                        plan.batched_copy_count ==
                            pending_buffer_copies.size(),
                    "Vulkan indirect copy batching planned {} of {} copies "
                    "({}).",
                    plan.batched_copy_count,
                    pending_buffer_copies.size(),
                    detail::buffer_copy_batch_status_name(plan.status));
                indirect_copy_records.clear();
                indirect_copy_records.reserve(plan.batched_copy_count);
                for (auto &&copy : pending_buffer_copies) {
                    indirect_copy_records.emplace_back(
                        detail::make_copy_memory_indirect_command(
                            copy.range));
                }
                auto const records_address = command_array->append(
                    indirect_copy_records.data(),
                    indirect_copy_records.size());
                if (records_address == 0u) {
                    // The command array ran out of records for this command
                    // buffer (the planner only promises free space at plan
                    // time), so the pre-existing per-copy path runs instead
                    // of reusing a record region that is already in flight.
                    for (auto &&copy : pending_buffer_copies) {
                        record_single_buffer_copy(copy);
                    }
                    break;
                }
                LUISA_ASSERT(
                    records_address == command_array_address,
                    "Vulkan indirect copy records were appended at {:#x} "
                    "but planned at {:#x}.",
                    records_address, command_array_address);
                if (indirect_copy_khr) {
                    auto info = detail::make_copy_memory_indirect_info(
                        plan, records_address);
                    vkCmdCopyMemoryIndirectKHR(_cmdbuffer, &info);
                } else {
                    // VK_NV_copy_memory_indirect spelling: the very same
                    // tightly packed record array, addressed directly.
                    vkCmdCopyMemoryIndirectNV(
                        _cmdbuffer,
                        records_address,
                        plan.batched_copy_count,
                        static_cast<uint32_t>(plan.stride));
                }
                // The command is really recorded now, so the extension is in
                // use and not merely negotiated.
                device()->mark_optional_extension_active(
                    detail::OptionalExtension::COPY_MEMORY_INDIRECT);
            } break;
            case detail::BufferCopyBatchMode::DEVICE_ADDRESS_REGIONS: {
                LUISA_ASSERT(
                    plan.batched_copy_count ==
                        pending_buffer_copies.size(),
                    "Vulkan device-address copy batching planned {} of {} "
                    "copies ({}).",
                    plan.batched_copy_count,
                    pending_buffer_copies.size(),
                    detail::buffer_copy_batch_status_name(plan.status));
                device_address_copy_regions.clear();
                device_address_copy_regions.reserve(
                    plan.batched_copy_count);
                for (auto &&copy : pending_buffer_copies) {
                    // The batch may reference imported buffers whose
                    // VK_BUFFER_USAGE_STORAGE_BUFFER_BIT cannot be attested,
                    // so the always-valid UNKNOWN flag is used
                    // (VUID-...-addressFlags-13100 / -13122 / -13123).
                    device_address_copy_regions.emplace_back(
                        detail::make_device_memory_copy_region(
                            copy.range, false));
                }
                auto info = detail::make_copy_device_memory_info(
                    plan.batched_copy_count,
                    device_address_copy_regions.data());
                vkCmdCopyMemoryKHR(_cmdbuffer, &info);
                device()->mark_optional_extension_active(
                    detail::OptionalExtension::DEVICE_ADDRESS_COMMANDS);
            } break;
            default: {
                // The pre-existing path: one command per copy, in order.
                for (auto &&copy : pending_buffer_copies) {
                    record_single_buffer_copy(copy);
                }
            } break;
        }
        pending_buffer_copies.clear();
    };
    auto flush_all_pending = [&]() {
        flush_pending_buffer_copies();
        flush_pending_upload();
        flush_pending_download();
    };
    // True when the copy was appended to the batch; otherwise the caller
    // records it immediately, exactly as before.
    auto accumulate_buffer_copy = [&](BufferCopyCommand const *c) noexcept {
        if (!indirect_copy_enabled && !device_address_commands_enabled) {
            // Neither optional extension is usable, so the pre-existing
            // one-command-per-copy path keeps running untouched (and no
            // device address is even queried).
            return false;
        }
        auto *src = reinterpret_cast<Buffer const *>(c->src_handle());
        auto *dst = reinterpret_cast<Buffer const *>(c->dst_handle());
        auto size = static_cast<uint64_t>(c->size());
        auto src_offset = static_cast<uint64_t>(c->src_offset());
        auto dst_offset = static_cast<uint64_t>(c->dst_offset());
        uint64_t src_address = 0u;
        uint64_t dst_address = 0u;
        if (!vk_ext_copy_detail::buffer_view_address(
                src, src_offset, size, src_address) ||
            !vk_ext_copy_detail::buffer_view_address(
                dst, dst_offset, size, dst_address)) {
            return false;
        }
        if (detail::check_buffer_copy_eligibility(
                src_address, dst_address, size) !=
            detail::BufferCopyEligibility::ELIGIBLE) {
            return false;
        }
        if (pending_buffer_copies.empty()) {
            // Nothing is batched yet, so this is exactly the pre-existing
            // "record the deferred staging copies, then the copy" order.
            flush_all_pending();
        }
        pending_buffer_copies.emplace_back(PendingBufferCopy{
            .range = detail::BufferCopyRange{
                .src_address = src_address,
                .dst_address = dst_address,
                .size = size},
            .src = src->vk_buffer(),
            .dst = dst->vk_buffer(),
            .src_offset = static_cast<VkDeviceSize>(src_offset),
            .dst_offset = static_cast<VkDeviceSize>(dst_offset)});
        if (pending_buffer_copies.size() >=
            detail::indirect_copy_command_capacity) {
            flush_pending_buffer_copies();
        }
        return true;
    };

    for (auto &&lst : cmd_lists) {
        luisa::vector<TimelineCmdInfo> tl_layer_cmds;
        dispatch_offsets->clear();
        auto delay_clear = vstd::scope_exit([&]() {
            scratch_buffer_alloc->soft_clear();
        });
        // Preprocess: record resources' states
        for (auto i = lst; i != nullptr; i = i->p_next) {
            auto cmd = i->cmd;
            switch (cmd->tag()) {
                case Command::Tag::EBufferUploadCommand: {
                    auto c = static_cast<BufferUploadCommand const *>(cmd);
                    resource_barrier->record(
                        BufferView{
                            reinterpret_cast<Buffer const *>(c->handle()),
                            c->offset(),
                            c->size()},
                        ResourceBarrier::Usage::kCopyDest);
                } break;
                case Command::Tag::EBufferDownloadCommand: {
                    auto c = static_cast<BufferDownloadCommand const *>(cmd);
                    resource_barrier->record(
                        BufferView{
                            reinterpret_cast<Buffer const *>(c->handle()),
                            c->offset(),
                            c->size()},
                        ResourceBarrier::Usage::kCopySource);
                } break;
                case Command::Tag::EBufferCopyCommand: {
                    auto c = static_cast<BufferCopyCommand const *>(cmd);
                    resource_barrier->record(
                        BufferView{
                            reinterpret_cast<Buffer const *>(c->dst_handle()),
                            c->dst_offset(),
                            c->size()},
                        ResourceBarrier::Usage::kCopyDest);
                    resource_barrier->record(
                        BufferView{
                            reinterpret_cast<Buffer const *>(c->src_handle()),
                            c->src_offset(),
                            c->size()},
                        ResourceBarrier::Usage::kCopySource);
                } break;
                case Command::Tag::EBufferToTextureCopyCommand: {
                    auto c = static_cast<BufferToTextureCopyCommand const *>(cmd);
                    resource_barrier->record(
                        TexView{
                            reinterpret_cast<Texture const *>(c->texture()),
                            c->level()},
                        ResourceBarrier::Usage::kCopyDest);
                    resource_barrier->record(
                        BufferView{
                            reinterpret_cast<Buffer const *>(c->buffer()),
                            c->buffer_offset(),
                            pixel_storage_size(c->storage(), c->size())},
                        ResourceBarrier::Usage::kCopySource);
                } break;
                case Command::Tag::EShaderDispatchCommand: {
                    auto c = static_cast<ShaderDispatchCommand const *>(cmd);
                    if (c->is_indirect()) {
                        auto indirect = validate_indirect_dispatch_source(c);
                        if (indirect.plan.command_count == 0u) { break; }
                        auto shader =
                            reinterpret_cast<Shader const *>(c->handle());
                        validate_indirect_dispatch_target(
                            c, shader, indirect, false);
                        preprocess_arguments(cmd, shader, c, false);
                        resource_barrier->record(
                            BufferView{indirect.source, 0u,
                                       indirect.source->byte_size()},
                            ResourceBarrier::Usage::kComputeRead);
                    } else {
                        auto shader =
                            reinterpret_cast<Shader const *>(c->handle());
                        preprocess_arguments(cmd, shader, c, false);
                    }
                } break;
                case Command::Tag::ETextureUploadCommand: {
                    auto c = static_cast<TextureUploadCommand const *>(cmd);
                    resource_barrier->record(
                        TexView{
                            reinterpret_cast<Texture const *>(c->handle()),
                            c->level()},
                        ResourceBarrier::Usage::kCopyDest);
                } break;
                case Command::Tag::ETextureDownloadCommand: {
                    auto c = static_cast<TextureDownloadCommand const *>(cmd);
                    resource_barrier->record(
                        TexView{
                            reinterpret_cast<Texture const *>(c->handle()),
                            c->level()},
                        ResourceBarrier::Usage::kCopySource);
                } break;
                case Command::Tag::ETextureCopyCommand: {
                    auto c = static_cast<TextureCopyCommand const *>(cmd);
                    resource_barrier->record(
                        TexView(
                            reinterpret_cast<Texture const *>(c->src_handle()),
                            c->src_level()),
                        ResourceBarrier::Usage::kCopySource);
                    resource_barrier->record(
                        TexView(
                            reinterpret_cast<Texture const *>(c->dst_handle()),
                            c->dst_level()),
                        ResourceBarrier::Usage::kCopyDest);
                } break;
                case Command::Tag::ETextureToBufferCopyCommand: {
                    auto c = static_cast<TextureToBufferCopyCommand const *>(cmd);
                    resource_barrier->record(
                        TexView{
                            reinterpret_cast<Texture const *>(c->texture()),
                            c->level()},
                        ResourceBarrier::Usage::kCopySource);
                    resource_barrier->record(
                        BufferView{
                            reinterpret_cast<Buffer const *>(c->buffer()),
                            c->buffer_offset(),
                            pixel_storage_size(c->storage(), c->size())},
                        ResourceBarrier::Usage::kCopyDest);
                } break;
                case Command::Tag::EAccelBuildCommand: {
                    auto c = static_cast<AccelBuildCommand const *>(cmd);
                    reinterpret_cast<Tlas *>(c->handle())->pre_build(*this, c->instance_count(), *write_desc_sets, *bindless_cache, c->modifications(), c->request());
                } break;
            case Command::Tag::EMeshBuildCommand: {
                auto c = static_cast<MeshBuildCommand const *>(cmd);
                if (c->is_cluster_build()) {
                    // Cluster (mega-geometry) builds go to the ClusterBlas
                    // sibling of Blas: VK_NV_cluster_acceleration_structure
                    // produces no VkAccelerationStructureKHR.
                    // TODO(dx): honour is_cluster_build() when the D3D12
                    // cluster AS path lands (NVAPI cluster build / DXR clusters).
                    reinterpret_cast<ClusterBlas *>(c->handle())->pre_build_clusters(*this, c);
                } else {
                    reinterpret_cast<Blas *>(c->handle())->pre_build(*this, c);
                }
            } break;
                case Command::Tag::ECurveBuildCommand: {
                } break;
                case Command::Tag::EMotionInstanceBuildCommand: {
                    // Already handled in the first pass (before reordering)
                } break;
                case Command::Tag::EProceduralPrimitiveBuildCommand: {
                    auto c = static_cast<ProceduralPrimitiveBuildCommand const *>(cmd);
                    reinterpret_cast<Blas *>(c->handle())->pre_build(*this, c);
                } break;
                case Command::Tag::EBindlessArrayUpdateCommand: {
                    auto c = static_cast<BindlessArrayUpdateCommand const *>(cmd);
                    reinterpret_cast<BindlessArray *>(c->handle())->pre_update(resource_barrier);
                } break;
                case Command::Tag::ECustomCommand: {
                    auto c = static_cast<CustomCommand const *>(cmd);
                    switch (c->custom_cmd_uuid()) {
                        case to_underlying(CustomCommandUUID::RASTER_CLEAR_DEPTH): {
                            auto cmd = static_cast<ClearDepthCommand const *>(c);
                            auto tex = reinterpret_cast<Texture const *>(cmd->handle());
                            resource_barrier->record(
                                TexView(tex, 0),
                                ResourceBarrier::Usage::kDepthClear);
                        } break;
                        case to_underlying(CustomCommandUUID::RASTER_CLEAR_RENDER_TARGET): {
                            auto cmd = static_cast<ClearRenderTargetCommand const *>(c);
                            auto tex = reinterpret_cast<Texture const *>(cmd->handle());
                            resource_barrier->record(
                                TexView(tex, cmd->level()),
                                ResourceBarrier::Usage::kRenderTargetClear);
                        } break;
                        case to_underlying(CustomCommandUUID::RASTER_DRAW_SCENE): {
                            auto cmd = static_cast<DrawRasterSceneCommand const *>(c);
                            if (cmd->is_native_shader()) {
                                // No `RasterShader`, no saved arguments: record
                                // the draw's declared usages instead (same rule
                                // as the NATIVE_SHADER_DISPATCH arm below, with
                                // the raster stage's barrier usages).
                                stream_raster_detail::record_native_raster_usages(
                                    resource_barrier, cmd);
                            } else {
                                auto shader = reinterpret_cast<RasterShader *>(cmd->handle());
                                // `i->cmd` is this draw's command identity, the
                                // one the sizing pass planned (the local `cmd`
                                // shadows it with the typed pointer).
                                preprocess_arguments(i->cmd, shader, cmd, true);
                            }
                            for (auto &i : cmd->rtv_texs()) {
                                auto tex = reinterpret_cast<Texture const *>(i.handle);
                                resource_barrier->record(
                                    TexView(tex, i.level),
                                    ResourceBarrier::Usage::kRenderTarget);
                            }
                            if (cmd->dsv_tex().handle != invalid_resource_handle) {
                                auto tex = reinterpret_cast<Texture const *>(cmd->dsv_tex().handle);
                                resource_barrier->record(
                                    TexView(tex, cmd->dsv_tex().level),
                                    ResourceBarrier::Usage::kDepthWrite);
                            }
                        } break;
                        case to_underlying(CustomCommandUUID::CUSTOM_DISPATCH): {
                            auto custom_cmd = static_cast<VKCustomCmd const *>(c);
                            for (auto &&i : const_cast<VKCustomCmd *>(custom_cmd)->get_resource_usages()) {
                                record_custom_resource_usage(
                                    resource_barrier, i);
                            }
                        } break;
                        case to_underlying(CustomCommandUUID::VK_CUDA_LAUNCH_KERNEL): {
                            auto cmd = static_cast<vk_cuda_interop::CudaKernelLaunchCommand const *>(c);
                            cmd->traverse_arguments([&]<typename T>(T const &arg, Usage usage) noexcept {
                                auto barrier_usage =
                                    (luisa::to_underlying(usage) &
                                     luisa::to_underlying(Usage::WRITE)) != 0u ?
                                        ResourceBarrier::Usage::kComputeUAV :
                                        ResourceBarrier::Usage::kComputeRead;
                                if constexpr (std::is_same_v<T, Argument::Buffer>) {
                                    LUISA_ASSERT(arg.handle != 0u,
                                                 "Vulkan CUDA kernel launch contains a null buffer handle.");
                                    auto buffer = reinterpret_cast<Buffer const *>(arg.handle);
                                    resource_barrier->record(
                                        BufferView(buffer, arg.offset, arg.size),
                                        barrier_usage);
                                } else if constexpr (std::is_same_v<T, Argument::Texture>) {
                                    LUISA_ASSERT(arg.handle != 0u,
                                                 "Vulkan CUDA kernel launch contains a null texture handle.");
                                    auto tex = reinterpret_cast<Texture const *>(arg.handle);
                                    resource_barrier->record(
                                        TexView(tex, arg.level),
                                        barrier_usage);
                                } else {
                                    LUISA_ERROR_WITH_LOCATION(
                                        "Bindless-array and accel arguments are not "
                                        "supported for CUDA kernel launch.");
                                }
                            });
                        } break;
                        case to_underlying(CustomCommandUUID::NATIVE_SHADER_DISPATCH): {
                            // A native shader dispatch declares its resource usages
                            // through traverse_arguments; they are the barrier
                            // contract (see the header of native_shader_ext.h).
                            auto cmd = static_cast<NativeShaderDispatchCommand const *>(c);
                            cmd->traverse_arguments([&]<typename T>(T const &arg, Usage usage) noexcept {
                                auto barrier_usage =
                                    (luisa::to_underlying(usage) &
                                     luisa::to_underlying(Usage::WRITE)) != 0u ?
                                        ResourceBarrier::Usage::kComputeUAV :
                                        ResourceBarrier::Usage::kComputeRead;
                                if constexpr (std::is_same_v<T, Argument::Buffer>) {
                                    LUISA_ASSERT(arg.handle != 0u,
                                                 "Native shader dispatch contains a null buffer handle.");
                                    auto buffer = reinterpret_cast<Buffer const *>(arg.handle);
                                    resource_barrier->record(
                                        BufferView(buffer, arg.offset, arg.size),
                                        barrier_usage);
                                } else {
                                    LUISA_ERROR_WITH_LOCATION(
                                        "Native shader dispatches only support buffer "
                                        "arguments on the Vulkan backend.");
                                }
                            });
                        } break;
                        case to_underlying(CustomCommandUUID::BATCH_BUFFER_TO_TEXTURE_COPY): {
                            auto bc = static_cast<BatchBufferToTextureCopyCommand const *>(c);
                            if (bc->regions().empty()) { break; }
                            auto *buf = reinterpret_cast<Buffer const *>(bc->buffer());
                            auto *tex = reinterpret_cast<Texture const *>(bc->texture());
                            resource_barrier->record(
                                BufferView{buf, bc->buffer_range_offset(), bc->buffer_range_size()},
                                ResourceBarrier::Usage::kCopySource);
                            for (uint32_t mip = bc->min_mip_level(); mip <= bc->max_mip_level(); ++mip) {
                                resource_barrier->record(
                                    TexView{tex, mip},
                                    ResourceBarrier::Usage::kCopyDest);
                            }
                        } break;
                        case to_underlying(CustomCommandUUID::STREAM_SIGNAL_EVENT): {
                        } break;
                        // NOTE: unimplemented command type — extend as new CustomCommandUUID
                        // values are added.
                        default: {
                            LUISA_ERROR("Command type not supported.");
                        } break;
                    }
                } break;
                default: break;
            }
        }
        resource_barrier->update_states(_cmdbuffer);

        size_t dispatch_offset_index = 0u;
        auto set_dispatch_args = [&](
                                     Command const *identity,
                                     BindPropVisitor &visitor,
                                     ShaderDispatchCommandBase const *c,
                                     Shader const *shader,
                                     const Buffer *indirect_source = nullptr) {
            LUISA_ASSERT(
                dispatch_offset_index < dispatch_offsets->size(),
                "Vulkan descriptor binding requested argument block {} from "
                "a {}-entry preprocessing table.",
                dispatch_offset_index, dispatch_offsets->size());
            auto &block = (*dispatch_offsets)[dispatch_offset_index++];
            // The preprocessing pass filled the table in emission order, so the
            // block at this position must belong to the command being encoded.
            // Checking the identity turns "the offsets line up" into "the layout
            // is this command's layout", which is what P1.4 hands over.
            LUISA_ASSERT(
                block.command == identity,
                "Vulkan descriptor binding is about to bind another command's "
                "argument block: the preprocessing and encoding passes of this "
                "layer disagree about the command order.");
            auto relative_offset = block.offset;
            auto descriptor_range = block.size;
            LUISA_ASSERT(
                relative_offset <= arg_buffer.size_bytes &&
                    descriptor_range <=
                        arg_buffer.size_bytes - relative_offset,
                "Vulkan dispatch argument block offset {} and size {} exceed "
                "the {}-byte upload allocation.",
                relative_offset, descriptor_range,
                arg_buffer.size_bytes);
            uint desc_index = 0;
            visitor.cmdbuffer = this;
            // Plan P2.3: this call owns the write list of the dispatch it encodes, so it
            // starts empty (a push consumes it, and one dispatch pushes twice at most).
            write_desc_sets->clear();
            // A push shader has no VkDescriptorSet: its set 0 lives in the command
            // buffer's push descriptor bank, so nothing is allocated for it.
            // Phase 3B (opt A): the interning id of this shader's set-0 layout is what
            // lets the state reuse the set it allocated for the batch before this one
            // instead of calling vkAllocateDescriptorSets again.
            visitor.cached_desc_entry = nullptr;
            visitor.desc_set = shader->uses_push_descriptors() ?
                                   VK_NULL_HANDLE :
                                   _state->allocate_descriptor_set(
                                       shader->desc_set_layout_id(0u),
                                       shader->desc_set_layout()[0],
                                       reinterpret_cast<uint64_t>(shader),
                                       &visitor.cached_desc_entry);
            if (descriptor_range > 0u) {
                LUISA_ASSERT(
                    arg_buffer.buffer != nullptr,
                    "Vulkan dispatch has a nonempty argument block without "
                    "an upload buffer.");
                constexpr auto max_device_offset =
                    std::numeric_limits<VkDeviceSize>::max();
                LUISA_ASSERT(
                    arg_buffer.offset <= max_device_offset &&
                        relative_offset <= max_device_offset - arg_buffer.offset,
                    "Vulkan argument descriptor offset {} + {} is not "
                    "representable by VkDeviceSize.",
                    arg_buffer.offset, relative_offset);
                auto descriptor_offset =
                    static_cast<VkDeviceSize>(arg_buffer.offset) +
                    static_cast<VkDeviceSize>(relative_offset);
                LUISA_ASSERT(
                    descriptor_range <= max_device_offset &&
                        descriptor_offset <= arg_buffer.buffer->byte_size() &&
                        descriptor_range <=
                            arg_buffer.buffer->byte_size() - descriptor_offset,
                    "Vulkan argument descriptor offset {} and range {} exceed "
                    "the backing upload buffer size {}.",
                    descriptor_offset, descriptor_range,
                    arg_buffer.buffer->byte_size());
                auto arg_buffer_info = temp_desc->allocate_memory<VkDescriptorBufferInfo>();
                *arg_buffer_info = VkDescriptorBufferInfo{
                    arg_buffer.buffer->vk_buffer(),
                    descriptor_offset,
                    static_cast<VkDeviceSize>(descriptor_range)};
                write_desc_sets->emplace_back(VkWriteDescriptorSet{
                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    nullptr,
                    visitor.desc_set,
                    desc_index++,
                    0,
                    1,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    nullptr,
                    arg_buffer_info,
                    nullptr});
            }
            if (shader->has_constant_ubo()) {
                auto ubo_info = temp_desc->allocate_memory<VkDescriptorBufferInfo>();
                *ubo_info = VkDescriptorBufferInfo{
                    shader->constant_ubo()->vk_buffer(),
                    0,
                    shader->constant_ubo()->byte_size()};
                write_desc_sets->emplace_back(VkWriteDescriptorSet{
                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    nullptr,
                    visitor.desc_set,
                    desc_index++,
                    0,
                    1,
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    nullptr,
                    ubo_info,
                    nullptr});
            }
            visitor.desc_index = desc_index;
            visitor.img_views = &_state->img_views;
            visitor.arguments =
                SavedArgumentCursor{shader->saved_arguments()};
            visitor.bindings = shader->binds();
            visitor.shader = shader;
            visitor.codegen_dialect = shader->codegen_dialect();
            visitor.uav_usage =
                shader->shader_tag() == Shader::ShaderTag::kRasterShader ?
                    ResourceBarrier::Usage::kRasterUAV :
                    ResourceBarrier::Usage::kComputeUAV;
            decode_cmd(shader->captured(), visitor);
            decode_cmd(c->arguments(), visitor);
            visitor.arguments.finish("descriptor binding");
            if (shader->uses_indirect_dispatch()) {
                auto source_view = BufferView{};
                if (indirect_source != nullptr) {
                    LUISA_ASSERT(
                        indirect_source->is_indirect_dispatch_buffer(),
                        "Vulkan indirect metadata source has the wrong buffer type.");
                    source_view = BufferView{
                        indirect_source, 0u,
                        indirect_source->byte_size()};
                } else {
                    auto *dummy = device()->indirect_dispatch_dummy();
                    LUISA_ASSERT(
                        dummy != nullptr &&
                            dummy->byte_size() >=
                                IndirectDispatchLayout::header_size +
                                    IndirectDispatchLayout::record_size,
                        "Vulkan direct dispatch has no valid persistent "
                        "indirect-metadata descriptor.");
                    source_view = BufferView{dummy};
                }
                auto *binding = visitor.binding_at(visitor.desc_index);
                LUISA_ASSERT(
                    binding != nullptr &&
                        binding->type == hlsl::ShaderVariableType::SPIRVIndirectDispatch,
                    "Vulkan native SPIR-V shader is missing its indirect "
                    "metadata descriptor at binding {}.",
                    visitor.desc_index);
                auto descriptor =
                    temp_desc->allocate_memory<VkDescriptorBufferInfo>();
                *descriptor = VkDescriptorBufferInfo{
                    source_view.buffer->vk_buffer(), source_view.offset,
                    source_view.size_bytes};
                write_desc_sets->emplace_back(VkWriteDescriptorSet{
                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    nullptr,
                    visitor.desc_set,
                    visitor.desc_index++,
                    0u,
                    1u,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    nullptr,
                    descriptor,
                    nullptr});
            } else {
                LUISA_ASSERT(
                    indirect_source == nullptr,
                    "Vulkan indirect dispatch cannot bind a shader without "
                    "the native logical-metadata descriptor.");
            }
            return visitor.desc_index;
        };
        auto push_shader_constants = [&]<typename T>(
                                         const Shader *shader,
                                         VkShaderStageFlags stage_flags,
                                         uint32_t offset,
                                         const T &value) {
            static_assert(sizeof(T) <=
                          std::numeric_limits<uint32_t>::max());
            constexpr auto size = static_cast<uint32_t>(sizeof(T));
            LUISA_ASSERT(
                detail::valid_push_constant_write(
                    offset, size, shader->push_constant_size()),
                "Vulkan push-constant write [{}, {}) exceeds or is not "
                "aligned to the shader range [0, {}).",
                offset, static_cast<uint64_t>(offset) + size,
                shader->push_constant_size());
            vkCmdPushConstants(
                _cmdbuffer, shader->pipeline_layout(), stage_flags,
                offset, size, &value);
        };
        // The shader is passed mutable because the set-0 descriptor-update
        // templates are cached on it (plan P2.2).
        auto bind_shader_desc = [&](BindPropVisitor &visitor, Shader *shader, VkPipelineBindPoint bind_point) {
            LUISA_ASSERT(
                visitor.desc_index ==
                    shader->local_descriptor_binding_count(),
                "Vulkan dispatch consumed {} local descriptor bindings but "
                "the validated shader interface requires {}.",
                visitor.desc_index,
                shader->local_descriptor_binding_count());
            // Plan P2.3: the shader decided at creation whether set 0 is a push
            // descriptor set. The flag describes the *layout object* this shader
            // was built with, so this branch cannot change for a given shader and
            // a frame can never mix the two paths for one set 0.
            if (shader->uses_push_descriptors()) {
                // No vkAllocateDescriptorSets, no vkUpdateDescriptorSets, no
                // template and no set-0 vkCmdBindDescriptorSets: one
                // vkCmdPushDescriptorSetKHR carries exactly the write list the
                // classic path would have written (Vulkan ignores each write's
                // dstSet for a push, so the list is passed through unchanged -
                // including a legal pNext such as an acceleration structure).
                LUISA_ASSERT(
                    !write_desc_sets->empty(),
                    "Vulkan push-descriptor shader produced no set-0 writes, "
                    "but a push call requires at least one.");
                auto push = detail::push_descriptor_set(
                    detail::PushDescriptorRecordRequest{
                        .command_buffer = _cmdbuffer,
                        .bind_point = bind_point,
                        .layout = shader->pipeline_layout(),
                        .set = 0u,
                        .max_push_descriptors =
                            device()->max_push_descriptors(),
                        .api = device()->push_descriptor_api(),
                        .allow_push = true},
                    luisa::span<const VkWriteDescriptorSet>{
                        write_desc_sets->data(),
                        write_desc_sets->size()});
                LUISA_ASSERT(
                    push.pushed,
                    "Vulkan push descriptor set 0 was refused at record time "
                    "({}); the shader was created with a push layout, so this "
                    "cannot fall back to a classic set mid-recording.",
                    detail::push_descriptor_status_name(push.status));
                // One push consumes the write list, so nothing leaks into the
                // next dispatch. An indirect dispatch re-pushes its own
                // retained list (see `indirect_repush_writes` at its call site).
                write_desc_sets->clear();
                // Exactly the state a set-0 bind would have recorded, and only
                // for slot 0: the global sets keep their tracked handles, which
                // is what lets their bind below stay skippable across dispatches.
                _state->bind_state.push_descriptor_sets(
                    bind_point,
                    detail::handle_to_uint64(shader->pipeline_layout()),
                    0u);
                desc_sets->clear();
            } else {
                if (!write_desc_sets->empty()) {
                    static const bool diff_desc_enabled =
                        !luisa::compute::detail::env_flag("LUISA_VK_DISABLE_DIFF_DESC_UPDATE");
                    if (diff_desc_enabled && visitor.cached_desc_entry != nullptr &&
                        !device()->descriptor_update_template_capable()) {
                        auto &cached_bindings = visitor.cached_desc_entry->bindings;
                        if (device()->resource_destroy_generation() !=
                            _state->cached_resource_generation) {
                            cached_bindings.clear();
                        }
                        uint32_t max_binding = 0u;
                        for (auto const &w : *write_desc_sets) {
                            if (w.dstBinding > max_binding) {
                                max_binding = w.dstBinding;
                            }
                        }
                        if (cached_bindings.size() <= max_binding) {
                            cached_bindings.resize(
                                static_cast<size_t>(max_binding) + 1u);
                        }
                        size_t keep = 0u;
                        for (size_t wi = 0u; wi < write_desc_sets->size(); ++wi) {
                            auto const &w = (*write_desc_sets)[wi];
                            bool unchanged = false;
                            if (w.descriptorCount == 1u && w.dstArrayElement == 0u) {
                                if ((w.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
                                     w.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) &&
                                    w.pNext == nullptr && w.pBufferInfo != nullptr) {
                                    CommandBufferState::CachedDescriptorBinding cur{
                                        w.descriptorType,
                                        detail::handle_to_uint64(w.pBufferInfo->buffer),
                                        w.pBufferInfo->offset,
                                        w.pBufferInfo->range};
                                    if (cached_bindings[w.dstBinding] == cur) {
                                        unchanged = true;
                                    } else {
                                        cached_bindings[w.dstBinding] = cur;
                                    }
                                } else if (w.descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR &&
                                           w.pNext != nullptr) {
                                    auto const *as_write =
                                        reinterpret_cast<VkWriteDescriptorSetAccelerationStructureKHR const *>(w.pNext);
                                    if (as_write->sType ==
                                            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR &&
                                        as_write->accelerationStructureCount == 1u &&
                                        as_write->pAccelerationStructures != nullptr) {
                                        CommandBufferState::CachedDescriptorBinding cur{
                                            w.descriptorType,
                                            detail::handle_to_uint64(as_write->pAccelerationStructures[0]),
                                            0u,
                                            0u};
                                        if (cached_bindings[w.dstBinding] == cur) {
                                            unchanged = true;
                                        } else {
                                            cached_bindings[w.dstBinding] = cur;
                                        }
                                    } else {
                                        cached_bindings[w.dstBinding] = {};
                                    }
                                } else {
                                    cached_bindings[w.dstBinding] = {};
                                }
                            } else if (w.dstBinding < cached_bindings.size()) {
                                cached_bindings[w.dstBinding] = {};
                            }
                            if (!unchanged) {
                                if (keep != wi) {
                                    (*write_desc_sets)[keep] = w;
                                }
                                ++keep;
                            }
                        }
                        write_desc_sets->resize(keep);
                        if (!write_desc_sets->empty()) {
                            LUISA_ASSERT(
                                write_desc_sets->size() <=
                                    std::numeric_limits<uint32_t>::max(),
                                "Vulkan descriptor-write count is not representable.");
                            vkUpdateDescriptorSets(
                                device()->logic_device(),
                                static_cast<uint32_t>(write_desc_sets->size()),
                                write_desc_sets->data(), 0u, nullptr);
                            submit_profile::record_descriptor_set_update();
                            write_desc_sets->clear();
                        }
                    } else {
                        LUISA_ASSERT(
                            write_desc_sets->size() <=
                                std::numeric_limits<uint32_t>::max(),
                            "Vulkan descriptor-write count is not representable.");
                        // Plan P2.2: the one funnel for this dispatch's set-0 update.
                        // It issues either one vkUpdateDescriptorSetWithTemplate (a
                        // cached template created from exactly this write shape) or,
                        // byte for byte, the pre-P2.2 vkUpdateDescriptorSets call with
                        // the same writes - never both and never a partial write. The
                        // decision depends on the device capability latched at creation
                        // and on the shape, never on a counter or a measurement.
                        detail::DescriptorSetUpdateRequest update_request{
                            .device = device()->logic_device(),
                            .set = visitor.desc_set,
                            .layout = shader->desc_set_layout()[0u],
                            .bind_point = bind_point,
                            .api = device()->descriptor_update_template_api(),
                            .allow_templates =
                                device()->descriptor_update_template_capable()};
                        auto update = detail::update_descriptor_set(
                            update_request, shader->descriptor_update_templates(),
                            luisa::span<const VkWriteDescriptorSet>{
                                write_desc_sets->data(), write_desc_sets->size()});
                        // "Active" in VulkanExtensionStatusExt means "the backend really
                        // uses this path", so it is marked where a template is actually
                        // created instead of on every dispatch.
                        if (update.created_template) {
                            device()->mark_optional_extension_active(
                                detail::OptionalExtension::DESCRIPTOR_UPDATE_TEMPLATE);
                        }
                        write_desc_sets->clear();
                    }
                }
                desc_sets->clear();
                desc_sets->push_back(visitor.desc_set);
            }
            desc_sets->push_back(device()->sampler_set());
            if (shader->use_buffer_bindless()) {
                desc_sets->push_back(device()->bdls_buffer_set());
            }
            if (shader->use_tex2d_bindless()) {
                desc_sets->push_back(device()->bdls_tex2d_set());
            }
            if (shader->use_tex3d_bindless()) {
                desc_sets->push_back(device()->bdls_tex3d_set());
            }
            // Plan P2.3: on the push path set 0 is not part of the bind, so the
            // assembled list is exactly the global sets (sampler + bindless heaps)
            // and starts at set 1.
            auto const bound_set_count =
                desc_sets->size() +
                (shader->uses_push_descriptors() ? 1u : 0u);
            LUISA_ASSERT(
                bound_set_count == shader->desc_set_layout().size() &&
                    desc_sets->size() <=
                        std::numeric_limits<uint32_t>::max(),
                "Vulkan dispatch assembled {} descriptor sets for a "
                "validated pipeline layout with {} sets (set 0 {}).",
                desc_sets->size(), shader->desc_set_layout().size(),
                shader->uses_push_descriptors() ? "pushed" : "bound");
            // P1.2: skipped only when this exact layout object and every set index
            // of the range already hold the requested handle (plan P2.3: the
            // tracker is per slot, so the global sets below are no longer re-bound
            // just because the local set 0 changed). With the push path the first
            // set is 1 and the local set is not part of the bind at all.
            detail::cmd_bind_descriptor_sets(
                _state->bind_state,
                _cmdbuffer,
                bind_point,
                shader->pipeline_layout(),
                shader->uses_push_descriptors() ? 1u : 0u,
                static_cast<uint32_t>(desc_sets->size()),
                desc_sets->data(),
                0u,
                nullptr);
        };
        auto prepare_indirect_dispatch = [&](
                                             const Buffer *source,
                                             IndirectDispatchPlan plan,
                                             uint3 target_block_size) {
            LUISA_ASSERT(source != nullptr &&
                             source->is_indirect_dispatch_buffer(),
                         "Vulkan indirect preparation requires an indirect source buffer.");
            if (plan.command_count == 0u) { return BufferView{}; }
            LUISA_ASSERT(
                plan.scratch_size_bytes <=
                    device()->properties().limits.maxStorageBufferRange,
                "Vulkan indirect command scratch range {} exceeds "
                "maxStorageBufferRange {}.",
                plan.scratch_size_bytes,
                device()->properties().limits.maxStorageBufferRange);
            auto descriptor_alignment = std::max<size_t>(
                16u,
                device()->properties().limits.minStorageBufferOffsetAlignment);
            auto scratch = scratch_buffer_alloc->allocate(
                plan.scratch_size_bytes, descriptor_alignment);
            auto command_buffer = BufferView{
                reinterpret_cast<const Buffer *>(scratch.handle),
                scratch.offset, plan.scratch_size_bytes};
            resource_barrier->record(
                command_buffer,
                ResourceBarrier::Usage::kComputeUAV);
            resource_barrier->update_states(_cmdbuffer);

            auto prepare_shader =
                device()->prepare_indirect_kernel.get(device());
            // Phase 3B (opt A): the indirect-preparation kernel's set is deliberately
            // not cached (id 0): it is bound once per indirect command and the state
            // records the "uncached allocation" so its pools are reset on retirement.
            VkDescriptorSet prepare_set = _state->allocate_descriptor_set(
                0u, prepare_shader->desc_set_layout()[0]);
            std::array descriptor_infos{
                VkDescriptorBufferInfo{
                    source->vk_buffer(), 0u, source->byte_size()},
                VkDescriptorBufferInfo{
                    command_buffer.buffer->vk_buffer(),
                    command_buffer.offset, command_buffer.size_bytes}};
            std::array descriptor_writes{
                VkWriteDescriptorSet{
                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    nullptr,
                    prepare_set,
                    0u,
                    0u,
                    1u,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    nullptr,
                    &descriptor_infos[0],
                    nullptr},
                VkWriteDescriptorSet{
                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    nullptr,
                    prepare_set,
                    1u,
                    0u,
                    1u,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    nullptr,
                    &descriptor_infos[1],
                    nullptr}};
            LUISA_ASSERT(
                descriptor_writes.size() ==
                    prepare_shader->local_descriptor_binding_count(),
                "Vulkan indirect preparation consumed {} local descriptor "
                "bindings but its validated interface requires {}.",
                descriptor_writes.size(),
                prepare_shader->local_descriptor_binding_count());
            vkUpdateDescriptorSets(
                device()->logic_device(),
                static_cast<uint32_t>(descriptor_writes.size()),
                descriptor_writes.data(), 0u, nullptr);
            // Counter-only: the indirect preparation kernel is a real compute
            // dispatch with its own update, set bind and pipeline bind (F2/F3);
            // the two binds go through the P1.2 tracker, so the counters describe
            // the calls that are actually recorded.
            submit_profile::record_descriptor_set_update();
            detail::cmd_bind_descriptor_sets(
                _state->bind_state, _cmdbuffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                prepare_shader->pipeline_layout(), 0u, 1u,
                &prepare_set, 0u, nullptr);
            detail::cmd_bind_pipeline(
                _state->bind_state, _cmdbuffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                prepare_shader->pipeline(),
                prepare_shader->pipeline_layout());

            auto &&limits = device()->properties().limits;
            LUISA_ASSERT(
                limits.maxComputeWorkGroupCount[0] != 0u &&
                    target_block_size.x != 0u &&
                    target_block_size.y != 0u &&
                    target_block_size.z != 0u,
                "Vulkan device or target shader reports a zero compute "
                "workgroup dimension.");
            auto max_commands_per_dispatch =
                static_cast<uint64_t>(
                    limits.maxComputeWorkGroupCount[0]) *
                IndirectDispatchLayout::prepare_block_size;
            auto command_base = uint64_t{0u};
            while (command_base < plan.command_count) {
                auto remaining =
                    static_cast<uint64_t>(plan.command_count) -
                    command_base;
                auto chunk = std::min(
                    remaining, max_commands_per_dispatch);
                IndirectDispatchPrepareConstants constants{
                    .command_count = plan.command_count,
                    .source_record_offset =
                        plan.source_record_offset,
                    .target_block_size_x = target_block_size.x,
                    .target_block_size_y = target_block_size.y,
                    .target_block_size_z = target_block_size.z,
                    .max_group_count_x = static_cast<uint32_t>(
                        std::min<uint64_t>(
                            limits.maxComputeWorkGroupCount[0],
                            indirect_dispatch_max_group_count_for_uint32_global_id(
                                target_block_size.x))),
                    .max_group_count_y = static_cast<uint32_t>(
                        std::min<uint64_t>(
                            limits.maxComputeWorkGroupCount[1],
                            indirect_dispatch_max_group_count_for_uint32_global_id(
                                target_block_size.y))),
                    .max_group_count_z = static_cast<uint32_t>(
                        std::min<uint64_t>(
                            limits.maxComputeWorkGroupCount[2],
                            indirect_dispatch_max_group_count_for_uint32_global_id(
                                target_block_size.z))),
                    .command_base =
                        static_cast<uint32_t>(command_base)};
                push_shader_constants(
                    prepare_shader, VK_SHADER_STAGE_COMPUTE_BIT,
                    0u, constants);
                auto group_count = static_cast<uint32_t>(
                    chunk / IndirectDispatchLayout::prepare_block_size +
                    (chunk % IndirectDispatchLayout::prepare_block_size != 0u));
                vkCmdDispatch(
                    _cmdbuffer, group_count, 1u, 1u);
                command_base += chunk;
            }

            resource_barrier->record(
                command_buffer,
                ResourceBarrier::Usage::kIndirectArgs);
            resource_barrier->update_states(_cmdbuffer);
            return command_buffer;
        };
        // Post process: actual start record command to commandbuffer
        for (auto i = lst; i != nullptr; i = i->p_next) {
            auto cmd = i->cmd;
            if (tl_active) {
                TimelineCmdInfo info{};
                info.tag = static_cast<uint32_t>(cmd->tag());
                switch (cmd->tag()) {
                    case Command::Tag::EBufferUploadCommand: {
                        auto c = static_cast<BufferUploadCommand const *>(cmd);
                        info.handle = c->handle();
                        info.bytes = c->size();
                    } break;
                    case Command::Tag::EBufferDownloadCommand: {
                        auto c = static_cast<BufferDownloadCommand const *>(cmd);
                        info.handle = c->handle();
                        info.bytes = c->size();
                    } break;
                    case Command::Tag::EBufferCopyCommand: {
                        auto c = static_cast<BufferCopyCommand const *>(cmd);
                        info.handle = c->src_handle();
                        info.dst_handle = c->dst_handle();
                        info.bytes = c->size();
                    } break;
                    case Command::Tag::EBufferToTextureCopyCommand: {
                        auto c = static_cast<BufferToTextureCopyCommand const *>(cmd);
                        info.handle = c->buffer();
                        info.dst_handle = c->texture();
                    } break;
                    case Command::Tag::EShaderDispatchCommand: {
                        auto c = static_cast<ShaderDispatchCommand const *>(cmd);
                        info.handle = c->handle();
                        if (c->is_indirect()) {
                            info.dx = 1u;
                        } else if (c->is_multiple_dispatch()) {
                            info.dx = static_cast<uint32_t>(c->dispatch_sizes().size());
                        } else {
                            auto ds = c->dispatch_size();
                            info.dx = ds.x;
                            info.dy = ds.y;
                            info.dz = ds.z;
                        }
                    } break;
                    case Command::Tag::EBindlessArrayUpdateCommand: {
                        auto c = static_cast<BindlessArrayUpdateCommand const *>(cmd);
                        info.handle = c->handle();
                    } break;
                    case Command::Tag::ECustomCommand: {
                        auto c = static_cast<CustomCommand const *>(cmd);
                        if (c->custom_cmd_uuid() == to_underlying(CustomCommandUUID::BATCH_BUFFER_TO_TEXTURE_COPY)) {
                            auto bc = static_cast<BatchBufferToTextureCopyCommand const *>(c);
                            info.tag = static_cast<uint32_t>(Command::Tag::EBufferToTextureCopyCommand);
                            info.handle = bc->buffer();
                            info.dst_handle = bc->texture();
                            info.bytes = bc->buffer_range_size();
                            info.dx = static_cast<uint32_t>(bc->regions().size());
                        }
                    } break;
                    default: break;
                }
                tl_layer_cmds.emplace_back(info);
            }
            switch (cmd->tag()) {
                case Command::Tag::EBufferUploadCommand: {
                    auto c = static_cast<BufferUploadCommand const *>(cmd);
                    auto chunk = _state->upload_alloc.allocate(c->size(), 16);
                    static_cast<UploadBuffer const *>(chunk.buffer)->copy_from(c->data(), chunk.offset, c->size());
                    VkBuffer src = chunk.buffer->vk_buffer();
                    VkBuffer dst = reinterpret_cast<Buffer const *>(c->handle())->vk_buffer();
                    auto &regions = pending_upload.copies[BufferPair{src, dst}];
                    regions.push_back({VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                                       nullptr,
                                       chunk.offset,
                                       c->offset(),
                                       c->size()});
                } break;
                case Command::Tag::EBufferDownloadCommand: {
                    auto c = static_cast<BufferDownloadCommand const *>(cmd);
                    auto chunk = _state->readback_alloc.allocate(c->size(), 16);
                    _state->callbacks.emplace_back([chunk, data = c->data(), size = c->size()]() {
                        static_cast<ReadbackBuffer const *>(chunk.buffer)->copy_to(data, chunk.offset, size);
                    });
                    VkBuffer src = reinterpret_cast<Buffer const *>(c->handle())->vk_buffer();
                    VkBuffer dst = chunk.buffer->vk_buffer();
                    auto &regions = pending_download.copies[BufferPair{src, dst}];
                    regions.push_back({VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                                       nullptr,
                                       c->offset(),
                                       chunk.offset,
                                       c->size()});
                } break;
                case Command::Tag::EBufferCopyCommand: {
                    auto c = static_cast<BufferCopyCommand const *>(cmd);
                    if (accumulate_buffer_copy(c)) { break; }
                    flush_all_pending();
                    VkBufferCopy2 buffer_copy{
                        VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                        nullptr,
                        c->src_offset(),
                        c->dst_offset(),
                        c->size()};
                    VkCopyBufferInfo2 copy_info2{
                        VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                        nullptr,
                        reinterpret_cast<Buffer const *>(c->src_handle())->vk_buffer(),
                        reinterpret_cast<Buffer const *>(c->dst_handle())->vk_buffer(),
                        1,
                        &buffer_copy};
                    detail::cmd_copy_buffer(_cmdbuffer, device(), &copy_info2);
                } break;
                case Command::Tag::EBufferToTextureCopyCommand: {
                    flush_all_pending();
                    auto c = static_cast<BufferToTextureCopyCommand const *>(cmd);
                    auto tex = reinterpret_cast<Texture const *>(c->texture());
                    int3 tex_offset = make_int3(c->texture_offset());
                    auto size = c->size();
                    VkBufferImageCopy2 region{
                        VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                        nullptr,
                        c->buffer_offset(),
                        0,
                        0,
                        VkImageSubresourceLayers{tex->get_aspect(), c->level(), 0, 1},
                        VkOffset3D{tex_offset.x, tex_offset.y, tex_offset.z},
                        VkExtent3D{size.x, size.y, size.z}};

                    VkCopyBufferToImageInfo2 copy_info{
                        VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
                        nullptr,
                        reinterpret_cast<Buffer const *>(c->buffer())->vk_buffer(),
                        tex->vk_image(),
                        resource_barrier->get_layout(tex, c->level()),
                        1,
                        &region};
                    detail::cmd_copy_buffer_to_image(_cmdbuffer, device(), &copy_info);
                } break;
                case Command::Tag::EShaderDispatchCommand: {
                    flush_all_pending();
                    auto c = static_cast<ShaderDispatchCommand const *>(cmd);
                    auto indirect_plan = IndirectDispatchPlan{};
                    const Buffer *indirect_source = nullptr;
                    auto indirect = ValidatedIndirectDispatch{};
                    if (c->is_indirect()) {
                        indirect = validate_indirect_dispatch_source(c);
                        indirect_plan = indirect.plan;
                        indirect_source = indirect.source;
                        if (indirect_plan.command_count == 0u) {
                            // The preprocessing phase likewise skipped this
                            // command, so it owns no argument-offset entry and
                            // needs no descriptor, pipeline, or source read.
                            break;
                        }
                    }
                    auto shader = reinterpret_cast<Shader *>(c->handle());
                    if (c->is_indirect()) {
                        validate_indirect_dispatch_target(
                            c, shader, indirect, true);
                    }
                    // Keep the shader's pipeline alive until this command buffer completes.
                    if (auto *ref = shader->pipeline_ref()) {
                        _state->retain_pipeline(ref);
                    }
                    // Counter-only: this ShaderDispatchCommand is about to be
                    // recorded. A statically empty indirect range returned above
                    // and is therefore not counted (F2/F3/F6).
                    submit_profile::record_dispatch();
                    bool is_rt_shader = (shader->shader_tag() == Shader::ShaderTag::kRayTracingShader);
                    BindPropVisitor visitor{};
                    visitor.device = device();
                    set_dispatch_args(
                        cmd, visitor, c, shader, indirect_source);
                    constexpr size_t max_printer_count = 1024ull * 1024ull;
                    BufferView count_buffer;
                    BufferView data_buffer;
                    if (!shader->printers().empty()) {
                        auto count_chunk = scratch_buffer_alloc->allocate(4, 16);
                        auto data_chunk = scratch_buffer_alloc->allocate(max_printer_count, 16);
                        count_buffer = BufferView(
                            reinterpret_cast<Buffer const *>(count_chunk.handle),
                            count_chunk.offset,
                            4);
                        auto upload_buffer = states()->upload_alloc.allocate(4);
                        uint zero = 0;
                        static_cast<UploadBuffer const *>(upload_buffer.buffer)->copy_from(&zero, upload_buffer.offset, 4);

                        resource_barrier->record(
                            count_buffer,
                            ResourceBarrier::Usage::kCopyDest);
                        resource_barrier->update_states(_cmdbuffer);
                        VkBufferCopy2 buffer_copy{
                            VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                            nullptr,
                            upload_buffer.offset,
                            count_buffer.offset,
                            4};
                        VkCopyBufferInfo2 copy_info2{
                            VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                            nullptr,
                            upload_buffer.buffer->vk_buffer(),
                            count_buffer.buffer->vk_buffer(),
                            1,
                            &buffer_copy};
                        detail::cmd_copy_buffer(_cmdbuffer, device(), &copy_info2);

                        data_buffer = BufferView{
                            reinterpret_cast<Buffer const *>(data_chunk.handle),
                            data_chunk.offset,
                            max_printer_count};
                        // bind counter
                        {
                            auto idx = visitor.desc_index++;
                            auto buffer_descs = temp_desc->allocate_memory<VkDescriptorBufferInfo>();
                            *buffer_descs = VkDescriptorBufferInfo{
                                count_buffer.buffer->vk_buffer(),
                                count_buffer.offset,
                                4};
                            write_desc_sets->emplace_back(VkWriteDescriptorSet{
                                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                nullptr,
                                visitor.desc_set,
                                idx,
                                0,
                                1,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                nullptr,
                                buffer_descs,
                                nullptr});
                        }
                        // bind data
                        {
                            auto idx = visitor.desc_index++;
                            auto buffer_descs = temp_desc->allocate_memory<VkDescriptorBufferInfo>();
                            *buffer_descs = VkDescriptorBufferInfo{
                                data_buffer.buffer->vk_buffer(),
                                data_buffer.offset,
                                max_printer_count};
                            write_desc_sets->emplace_back(VkWriteDescriptorSet{
                                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                nullptr,
                                visitor.desc_set,
                                idx,
                                0,
                                1,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                nullptr,
                                buffer_descs,
                                nullptr});
                        }
                        resource_barrier->record(
                            data_buffer,
                            ResourceBarrier::Usage::kComputeUAV);
                        resource_barrier->record(
                            count_buffer,
                            ResourceBarrier::Usage::kComputeUAV);
                        resource_barrier->update_states(_cmdbuffer);
                    }
                    auto bind_point = is_rt_shader ? VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR : VK_PIPELINE_BIND_POINT_COMPUTE;
                    auto push_stage = is_rt_shader ?
                                          static_cast<VkShaderStageFlags>(VK_SHADER_STAGE_RAYGEN_BIT_KHR) :
                                          static_cast<VkShaderStageFlags>(VK_SHADER_STAGE_COMPUTE_BIT);
                    // Get pipeline and block_size from the correct shader type
                    VkPipeline vk_pipeline;
                    uint3 blk;
                    if (is_rt_shader) {
                        auto rt = static_cast<RayTracingShader const *>(shader);
                        vk_pipeline = rt->pipeline();
                        blk = rt->block_size();
                    } else {
                        auto cs = static_cast<ComputeShader const *>(shader);
                        vk_pipeline = cs->pipeline();
                        blk = cs->block_size();
                    }
                    // Plan P2.3: an indirect dispatch binds set 0 twice (as the prologue of
                    // the command, and again after the preparation kernel disturbed the push
                    // descriptor bank), and one push consumes its write list. The list of this
                    // command is therefore retained here - only for indirect commands - and
                    // restored for the second push.
                    vstd::vector<VkWriteDescriptorSet> indirect_repush_writes;
                    if (c->is_indirect()) {
                        indirect_repush_writes.assign(write_desc_sets->begin(),
                                                        write_desc_sets->end());
                    }
                    bind_shader_desc(visitor, shader, bind_point);
                    // P1.2: a repeated dispatch of one shader in one command
                    // buffer binds its pipeline once; the bind is skipped only for
                    // the exact same VkPipeline handle at this bind point.
                    detail::cmd_bind_pipeline(
                        _state->bind_state, _cmdbuffer, bind_point,
                        vk_pipeline, shader->pipeline_layout());
                    auto calc = [](uint disp, uint thd) {
                        auto group_count = indirect_dispatch_group_count(
                            disp, thd);
                        LUISA_ASSERT(group_count.valid_block_size,
                                     "Vulkan compute block dimension is zero.");
                        return group_count.value;
                    };
                    auto validate_group_count = [&](uint3 group_count) {
                        if (is_rt_shader) { return; }
                        auto &&limits = device()->properties().limits;
                        auto representable = [](uint block_size) noexcept {
                            return indirect_dispatch_max_group_count_for_uint32_global_id(
                                block_size);
                        };
                        LUISA_ASSERT(
                            group_count.x <=
                                    limits.maxComputeWorkGroupCount[0] &&
                                group_count.y <=
                                    limits.maxComputeWorkGroupCount[1] &&
                                group_count.z <=
                                    limits.maxComputeWorkGroupCount[2] &&
                                group_count.x <= representable(blk.x) &&
                                group_count.y <= representable(blk.y) &&
                                group_count.z <= representable(blk.z),
                            "Vulkan direct dispatch group count ({}, {}, {}) "
                            "exceeds device or uint32 global-ID limits.",
                            group_count.x, group_count.y, group_count.z);
                    };
                    auto push_direct = [&](uint3 dispatch_size,
                                           uint kernel_id) {
                        if (is_rt_shader) {
                            auto value = make_uint4(
                                dispatch_size, kernel_id);
                            push_shader_constants(
                                shader, push_stage, 0u, value);
                        } else {
                            IndirectDispatchPushConstants value{
                                .logical_size_x = dispatch_size.x,
                                .logical_size_y = dispatch_size.y,
                                .logical_size_z = dispatch_size.z,
                                .kernel_id = kernel_id,
                                .mode = static_cast<uint32_t>(
                                    IndirectDispatchMode::DIRECT)};
                            push_shader_constants(
                                shader, push_stage, 0u, value);
                        }
                        // Software (fallback) ray tracing: the region base of
                        // every fallback `accel` argument travels in the same
                        // push-constant block, right after the dispatch block
                        // the codegen declares first (32 + 4 * i, see
                        // fallback_rtx_header.bytes and entry_points.cpp) - one
                        // 32-bit word per argument, in the argument order the
                        // property table reserved the words in.
                        auto expected_base_count =
                            shader->push_constant_size() >
                                    fallback_accel_base_push_offset ?
                                (shader->push_constant_size() -
                                 fallback_accel_base_push_offset) /
                                    4u :
                                0u;
                        LUISA_ASSERT(
                            visitor.fallback_accel_bases.size() ==
                                expected_base_count,
                            "Vulkan fallback ray tracing pushed {} region "
                            "bases, but the shader's push-constant block "
                            "({} bytes) expects {}.",
                            visitor.fallback_accel_bases.size(),
                            shader->push_constant_size(),
                            expected_base_count);
                        for (auto i = 0u;
                             i < visitor.fallback_accel_bases.size(); ++i) {
                            push_shader_constants(
                                shader, push_stage,
                                fallback_accel_base_push_offset + 4u * i,
                                visitor.fallback_accel_bases[i]);
                        }
                    };
                    if (c->is_indirect()) {
                        auto commands = prepare_indirect_dispatch(
                            indirect_source, indirect_plan, blk);
                        if (indirect_plan.command_count != 0u) {
                    // Preparation binds its own compute pipeline and
                    // descriptor set. Restore the target contract once,
                    // then select one absolute source record per command.
                    // The prologue push consumed the write list; restore this command's own
                    // list so the second bind pushes exactly the same descriptors again.
                    if (shader->uses_push_descriptors() && !indirect_repush_writes.empty()) {
                        write_desc_sets->assign(indirect_repush_writes.begin(),
                                                indirect_repush_writes.end());
                    }
                    bind_shader_desc(
                        visitor, shader,
                        VK_PIPELINE_BIND_POINT_COMPUTE);
                    // P1.2: the preparation kernel bound its own pipeline and set
                    // on this bind point, so restoring the target contract is a
                    // real rebind (the tracker sees the difference).
                    detail::cmd_bind_pipeline(
                        _state->bind_state,
                        _cmdbuffer,
                        VK_PIPELINE_BIND_POINT_COMPUTE,
                        vk_pipeline, shader->pipeline_layout());
                    // VK_KHR_device_address_commands: the very same prepared
                    // dispatch header can be consumed through
                    // vkCmdDispatchIndirect2KHR, which addresses it by
                    // VkDeviceAddress instead of by (VkBuffer, offset). The
                    // gate and the address/range validation live in
                    // device_address_command_plan.h; a rejected plan keeps the
                    // pre-existing vkCmdDispatchIndirect below, so the indirect
                    // metadata contract (same header buffer, same SPIR-V side)
                    // is unchanged either way.
                    auto const device_address_dispatch_enabled =
                        device()->optional_extension_enabled(
                            detail::OptionalExtension::DEVICE_ADDRESS_COMMANDS) &&
                        vkCmdDispatchIndirect2KHR != nullptr;
                    // The prepared command array is a backend-owned storage
                    // buffer, so its VK_BUFFER_USAGE_STORAGE_BUFFER_BIT is
                    // attestable and VK_ADDRESS_COMMAND_STORAGE_BUFFER_USAGE_BIT_KHR
                    // is the exact flag.
                    auto const prepared_command_address =
                        commands.buffer->device_address_capable() ?
                            commands.buffer->get_device_address() :
                            0u;
                    for (auto command_index = 0u;
                         command_index <
                         indirect_plan.command_count;
                         ++command_index) {
                        auto source_record_index =
                            indirect_plan.source_record_offset +
                            command_index;
                        IndirectDispatchPushConstants value{
                            .mode = static_cast<uint32_t>(
                                IndirectDispatchMode::INDIRECT),
                            .source_record_index =
                                source_record_index};
                        push_shader_constants(
                            shader, VK_SHADER_STAGE_COMPUTE_BIT,
                            0u, value);
                        auto command_offset =
                            static_cast<VkDeviceSize>(
                                commands.offset) +
                            static_cast<VkDeviceSize>(
                                command_index) *
                                IndirectDispatchLayout::
                                    vulkan_command_size;
                        // Every prepared record is one whole
                        // vulkan_command_size block inside the scratch buffer
                        // (validated by prepare_indirect_dispatch), and that
                        // block is larger than sizeof(VkDispatchIndirectCommand).
                        auto device_address_plan =
                            detail::plan_device_address_indirect_dispatch(
                                device_address_dispatch_enabled,
                                prepared_command_address != 0u,
                                prepared_command_address,
                                static_cast<uint64_t>(command_offset),
                                static_cast<uint64_t>(
                                    IndirectDispatchLayout::vulkan_command_size));
                        if (device_address_plan) {
                            auto info =
                                detail::make_dispatch_indirect_2_info(
                                    device_address_plan);
                            vkCmdDispatchIndirect2KHR(_cmdbuffer, &info);
                            device()->mark_optional_extension_active(
                                detail::OptionalExtension::DEVICE_ADDRESS_COMMANDS);
                        } else {
                            vkCmdDispatchIndirect(
                                _cmdbuffer,
                                commands.buffer->vk_buffer(),
                                command_offset);
                        }
                    }
                        }
                    } else if (c->is_multiple_dispatch()) {
                        uint idx = 0;
                        for (auto &disp_size : c->dispatch_sizes()) {
                            push_direct(disp_size, idx);
                            ++idx;
                            if (is_rt_shader) {
                                auto rt = static_cast<RayTracingShader const *>(shader);
                                vkCmdTraceRaysKHR(_cmdbuffer,
                                                  &rt->raygen_region(), &rt->miss_region(),
                                                  &rt->hit_region(), &rt->callable_region(),
                                                  disp_size.x, disp_size.y, disp_size.z);
                            } else {
                                auto group_count = make_uint3(
                                    calc(disp_size.x, blk.x),
                                    calc(disp_size.y, blk.y),
                                    calc(disp_size.z, blk.z));
                                validate_group_count(group_count);
                                vkCmdDispatch(
                                    _cmdbuffer, group_count.x,
                                    group_count.y, group_count.z);
                            }
                        }
                    } else {
                        auto disp_size = c->dispatch_size();
                        push_direct(disp_size, 0u);
                        if (is_rt_shader) {
                            auto rt = static_cast<RayTracingShader const *>(shader);
                            vkCmdTraceRaysKHR(_cmdbuffer,
                                              &rt->raygen_region(), &rt->miss_region(),
                                              &rt->hit_region(), &rt->callable_region(),
                                              disp_size.x, disp_size.y, disp_size.z);
                        } else {
                            auto group_count = make_uint3(
                                calc(disp_size.x, blk.x),
                                calc(disp_size.y, blk.y),
                                calc(disp_size.z, blk.z));
                            validate_group_count(group_count);
                            vkCmdDispatch(
                                _cmdbuffer, group_count.x,
                                group_count.y, group_count.z);
                        }
                    }
                    if (logger && !shader->printers().empty()) {
                        resource_barrier->record(
                            count_buffer,
                            ResourceBarrier::Usage::kCopySource);
                        resource_barrier->record(
                            data_buffer,
                            ResourceBarrier::Usage::kCopySource);
                        resource_barrier->update_states(_cmdbuffer);
                        auto counter_readback = states()->readback_alloc.allocate(4, 16);
                        auto data_readback = states()->readback_alloc.allocate(max_printer_count, 16);
                        // Copy counter
                        {
                            VkBufferCopy2 buffer_copy{
                                VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                                nullptr,
                                count_buffer.offset,
                                counter_readback.offset,
                                4};
                            VkCopyBufferInfo2 copy_info2{
                                VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                                nullptr,
                                count_buffer.buffer->vk_buffer(),
                                reinterpret_cast<Buffer const *>(counter_readback.buffer)->vk_buffer(),
                                1,
                                &buffer_copy};

                            detail::cmd_copy_buffer(_cmdbuffer, device(), &copy_info2);
                        }
                        // Copy data
                        {
                            VkBufferCopy2 buffer_copy{
                                VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                                nullptr,
                                data_buffer.offset,
                                data_readback.offset,
                                max_printer_count};
                            VkCopyBufferInfo2 copy_info2{
                                VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                                nullptr,
                                data_buffer.buffer->vk_buffer(),
                                reinterpret_cast<Buffer const *>(data_readback.buffer)->vk_buffer(),
                                1,
                                &buffer_copy};
                            detail::cmd_copy_buffer(_cmdbuffer, device(), &copy_info2);
                        }
                        states()->callbacks.emplace_back(
                            [printers = shader->printers(),
                             logger = this->logger,
                             counter_readback,
                             data_readback]() {
                                uint counter = 0;
                                static_cast<ReadbackBuffer const *>(counter_readback.buffer)->copy_to(&counter, counter_readback.offset, 4);
                                luisa::vector<std::byte> data;
                                luisa::enlarge_by(data, counter);
                                static_cast<ReadbackBuffer const *>(data_readback.buffer)->copy_to(data.data(), data_readback.offset, counter);
                                size_t offset = 0;
                                const auto ptr = data.data();
                                const auto end = data.size();
                                while (offset < end) {
                                    uint flagTypeIdx = *reinterpret_cast<uint32_t *>(ptr + offset);
                                    auto &type = printers[flagTypeIdx];
                                    ShaderPrintFormatter formatter{type.first, type.second, false};
                                    luisa::string result;
                                    auto align = std::max<size_t>(4, type.second->alignment());
                                    formatter(result, {ptr + offset + align, type.second->size()});
                                    size_t ele_size = align + type.second->size();
                                    ele_size = ((ele_size + 15ull) & (~15ull));
                                    offset += ele_size;
                                    (*logger)(result);
                                }
                            });
                    }
                } break;
                case Command::Tag::ETextureUploadCommand: {
                    flush_all_pending();
                    auto c = static_cast<TextureUploadCommand const *>(cmd);
                    auto pixel_size = pixel_storage_size(c->storage(), c->size());
                    auto buffer = _state->upload_alloc.allocate(pixel_size, 16);
                    static_cast<UploadBuffer const *>(buffer.buffer)->copy_from(c->data(), buffer.offset, pixel_size);
                    auto tex = reinterpret_cast<Texture const *>(c->handle());
                    int3 tex_offset = make_int3(c->offset());
                    auto size = c->size();
                    VkBufferImageCopy2 region{
                        VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                        nullptr,
                        buffer.offset,
                        0,
                        0,
                        VkImageSubresourceLayers{tex->get_aspect(), c->level(), 0, 1},
                        VkOffset3D{tex_offset.x, tex_offset.y, tex_offset.z},
                        VkExtent3D{size.x, size.y, size.z}};

                    VkCopyBufferToImageInfo2 copy_info{
                        VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
                        nullptr,
                        buffer.buffer->vk_buffer(),
                        tex->vk_image(),
                        resource_barrier->get_layout(tex, c->level()),
                        1,
                        &region};
                    detail::cmd_copy_buffer_to_image(_cmdbuffer, device(), &copy_info);
                } break;
                case Command::Tag::ETextureDownloadCommand: {
                    flush_all_pending();
                    auto c = static_cast<TextureDownloadCommand const *>(cmd);
                    auto pixel_size = pixel_storage_size(c->storage(), c->size());
                    auto buffer = _state->readback_alloc.allocate(pixel_size, 16);
                    _state->callbacks.emplace_back([buffer = buffer.buffer,
                                                    offset = buffer.offset,
                                                    pixel_size,
                                                    data = c->data()]() {
                        static_cast<ReadbackBuffer const *>(buffer)->copy_to(data, offset, pixel_size);
                    });
                    auto tex = reinterpret_cast<Texture const *>(c->handle());
                    int3 tex_offset = make_int3(c->offset());
                    auto size = c->size();
                    VkBufferImageCopy2 region{
                        VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                        nullptr,
                        buffer.offset,
                        0,
                        0,
                        VkImageSubresourceLayers{tex->get_aspect(), c->level(), 0, 1},
                        VkOffset3D{tex_offset.x, tex_offset.y, tex_offset.z},
                        VkExtent3D{size.x, size.y, size.z}};
                    VkCopyImageToBufferInfo2 info{
                        VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
                        nullptr,
                        tex->vk_image(),
                        resource_barrier->get_layout(tex, c->level()),
                        buffer.buffer->vk_buffer(),
                        1,
                        &region};
                    detail::cmd_copy_image_to_buffer(_cmdbuffer, device(), &info);
                } break;
                case Command::Tag::ETextureCopyCommand: {
                    flush_all_pending();
                    auto c = static_cast<TextureCopyCommand const *>(cmd);
                    auto src_tex = reinterpret_cast<Texture const *>(c->src_handle());
                    int3 src_tex_offset = make_int3(c->src_offset());
                    int3 dst_tex_offset = make_int3(c->dst_offset());
                    auto dst_tex = reinterpret_cast<Texture const *>(c->dst_handle());
                    auto size = c->size();
                    VkImageCopy2 copy{
                        VK_STRUCTURE_TYPE_IMAGE_COPY_2,
                        nullptr,
                        VkImageSubresourceLayers{src_tex->get_aspect(), c->src_level(), 0, 1},
                        VkOffset3D{src_tex_offset.x, src_tex_offset.y, src_tex_offset.z},
                        VkImageSubresourceLayers{dst_tex->get_aspect(), c->dst_level(), 0, 1},
                        VkOffset3D{dst_tex_offset.x, dst_tex_offset.y, dst_tex_offset.z},
                        VkExtent3D{size.x, size.y, size.z}};
                    VkCopyImageInfo2 info{
                        VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2,
                        nullptr,
                        src_tex->vk_image(),
                        resource_barrier->get_layout(src_tex, c->src_level()),
                        dst_tex->vk_image(),
                        resource_barrier->get_layout(dst_tex, c->dst_level()),
                        1,
                        &copy};
                    detail::cmd_copy_image(_cmdbuffer, device(), &info);
                } break;
                case Command::Tag::ETextureToBufferCopyCommand: {
                    flush_all_pending();
                    auto c = static_cast<TextureToBufferCopyCommand const *>(cmd);
                    auto tex = reinterpret_cast<Texture const *>(c->texture());
                    int3 tex_offset = make_int3(c->texture_offset());
                    auto size = c->size();
                    VkBufferImageCopy2 region{
                        VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                        nullptr,
                        c->buffer_offset(),
                        0,
                        0,
                        VkImageSubresourceLayers{tex->get_aspect(), c->level(), 0, 1},
                        VkOffset3D{tex_offset.x, tex_offset.y, tex_offset.z},
                        VkExtent3D{size.x, size.y, size.z}};
                    VkCopyImageToBufferInfo2 info{
                        VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
                        nullptr,
                        tex->vk_image(),
                        resource_barrier->get_layout(tex, c->level()),
                        reinterpret_cast<Buffer const *>(c->buffer())->vk_buffer(),
                        1,
                        &region};
                    detail::cmd_copy_image_to_buffer(_cmdbuffer, device(), &info);
                } break;
                case Command::Tag::EAccelBuildCommand: {
                    flush_all_pending();
                    auto c = static_cast<AccelBuildCommand const *>(cmd);
                    reinterpret_cast<Tlas *>(c->handle())->build(*this, c->instance_count());
                    // resource_barrier->record(
                    //     BufferView{&bf},
                    //     ResourceBarrier::Usage::kCopySource);
                    // resource_barrier->update_states(_cmdbuffer);
                    // luisa::vector<VkAccelerationStructureInstanceKHR> vec(bf.byte_size() / sizeof(VkAccelerationStructureInstanceKHR));
                    // auto chunk = _state->readback_alloc.allocate(vec.size_bytes(), 16);

                    // VkBufferCopy2 buffer_copy{
                    //     VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                    //     nullptr,
                    //     0,
                    //     chunk.offset,
                    //     vec.size_bytes()};
                    // int x = 0;
                    // VkCopyBufferInfo2 copy_info2{
                    //     VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                    //     nullptr,
                    //     bf.vk_buffer(),
                    //     chunk.buffer->vk_buffer(),
                    //     1,
                    //     &buffer_copy};
                    // vkCmdCopyBuffer2(
                    //     _cmdbuffer,
                    //     &copy_info2);
                    // _state->callbacks.emplace_back([chunk, vec = std::move(vec)]() mutable {
                    //     static_cast<ReadbackBuffer const *>(chunk.buffer)->copy_to(vec.data(), chunk.offset, vec.size_bytes());
                    //     for (auto &i : vec) {
                    //         LUISA_INFO(
                    //             "Matrix: {}\n{}\n{}\ninstanceCustomIndex{}\nmask{}\ninstanceShaderBindingTableRecordOffset{}\nflags{}\naccelerationStructureReference{}",
                    //             (float4&)i.transform.matrix[0],
                    //             (float4&)i.transform.matrix[1],
                    //             (float4&)i.transform.matrix[2],
                    //             (uint)i.instanceCustomIndex,
                    //             (uint)i.mask,
                    //             (uint)i.instanceShaderBindingTableRecordOffset,
                    //             (uint)i.flags,
                    //             i.accelerationStructureReference
                    //         );
                    //     }
                    // });
                } break;
                case Command::Tag::EMeshBuildCommand: {
                    flush_all_pending();
                auto c = static_cast<MeshBuildCommand const *>(cmd);
                if (c->is_cluster_build()) {
                    // TODO(dx): honour is_cluster_build() when the D3D12
                    // cluster AS path lands (NVAPI cluster build / DXR clusters).
                    reinterpret_cast<ClusterBlas *>(c->handle())->build_clusters(*this, c);
                } else {
                    reinterpret_cast<Blas *>(c->handle())->build(*this, c);
                }
            } break;
                case Command::Tag::ECurveBuildCommand: {
                } break;
                case Command::Tag::EMotionInstanceBuildCommand: {
                    flush_all_pending();
                    // Motion instance build (execute): no GPU work needed,
                    // keyframes were already stored in preprocess pass
                } break;
                case Command::Tag::EProceduralPrimitiveBuildCommand: {
                    flush_all_pending();
                    auto c = static_cast<ProceduralPrimitiveBuildCommand const *>(cmd);
                    reinterpret_cast<Blas *>(c->handle())->build(*this, c);
                } break;
                case Command::Tag::EBindlessArrayUpdateCommand: {
                    flush_all_pending();
                    auto c = static_cast<BindlessArrayUpdateCommand const *>(cmd);
                    auto bdls = reinterpret_cast<BindlessArray *>(c->handle());
                    c->visit_modifications([&](auto &&t) {
                        bdls->update(this, *write_desc_sets, *bindless_cache, luisa::span{t});
                    });
                    // LOG bindless indices

                    // auto &bf = reinterpret_cast<BindlessArray *>(c->handle())->indices_buffer();
                    // resource_barrier->record(
                    //     BufferView{&bf},
                    //     ResourceBarrier::Usage::kCopySource);
                    // resource_barrier->update_states(_cmdbuffer);

                    // luisa::vector<std::array<uint, 3>> vec(3);
                    // auto chunk = _state->readback_alloc.allocate(vec.size(), 16);
                    // VkBufferCopy2 buffer_copy{
                    //     VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                    //     nullptr,
                    //     0,
                    //     chunk.offset,
                    //     vec.size_bytes()};
                    // int x = 0;
                    // VkCopyBufferInfo2 copy_info2{
                    //     VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                    //     nullptr,
                    //     bf.vk_buffer(),
                    //     chunk.buffer->vk_buffer(),
                    //     1,
                    //     &buffer_copy};
                    // vkCmdCopyBuffer2(
                    //     _cmdbuffer,
                    //     &copy_info2);
                    // _state->callbacks.emplace_back([chunk, vec = std::move(vec)]() mutable {
                    //     static_cast<ReadbackBuffer const *>(chunk.buffer)->copy_to(vec.data(), chunk.offset, vec.size_bytes());
                    //     for (auto &i : vec) {
                    //         LUISA_INFO(uint3(i[0], i[1] & ((1u<<24u) - 1), i[2] & ((1u<<24u) - 1)));
                    //     }
                    // });
                } break;
                case Command::Tag::ECustomCommand: {
                    flush_all_pending();
                    auto c = static_cast<CustomCommand const *>(cmd);
                    switch (c->custom_cmd_uuid()) {
                        case to_underlying(CustomCommandUUID::RASTER_CLEAR_DEPTH): {
                            auto cmd = static_cast<ClearDepthCommand const *>(c);
                            auto tex = reinterpret_cast<Texture *>(cmd->handle());
                            VkClearDepthStencilValue depth_stencil_value{
                                .depth = cmd->value(),
                                .stencil = 0};
                            auto depth_format = tex->depth_format();
                            VkImageSubresourceRange sub_range{
                                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                                .baseMipLevel = 0,
                                .levelCount = 1,
                                .baseArrayLayer = 0,
                                .layerCount = 1};
                            if (depth_format == DepthFormat::D24S8 || depth_format == DepthFormat::D32S8A24) {
                                sub_range.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
                            }
                            vkCmdClearDepthStencilImage(
                                _cmdbuffer,
                                tex->vk_image(),
                                resource_barrier->get_layout(tex, 0),
                                &depth_stencil_value,
                                1,
                                &sub_range);
                        } break;
                        case to_underlying(CustomCommandUUID::RASTER_CLEAR_RENDER_TARGET): {
                            auto cmd = static_cast<ClearRenderTargetCommand const *>(c);
                            auto tex = reinterpret_cast<Texture *>(cmd->handle());
                            VkClearColorValue color_value;
                            float4 values = cmd->value();
                            std::memcpy(color_value.float32, &values, sizeof(float4));
                            VkImageSubresourceRange sub_range{
                                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .baseMipLevel = cmd->level(),
                                .levelCount = 1,
                                .baseArrayLayer = 0,
                                .layerCount = 1};
                            vkCmdClearColorImage(
                                _cmdbuffer,
                                tex->vk_image(),
                                resource_barrier->get_layout(tex, cmd->level()),
                                &color_value,
                                1,
                                &sub_range);
                        } break;
                        case to_underlying(CustomCommandUUID::RASTER_DRAW_SCENE): {
                            auto cmd = static_cast<DrawRasterSceneCommand const *>(c);
                            // A DSL raster draw binds its `RasterShader` through
                            // the saved-argument ABI; a native one has no such
                            // shader and asks its NativeRasterShader to bind
                            // itself. Everything from the framebuffer on is
                            // shared, so only the pipeline and the render pass it
                            // was built for are resolved here.
                            auto shader = static_cast<RasterShader *>(nullptr);
                            auto native_shader = static_cast<NativeRasterShader *>(nullptr);
                            VkRenderPass render_pass = VK_NULL_HANDLE;
                            // Vulkan bakes the vertex binding stride into the
                            // pipeline, so it must come from the meshes'
                            // VertexBufferViews (like on DX), not from the
                            // mesh format's attribute sizes. All meshes of
                            // one draw share the pipeline and must therefore
                            // agree on the per-binding strides.
                            luisa::fixed_vector<uint, 4> strides;
                            for (auto &mesh : cmd->scene()) {
                                auto vb = mesh.vertex_buffers();
                                if (strides.empty()) {
                                    for (auto &i : vb) {
                                        strides.emplace_back(static_cast<uint>(i.stride()));
                                    }
                                } else {
                                    LUISA_ASSERT(
                                        strides.size() == vb.size(),
                                        "Raster meshes of one draw disagree on the vertex stream count ({} vs {}).",
                                        strides.size(), vb.size());
                                    for (auto binding : vstd::range(vb.size())) {
                                        LUISA_ASSERT(
                                            strides[binding] == vb[binding].stride(),
                                            "Raster meshes of one draw disagree on the stride of vertex stream {} ({} vs {}).",
                                            binding, strides[binding], vb[binding].stride());
                                    }
                                }
                            }
                            if (cmd->is_native_shader()) {
                                // The native raster shader binds its own
                                // graphics pipeline, this draw's descriptor sets
                                // and the application's uniform block, and hands
                                // back the render pass the pipeline was built
                                // for. VK_NULL_HANDLE means "already reported":
                                // skip the draw rather than begin a render pass
                                // against a null handle.
                                render_pass = encode_native_raster_draw(
                                    device(), _state.get(), _cmdbuffer, cmd,
                                    strides, &native_shader);
                                if (render_pass == VK_NULL_HANDLE) {
                                    LUISA_ERROR_WITH_LOCATION(
                                        "A native raster draw could not be "
                                        "encoded (shader handle {}); the draw is "
                                        "skipped.",
                                        cmd->handle());
                                    break;
                                }
                            } else {
                                shader = reinterpret_cast<RasterShader *>(cmd->handle());
                                auto pipe = shader->create_pipeline(cmd->rtv_texs(), cmd->dsv_tex(), cmd->mesh_format(), cmd->raster_state(), strides);
                                BindPropVisitor visitor{};
                                visitor.device = device();
                                // bind arguments (`i->cmd` is this draw's command
                                // identity, which the local `cmd` shadows)
                                set_dispatch_args(i->cmd, visitor, cmd, shader);
                                bind_shader_desc(visitor, shader, VK_PIPELINE_BIND_POINT_GRAPHICS);
                                // P1.2: the graphics bind point is tracked
                                // separately from compute/RT, so a draw only
                                // skips a pipeline bind that is really the same
                                // VkPipeline at the graphics bind point.
                                detail::cmd_bind_pipeline(
                                    _state->bind_state, _cmdbuffer,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipe.pipeline, shader->pipeline_layout());
                                render_pass = pipe.render_pass;
                            }
                            // framebuffer
                            VkFramebuffer fb;
                            uint img_view_offset = _state->img_views.size();
                            auto emplace_img_view = [&](VkImageView &img_view, Texture *tex, uint level) {
                                VkImageViewCreateInfo imgview_create_info{
                                    VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                    nullptr,
                                    0,
                                    tex->vk_image(),
                                    VkImageViewType(tex->dimension() - 1),
                                    Texture::to_vk_format(tex->format()),
                                    VkComponentMapping{VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY},
                                    VkImageSubresourceRange{
                                        tex->get_aspect(),
                                        level,
                                        1,
                                        0,
                                        1}};
                                VK_CHECK_RESULT(vkCreateImageView(device()->logic_device(), &imgview_create_info, Device::alloc_callbacks(), &img_view));
                            };
                            auto mip_resolution = [](Texture *tex, uint level) noexcept {
                                LUISA_ASSERT(
                                    level < tex->mip(),
                                    "Raster attachment mip level {} is outside [0, {}) for a {}x{} texture.",
                                    level, tex->mip(), tex->size().x, tex->size().y);
                                return make_uint2(
                                    std::max(tex->size().x >> level, 1u),
                                    std::max(tex->size().y >> level, 1u));
                            };
                            auto resolution = uint2{};
                            auto has_resolution = false;
                            auto include_attachment_resolution = [&](Texture *tex, uint level) noexcept {
                                auto extent = mip_resolution(tex, level);
                                if (!has_resolution) {
                                    resolution = extent;
                                    has_resolution = true;
                                } else {
                                    LUISA_ASSERT(
                                        resolution.x == extent.x &&
                                            resolution.y == extent.y,
                                        "Raster attachments have mismatched mip extents: expected {}x{}, got {}x{} at mip {}.",
                                        resolution.x, resolution.y,
                                        extent.x, extent.y, level);
                                }
                            };
                            for (auto &i : cmd->rtv_texs()) {
                                auto &img_view = _state->img_views.emplace_back();
                                auto tex = reinterpret_cast<Texture *>(i.handle);
                                emplace_img_view(img_view, tex, i.level);
                                include_attachment_resolution(tex, i.level);
                                VkClearValue clear_value;
                                std::memset(&clear_value, 0, sizeof(VkClearValue));
                            }
                            if (cmd->dsv_tex().handle != invalid_resource_handle) {
                                auto &img_view = _state->img_views.emplace_back();
                                auto tex = reinterpret_cast<Texture *>(cmd->dsv_tex().handle);
                                emplace_img_view(img_view, tex, cmd->dsv_tex().level);
                                include_attachment_resolution(
                                    tex, cmd->dsv_tex().level);
                                VkClearValue clear_value;
                                std::memset(&clear_value, 0, sizeof(VkClearValue));
                                clear_value.depthStencil.depth = 0.f;
                            }
                            LUISA_ASSERT(
                                has_resolution,
                                "Raster draw command has no color or depth attachment.");

                            VkFramebufferCreateInfo framebuffer_create_info{
                                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                .renderPass = render_pass,
                                .attachmentCount = (uint)(_state->img_views.size() - img_view_offset),
                                .pAttachments = _state->img_views.data() + img_view_offset,
                                .width = resolution.x,
                                .height = resolution.y,
                                .layers = 1};
                            VK_CHECK_RESULT(vkCreateFramebuffer(
                                device()->logic_device(),
                                &framebuffer_create_info,
                                Device::alloc_callbacks(),
                                &fb));
                            VkRenderPassBeginInfo begin_pass_info{
                                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                .renderPass = render_pass,
                                .framebuffer = fb,
                                .renderArea = VkRect2D{
                                    .offset = VkOffset2D{0, 0},
                                    .extent = VkExtent2D{resolution.x, resolution.y}},
                                .clearValueCount = 0,
                                .pClearValues = nullptr};
                            vkCmdBeginRenderPass(_cmdbuffer, &begin_pass_info, VK_SUBPASS_CONTENTS_INLINE);
                            auto &&cmd_vp = cmd->viewport();
                            VkViewport viewport{};
                            viewport.x = static_cast<float>(cmd_vp.start.x);
                            viewport.y = static_cast<float>(cmd_vp.start.y);
                            viewport.width = std::max(1.f, static_cast<float>(cmd_vp.size.x));
                            viewport.height = std::max(1.f, static_cast<float>(cmd_vp.size.y));
                            viewport.minDepth = 0.0f;
                            viewport.maxDepth = 1.0f;
                            vkCmdSetViewport(_cmdbuffer, 0, 1, &viewport);

                            VkRect2D scissor{};
                            scissor.offset = {(int)cmd_vp.start.x, (int)cmd_vp.start.y};
                            scissor.extent = {cmd_vp.size.x, cmd_vp.size.y};
                            vkCmdSetScissor(_cmdbuffer, 0, 1, &scissor);
                            vstd::fixed_vector<VkBuffer, 4> vertex_buffers;
                            vstd::fixed_vector<VkDeviceSize, 4> vertex_buffer_offsets;
                            for (auto &mesh : cmd->scene()) {
                                auto vb = mesh.vertex_buffers();
                                vertex_buffers.clear();
                                vertex_buffer_offsets.clear();
                                vertex_buffers.reserve(vb.size());
                                vertex_buffer_offsets.reserve(vb.size());
                                for (auto &i : vb) {
                                    vertex_buffers.emplace_back(reinterpret_cast<Buffer *>(i.handle())->vk_buffer());
                                    vertex_buffer_offsets.emplace_back(i.offset());
                                }
                                vkCmdBindVertexBuffers(_cmdbuffer, 0, vb.size(), vertex_buffers.data(), vertex_buffer_offsets.data());
                                auto before_draw = [&]() {
                                    if (cmd->is_native_shader()) {
                                        // A native raster shader reads the object
                                        // id from the slot it reserved at
                                        // push-constant offset 0 (nothing to do
                                        // when it reserved none).
                                        if (native_shader->object_id_bytes() != 0u) {
                                            native_shader->push_object_id(
                                                _cmdbuffer, mesh.object_id());
                                        }
                                    } else {
                                        uint value = mesh.object_id();
                                        push_shader_constants(
                                            shader,
                                            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                            0u, value);
                                    }
                                };
                                luisa::visit([&]<typename T>(T const &t) {
                                    // Draw
                                    if constexpr (std::is_integral_v<T>) {
                                        before_draw();
                                        vkCmdDraw(
                                            _cmdbuffer,
                                            t,
                                            mesh.instance_count(),
                                            mesh.vertex_offset(),
                                            mesh.base_instance());
                                    } else {
                                        auto buffer = reinterpret_cast<Buffer *>(t.handle());
                                        vkCmdBindIndexBuffer(_cmdbuffer, buffer->vk_buffer(), t.offset_bytes(), VK_INDEX_TYPE_UINT32);
                                        before_draw();
                                        vkCmdDrawIndexed(
                                            _cmdbuffer,
                                            t.size_bytes() / sizeof(uint),
                                            mesh.instance_count(),
                                            0,
                                            mesh.vertex_offset(),
                                            mesh.base_instance());
                                    }
                                    // Draw indexed
                                },
                                             mesh.index());
                            }
                            vkCmdEndRenderPass(_cmdbuffer);
                            _state->dispose_pool.emplace_back(fb, [](Stream *stream, CommandBufferState *state, void *ptr) {
                                vkDestroyFramebuffer(
                                    stream->device()->logic_device(),
                                    static_cast<VkFramebuffer>(ptr),
                                    Device::alloc_callbacks());
                            });
                            for (auto &i : cmd->rtv_texs()) {
                                auto tex = reinterpret_cast<Texture const *>(i.handle);
                                resource_barrier->force_refresh_layout(
                                    tex, i.level,
                                    VK_IMAGE_LAYOUT_GENERAL);
                            }
                            if (cmd->dsv_tex().handle != invalid_resource_handle) {
                                auto tex = reinterpret_cast<Texture const *>(cmd->dsv_tex().handle);
                                resource_barrier->force_refresh_layout(
                                    tex, cmd->dsv_tex().level,
                                    VK_IMAGE_LAYOUT_GENERAL);
                            }

                        } break;
                        case to_underlying(CustomCommandUUID::CUSTOM_DISPATCH): {
                            static_cast<VKCustomCmd const *>(c)->execute(
                                device()->physical_device(),
                                device()->logic_device(),
                                _stream.queue(),
                                _cmdbuffer,
                                _state->desc_pool);
                            // P1.2: `execute` receives the raw command buffer and
                            // may bind anything, so nothing this tracker believed
                            // about the recorded bind state survives it.
                            _state->bind_state.invalidate_all();
                        } break;
                        case to_underlying(CustomCommandUUID::VK_CUDA_LAUNCH_KERNEL): {
#ifdef LUISA_VULKAN_ENABLE_CUDA_INTEROP
                            cuda_launch_kernel(
                                device(), _cmdbuffer,
                                static_cast<vk_cuda_interop::CudaKernelLaunchCommand const *>(c));
                            // P1.2: the CUDA launch is foreign code on our command
                            // buffer. VK_NV_cuda_kernel_launch binds no Vulkan
                            // pipeline, but "no bind we know of" is not something
                            // the tracker may assume, so the state is flushed.
                            _state->bind_state.invalidate_all();
#else
                            LUISA_ERROR(
                                "VK_CUDA_LAUNCH_KERNEL requires the vk backend "
                                "built with lc_vk_cuda_interop (xmake) / "
                                "LUISA_COMPUTE_ENABLE_VK_CUDA_INTEROP (cmake).");
#endif
                        } break;
                        case to_underlying(CustomCommandUUID::NATIVE_SHADER_DISPATCH): {
                            encode_native_shader_dispatch(
                                device(), _state.get(), _cmdbuffer,
                                static_cast<NativeShaderDispatchCommand const *>(c));
                        } break;
                        case to_underlying(CustomCommandUUID::BATCH_BUFFER_TO_TEXTURE_COPY): {
                            auto bc = static_cast<BatchBufferToTextureCopyCommand const *>(c);
                            auto const &regions = bc->regions();
                            if (regions.empty()) { break; }
                            auto *buf = reinterpret_cast<Buffer const *>(bc->buffer());
                            auto *tex = reinterpret_cast<Texture const *>(bc->texture());
                            auto vk_buf = buf->vk_buffer();
                            auto vk_img = tex->vk_image();
                            auto aspect = tex->get_aspect();
                            luisa::fixed_vector<VkBufferImageCopy2, 64> vk_regions;
                            const uint32_t min_mip = bc->min_mip_level();
                            const uint32_t max_mip = bc->max_mip_level();
                            for (uint32_t mip = min_mip; mip <= max_mip; ++mip) {
                                vk_regions.clear();
                                for (auto const &r : regions) {
                                    if (r.mip_level != mip) { continue; }
                                    vk_regions.push_back(VkBufferImageCopy2{
                                        VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                                        nullptr,
                                        r.buffer_offset,
                                        r.buffer_row_length,
                                        r.buffer_image_height,
                                        VkImageSubresourceLayers{aspect, r.mip_level, 0u, 1u},
                                        VkOffset3D{
                                            static_cast<int32_t>(r.texture_offset.x),
                                            static_cast<int32_t>(r.texture_offset.y),
                                            static_cast<int32_t>(r.texture_offset.z)},
                                        VkExtent3D{r.texture_extent.x, r.texture_extent.y, r.texture_extent.z}});
                                }
                                if (vk_regions.empty()) { continue; }
                                VkCopyBufferToImageInfo2 copy_info{
                                    VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
                                    nullptr,
                                    vk_buf,
                                    vk_img,
                                    resource_barrier->get_layout(tex, mip),
                                    static_cast<uint32_t>(vk_regions.size()),
                                    vk_regions.data()};
                                detail::cmd_copy_buffer_to_image(_cmdbuffer, device(), &copy_info);
                            }
                        } break;
                        case to_underlying(CustomCommandUUID::STREAM_SIGNAL_EVENT): {
                        } break;
                        // NOTE: unimplemented command type — extend as new CustomCommandUUID
                        // values are added.
                        default: {
                            LUISA_ERROR("Command type not supported.");
                        } break;
                    }
                } break;
                default: break;
            }
        }
        LUISA_ASSERT(
            dispatch_offset_index == dispatch_offsets->size(),
            "Vulkan descriptor binding consumed {} argument blocks, but "
            "preprocessing produced {}.",
            dispatch_offset_index, dispatch_offsets->size());
        flush_all_pending();
        if (tl_active) {
            uint32_t q_end = 0xffffffffu;
            if (_state->timestamp_cursor + 1u < _state->timestamp_pool_capacity) {
                q_end = _state->timestamp_cursor++;
                vkCmdWriteTimestamp(_cmdbuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, _state->timestamp_pool, q_end);
            } else {
                ++_state->tl_truncated_spans;
            }
            _state->tl_cmd_count += static_cast<uint32_t>(tl_layer_cmds.size());
            ++_state->tl_layer_count;
            _state->tl_layers.emplace_back(TimelineLayerRecord{
                .layer_idx = tl_layer_idx,
                .q_begin = tl_prev_q_end,
                .q_end = q_end,
                .cmds = std::move(tl_layer_cmds)});
            tl_prev_q_end = q_end;
            ++tl_layer_idx;
        }
    }
    if (scratch_buffer_alloc->allocated_buffer().size() > 4u) {
        scratch_buffer_alloc->clear();
    }
    LUISA_ASSERT(
        preprocessed_argument_buffer_layout.size() ==
                planned_argument_buffer_layout.size() &&
            uniform_data->size() == uniform_buffer_size,
        "Vulkan argument-buffer sizing/preprocessing mismatch: planned {} "
        "bytes, preprocessed {} bytes, emitted {} bytes.",
        planned_argument_buffer_layout.size(),
        preprocessed_argument_buffer_layout.size(),
        uniform_data->size());
    // P1.4 completeness: every command the sizing pass planned was preprocessed
    // exactly once, and nothing was preprocessed twice. Together with the
    // identity check of the emission pass this closes the association in both
    // directions.
    LUISA_ASSERT(
        argument_scratch.plans.consumed() == argument_scratch.plans.size(),
        "Vulkan argument planning recorded {} plan(s), but preprocessing "
        "consumed {}; every planned dispatch must be preprocessed exactly "
        "once.",
        argument_scratch.plans.size(), argument_scratch.plans.consumed());
    if (uniform_buffer_size > 0) {
        static_cast<UploadBuffer const *>(arg_buffer.buffer)->copy_from(uniform_data->data(), arg_buffer.offset, uniform_data->size());
    }
    _state->upload_alloc.flush_host();
    _state->readback_alloc.flush_host();
}

}// namespace lc::vk
