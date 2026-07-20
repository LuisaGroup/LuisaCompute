#include "stream.h"
#include "device.h"
#include "compute_shader.h"
#include "bindless_array.h"
#include <luisa/core/logging.h>
#include "log.h"
#include "blas.h"
#include "tlas.h"
#include "motion_instance.h"
#include "rt_shader.h"
#include "swapchain.h"
#include "sparse_buffer.h"
#include "sparse_binding_plan.h"
#include "sparse_heap.h"
#include "timeline_semaphore_plan.h"
#include "indirect_buffer.h"
#include "descriptor_interface_plan.h"
#include "queue_family_contract.h"
#include "resource_barrier_contract.h"
#include <luisa/runtime/swapchain.h>
#include <luisa/backends/ext/vk_custom_cmd.h>
#include "../common/argument_block_layout.h"
#include "../common/shader_print_formatter.h"
#include "raster_shader.h"
#include <bit>
#include <limits>
namespace lc::vk {
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
    return std::bit_cast<uint64_t>(buffer->vk_buffer());
}

uint64_t ReorderFuncTable::canonical_texture_handle(
    uint64_t handle) const noexcept {
    auto texture = reinterpret_cast<Texture const *>(handle);
    return std::bit_cast<uint64_t>(texture->vk_image());
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
    vstd::span<const hlsl::Property> bindings,
    uint32_t &descriptor_index, Usage usage,
    const char *phase) noexcept {
    auto descriptor_roles = detail::texture_descriptor_roles(usage);
    auto result = TextureDescriptorBindings{};
    auto consume = [&](hlsl::ShaderVariableType expected,
                       uint32_t &role_binding,
                       const char *role_name) noexcept {
        auto index = descriptor_index;
        auto *property = detail::find_local_descriptor_property(
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

struct PlannedArgumentBlock {
    ArgumentBlockLayout layout;
    size_t buffer_metadata_count{};
    ArgumentBlockTrailerPlacement trailer{};
};

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
    auto count = size_t{0u};
    for (auto &argument : shader->saved_arguments()) {
        if (argument.has_buffer_metadata()) {
            auto next = static_cast<size_t>(
                            argument.buffer_metadata_index()) +
                        1u;
            count = std::max(count, next);
        }
    }
    return count;
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

[[nodiscard]] static PlannedArgumentBlock plan_argument_block(
    const Shader *shader, const ShaderDispatchCommandBase *command,
    size_t descriptor_range_limit) noexcept {
    PlannedArgumentBlock plan{
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
    const PlannedArgumentBlock &plan) noexcept {
    LUISA_ASSERT(
        static_cast<bool>(plan.layout),
        "Vulkan per-dispatch argument block is invalid for the device's "
        "maxStorageBufferRange ({} bytes): {}.",
        plan.layout.limit(),
        argument_block_layout_status_name(plan.layout.status()));
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
    uint32_t descriptor_index{};
    detail::ShaderCodegenDialect codegen_dialect{
        detail::ShaderCodegenDialect::HLSL_SPIRV};
    [[nodiscard]] const hlsl::Property *binding_at(
        uint32_t index) const noexcept {
        return detail::find_local_descriptor_property(bindings, index);
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
                    res, bf.offset, bf.size, argument.struct_size)
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
            bindings, descriptor_index, argument.var_usage,
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
    uint desc_index;
    vstd::vector<VkImageView> *img_views;
    SavedArgumentCursor arguments;
    vstd::span<const hlsl::Property> bindings;
    detail::ShaderCodegenDialect codegen_dialect{
        detail::ShaderCodegenDialect::HLSL_SPIRV};
    ResourceBarrier::Usage uav_usage;
    [[nodiscard]] const hlsl::Property *binding_at(uint index) const noexcept {
        return detail::find_local_descriptor_property(bindings, index);
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
            bindings, desc_index, argument.var_usage,
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
    delete reinterpret_cast<Pack *>(handle);
}
template<typename Pack>
auto Visitor<Pack>::create(uint64 size) -> Pack * {
    return new Pack{device, size};
}
template<typename T>
void BufferAllocator<T>::clear() {
    // Soft clear: reset positions without freeing buffers
    // to keep staging buffers warm across dispatches.
    // Cap total warm capacity to 4MB to avoid unbounded growth.
    static constexpr size_t kMaxWarmCapacity = 4ull * 1024ull * 1024ull;
    size_t total_size = 0;
    for (auto &buf : alloc.allocated_buffer()) {
        total_size += buf.fullSize;
    }
    if (total_size <= kMaxWarmCapacity) {
        alloc.soft_clear();
    } else {
        alloc.dispose();
    }
    // Keep large_buffers for reuse, but limit count.
    if (large_buffers.size() > 8) {
        large_buffers.clear();
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
BufferView BufferAllocator<T>::allocate(size_t size) {
    if (size <= kLargeBufferSize) [[likely]] {
        auto chunk = alloc.allocate(size);
        return BufferView(reinterpret_cast<T const *>(chunk.handle), chunk.offset, size);
    } else {
        auto &v = large_buffers.emplace_back(visitor.create(size));
        return BufferView(v.get(), 0, size);
    }
}

template<typename T>
BufferView BufferAllocator<T>::allocate(size_t size, size_t align) {
    if (size <= kLargeBufferSize) [[likely]] {
        auto chunk = alloc.allocate(size, align);
        return BufferView(reinterpret_cast<T const *>(chunk.handle), chunk.offset, size);
    } else {
        auto &v = large_buffers.emplace_back(visitor.create(size));
        return BufferView(v.get(), 0, size);
    }
}
}// namespace temp_buffer

static constexpr size_t kTempSize = 1024ull * 1024ull;
CommandBufferState::CommandBufferState()
    : upload_alloc(kTempSize),
      readback_alloc(kTempSize) {
}
void CommandBufferState::init(Device &device, StreamTag tag) {
    this->device = &device;
    upload_alloc.visitor.device = &device;
    readback_alloc.visitor.device = &device;
    {
        VkDescriptorPoolSize pool_sizes[6];
        pool_sizes[0].descriptorCount = 65536;
        pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_sizes[1].descriptorCount = 65536;
        pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        pool_sizes[2].descriptorCount = 65536;
        pool_sizes[2].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        pool_sizes[3].descriptorCount = 65536;
        pool_sizes[3].type = VK_DESCRIPTOR_TYPE_SAMPLER;
        pool_sizes[4].descriptorCount = 65536;
        pool_sizes[4].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        auto pool_size_count = 5u;
        if (device.enable_raytracing()) {
            pool_sizes[pool_size_count].descriptorCount = 65536;
            pool_sizes[pool_size_count].type =
                VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            ++pool_size_count;
        }
        VkDescriptorPoolCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .flags = 0,
            .maxSets = 262144,
            .poolSizeCount = pool_size_count,
            .pPoolSizes = pool_sizes};
        VK_CHECK_RESULT(vkCreateDescriptorPool(device.logic_device(), &createInfo, Device::alloc_callbacks(), &desc_pool));
    }
    if (!pool) {
        VkCommandPoolCreateInfo pool_ci{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
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
    }
}
CommandBufferState::~CommandBufferState() {
    vkDestroyCommandPool(device->logic_device(), pool, Device::alloc_callbacks());
    vkDestroyDescriptorPool(device->logic_device(), desc_pool, Device::alloc_callbacks());
}
void CommandBufferState::reset(Stream *stream, Device &device) {
    for (auto &i : callbacks) {
        i();
    }
    callbacks.clear();
    for (auto &i : dispose_pool) {
        i.second(stream, this, i.first);
    }
    dispose_pool.clear();
    upload_alloc.clear();
    readback_alloc.clear();
    for (auto i : img_views) {
        vkDestroyImageView(device.logic_device(), i, Device::alloc_callbacks());
    }
    img_views.clear();
    VK_CHECK_RESULT(vkResetDescriptorPool(device.logic_device(), desc_pool, 0));
}
bool CommandBuffer::retire_and_recycle() {
    auto retirement = detail::plan_command_buffer_retirement(_ownership);
    if (retirement.reset_native_buffer) {
        VK_CHECK_RESULT(vkResetCommandBuffer(_cmdbuffer, 0));
    }
    _state->reset(&_stream, *device());
    return retirement.recycle_native_buffer;
}

Stream::Stream(Device *device, StreamTag tag)
    : Resource{device},
      _evt(device),
      logger([](luisa::string_view str) {
          LUISA_INFO("[DEVICE] {}", str);
      }),
      reorder({}),
      _thd([this]() {
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
                          } else if constexpr (std::is_same_v<T, NotifyEvt>) {
                              t.evt->_notify(t.value);
                          } else if constexpr (std::is_same_v<T, CommandBuffer>) {
                              if (t.retire_and_recycle()) {
                                  _cmdbuffers.enqueue(std::move(t));
                              }
                          }
                      });
              }
          };
          while (_enabled) {
              loop_cmd();
              while (_enabled && _exec.length() == 0) {
                  std::this_thread::yield();
              }
          }
          loop_cmd();
      }),
      _temp_desc(65536, &_temp_desc_visitor, 2),
      _scratch_buffer_alloc(kTempSize, &_scratch_buffer_alloc_visitor),
      _stream_tag(tag) {
    switch (tag) {
        case StreamTag::GRAPHICS:
            _queue = device->graphics_queue();
            _resource_barrier.queue_type = ResourceBarrier::QueueType::GRAPHICS;
            _queue_mtx = &device->graphics_queue_mtx();
            break;
        case StreamTag::COPY:
            _resource_barrier.queue_type = ResourceBarrier::QueueType::COPY;
            _queue = device->copy_queue();
            _queue_mtx = &device->copy_queue_mtx();
            break;
        case StreamTag::COMPUTE:
            _resource_barrier.queue_type = ResourceBarrier::QueueType::COMPUTE;
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
        std::lock_guard lck{_mtx};
        _enabled = false;
    }
    _thd.join();
    _scratch_buffer_alloc_visitor.buffers.clear();
    while (auto p = _cmdbuffers.dequeue()) {
    }
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

void Stream::present(
    Texture const *tex,
    uint mip,
    Swapchain *swapchain,
    bool inqueue_limit) {
    std::lock_guard lck{_dispatch_mtx};
    _temp_desc.clear();
    if (inqueue_limit) {
        if (_evt.last_fence() > 2) {
            _evt.sync(_evt.last_fence() - 2);
        }
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
            _mtx.unlock();
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
            static const uint64_t zero_signal_values[1] = {0u};
            timeline_info.pSignalSemaphoreValues = zero_signal_values;
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.pNext = &timeline_info;
            submit_info.waitSemaphoreCount = static_cast<uint32_t>(wait_semaphores.size());
            submit_info.pWaitSemaphores = wait_semaphores.data();
            submit_info.pWaitDstStageMask = wait_stages_all.data();
            submit_info.signalSemaphoreCount = present_cmd.signal_semaphores.size();
            submit_info.pSignalSemaphores = present_cmd.signal_semaphores.data();
            auto _cmdbuffer = cmdbuffer.cmdbuffer();
            if (_execute_external_command_buffer(_cmdbuffer)) {
                submit_info.commandBufferCount = 0;
                submit_info.pCommandBuffers = nullptr;
            } else {
                submit_info.commandBufferCount = 1;
                submit_info.pCommandBuffers = &_cmdbuffer;
            }
            _queue_mtx->lock();
            VK_CHECK_RESULT(vkQueueSubmit(_queue, 1u, &submit_info, vk_fence));
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
                LUISA_ERROR_WITH_LOCATION("Failed to present swapchain image: {}.", luisa::to_string(result));
            }
        }
        _evt._signal(*this, fence);
        _mtx.lock();
        _exec.enqueue(SyncExt{
            .evt = &_evt,
            .value = fence});
        _exec.enqueue(std::move(cmdbuffer));
        _exec.enqueue(NotifyEvt{
            .evt = &_evt,
            .value = fence});

        _mtx.unlock();
    }
}
void Stream::dispatch(
    vstd::span<const luisa::unique_ptr<Command>> cmds,
    luisa::vector<luisa::move_only_function<void()>> &&callbacks,
    vstd::span<const SwapchainPresent> presents,
    bool inqueue_limit) {
    std::lock_guard lck{_dispatch_mtx};
    PresentCommand present_cmd;
    luisa::fixed_vector<VkSwapchainKHR, 1> vk_swapchains;
    _temp_desc.clear();
    if (cmds.empty() && callbacks.empty() && presents.empty()) {
        return;
    }
    if (inqueue_limit) {
        if (_evt.last_fence() > 2) {
            _evt.sync(_evt.last_fence() - 2);
        }
    }
    auto fence_plan = detail::plan_timeline_value_increment(
        _evt.last_fence(), 1u);
    LUISA_ASSERT(
        static_cast<bool>(fence_plan),
        "Vulkan stream timeline fence overflow during dispatch.");
    auto fence = fence_plan.value;
    if (!cmds.empty() || !presents.empty()) {
        CommandBuffer cmdbuffer = [&]() {
            auto p = _cmdbuffers.dequeue();
            if (p) return std::move(*p);
            return CommandBuffer{*this};
        }();
        _scratch_buffer_alloc_visitor.cmdbuffer = &cmdbuffer;
        _scratch_buffer_alloc_visitor.device = device();

        auto cb = cmdbuffer.cmdbuffer();
        auto cb_ptr = &cb;
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
        cmdbuffer.begin();
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
        if (device()->config_ext()) {
            // Bindless updates have now advanced the encoded descriptor map.
            // Expand the after-state against this final snapshot, whereas the
            // before-state above deliberately used the initial snapshot.
            auto after_states = device()->config_ext()->after_states(
                reinterpret_cast<uint64_t>(this));
            for (auto &i : after_states) {
                set_config_resource_restore_state(&_resource_barrier, i);
            }
        }
        _resource_barrier.restore_states(cmdbuffer.cmdbuffer());
        cmdbuffer.end();

        if (!presents.empty() && !present_failed) {
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.waitSemaphoreCount = present_cmd.submit_wait_semaphores.size();
            submit_info.pWaitSemaphores = present_cmd.submit_wait_semaphores.data();
            submit_info.pWaitDstStageMask = present_cmd.wait_stages.data();
            submit_info.signalSemaphoreCount = present_cmd.signal_semaphores.size();
            submit_info.pSignalSemaphores = present_cmd.signal_semaphores.data();
            if (_execute_external_command_buffer(cb)) {
                // External execution - submit with 0 command buffers
                submit_info.commandBufferCount = 0;
                submit_info.pCommandBuffers = nullptr;
            } else {
                submit_info.commandBufferCount = 1;
                submit_info.pCommandBuffers = &cb;
            }
            _queue_mtx->lock();
            VK_CHECK_RESULT(vkQueueSubmit(_queue, 1u, &submit_info, present_cmd.submit_fences[0]));
            for (auto i : vstd::range(1u, present_cmd.submit_fences.size())) {
                VK_CHECK_RESULT(vkQueueSubmit(_queue, 0, nullptr, present_cmd.submit_fences[i]));
            }
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
                LUISA_ERROR_WITH_LOCATION("Failed to present swapchain image: {}.", luisa::to_string(result));
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
                VkSubmitInfo submit_info{};
                submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submit_info.waitSemaphoreCount = present_cmd.submit_wait_semaphores.size();
                submit_info.pWaitSemaphores = present_cmd.submit_wait_semaphores.data();
                submit_info.pWaitDstStageMask = present_cmd.wait_stages.data();
                submit_info.signalSemaphoreCount = present_cmd.signal_semaphores.size();
                submit_info.pSignalSemaphores = present_cmd.signal_semaphores.data();
                if (_execute_external_command_buffer(cb)) {
                    submit_info.commandBufferCount = 0;
                    submit_info.pCommandBuffers = nullptr;
                } else {
                    submit_info.commandBufferCount = 1;
                    submit_info.pCommandBuffers = &cb;
                }
                _queue_mtx->lock();
                VK_CHECK_RESULT(vkQueueSubmit(_queue, 1u, &submit_info, present_cmd.submit_fences[0]));
                for (auto i : vstd::range(1u, present_cmd.submit_fences.size())) {
                    VK_CHECK_RESULT(vkQueueSubmit(_queue, 0, nullptr, present_cmd.submit_fences[i]));
                }
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
                    LUISA_ERROR_WITH_LOCATION("Failed to present swapchain image: {}.", luisa::to_string(result));
                }

                // Use normal completion path for the submitted prefix
                cb_ptr = nullptr;
                _evt._signal(*this, fence, cb_ptr);
                _mtx.lock();
                _exec.enqueue(SyncExt{
                    .evt = &_evt,
                    .value = fence});
                _exec.enqueue(std::move(cmdbuffer));
            } else {
                // No successful swapchains - if there are commands, submit them without present
                if (!cmds.empty()) {
                    // Check for external command buffer execution (same as normal path)
                    auto cb_ptr = &cb;
                    if (_execute_external_command_buffer(cb)) {
                        cb_ptr = nullptr;
                    }
                    _evt._signal(*this, fence, cb_ptr);
                    _mtx.lock();
                    _exec.enqueue(SyncExt{
                        .evt = &_evt,
                        .value = fence});
                    _exec.enqueue(std::move(cmdbuffer));
                } else {
                    // No commands and no successful presents - cleanup without submitting
                    cmdbuffer.states()->reset(this, *device());
                    _mtx.lock();
                    // Update fence to prevent reuse, then notify
                    _evt._update_fence(fence);
                }
                // Enqueue callbacks and completion notification
                if (!callbacks.empty()) {
                    _exec.enqueue(std::move(callbacks));
                }
                _exec.enqueue(NotifyEvt{
                    .evt = &_evt,
                    .value = fence});
                _mtx.unlock();
                return;
            }

            // Enqueue callbacks and completion notification for successful prefix
            if (!callbacks.empty()) {
                _exec.enqueue(std::move(callbacks));
            }
            _exec.enqueue(NotifyEvt{
                .evt = &_evt,
                .value = fence});

            _mtx.unlock();
            return;
        }
        // Command buffer already submitted in present path, don't submit again
        if (!presents.empty()) {
            cb_ptr = nullptr;
        }
        if (cb_ptr && _execute_external_command_buffer(cb)) {
            cb_ptr = nullptr;
        }
        _evt._signal(*this, fence, cb_ptr);
        _mtx.lock();
        _exec.enqueue(SyncExt{
            .evt = &_evt,
            .value = fence});
        _exec.enqueue(std::move(cmdbuffer));
    } else {
        _evt._update_fence(fence);
        _mtx.lock();
    }
    if (!callbacks.empty()) {
        _exec.enqueue(std::move(callbacks));
    }
    _exec.enqueue(NotifyEvt{
        .evt = &_evt,
        .value = fence});

    _mtx.unlock();
}
void Stream::update_sparse_resources(luisa::vector<SparseUpdateTile> &&textures_update) noexcept {
    std::lock_guard lck{_dispatch_mtx};
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
    _mtx.unlock();
}
void Stream::sync() {
    _evt.sync(_evt.last_fence());
}
CommandBuffer::CommandBuffer(Stream &stream) noexcept
    : Resource(stream.device()),
      _stream(stream),
      _state(vstd::make_unique<CommandBufferState>()) {
    _state->init(*stream.device(), stream.stream_tag());
    _cmdbuffer = nullptr;
    if (device()->config_ext()) {
        _cmdbuffer = device()->config_ext()->borrow_command_buffer(stream.stream_tag());
    }
    if (_cmdbuffer) {
        _ownership = detail::CommandBufferOwnership::BORROWED;
    } else {
        _ownership = detail::CommandBufferOwnership::BACKEND;
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
void CommandBuffer::begin() {
    VkCommandBufferBeginInfo bi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };
    VK_CHECK_RESULT(vkBeginCommandBuffer(_cmdbuffer, &bi));
}
void CommandBuffer::end() {
    VK_CHECK_RESULT(vkEndCommandBuffer(_cmdbuffer));
}
CommandBuffer::CommandBuffer(CommandBuffer &&rhs) noexcept
    : Resource(std::move(rhs)),
      _stream(rhs._stream),         // NOLINT(bugprone-use-after-move)
      _cmdbuffer(rhs._cmdbuffer),   // NOLINT(bugprone-use-after-move)
      _state(std::move(rhs._state)),// NOLINT(bugprone-use-after-move)
      _ownership(rhs._ownership) {  // NOLINT(bugprone-use-after-move)
    rhs._cmdbuffer = nullptr;
}
void Stream::signal(Event *event, uint64_t value) {
    std::lock_guard lck{_dispatch_mtx};
    event->_signal(*this, value);
    _mtx.lock();
    _exec.enqueue(SyncExt{event, value});
    _exec.enqueue(NotifyEvt{event, value});
    _mtx.unlock();
}
void Stream::wait(Event *event, uint64_t value) {
    std::lock_guard lck{_dispatch_mtx};
    event->_wait(*this, value);
}
void CommandBuffer::execute(vstd::span<const luisa::unique_ptr<Command>> cmds) {
    // collect argument buffer
    const auto argument_buffer_alignment = std::max<size_t>(
        32u, device()->properties().limits.minStorageBufferOffsetAlignment);
    const auto argument_buffer_range_limit = static_cast<size_t>(
        device()->properties().limits.maxStorageBufferRange);
    ArgumentBlockLayout planned_argument_buffer_layout;
    auto dispatch_shader = [&](ShaderDispatchCommandBase const *c, Shader const *shader) {
        auto plan = plan_argument_block(
            shader, c, argument_buffer_range_limit);
        validate_argument_block_plan(plan);
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
    for (auto &&command : cmds) {
        if (command->tag() == Command::Tag::EShaderDispatchCommand) {
            auto c = static_cast<ShaderDispatchCommand const *>(command.get());
            if (c->is_indirect()) {
                auto indirect = validate_indirect_dispatch_source(c);
                if (indirect.plan.command_count == 0u) {
                    // A statically empty indirect range has no target/source
                    // resource use and must not perturb reorder boundaries.
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
                auto c = static_cast<ShaderDispatchCommand const *>(command.get());
                auto shader = reinterpret_cast<Shader const *>(c->handle());
                dispatch_shader(c, shader);
            } break;
            case Command::Tag::ECustomCommand: {
                auto cmd = static_cast<CustomCommand const *>(command.get());
                if (cmd->custom_cmd_uuid() == to_underlying(CustomCommandUUID::RASTER_DRAW_SCENE)) {
                    auto c = static_cast<DrawRasterSceneCommand const *>(cmd);
                    auto shader = reinterpret_cast<Shader const *>(c->handle());
                    dispatch_shader(c, shader);
                }
            } break;
            case Command::Tag::EMotionInstanceBuildCommand: {
                // Handle motion instance build early (before reordering/preprocess)
                // to ensure child is set before TLAS build references it
                auto c = static_cast<MotionInstanceBuildCommand const *>(command.get());
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
    auto clear_reorder = vstd::scope_exit([&] {
        _stream.reorder.clear();
    });
    uniform_data->clear();
    uniform_data->reserve(uniform_buffer_size);
    BufferView arg_buffer;
    if (uniform_buffer_size > 0) {
        arg_buffer = _state->upload_alloc.allocate(
            uniform_buffer_size, argument_buffer_alignment);
    }
    ArgumentBlockLayout preprocessed_argument_buffer_layout;
    auto preprocess_arguments = [this, argument_buffer_alignment,
                                 argument_buffer_range_limit,
                                 &preprocessed_argument_buffer_layout](
                                    Shader const *shader,
                                    ShaderDispatchCommandBase const *c,
                                    bool is_raster) {
        auto plan = plan_argument_block(
            shader, c, argument_buffer_range_limit);
        validate_argument_block_plan(plan);
        size_t block_offset = 0u;
        if (!preprocessed_argument_buffer_layout.append_padded(
                plan.layout.size(), argument_buffer_alignment,
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

        vstd::vector<StorageBufferMetadata> buffer_metadata(
            plan.buffer_metadata_count);
        vstd::vector<uint32_t> validation_values;
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
                    shader, plan.buffer_metadata_count),
                emitted_trailer)) {
            LUISA_ERROR_WITH_LOCATION(
                "Vulkan argument preprocessing could not append its "
                "checked trailer layout: {}.",
                argument_block_layout_status_name(
                    emitted_layout.status()));
        }
        LUISA_ASSERT(
            emitted_layout.size() == plan.layout.size() &&
                emitted_trailer.metadata_offset ==
                    plan.trailer.metadata_offset &&
                emitted_trailer.metadata_size ==
                    plan.trailer.metadata_size &&
                emitted_trailer.validation_offset ==
                    plan.trailer.validation_offset &&
                emitted_trailer.validation_size ==
                    plan.trailer.validation_size,
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
        dispatch_offsets->emplace_back(
            block_offset, emitted_layout.size());
    };
    for (auto &&lst : cmd_lists) {
        dispatch_offsets->clear();
        auto delay_clear = vstd::scope_exit([&]() {
            scratch_buffer_alloc->clear();
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
                        preprocess_arguments(shader, c, false);
                        resource_barrier->record(
                            BufferView{indirect.source, 0u,
                                       indirect.source->byte_size()},
                            ResourceBarrier::Usage::kComputeRead);
                    } else {
                        auto shader =
                            reinterpret_cast<Shader const *>(c->handle());
                        preprocess_arguments(shader, c, false);
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
                    reinterpret_cast<Blas *>(c->handle())->pre_build(*this, c);
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
                            auto shader = reinterpret_cast<RasterShader *>(cmd->handle());
                            preprocess_arguments(shader, cmd, true);
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
                                     BindPropVisitor &visitor,
                                     ShaderDispatchCommandBase const *c,
                                     Shader const *shader,
                                     const Buffer *indirect_source = nullptr) {
            LUISA_ASSERT(
                dispatch_offset_index < dispatch_offsets->size(),
                "Vulkan descriptor binding requested argument block {} from "
                "a {}-entry preprocessing table.",
                dispatch_offset_index, dispatch_offsets->size());
            auto [relative_offset, descriptor_range] =
                (*dispatch_offsets)[dispatch_offset_index++];
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
            VkDescriptorSetAllocateInfo alloc_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = _state->desc_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = shader->desc_set_layout().data()};
            VK_CHECK_RESULT(
                vkAllocateDescriptorSets(
                    device()->logic_device(),
                    &alloc_info,
                    &visitor.desc_set));
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
        auto bind_shader_desc = [&](BindPropVisitor &visitor, Shader const *shader, VkPipelineBindPoint bind_point) {
            LUISA_ASSERT(
                visitor.desc_index ==
                    shader->local_descriptor_binding_count(),
                "Vulkan dispatch consumed {} local descriptor bindings but "
                "the validated shader interface requires {}.",
                visitor.desc_index,
                shader->local_descriptor_binding_count());
            if (!write_desc_sets->empty()) {
                LUISA_ASSERT(
                    write_desc_sets->size() <=
                        std::numeric_limits<uint32_t>::max(),
                    "Vulkan descriptor-write count is not representable.");
                vkUpdateDescriptorSets(
                    device()->logic_device(),
                    static_cast<uint32_t>(write_desc_sets->size()),
                    write_desc_sets->data(), 0,
                    nullptr);
                write_desc_sets->clear();
            }
            desc_sets->clear();
            desc_sets->push_back(visitor.desc_set);
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
            LUISA_ASSERT(
                desc_sets->size() == shader->desc_set_layout().size() &&
                    desc_sets->size() <=
                        std::numeric_limits<uint32_t>::max(),
                "Vulkan dispatch assembled {} descriptor sets for a "
                "validated pipeline layout with {} sets.",
                desc_sets->size(), shader->desc_set_layout().size());
            vkCmdBindDescriptorSets(
                _cmdbuffer,
                bind_point,
                shader->pipeline_layout(),
                0,
                static_cast<uint32_t>(desc_sets->size()),
                desc_sets->data(),
                0,
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
            VkDescriptorSet prepare_set{};
            VkDescriptorSetAllocateInfo alloc_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = _state->desc_pool,
                .descriptorSetCount = 1u,
                .pSetLayouts =
                    prepare_shader->desc_set_layout().data()};
            VK_CHECK_RESULT(vkAllocateDescriptorSets(
                device()->logic_device(), &alloc_info, &prepare_set));
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
            vkCmdBindDescriptorSets(
                _cmdbuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                prepare_shader->pipeline_layout(), 0u, 1u,
                &prepare_set, 0u, nullptr);
            vkCmdBindPipeline(
                _cmdbuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                prepare_shader->pipeline());

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
            for (auto &[buffer_pair, regions] : pending_upload.copies) {
                if (regions.empty()) continue;
                VkCopyBufferInfo2 copy_info2{
                    VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                    nullptr,
                    buffer_pair.src,
                    buffer_pair.dst,
                    static_cast<uint32_t>(regions.size()),
                    regions.data()};
                vkCmdCopyBuffer2(_cmdbuffer, &copy_info2);
            }
            pending_upload.copies.clear();
        };
        auto flush_pending_download = [&]() {
            for (auto &[buffer_pair, regions] : pending_download.copies) {
                if (regions.empty()) continue;
                VkCopyBufferInfo2 copy_info2{
                    VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                    nullptr,
                    buffer_pair.src,
                    buffer_pair.dst,
                    static_cast<uint32_t>(regions.size()),
                    regions.data()};
                vkCmdCopyBuffer2(_cmdbuffer, &copy_info2);
            }
            pending_download.copies.clear();
        };
        auto flush_all_pending = [&]() {
            flush_pending_upload();
            flush_pending_download();
        };

        // Post process: actual start record command to commandbuffer
        for (auto i = lst; i != nullptr; i = i->p_next) {
            auto cmd = i->cmd;
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
                    flush_all_pending();
                    auto c = static_cast<BufferCopyCommand const *>(cmd);
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
                    vkCmdCopyBuffer2(
                        _cmdbuffer,
                        &copy_info2);
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
                    vkCmdCopyBufferToImage2(_cmdbuffer, &copy_info);
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
                        _state->dispose_after_flush(PipelineRefHolder{ref});
                    }
                    bool is_rt_shader = (shader->shader_tag() == Shader::ShaderTag::kRayTracingShader);
                    BindPropVisitor visitor{};
                    set_dispatch_args(
                        visitor, c, shader, indirect_source);
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
                        vkCmdCopyBuffer2(
                            _cmdbuffer,
                            &copy_info2);

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
                    bind_shader_desc(visitor, shader, bind_point);
                    vkCmdBindPipeline(_cmdbuffer, bind_point, vk_pipeline);
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
                    };
                    if (c->is_indirect()) {
                        auto commands = prepare_indirect_dispatch(
                            indirect_source, indirect_plan, blk);
                        if (indirect_plan.command_count != 0u) {
                            // Preparation binds its own compute pipeline and
                            // descriptor set. Restore the target contract once,
                            // then select one absolute source record per command.
                            bind_shader_desc(
                                visitor, shader,
                                VK_PIPELINE_BIND_POINT_COMPUTE);
                            vkCmdBindPipeline(
                                _cmdbuffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                vk_pipeline);
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
                                vkCmdDispatchIndirect(
                                    _cmdbuffer,
                                    commands.buffer->vk_buffer(),
                                    command_offset);
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

                            vkCmdCopyBuffer2(
                                _cmdbuffer,
                                &copy_info2);
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
                            vkCmdCopyBuffer2(
                                _cmdbuffer,
                                &copy_info2);
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
                    vkCmdCopyBufferToImage2(_cmdbuffer, &copy_info);
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
                    vkCmdCopyImageToBuffer2(
                        _cmdbuffer,
                        &info);
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
                    vkCmdCopyImage2(
                        _cmdbuffer,
                        &info);
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
                    vkCmdCopyImageToBuffer2(
                        _cmdbuffer,
                        &info);
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
                    reinterpret_cast<Blas *>(c->handle())->build(*this, c);
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
                            auto shader = reinterpret_cast<RasterShader *>(cmd->handle());
                            auto pipe = shader->create_pipeline(cmd->rtv_texs(), cmd->dsv_tex(), cmd->mesh_format(), cmd->raster_state());
                            BindPropVisitor visitor{};
                            // bind arguments
                            set_dispatch_args(visitor, cmd, shader);
                            bind_shader_desc(visitor, shader, VK_PIPELINE_BIND_POINT_GRAPHICS);
                            vkCmdBindPipeline(_cmdbuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipeline);
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
                                .renderPass = pipe.render_pass,
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
                                .renderPass = pipe.render_pass,
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
                                    uint value = mesh.object_id();
                                    push_shader_constants(
                                        shader,
                                        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                        0u, value);
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
                                            0);
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
                                            0);
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
    if (uniform_buffer_size > 0) {
        static_cast<UploadBuffer const *>(arg_buffer.buffer)->copy_from(uniform_data->data(), arg_buffer.offset, uniform_data->size());
    }
    for (auto &i : _state->upload_alloc.alloc.allocated_buffer()) {
        reinterpret_cast<UploadBuffer *>(i.handle)->flush_host();
    }
    for (auto &i : _state->readback_alloc.alloc.allocated_buffer()) {
        reinterpret_cast<ReadbackBuffer *>(i.handle)->flush_host();
    }
    for (auto &i : _state->upload_alloc.large_buffers) {
        i->flush_host();
    }
    for (auto &i : _state->readback_alloc.large_buffers) {
        i->flush_host();
    }
}

}// namespace lc::vk
