// Test for backend-independent command reordering.
// This test covers:
// - bindless resource snapshots and saved read/write usage
// - canonical buffer/texture identities across backend wrapper aliases
// - range-sensitive buffer and texture hazards, including mip suffixes
// - raster captured-binding usage indexing

#include "ut/ut.hpp"

#include "command_reorder_visitor.h"
#include "bindless_update_contract.h"

#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

struct FakeReorderState {
    struct Resource {
        uint64_t handle;
        bool is_buffer;
    };
    std::unordered_map<uint64_t, Usage> shader_usages;
    std::unordered_map<uint64_t, std::vector<Usage>> indexed_shader_usages;
    std::unordered_map<uint64_t, std::vector<Argument>> shader_captured_bindings;
    std::unordered_map<uint64_t, std::vector<Argument>> raster_shader_captured_bindings;
    std::unordered_map<uint64_t, std::vector<Resource>> bindless_resources;
    std::unordered_map<
        uint64_t,
        std::map<std::pair<size_t, uint8_t>, Resource>>
        bindless_slot_resources;
    std::unordered_map<uint64_t, uint64_t> buffer_aliases;
    std::unordered_map<uint64_t, uint64_t> texture_aliases;
};

struct FakeReorderFuncTable {
    std::shared_ptr<FakeReorderState> state;

    [[nodiscard]] uint64_t canonical_buffer_handle(
        uint64_t handle) const noexcept {
        if (auto iter = state->buffer_aliases.find(handle);
            iter != state->buffer_aliases.end()) {
            return iter->second;
        }
        return handle;
    }

    [[nodiscard]] uint64_t canonical_texture_handle(
        uint64_t handle) const noexcept {
        if (auto iter = state->texture_aliases.find(handle);
            iter != state->texture_aliases.end()) {
            return iter->second;
        }
        return handle;
    }

    void traverse_bindless_resources(
        uint64_t bindless_handle,
        ReorderBindlessResourceVisitor visitor) const noexcept {
        if (auto iter = state->bindless_resources.find(bindless_handle);
            iter != state->bindless_resources.end()) {
            for (auto resource : iter->second) {
                visitor(resource.handle, resource.is_buffer);
            }
        }
        if (auto iter = state->bindless_slot_resources.find(bindless_handle);
            iter != state->bindless_slot_resources.end()) {
            for (auto &&[slot, resource] : iter->second) {
                static_cast<void>(slot);
                visitor(resource.handle, resource.is_buffer);
            }
        }
    }

    [[nodiscard]] Usage get_usage(
        uint64_t shader_handle, size_t argument_index) const noexcept {
        if (auto iter = state->indexed_shader_usages.find(shader_handle);
            iter != state->indexed_shader_usages.end()) {
            return iter->second.at(argument_index);
        }
        return state->shader_usages.at(shader_handle);
    }

private:
    void update_slot(
        uint64_t bindless_handle, size_t slot, uint8_t kind,
        BindlessArrayUpdateCommand::Operation operation,
        uint64_t resource_handle, bool is_buffer) const noexcept {
        auto &slots = state->bindless_slot_resources[bindless_handle];
        auto key = std::pair{slot, kind};
        switch (operation) {
            case BindlessArrayUpdateCommand::Operation::NONE: break;
            case BindlessArrayUpdateCommand::Operation::EMPLACE:
                slots.insert_or_assign(
                    key, FakeReorderState::Resource{
                             .handle = resource_handle,
                             .is_buffer = is_buffer});
                break;
            case BindlessArrayUpdateCommand::Operation::REMOVE:
                slots.erase(key);
                break;
        }
    }

public:
    void update_bindless(
        uint64_t handle,
        luisa::span<const BindlessArrayUpdateCommand::Modification> mods) const noexcept {
        for (auto &&mod : mods) {
            update_slot(handle, mod.slot, 0u, mod.buffer.op,
                        mod.buffer.handle, true);
            update_slot(handle, mod.slot, 1u, mod.tex2d.op,
                        mod.tex2d.handle, false);
            update_slot(handle, mod.slot, 2u, mod.tex3d.op,
                        mod.tex3d.handle, false);
        }
    }

    void update_bindless(
        uint64_t handle,
        luisa::span<const BindlessArrayUpdateCommand::BufferModification> mods) const noexcept {
        for (auto &&mod : mods) {
            update_slot(handle, mod.slot, 0u, mod.buffer.op,
                        mod.buffer.handle, true);
        }
    }

    void update_bindless(
        uint64_t handle,
        luisa::span<const BindlessArrayUpdateCommand::Texture2DModification> mods) const noexcept {
        for (auto &&mod : mods) {
            update_slot(handle, mod.slot, 1u, mod.tex2d.op,
                        mod.tex2d.handle, false);
        }
    }

    void update_bindless(
        uint64_t handle,
        luisa::span<const BindlessArrayUpdateCommand::Texture3DModification> mods) const noexcept {
        for (auto &&mod : mods) {
            update_slot(handle, mod.slot, 2u, mod.tex3d.op,
                        mod.tex3d.handle, false);
        }
    }

    [[nodiscard]] luisa::span<const Argument>
    shader_bindings(uint64_t shader_handle) const noexcept {
        if (auto iter = state->shader_captured_bindings.find(shader_handle);
            iter != state->shader_captured_bindings.end()) {
            return luisa::span<const Argument>{iter->second.data(), iter->second.size()};
        }
        return {};
    }

    [[nodiscard]] luisa::span<const Argument>
    raster_shader_bindings(uint64_t shader_handle) const noexcept {
        if (auto iter = state->raster_shader_captured_bindings.find(shader_handle);
            iter != state->raster_shader_captured_bindings.end()) {
            return luisa::span<const Argument>{iter->second.data(), iter->second.size()};
        }
        return {};
    }

    // todo: add work graph to this test
    [[nodiscard]] luisa::span<const Argument>
    work_graph_bindings(uint64_t shader_handle) const noexcept {
        return {}; 
    }
};

struct IncompleteBindlessUpdateFuncTable : FakeReorderFuncTable {
    void update_bindless(
        uint64_t,
        luisa::span<const BindlessArrayUpdateCommand::Modification>) const noexcept {}
};

static_assert(ReorderFuncTable<FakeReorderFuncTable>);
static_assert(!ReorderFuncTable<IncompleteBindlessUpdateFuncTable>);

using Reorder = CommandReorderVisitor<FakeReorderFuncTable, true>;

[[nodiscard]] ShaderDispatchCommand make_bindless_dispatch(
    uint64_t shader_handle, uint64_t bindless_handle) {
    Argument argument{
        .tag = Argument::Tag::BINDLESS_ARRAY,
        .bindless_array = {bindless_handle}};
    luisa::vector<std::byte> argument_buffer(sizeof(argument));
    std::memcpy(argument_buffer.data(), &argument, sizeof(argument));
    return ShaderDispatchCommand{
        shader_handle, std::move(argument_buffer), 1u,
        uint3{1u, 1u, 1u}};
}

[[nodiscard]] ShaderDispatchCommand make_buffer_dispatch(
    uint64_t shader_handle, uint64_t buffer_handle,
    size_t offset, size_t size) {
    Argument argument{
        .tag = Argument::Tag::BUFFER,
        .buffer = {buffer_handle, offset, size}};
    luisa::vector<std::byte> argument_buffer(sizeof(argument));
    std::memcpy(argument_buffer.data(), &argument, sizeof(argument));
    return ShaderDispatchCommand{
        shader_handle, std::move(argument_buffer), 1u,
        uint3{1u, 1u, 1u}};
}

[[nodiscard]] ShaderDispatchCommand make_texture_dispatch(
    uint64_t shader_handle, uint64_t texture_handle,
    uint32_t level) {
    Argument argument{
        .tag = Argument::Tag::TEXTURE,
        .texture = {texture_handle, level}};
    luisa::vector<std::byte> argument_buffer(sizeof(argument));
    std::memcpy(argument_buffer.data(), &argument, sizeof(argument));
    return ShaderDispatchCommand{
        shader_handle, std::move(argument_buffer), 1u,
        uint3{1u, 1u, 1u}};
}

[[nodiscard]] DrawRasterSceneCommand make_raster_buffer_dispatch(
    uint64_t shader_handle, uint64_t buffer_handle,
    size_t offset, size_t size, MeshFormat const &mesh_format) {
    Argument argument{
        .tag = Argument::Tag::BUFFER,
        .buffer = {buffer_handle, offset, size}};
    luisa::vector<std::byte> argument_buffer(sizeof(argument));
    std::memcpy(argument_buffer.data(), &argument, sizeof(argument));
    std::array<Argument::Texture, 8u> render_targets{};
    Argument::Texture depth_target{
        .handle = std::numeric_limits<uint64_t>::max(),
        .level = 0u};
    return DrawRasterSceneCommand{
        shader_handle, std::move(argument_buffer), 1u, render_targets, 0u, depth_target, {}, Viewport{0u, 0u, 1u, 1u}, RasterState{}, &mesh_format};
}

[[nodiscard]] DrawRasterSceneCommand make_empty_raster_dispatch(
    uint64_t shader_handle, MeshFormat const &mesh_format) {
    std::array<Argument::Texture, 8u> render_targets{};
    Argument::Texture depth_target{
        .handle = std::numeric_limits<uint64_t>::max(),
        .level = 0u};
    return DrawRasterSceneCommand{
        shader_handle, {}, 0u, render_targets, 0u, depth_target, {}, Viewport{0u, 0u, 1u, 1u}, RasterState{}, &mesh_format};
}

[[nodiscard]] size_t direct_read_then_bindless(
    Usage bindless_usage) {
    constexpr auto resource = 11u;
    constexpr auto heap = 21u;
    constexpr auto shader = 31u;
    auto state = std::make_shared<FakeReorderState>();
    state->shader_usages.emplace(shader, bindless_usage);
    state->bindless_resources.emplace(
        heap, std::vector{FakeReorderState::Resource{
                  .handle = resource, .is_buffer = true}});
    Reorder reorder{FakeReorderFuncTable{state}};
    std::byte destination{};
    BufferDownloadCommand direct_read{resource, 0u, 1u, &destination};
    auto dispatch = make_bindless_dispatch(shader, heap);
    reorder.visit(&direct_read);
    reorder.visit(&dispatch);
    return reorder.command_lists().size();
}

[[nodiscard]] size_t aliased_buffer_access_layers(
    bool first_writes, bool second_writes,
    size_t first_offset = 0u, size_t second_offset = 0u) {
    constexpr auto owner = 41u;
    constexpr auto alias = 42u;
    constexpr auto access_size = 4u;
    auto state = std::make_shared<FakeReorderState>();
    state->buffer_aliases.emplace(alias, owner);
    Reorder reorder{FakeReorderFuncTable{state}};
    std::array<std::byte, access_size> first_data{};
    std::array<std::byte, access_size> second_data{};
    BufferUploadCommand first_write{
        owner, first_offset, access_size, first_data.data()};
    BufferDownloadCommand first_read{
        owner, first_offset, access_size, first_data.data()};
    BufferUploadCommand second_write{
        alias, second_offset, access_size, second_data.data()};
    BufferDownloadCommand second_read{
        alias, second_offset, access_size, second_data.data()};
    if (first_writes) {
        reorder.visit(&first_write);
    } else {
        reorder.visit(&first_read);
    }
    if (second_writes) {
        reorder.visit(&second_write);
    } else {
        reorder.visit(&second_read);
    }
    return reorder.command_lists().size();
}

[[nodiscard]] size_t command_layer(
    Reorder const &reorder, Command const *command) noexcept {
    auto layers = reorder.command_lists();
    for (auto layer = 0u; layer < layers.size(); ++layer) {
        for (auto link = layers[layer]; link != nullptr;
             link = link->p_next) {
            if (link->cmd == command) { return layer; }
        }
    }
    return std::numeric_limits<size_t>::max();
}

class IsolatedBindlessCustomCommand final : public CustomDispatchCommand {
private:
    Argument::BindlessArray _argument;

public:
    explicit IsolatedBindlessCustomCommand(uint64_t handle) noexcept
        : _argument{handle} {}

    [[nodiscard]] uint64_t custom_cmd_uuid() const noexcept override {
        return static_cast<uint64_t>(
            CustomCommandUUID::CUSTOM_DISPATCH);
    }

    [[nodiscard]] StreamTag stream_tag() const noexcept override {
        return StreamTag::COMPUTE;
    }

    [[nodiscard]] uint3 max_dispatch_size() const noexcept override {
        return make_uint3(1u);
    }

    [[nodiscard]] bool
    requires_resource_state_isolation() const noexcept override {
        return true;
    }

    void traverse_arguments(
        MutableArgumentVisitor &visitor) noexcept override {
        visitor.visit(_argument, Usage::READ);
    }

    void traverse_arguments(
        ArgumentVisitor &visitor) const noexcept override {
        visitor.visit(_argument, Usage::READ);
    }
};

}// namespace

int main() {
    "bindless_write_waits_for_prior_direct_read"_test = [] {
        expect(direct_read_then_bindless(Usage::WRITE) == 2u);
    };

    "bindless_read_can_share_prior_direct_read_layer"_test = [] {
        expect(direct_read_then_bindless(Usage::READ) == 1u);
    };

    "isolated_native_bindless_state_has_its_own_resource_layer"_test = [] {
        constexpr auto texture = 14u;
        constexpr auto heap = 24u;
        auto state = std::make_shared<FakeReorderState>();
        state->bindless_resources.emplace(
            heap, std::vector{FakeReorderState::Resource{
                      .handle = texture, .is_buffer = false}});
        Reorder reorder{FakeReorderFuncTable{state}};
        std::array<std::byte, 4u> before_data{};
        std::array<std::byte, 4u> after_data{};
        TextureDownloadCommand before{
            texture, PixelStorage::BYTE4, 0u,
            make_uint3(1u), before_data.data()};
        IsolatedBindlessCustomCommand native{heap};
        TextureDownloadCommand after{
            texture, PixelStorage::BYTE4, 0u,
            make_uint3(1u), after_data.data()};

        before.accept(reorder);
        native.accept(reorder);
        after.accept(reorder);

        expect(command_layer(reorder, &before) == 0u);
        expect(command_layer(reorder, &native) == 1u)
            << "an exact native state must not merge with a prior read";
        expect(command_layer(reorder, &after) == 2u)
            << "a following abstract read must start after the native state";
    };

    "direct_read_waits_for_prior_bindless_write_snapshot"_test = [] {
        constexpr auto resource = 12u;
        constexpr auto heap = 22u;
        constexpr auto shader = 32u;
        auto state = std::make_shared<FakeReorderState>();
        state->shader_usages.emplace(shader, Usage::WRITE);
        state->bindless_resources.emplace(
            heap, std::vector{FakeReorderState::Resource{
                      .handle = resource, .is_buffer = true}});
        Reorder reorder{FakeReorderFuncTable{state}};
        auto dispatch = make_bindless_dispatch(shader, heap);
        reorder.visit(&dispatch);

        // Mutating the heap after the dispatch must not erase the resource
        // dependency snapshotted for that dispatch.
        state->bindless_resources[heap].clear();
        std::byte destination{};
        BufferDownloadCommand direct_read{
            resource, 0u, 1u, &destination};
        reorder.visit(&direct_read);
        expect(reorder.command_lists().size() == 2u);
    };

    "distinct_bindless_arrays_order_shared_buffer"_test = [] {
        constexpr auto resource = 13u;
        constexpr auto write_heap = 23u;
        constexpr auto read_heap = 24u;
        constexpr auto write_shader = 33u;
        constexpr auto read_shader = 34u;
        auto state = std::make_shared<FakeReorderState>();
        state->shader_usages.emplace(write_shader, Usage::WRITE);
        state->shader_usages.emplace(read_shader, Usage::READ);
        auto binding = std::vector{FakeReorderState::Resource{
            .handle = resource, .is_buffer = true}};
        state->bindless_resources.emplace(write_heap, binding);
        state->bindless_resources.emplace(read_heap, binding);
        Reorder reorder{FakeReorderFuncTable{state}};
        auto write = make_bindless_dispatch(write_shader, write_heap);
        auto read = make_bindless_dispatch(read_shader, read_heap);
        reorder.visit(&write);
        reorder.visit(&read);
        expect(reorder.command_lists().size() == 2u);
    };

    "bindless_texture_remains_read_only_for_buffer_write_usage"_test = [] {
        constexpr auto texture = 14u;
        constexpr auto heap = 25u;
        constexpr auto shader = 35u;
        auto state = std::make_shared<FakeReorderState>();
        state->shader_usages.emplace(shader, Usage::WRITE);
        state->bindless_resources.emplace(
            heap, std::vector{FakeReorderState::Resource{
                      .handle = texture, .is_buffer = false}});
        Reorder reorder{FakeReorderFuncTable{state}};
        auto dispatch = make_bindless_dispatch(shader, heap);
        reorder.visit(&dispatch);
        std::array<std::byte, 4u> destination{};
        TextureDownloadCommand direct_read{
            texture, PixelStorage::BYTE4, 0u,
            uint3{1u, 1u, 1u}, destination.data()};
        reorder.visit(&direct_read);
        expect(reorder.command_lists().size() == 1u);
    };

    "bindless_texture_read_waits_for_prior_upload"_test = [] {
        constexpr auto texture = 15u;
        constexpr auto heap = 26u;
        constexpr auto shader = 36u;
        auto state = std::make_shared<FakeReorderState>();
        state->shader_usages.emplace(shader, Usage::READ);
        state->bindless_resources.emplace(
            heap, std::vector{FakeReorderState::Resource{
                      .handle = texture, .is_buffer = false}});
        Reorder reorder{FakeReorderFuncTable{state}};
        std::array<std::byte, 4u> source{};
        TextureUploadCommand upload{
            texture, PixelStorage::BYTE4, 5u,
            uint3{1u, 1u, 1u}, source.data()};
        auto dispatch = make_bindless_dispatch(shader, heap);
        reorder.visit(&upload);
        reorder.visit(&dispatch);
        expect(reorder.command_lists().size() == 2u);
    };

    "canonical_buffer_owner_write_alias_read"_test = [] {
        expect(aliased_buffer_access_layers(true, false) == 2u);
    };

    "canonical_buffer_owner_read_alias_write"_test = [] {
        expect(aliased_buffer_access_layers(false, true) == 2u);
    };

    "canonical_buffer_owner_write_alias_write"_test = [] {
        expect(aliased_buffer_access_layers(true, true) == 2u);
    };

    "canonical_buffer_disjoint_alias_ranges_share_layer"_test = [] {
        expect(aliased_buffer_access_layers(true, false, 0u, 4u) == 1u);
    };

    "canonical_bindless_buffer_alias_orders_direct_alias"_test = [] {
        constexpr auto owner = 51u;
        constexpr auto bindless_alias = 52u;
        constexpr auto direct_alias = 53u;
        constexpr auto heap = 54u;
        constexpr auto shader = 55u;
        auto state = std::make_shared<FakeReorderState>();
        state->buffer_aliases.emplace(bindless_alias, owner);
        state->buffer_aliases.emplace(direct_alias, owner);
        state->shader_usages.emplace(shader, Usage::WRITE);
        state->bindless_resources.emplace(
            heap, std::vector{FakeReorderState::Resource{
                      .handle = bindless_alias, .is_buffer = true}});
        Reorder reorder{FakeReorderFuncTable{state}};
        auto dispatch = make_bindless_dispatch(shader, heap);
        reorder.visit(&dispatch);
        std::array<std::byte, 4u> destination{};
        BufferDownloadCommand direct_read{
            direct_alias, 0u, destination.size(), destination.data()};
        reorder.visit(&direct_read);
        expect(reorder.command_lists().size() == 2u);
    };

    "canonical_shader_buffer_alias_orders_direct_owner"_test = [] {
        constexpr auto owner = 56u;
        constexpr auto shader_alias = 57u;
        constexpr auto shader = 58u;
        auto state = std::make_shared<FakeReorderState>();
        state->buffer_aliases.emplace(shader_alias, owner);
        state->shader_usages.emplace(shader, Usage::READ);
        Reorder reorder{FakeReorderFuncTable{state}};
        std::array<std::byte, 4u> source{};
        BufferUploadCommand direct_write{
            owner, 0u, source.size(), source.data()};
        auto dispatch = make_buffer_dispatch(
            shader, shader_alias, 0u, source.size());
        reorder.visit(&direct_write);
        reorder.visit(&dispatch);
        expect(reorder.command_lists().size() == 2u);
    };

    "canonical_texture_owner_write_alias_read"_test = [] {
        constexpr auto owner = 61u;
        constexpr auto alias = 62u;
        auto state = std::make_shared<FakeReorderState>();
        state->texture_aliases.emplace(alias, owner);
        Reorder reorder{FakeReorderFuncTable{state}};
        std::array<std::byte, 4u> source{};
        std::array<std::byte, 4u> destination{};
        TextureUploadCommand write{
            owner, PixelStorage::BYTE4, 0u,
            uint3{1u, 1u, 1u}, source.data()};
        TextureDownloadCommand read{
            alias, PixelStorage::BYTE4, 0u,
            uint3{1u, 1u, 1u}, destination.data()};
        reorder.visit(&write);
        reorder.visit(&read);
        expect(reorder.command_lists().size() == 2u);
    };

    "buffer_and_texture_identity_namespaces_are_distinct"_test = [] {
        constexpr auto shared_bits = 71u;
        auto state = std::make_shared<FakeReorderState>();
        Reorder reorder{FakeReorderFuncTable{state}};
        std::array<std::byte, 4u> source{};
        std::array<std::byte, 4u> destination{};
        BufferUploadCommand buffer_write{
            shared_bits, 0u, source.size(), source.data()};
        TextureDownloadCommand texture_read{
            shared_bits, PixelStorage::BYTE4, 0u,
            uint3{1u, 1u, 1u}, destination.data()};
        reorder.visit(&buffer_write);
        reorder.visit(&texture_read);
        expect(reorder.command_lists().size() == 1u);
    };

    "sampled_texture_read_uses_base_mip_suffix"_test = [] {
        constexpr auto texture = 72u;
        constexpr auto shader = 73u;
        auto state = std::make_shared<FakeReorderState>();
        state->shader_usages.emplace(shader, Usage::READ);
        Reorder reorder{FakeReorderFuncTable{state}};
        std::array<std::byte, 4u> source{};
        TextureUploadCommand upload{
            texture, PixelStorage::BYTE4, 5u,
            uint3{1u, 1u, 1u}, source.data()};
        auto sample = make_texture_dispatch(shader, texture, 2u);
        reorder.visit(&upload);
        reorder.visit(&sample);
        expect(reorder.command_lists().size() == 2u);
    };

    "sampled_texture_read_excludes_lower_mips"_test = [] {
        constexpr auto texture = 74u;
        constexpr auto shader = 75u;
        auto state = std::make_shared<FakeReorderState>();
        state->shader_usages.emplace(shader, Usage::READ);
        Reorder reorder{FakeReorderFuncTable{state}};
        std::array<std::byte, 4u> source{};
        TextureUploadCommand upload{
            texture, PixelStorage::BYTE4, 1u,
            uint3{1u, 1u, 1u}, source.data()};
        auto sample = make_texture_dispatch(shader, texture, 2u);
        reorder.visit(&upload);
        reorder.visit(&sample);
        expect(reorder.command_lists().size() == 1u);
    };

    "storage_texture_write_uses_only_base_mip"_test = [] {
        constexpr auto texture = 76u;
        constexpr auto shader = 77u;
        auto state = std::make_shared<FakeReorderState>();
        state->shader_usages.emplace(shader, Usage::WRITE);
        Reorder reorder{FakeReorderFuncTable{state}};
        std::array<std::byte, 4u> source{};
        TextureUploadCommand upload{
            texture, PixelStorage::BYTE4, 5u,
            uint3{1u, 1u, 1u}, source.data()};
        auto write = make_texture_dispatch(shader, texture, 2u);
        reorder.visit(&upload);
        reorder.visit(&write);
        expect(reorder.command_lists().size() == 1u);
    };

    "read_write_texture_splits_sampled_suffix_and_storage_base"_test = [] {
        constexpr auto texture = 78u;
        constexpr auto shader = 79u;
        auto state = std::make_shared<FakeReorderState>();
        state->shader_usages.emplace(shader, Usage::READ_WRITE);
        std::array<std::byte, 4u> data{};

        Reorder suffix_reorder{FakeReorderFuncTable{state}};
        TextureUploadCommand upper_mip_write{
            texture, PixelStorage::BYTE4, 5u,
            uint3{1u, 1u, 1u}, data.data()};
        auto read_write = make_texture_dispatch(shader, texture, 2u);
        suffix_reorder.visit(&upper_mip_write);
        suffix_reorder.visit(&read_write);
        expect(suffix_reorder.command_lists().size() == 2u);

        Reorder base_reorder{FakeReorderFuncTable{state}};
        TextureDownloadCommand base_mip_read{
            texture, PixelStorage::BYTE4, 2u,
            uint3{1u, 1u, 1u}, data.data()};
        auto second_read_write = make_texture_dispatch(shader, texture, 2u);
        base_reorder.visit(&base_mip_read);
        base_reorder.visit(&second_read_write);
        expect(base_reorder.command_lists().size() == 2u);
    };

    "buffer_ranges_are_safe_near_uint64_limit"_test = [] {
        constexpr auto buffer = 80u;
        constexpr auto max_offset = std::numeric_limits<size_t>::max();
        std::array<std::byte, 8u> data{};

        Reorder disjoint_reorder{FakeReorderFuncTable{
            std::make_shared<FakeReorderState>()}};
        BufferUploadCommand lower_write{
            buffer, max_offset - 8u, 4u, data.data()};
        BufferDownloadCommand upper_read{
            buffer, max_offset - 4u, 4u, data.data()};
        disjoint_reorder.visit(&lower_write);
        disjoint_reorder.visit(&upper_read);
        expect(disjoint_reorder.command_lists().size() == 1u);

        Reorder overlap_reorder{FakeReorderFuncTable{
            std::make_shared<FakeReorderState>()}};
        BufferUploadCommand full_tail_write{
            buffer, max_offset - 8u, 8u, data.data()};
        BufferDownloadCommand tail_read{
            buffer, max_offset - 1u, 1u, data.data()};
        overlap_reorder.visit(&full_tail_write);
        overlap_reorder.visit(&tail_read);
        expect(overlap_reorder.command_lists().size() == 2u);

        if constexpr (sizeof(size_t) >= sizeof(uint64_t)) {
            constexpr auto signed_boundary = static_cast<size_t>(
                std::numeric_limits<int64_t>::max());
            Reorder signed_crossing_reorder{FakeReorderFuncTable{
                std::make_shared<FakeReorderState>()}};
            BufferUploadCommand crossing_write{
                buffer, signed_boundary - 1u, 4u, data.data()};
            BufferDownloadCommand crossing_read{
                buffer, signed_boundary + 1u, 1u, data.data()};
            signed_crossing_reorder.visit(&crossing_write);
            signed_crossing_reorder.visit(&crossing_read);
            expect(signed_crossing_reorder.command_lists().size() == 2u)
                << "a half-open range crossing INT64_MAX must remain an "
                   "overlapping unsigned resource range";
        }
    };

    "texture_copy_buffer_hazard_size_uses_checked_wide_products"_test = [] {
        if constexpr (sizeof(size_t) > sizeof(uint32_t)) {
            constexpr auto buffer = 203u;
            constexpr auto texture = 204u;
            Reorder reorder{FakeReorderFuncTable{
                std::make_shared<FakeReorderState>()}};
            BufferToTextureCopyCommand texture_copy{
                buffer, 0u, texture, PixelStorage::BYTE1, 0u,
                uint3{65536u, 65536u, 1u}};
            std::byte source{};
            BufferUploadCommand overlapping_write{
                buffer,
                static_cast<size_t>(uint64_t{1u} << 31u),
                1u, &source};
            reorder.visit(&texture_copy);
            reorder.visit(&overlapping_write);
            expect(reorder.command_lists().size() == 2u)
                << "the 4-GiB texture-copy source range must not wrap to "
                   "an empty 32-bit product";
        }
    };

    "independent_commands_preserve_source_order_within_a_layer"_test = [] {
        auto state = std::make_shared<FakeReorderState>();
        Reorder reorder{FakeReorderFuncTable{state}};
        std::array<std::byte, 4u> first_destination{};
        std::array<std::byte, 4u> second_destination{};
        BufferDownloadCommand first{
            201u, 0u, first_destination.size(),
            first_destination.data()};
        BufferDownloadCommand second{
            202u, 0u, second_destination.size(),
            second_destination.data()};
        reorder.visit(&first);
        reorder.visit(&second);
        auto layers = reorder.command_lists();
        expect(layers.size() == 1u);
        auto link = layers.front();
        expect(link != nullptr);
        if (link != nullptr) {
            expect(link->cmd == &first);
            link = link->p_next;
        }
        expect(link != nullptr);
        if (link != nullptr) {
            expect(link->cmd == &second);
            expect(link->p_next == nullptr);
        }
    };

    "raster_captured_resource_participates_in_reordering"_test = [] {
        constexpr auto buffer = 81u;
        constexpr auto shader = 82u;
        auto state = std::make_shared<FakeReorderState>();
        state->indexed_shader_usages.emplace(
            shader, std::vector{Usage::READ});
        state->raster_shader_captured_bindings.emplace(
            shader, std::vector{Argument{
                        .tag = Argument::Tag::BUFFER,
                        .buffer = {buffer, 0u, 4u}}});
        Reorder reorder{FakeReorderFuncTable{state}};
        std::array<std::byte, 4u> source{};
        BufferUploadCommand upload{
            buffer, 0u, source.size(), source.data()};
        MeshFormat mesh_format;
        auto draw = make_empty_raster_dispatch(shader, mesh_format);
        reorder.visit(&upload);
        reorder.visit(&draw);
        expect(reorder.command_lists().size() == 2u);
    };

    "raster_captured_uniform_advances_usage_index"_test = [] {
        constexpr auto buffer = 83u;
        constexpr auto shader = 84u;
        auto state = std::make_shared<FakeReorderState>();
        state->indexed_shader_usages.emplace(
            shader, std::vector{Usage::NONE, Usage::WRITE});
        state->raster_shader_captured_bindings.emplace(
            shader, std::vector{Argument{
                        .tag = Argument::Tag::UNIFORM,
                        .uniform = {0u, 4u, 4u}}});
        Reorder reorder{FakeReorderFuncTable{state}};
        std::array<std::byte, 4u> destination{};
        BufferDownloadCommand prior_read{
            buffer, 0u, destination.size(), destination.data()};
        MeshFormat mesh_format;
        auto draw = make_raster_buffer_dispatch(
            shader, buffer, 0u, destination.size(), mesh_format);
        reorder.visit(&prior_read);
        reorder.visit(&draw);
        expect(reorder.command_lists().size() == 2u);
    };

    "indirect_workload_is_charged_after_resource_layering"_test = [] {
        constexpr auto indirect_buffer = 91u;
        constexpr auto direct_buffer = 92u;
        constexpr auto indirect_shader = 93u;
        constexpr auto direct_shader = 94u;
        auto state = std::make_shared<FakeReorderState>();
        state->shader_usages.emplace(direct_shader, Usage::READ);
        Reorder reorder{FakeReorderFuncTable{state}};
        std::array<std::byte, 4u> data{};
        BufferUploadCommand indirect_source_write{
            indirect_buffer, 0u, data.size(), data.data()};
        ShaderDispatchCommand indirect{
            indirect_shader, {}, 0u, IndirectDispatchArg{.handle = indirect_buffer, .offset = 0u, .max_dispatch_size = std::numeric_limits<uint32_t>::max()}};
        BufferUploadCommand direct_source_write{
            direct_buffer, 0u, data.size(), data.data()};
        auto direct = make_buffer_dispatch(
            direct_shader, direct_buffer, 0u, data.size());

        reorder.visit(&indirect_source_write);
        reorder.visit(&indirect);
        reorder.visit(&direct_source_write);
        reorder.visit(&direct);
        expect(reorder.command_lists().size() == 3u)
            << "the oversized indirect dispatch must charge its dependency-selected layer";
    };

    "multiple_dispatch_workload_charges_every_dispatch_in_the_batch"_test = [] {
        auto state = std::make_shared<FakeReorderState>();
        Reorder reorder{FakeReorderFuncTable{state}};
        luisa::vector<uint3> dispatches;
        dispatches.emplace_back(1'100'000u, 1u, 1u);
        dispatches.emplace_back(1'100'000u, 1u, 1u);
        ShaderDispatchCommand batch{
            95u, {}, 0u, std::move(dispatches)};
        ShaderDispatchCommand following{
            96u, {}, 0u, uint3{1u, 1u, 1u}};
        reorder.visit(&batch);
        reorder.visit(&following);
        expect(command_layer(reorder, &batch) == 0u);
        expect(command_layer(reorder, &following) == 1u)
            << "a batch whose cumulative workload exceeds the layer budget "
               "must displace following independent work";
    };

    "bindless_update_contract_distinguishes_none_and_checks_slots"_test = [] {
        using namespace lc::bindless_update_detail;
        using Operation = BindlessArrayUpdateCommand::Operation;
        static_assert(valid_operation(Operation::NONE));
        static_assert(valid_operation(Operation::EMPLACE));
        static_assert(valid_operation(Operation::REMOVE));
        static_assert(!valid_operation(static_cast<Operation>(3u)));
        static_assert(!changes_slot(Operation::NONE));
        static_assert(changes_slot(Operation::EMPLACE));
        static_assert(changes_slot(Operation::REMOVE));
        static_assert(slot_in_bounds(3u, 4u));
        static_assert(!slot_in_bounds(4u, 4u));

        expect(!changes_slot(Operation::NONE))
            << "NONE must not dereference an existing backend slot";
        expect(!slot_in_bounds(4u, 4u));
    };

    "all_bindless_update_variants_feed_the_next_snapshot"_test = [] {
        using Update = BindlessArrayUpdateCommand;
        constexpr auto heap = 101u;
        constexpr auto shader = 102u;
        auto exercise_buffer = [&](auto modifications) {
            constexpr auto resource = 103u;
            auto state = std::make_shared<FakeReorderState>();
            state->shader_usages.emplace(shader, Usage::WRITE);
            Reorder reorder{FakeReorderFuncTable{state}};
            Update update{heap, std::move(modifications)};
            auto dispatch = make_bindless_dispatch(shader, heap);
            std::array<std::byte, 4u> destination{};
            BufferDownloadCommand read{
                resource, 0u, destination.size(), destination.data()};
            reorder.visit(&update);
            reorder.visit(&dispatch);
            reorder.visit(&read);
            expect(reorder.command_lists().size() == 3u);
        };
        auto exercise_texture = [&](auto modifications) {
            constexpr auto resource = 104u;
            auto state = std::make_shared<FakeReorderState>();
            state->shader_usages.emplace(shader, Usage::READ);
            Reorder reorder{FakeReorderFuncTable{state}};
            Update update{heap, std::move(modifications)};
            auto dispatch = make_bindless_dispatch(shader, heap);
            std::array<std::byte, 4u> source{};
            TextureUploadCommand write{
                resource, PixelStorage::BYTE4, 5u,
                uint3{1u, 1u, 1u}, source.data()};
            reorder.visit(&update);
            reorder.visit(&dispatch);
            reorder.visit(&write);
            expect(reorder.command_lists().size() == 3u);
        };

        luisa::vector<Update::Modification> general;
        general.emplace_back(
            0u, Update::ModifiedBuffer::emplace(103u, 0u, 4u),
            Update::ModifiedTexture{}, Update::ModifiedTexture{});
        exercise_buffer(std::move(general));

        luisa::vector<Update::BufferModification> buffers;
        buffers.emplace_back(
            0u, Update::ModifiedBuffer::emplace(103u, 0u, 4u));
        exercise_buffer(std::move(buffers));

        luisa::vector<Update::Texture2DModification> textures_2d;
        textures_2d.emplace_back(
            0u, Update::ModifiedTexture::emplace(
                    104u, Sampler::point_zero()));
        exercise_texture(std::move(textures_2d));

        luisa::vector<Update::Texture3DModification> textures_3d;
        textures_3d.emplace_back(
            0u, Update::ModifiedTexture::emplace(
                    104u, Sampler::point_zero()));
        exercise_texture(std::move(textures_3d));
    };

    "bindless_replacement_and_removal_change_future_snapshots"_test = [] {
        using Update = BindlessArrayUpdateCommand;
        constexpr auto heap = 111u;
        constexpr auto shader = 112u;
        constexpr auto old_resource = 113u;
        constexpr auto new_resource = 114u;

        {
            auto state = std::make_shared<FakeReorderState>();
            state->shader_usages.emplace(shader, Usage::WRITE);
            Reorder reorder{FakeReorderFuncTable{state}};
            luisa::vector<Update::BufferModification> initial;
            initial.emplace_back(
                0u, Update::ModifiedBuffer::emplace(
                        old_resource, 0u, 4u));
            Update initial_update{heap, std::move(initial)};
            auto old_dispatch = make_bindless_dispatch(shader, heap);
            luisa::vector<Update::BufferModification> replacement;
            replacement.emplace_back(
                0u, Update::ModifiedBuffer::emplace(
                        new_resource, 0u, 4u));
            Update replace_update{heap, std::move(replacement)};
            auto new_dispatch = make_bindless_dispatch(shader, heap);
            std::array<std::byte, 4u> old_data{};
            std::array<std::byte, 4u> new_data{};
            BufferDownloadCommand old_read{
                old_resource, 0u, old_data.size(), old_data.data()};
            BufferDownloadCommand new_read{
                new_resource, 0u, new_data.size(), new_data.data()};

            reorder.visit(&initial_update);
            reorder.visit(&old_dispatch);
            reorder.visit(&replace_update);
            reorder.visit(&new_dispatch);
            reorder.visit(&old_read);
            reorder.visit(&new_read);
            expect(command_layer(reorder, &old_read) == 2u)
                << "the replacement dispatch must stop naming the old slot resource";
            expect(command_layer(reorder, &new_read) == 4u)
                << "the replacement dispatch must name the new slot resource";
        }

        {
            auto state = std::make_shared<FakeReorderState>();
            state->shader_usages.emplace(shader, Usage::WRITE);
            Reorder reorder{FakeReorderFuncTable{state}};
            luisa::vector<Update::BufferModification> initial;
            initial.emplace_back(
                0u, Update::ModifiedBuffer::emplace(
                        old_resource, 0u, 4u));
            Update initial_update{heap, std::move(initial)};
            auto old_dispatch = make_bindless_dispatch(shader, heap);
            luisa::vector<Update::BufferModification> removal;
            removal.emplace_back(
                0u, Update::ModifiedBuffer::remove());
            Update remove_update{heap, std::move(removal)};
            auto empty_dispatch = make_bindless_dispatch(shader, heap);
            std::array<std::byte, 4u> data{};
            BufferDownloadCommand old_read{
                old_resource, 0u, data.size(), data.data()};

            reorder.visit(&initial_update);
            reorder.visit(&old_dispatch);
            reorder.visit(&remove_update);
            reorder.visit(&empty_dispatch);
            reorder.visit(&old_read);
            expect(command_layer(reorder, &old_read) == 2u)
                << "a removed slot must not participate in later dispatch snapshots";
        }
    };
}
