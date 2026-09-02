// End-to-end test for the C++ remote DeviceInterface, server, command codec,
// asynchronous completion, readback, and AST resource-binding remap.

#include "ut/ut.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_set>

#include <asio.hpp>

#include <luisa/backends/ext/remote_config_ext.h>
#include <luisa/luisa-compute.h>

#include "remote_blob_cache.h"
#include "remote_server.h"
#include "remote_transport.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::remote;
using namespace boost::ut;
using namespace boost::ut::literals;
using namespace std::chrono_literals;

namespace {

[[nodiscard]] uint64_t parse_u64(string_view text) {
    uint64_t value{};
    auto result = std::from_chars(
        text.data(), text.data() + text.size(), value);
    expect(result.ec == std::errc{});
    expect(result.ptr == text.data() + text.size());
    return value;
}

void write_test_shader_option(
    Writer &writer, const ShaderOption &option) noexcept {
    writer.write_bool(option.enable_cache);
    writer.write_bool(option.enable_fast_math);
    writer.write_bool(option.enable_debug_info);
    writer.write_bool(option.compile_only);
    writer.write_u32(option.max_registers);
    writer.write_bool(option.time_trace);
    writer.write_bool(option.enable_extended_accel_limits);
    writer.write_bool(option.enable_scalarizer);
    writer.write_bool(option.enable_ray_query_pipeline);
    writer.write_bool(option.force_ray_query_pipeline);
    writer.write_bool(option.enable_driver_optimization);
    writer.write_string(option.name);
}

struct MockState {
    std::atomic_bool ast_received{false};
    std::atomic_bool ast_binding_remapped{false};
    std::atomic_bool custom_callable_received{false};
    std::atomic_bool bindless_update_remapped{false};
    std::atomic_bool mesh_build_remapped{false};
    std::atomic_bool curve_build_remapped{false};
    std::atomic_bool procedural_build_remapped{false};
    std::atomic_bool motion_build_remapped{false};
    std::atomic_bool accel_build_remapped{false};
    std::atomic_bool indirect_buffer_created{false};
    std::atomic_bool indirect_writer_remapped{false};
    std::atomic_bool indirect_dispatch_remapped{false};
    std::atomic_uint texture_command_count{0u};
    std::atomic_uint live_buffers{0u};
};

struct MockBuffer {
    vector<std::byte> bytes;
    bool indirect_dispatch{};
    size_t indirect_dispatch_capacity{};
};

struct MockTexture {
    PixelStorage storage{};
    uint3 size{};
    vector<std::byte> bytes;
};

struct MockStream {
    DeviceInterface::StreamLogCallback log_callback;
};

struct MockShader {
    vector<Usage> usages;
    vector<Function::Binding> bindings;
    uint3 block_size{};
    bool indirect_writer{};
};

struct MockBindless {
    size_t size{};
    BindlessSlotType type{};
};

struct MockEvent {
    std::atomic_uint64_t value{0u};
};

struct MockPrimitive {
    Resource::Tag tag{};
    size_t keyframe_count{};
};

struct MockAccel {};

class MockDevice final : public DeviceInterface {

private:
    shared_ptr<MockState> _state;
    std::mutex _mutex;
    std::unordered_set<MockBuffer *> _buffers;
    std::unordered_set<MockTexture *> _textures;
    std::unordered_set<MockPrimitive *> _primitives;
    std::unordered_set<MockAccel *> _accels;

public:
    MockDevice(Context &&context, shared_ptr<MockState> state,
               string backend = "mock") noexcept
        : DeviceInterface{std::move(context)}, _state{std::move(state)} {
        _backend_name = std::move(backend);
    }

    ~MockDevice() noexcept override = default;

    void *native_handle() const noexcept override {
        return const_cast<MockDevice *>(this);
    }
    uint compute_warp_size() const noexcept override { return 32u; }
    size_t compute_max_shared_memory_size() const noexcept override { return 32768u; }
    uint64_t memory_granularity() const noexcept override { return 256u; }

    BufferCreationInfo create_buffer(
        const Type *element, size_t count, void *) noexcept override {
        auto indirect = element == Type::of<IndirectKernelDispatch>();
        auto stride = indirect                    ? 32u :
                      element == Type::of<void>() ? 1u :
                                                    element->size();
        auto buffer = new MockBuffer;
        buffer->bytes.resize(stride * count);
        buffer->indirect_dispatch = indirect;
        buffer->indirect_dispatch_capacity = indirect ? count : 0u;
        if (indirect) {
            _state->indirect_buffer_created.store(
                true, std::memory_order_release);
        }
        {
            std::scoped_lock lock{_mutex};
            _buffers.emplace(buffer);
        }
        _state->live_buffers.fetch_add(1u, std::memory_order_release);
        BufferCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(buffer);
        info.native_handle = buffer->bytes.data();
        info.element_stride = stride;
        info.total_size_bytes = buffer->bytes.size();
        return info;
    }

    void destroy_buffer(uint64_t handle) noexcept override {
        auto buffer = reinterpret_cast<MockBuffer *>(handle);
        {
            std::scoped_lock lock{_mutex};
            _buffers.erase(buffer);
        }
        delete buffer;
        _state->live_buffers.fetch_sub(1u, std::memory_order_release);
    }

    ResourceCreationInfo create_texture(
        PixelFormat format, uint, uint width, uint height, uint depth, uint,
        void *, bool, bool) noexcept override {
        auto texture = new MockTexture{
            .storage = pixel_format_to_storage(format),
            .size = uint3{width, height, depth}};
        auto size = checked_pixel_storage_size(
            texture->storage, texture->size);
        expect(static_cast<bool>(size));
        texture->bytes.resize(size.size);
        {
            std::scoped_lock lock{_mutex};
            _textures.emplace(texture);
        }
        return ResourceCreationInfo{
            .handle = reinterpret_cast<uint64_t>(texture),
            .native_handle = texture};
    }
    void destroy_texture(uint64_t handle) noexcept override {
        auto texture = reinterpret_cast<MockTexture *>(handle);
        {
            std::scoped_lock lock{_mutex};
            _textures.erase(texture);
        }
        delete texture;
    }
    ResourceCreationInfo create_bindless_array(
        size_t size, BindlessSlotType type) noexcept override {
        auto array = new MockBindless{size, type};
        return ResourceCreationInfo{
            .handle = reinterpret_cast<uint64_t>(array),
            .native_handle = array};
    }
    void destroy_bindless_array(uint64_t handle) noexcept override {
        delete reinterpret_cast<MockBindless *>(handle);
    }

    ResourceCreationInfo create_stream(StreamTag) noexcept override {
        auto stream = new MockStream;
        return ResourceCreationInfo{
            .handle = reinterpret_cast<uint64_t>(stream),
            .native_handle = stream};
    }
    void destroy_stream(uint64_t handle) noexcept override {
        delete reinterpret_cast<MockStream *>(handle);
    }
    void synchronize_stream(uint64_t) noexcept override {}

    void dispatch(uint64_t stream_handle, CommandList &&list) noexcept override {
        auto commands = list.steal_commands();
        auto callbacks = list.steal_callbacks();
        for (auto &&command : commands) {
            switch (command->tag()) {
                case Command::Tag::EBufferUploadCommand: {
                    auto upload = static_cast<BufferUploadCommand *>(command.get());
                    auto buffer = reinterpret_cast<MockBuffer *>(upload->handle());
                    std::memcpy(buffer->bytes.data() + upload->offset(),
                                upload->data(), upload->size());
                    break;
                }
                case Command::Tag::EBufferDownloadCommand: {
                    auto download = static_cast<BufferDownloadCommand *>(command.get());
                    auto buffer = reinterpret_cast<MockBuffer *>(download->handle());
                    std::memcpy(download->data(),
                                buffer->bytes.data() + download->offset(),
                                download->size());
                    break;
                }
                case Command::Tag::EBufferCopyCommand: {
                    auto copy = static_cast<BufferCopyCommand *>(command.get());
                    auto source = reinterpret_cast<MockBuffer *>(copy->src_handle());
                    auto destination = reinterpret_cast<MockBuffer *>(copy->dst_handle());
                    std::memmove(destination->bytes.data() + copy->dst_offset(),
                                 source->bytes.data() + copy->src_offset(),
                                 copy->size());
                    break;
                }
                case Command::Tag::ETextureUploadCommand: {
                    auto upload = static_cast<TextureUploadCommand *>(command.get());
                    auto texture = reinterpret_cast<MockTexture *>(upload->handle());
                    {
                        std::scoped_lock lock{_mutex};
                        expect(_textures.contains(texture));
                    }
                    expect(upload->storage() == texture->storage);
                    expect(all(upload->size() == texture->size));
                    expect(all(upload->offset() == 0u));
                    std::memcpy(texture->bytes.data(), upload->data(),
                                texture->bytes.size());
                    _state->texture_command_count.fetch_add(
                        1u, std::memory_order_relaxed);
                    break;
                }
                case Command::Tag::ETextureDownloadCommand: {
                    auto download = static_cast<TextureDownloadCommand *>(command.get());
                    auto texture = reinterpret_cast<MockTexture *>(download->handle());
                    {
                        std::scoped_lock lock{_mutex};
                        expect(_textures.contains(texture));
                    }
                    expect(download->storage() == texture->storage);
                    expect(all(download->size() == texture->size));
                    expect(all(download->offset() == 0u));
                    std::memcpy(download->data(), texture->bytes.data(),
                                texture->bytes.size());
                    _state->texture_command_count.fetch_add(
                        1u, std::memory_order_relaxed);
                    break;
                }
                case Command::Tag::ETextureCopyCommand: {
                    auto copy = static_cast<TextureCopyCommand *>(command.get());
                    auto source = reinterpret_cast<MockTexture *>(copy->src_handle());
                    auto destination = reinterpret_cast<MockTexture *>(copy->dst_handle());
                    {
                        std::scoped_lock lock{_mutex};
                        expect(_textures.contains(source));
                        expect(_textures.contains(destination));
                    }
                    expect(copy->storage() == source->storage);
                    expect(copy->storage() == destination->storage);
                    expect(all(copy->size() == source->size));
                    std::memcpy(destination->bytes.data(), source->bytes.data(),
                                source->bytes.size());
                    _state->texture_command_count.fetch_add(
                        1u, std::memory_order_relaxed);
                    break;
                }
                case Command::Tag::EBufferToTextureCopyCommand: {
                    auto copy = static_cast<BufferToTextureCopyCommand *>(command.get());
                    auto buffer = reinterpret_cast<MockBuffer *>(copy->buffer());
                    auto texture = reinterpret_cast<MockTexture *>(copy->texture());
                    {
                        std::scoped_lock lock{_mutex};
                        expect(_buffers.contains(buffer));
                        expect(_textures.contains(texture));
                    }
                    expect(copy->storage() == texture->storage);
                    expect(all(copy->size() == texture->size));
                    std::memcpy(texture->bytes.data(),
                                buffer->bytes.data() + copy->buffer_offset(),
                                texture->bytes.size());
                    _state->texture_command_count.fetch_add(
                        1u, std::memory_order_relaxed);
                    break;
                }
                case Command::Tag::ETextureToBufferCopyCommand: {
                    auto copy = static_cast<TextureToBufferCopyCommand *>(command.get());
                    auto buffer = reinterpret_cast<MockBuffer *>(copy->buffer());
                    auto texture = reinterpret_cast<MockTexture *>(copy->texture());
                    {
                        std::scoped_lock lock{_mutex};
                        expect(_buffers.contains(buffer));
                        expect(_textures.contains(texture));
                    }
                    expect(copy->storage() == texture->storage);
                    expect(all(copy->size() == texture->size));
                    std::memcpy(buffer->bytes.data() + copy->buffer_offset(),
                                texture->bytes.data(), texture->bytes.size());
                    _state->texture_command_count.fetch_add(
                        1u, std::memory_order_relaxed);
                    break;
                }
                case Command::Tag::EShaderDispatchCommand: {
                    auto dispatch = static_cast<ShaderDispatchCommand *>(command.get());
                    auto shader = reinterpret_cast<MockShader *>(dispatch->handle());
                    if (dispatch->is_indirect()) {
                        auto indirect = dispatch->indirect_dispatch();
                        auto buffer = reinterpret_cast<MockBuffer *>(
                            indirect.handle);
                        std::scoped_lock lock{_mutex};
                        _state->indirect_dispatch_remapped.store(
                            _buffers.contains(buffer) &&
                                buffer->indirect_dispatch &&
                                indirect.offset == 0u &&
                                indirect.max_dispatch_size == 1u,
                            std::memory_order_release);
                        break;
                    }
                    if (shader->indirect_writer) {
                        expect(dispatch->arguments().size() == 1u);
                        auto argument = dispatch->arguments().front();
                        auto buffer = reinterpret_cast<MockBuffer *>(
                            argument.buffer.handle);
                        std::scoped_lock lock{_mutex};
                        _state->indirect_writer_remapped.store(
                            argument.tag == Argument::Tag::BUFFER &&
                                _buffers.contains(buffer) &&
                                buffer->indirect_dispatch &&
                                argument.buffer.size ==
                                    buffer->indirect_dispatch_capacity,
                            std::memory_order_release);
                        break;
                    }
                    expect(shader->block_size.x == 32u);
                    expect(dispatch->arguments().size() == 2u);
                    auto output_arg = dispatch->arguments()[0u];
                    auto uniform_arg = dispatch->arguments()[1u];
                    expect(output_arg.tag == Argument::Tag::BUFFER);
                    expect(uniform_arg.tag == Argument::Tag::UNIFORM);
                    auto output = reinterpret_cast<MockBuffer *>(output_arg.buffer.handle);
                    uint base{};
                    auto uniform = dispatch->uniform(uniform_arg.uniform);
                    std::memcpy(&base, uniform.data(), sizeof(base));
                    auto count = dispatch->dispatch_size().x;
                    auto values = reinterpret_cast<uint *>(
                        output->bytes.data() + output_arg.buffer.offset);
                    for (auto i = 0u; i < count; i++) {
                        values[i] = base + i;
                    }
                    break;
                }
                case Command::Tag::EBindlessArrayUpdateCommand: {
                    auto update = static_cast<BindlessArrayUpdateCommand *>(command.get());
                    auto array = reinterpret_cast<MockBindless *>(update->handle());
                    expect(array->type == BindlessSlotType::MULTIPLE);
                    update->visit_modifications([&](auto const &modifications) {
                        expect(modifications.size() == 1u);
                        using Modification = typename std::remove_cvref_t<decltype(modifications)>::value_type;
                        if constexpr (std::is_same_v<Modification,
                                                     BindlessArrayUpdateCommand::Modification>) {
                            auto buffer = reinterpret_cast<MockBuffer *>(
                                modifications.front().buffer.handle);
                            std::scoped_lock lock{_mutex};
                            _state->bindless_update_remapped.store(
                                _buffers.contains(buffer), std::memory_order_release);
                        } else {
                            expect(false) << "unexpected bindless update variant";
                        }
                    });
                    break;
                }
                case Command::Tag::EMeshBuildCommand: {
                    auto build = static_cast<MeshBuildCommand *>(command.get());
                    auto mesh = reinterpret_cast<MockPrimitive *>(build->handle());
                    auto vertices = reinterpret_cast<MockBuffer *>(build->vertex_buffer());
                    auto triangles = reinterpret_cast<MockBuffer *>(build->triangle_buffer());
                    std::scoped_lock lock{_mutex};
                    _state->mesh_build_remapped.store(
                        _primitives.contains(mesh) &&
                            mesh->tag == Resource::Tag::MESH &&
                            _buffers.contains(vertices) &&
                            _buffers.contains(triangles),
                        std::memory_order_release);
                    break;
                }
                case Command::Tag::ECurveBuildCommand: {
                    auto build = static_cast<CurveBuildCommand *>(command.get());
                    auto curve = reinterpret_cast<MockPrimitive *>(build->handle());
                    auto control_points = reinterpret_cast<MockBuffer *>(build->cp_buffer());
                    auto segments = reinterpret_cast<MockBuffer *>(build->seg_buffer());
                    std::scoped_lock lock{_mutex};
                    _state->curve_build_remapped.store(
                        _primitives.contains(curve) &&
                            curve->tag == Resource::Tag::CURVE &&
                            _buffers.contains(control_points) &&
                            _buffers.contains(segments),
                        std::memory_order_release);
                    break;
                }
                case Command::Tag::EProceduralPrimitiveBuildCommand: {
                    auto build = static_cast<ProceduralPrimitiveBuildCommand *>(command.get());
                    auto primitive = reinterpret_cast<MockPrimitive *>(build->handle());
                    auto aabbs = reinterpret_cast<MockBuffer *>(build->aabb_buffer());
                    std::scoped_lock lock{_mutex};
                    _state->procedural_build_remapped.store(
                        _primitives.contains(primitive) &&
                            primitive->tag == Resource::Tag::PROCEDURAL_PRIMITIVE &&
                            _buffers.contains(aabbs),
                        std::memory_order_release);
                    break;
                }
                case Command::Tag::EMotionInstanceBuildCommand: {
                    auto build = static_cast<MotionInstanceBuildCommand *>(command.get());
                    auto instance = reinterpret_cast<MockPrimitive *>(build->handle());
                    auto child = reinterpret_cast<MockPrimitive *>(build->child());
                    std::scoped_lock lock{_mutex};
                    _state->motion_build_remapped.store(
                        _primitives.contains(instance) &&
                            instance->tag == Resource::Tag::MOTION_INSTANCE &&
                            build->keyframes().size() == instance->keyframe_count &&
                            _primitives.contains(child) &&
                            child->tag == Resource::Tag::MESH,
                        std::memory_order_release);
                    break;
                }
                case Command::Tag::EAccelBuildCommand: {
                    auto build = static_cast<AccelBuildCommand *>(command.get());
                    auto accel = reinterpret_cast<MockAccel *>(build->handle());
                    auto valid = false;
                    {
                        std::scoped_lock lock{_mutex};
                        valid = _accels.contains(accel) &&
                                build->modifications().size() == 4u;
                        for (auto &&modification : build->modifications()) {
                            if ((modification.flags &
                                 AccelBuildCommand::Modification::flag_primitive) != 0u) {
                                valid = valid && _primitives.contains(
                                                     reinterpret_cast<MockPrimitive *>(
                                                         modification.primitive));
                            }
                        }
                    }
                    _state->accel_build_remapped.store(
                        valid, std::memory_order_release);
                    break;
                }
                default:
                    expect(false) << "unexpected mock command";
                    break;
            }
        }
        auto stream = reinterpret_cast<MockStream *>(stream_handle);
        if (stream->log_callback) { stream->log_callback("mock-dispatch"); }
        for (auto &callback : callbacks) { callback(); }
    }

    void set_stream_log_callback(
        uint64_t stream_handle,
        const StreamLogCallback &callback) noexcept override {
        reinterpret_cast<MockStream *>(stream_handle)->log_callback = callback;
    }

    SwapchainCreationInfo create_swapchain(
        const SwapchainOption &, uint64_t) noexcept override {
        SwapchainCreationInfo info{};
        info.invalidate();
        return info;
    }
    void destroy_swapchain(uint64_t) noexcept override {}
    void present_display_in_stream(uint64_t, uint64_t, uint64_t) noexcept override {}

    ShaderCreationInfo create_shader(
        const ShaderOption &, Function kernel) noexcept override {
        auto shader = new MockShader;
        shader->block_size = kernel.block_size();
        shader->usages.reserve(kernel.arguments().size());
        for (auto argument : kernel.arguments()) {
            shader->usages.emplace_back(kernel.variable_usage(argument.uid()));
        }
        shader->bindings.assign(
            kernel.bound_arguments().begin(), kernel.bound_arguments().end());
        shader->indirect_writer = std::any_of(
            kernel.unbound_arguments().begin(),
            kernel.unbound_arguments().end(),
            [](auto argument) noexcept {
                return argument.type()->is_custom() &&
                       argument.type()->description() ==
                           ast_json_indirect_dispatch_buffer_type_name;
            });
        _state->ast_received.store(true, std::memory_order_release);
        _state->custom_callable_received.store(
            !kernel.custom_callables().empty(), std::memory_order_release);
        if (!shader->bindings.empty()) {
            auto binding = get<Function::BufferBinding>(shader->bindings.front());
            auto native_buffer = reinterpret_cast<MockBuffer *>(binding.handle);
            std::scoped_lock lock{_mutex};
            _state->ast_binding_remapped.store(
                _buffers.contains(native_buffer), std::memory_order_release);
        }
        ShaderCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(shader);
        info.native_handle = shader;
        info.block_size = kernel.block_size();
        return info;
    }
    ShaderCreationInfo load_shader(
        string_view, span<const Type *const>) noexcept override {
        return ShaderCreationInfo::make_invalid();
    }
    Usage shader_argument_usage(uint64_t handle, size_t index) noexcept override {
        auto shader = reinterpret_cast<MockShader *>(handle);
        return shader->usages.at(index);
    }
    void destroy_shader(uint64_t handle) noexcept override {
        delete reinterpret_cast<MockShader *>(handle);
    }

    ResourceCreationInfo create_event() noexcept override {
        auto event = new MockEvent;
        return ResourceCreationInfo{
            .handle = reinterpret_cast<uint64_t>(event),
            .native_handle = event};
    }
    void destroy_event(uint64_t handle) noexcept override {
        delete reinterpret_cast<MockEvent *>(handle);
    }
    void signal_event(uint64_t event, uint64_t, uint64_t value) noexcept override {
        reinterpret_cast<MockEvent *>(event)->value.store(value);
    }
    void wait_event(uint64_t event, uint64_t, uint64_t value) noexcept override {
        while (reinterpret_cast<MockEvent *>(event)->value.load() < value) {}
    }
    bool is_event_completed(uint64_t event, uint64_t value) const noexcept override {
        return reinterpret_cast<MockEvent *>(event)->value.load() >= value;
    }
    void synchronize_event(uint64_t event, uint64_t value) noexcept override {
        while (!is_event_completed(event, value)) {}
    }

    [[nodiscard]] ResourceCreationInfo _create_primitive(
        Resource::Tag tag, size_t keyframe_count = 0u) noexcept {
        auto primitive = new MockPrimitive{tag, keyframe_count};
        {
            std::scoped_lock lock{_mutex};
            _primitives.emplace(primitive);
        }
        return ResourceCreationInfo{
            .handle = reinterpret_cast<uint64_t>(primitive),
            .native_handle = primitive};
    }
    void _destroy_primitive(uint64_t handle) noexcept {
        auto primitive = reinterpret_cast<MockPrimitive *>(handle);
        {
            std::scoped_lock lock{_mutex};
            _primitives.erase(primitive);
        }
        delete primitive;
    }
    ResourceCreationInfo create_mesh(const AccelOption &) noexcept override {
        return _create_primitive(Resource::Tag::MESH);
    }
    void destroy_mesh(uint64_t handle) noexcept override {
        _destroy_primitive(handle);
    }
    ResourceCreationInfo create_procedural_primitive(
        const AccelOption &) noexcept override {
        return _create_primitive(Resource::Tag::PROCEDURAL_PRIMITIVE);
    }
    void destroy_procedural_primitive(uint64_t handle) noexcept override {
        _destroy_primitive(handle);
    }
    ResourceCreationInfo create_curve(const AccelOption &) noexcept override {
        return _create_primitive(Resource::Tag::CURVE);
    }
    void destroy_curve(uint64_t handle) noexcept override {
        _destroy_primitive(handle);
    }
    ResourceCreationInfo create_motion_instance(
        const AccelMotionOption &option) noexcept override {
        return _create_primitive(
            Resource::Tag::MOTION_INSTANCE, option.keyframe_count);
    }
    void destroy_motion_instance(uint64_t handle) noexcept override {
        _destroy_primitive(handle);
    }
    ResourceCreationInfo create_accel(const AccelOption &) noexcept override {
        auto accel = new MockAccel;
        {
            std::scoped_lock lock{_mutex};
            _accels.emplace(accel);
        }
        return ResourceCreationInfo{
            .handle = reinterpret_cast<uint64_t>(accel),
            .native_handle = accel};
    }
    void destroy_accel(uint64_t handle) noexcept override {
        auto accel = reinterpret_cast<MockAccel *>(handle);
        {
            std::scoped_lock lock{_mutex};
            _accels.erase(accel);
        }
        delete accel;
    }

    string query(string_view property) noexcept override {
        return property == "mock.property" ? "mock.value" : "";
    }
    DeviceExtension *extension(string_view) noexcept override { return nullptr; }
    void set_name(Resource::Tag, uint64_t, string_view) noexcept override {}
};

void test_remote_backend_e2e(const char *program_path) {
    auto state = make_shared<MockState>();
    auto native = make_shared<MockDevice>(Context{program_path}, state);
    ServerOptions server_options;
    server_options.port = 0u;
    server_options.token = "test-token";
    server_options.max_blob_cache_bytes = 1u * 1024u * 1024u;
    server_options.max_blob_entry_size = 1u * 1024u * 1024u;
    server_options.blob_cache_min_size = 1u;
    Server server{native, std::move(server_options)};
    std::thread server_thread{[&] { server.run(); }};

    {
        Context context{program_path};
        DeviceConfig config;
        config.extension = make_unique<RemoteDeviceConfigExt>(
            "127.0.0.1", server.port(), "test-token",
            2'000u, 5'000u, 16u * 1024u * 1024u,
            true, 1u);
        auto device = context.create_device("remote", &config, false);
        expect(device.compute_warp_size() == 32u);
        expect(device.compute_max_shared_memory_size() == 32768u);
        expect(device.memory_granularity() == 256u);
        expect(device.query("remote.native_backend") == "mock");
        expect(device.query("remote.blob_cache.enabled") == "true");
        expect(device.query("mock.property") == "mock.value");

        constexpr auto count = 16u;
        auto source = device.create_buffer<uint>(count);
        auto destination = device.create_buffer<uint>(count);
        auto stream = device.create_stream();
        vector<uint> upload(count);
        vector<uint> download(count, 0u);
        for (auto i = 0u; i < count; i++) { upload[i] = i * 3u; }
        bool callback_called = false;
        bool log_called = false;
        stream.set_log_callback([&](string_view message) {
            expect(message == "mock-dispatch");
            log_called = true;
        });
        stream << source.copy_from(span{upload})
               << source.copy_to(destination.view())
               << destination.copy_to(span{download})
               << [&] { callback_called = true; }
               << synchronize();
        expect(callback_called);
        expect(log_called);
        expect(download == upload);

        auto cache_hits = parse_u64(
            device.query("remote.blob_cache.hits"));
        auto uploaded_bytes = parse_u64(
            device.query("remote.blob_cache.uploaded_bytes"));
        std::fill(download.begin(), download.end(), 0u);
        stream << source.copy_from(span{upload})
               << source.copy_to(destination.view())
               << destination.copy_to(span{download})
               << synchronize();
        expect(download == upload);
        expect(parse_u64(device.query("remote.blob_cache.hits")) >
               cache_hits);
        expect(parse_u64(
                   device.query("remote.blob_cache.uploaded_bytes")) ==
               uploaded_bytes);

        constexpr auto image_size = 4u;
        vector<float> image_upload(image_size * image_size);
        vector<float> image_download(image_upload.size(), 0.0f);
        for (auto i = 0u; i < image_upload.size(); i++) {
            image_upload[i] = static_cast<float>(i) + 0.25f;
        }
        auto source_image = device.create_image<float>(
            PixelStorage::FLOAT1, image_size, image_size);
        auto destination_image = device.create_image<float>(
            PixelStorage::FLOAT1, image_size, image_size);
        stream << source_image.copy_from(span{image_upload})
               << source_image.copy_to(destination_image)
               << destination_image.copy_to(span{image_download})
               << synchronize();
        expect(image_download == image_upload);

        auto image_buffer = device.create_buffer<float>(image_upload.size());
        auto image_readback = device.create_buffer<float>(image_upload.size());
        std::fill(image_download.begin(), image_download.end(), 0.0f);
        stream << image_buffer.copy_from(span{image_upload})
               << source_image.copy_from(image_buffer)
               << source_image.copy_to(image_readback)
               << image_readback.copy_to(span{image_download})
               << synchronize();
        expect(image_download == image_upload);
        expect(state->texture_command_count.load(std::memory_order_acquire) == 5u);

        auto bindless = device.create_bindless_array(count);
        bindless.emplace_on_update(0u, source);
        stream << bindless.update() << synchronize();
        expect(state->bindless_update_remapped.load(std::memory_order_acquire));

        Callable twice = [](UInt value) noexcept { return value * 2u; };
        Kernel1D kernel = [&twice](BufferUInt output, UInt base) noexcept {
            set_block_size(32u);
            auto index = dispatch_id().x;
            output.write(index, twice(base) + index);
        };
        auto shader = device.compile(kernel);
        vector<uint> shader_output(count, 0u);
        callback_called = false;
        stream << shader(destination, 41u).dispatch(count)
               << destination.copy_to(span{shader_output})
               << [&] { callback_called = true; }
               << synchronize();
        expect(callback_called);
        for (auto i = 0u; i < count; i++) {
            expect(shader_output[i] == 41u + i);
        }
        expect(state->ast_received.load(std::memory_order_acquire));
        expect(state->custom_callable_received.load(std::memory_order_acquire));

        auto indirect = device.create_indirect_dispatch_buffer(4u);
        Kernel1D write_indirect = [](IndirectDispatchBufferVar commands) noexcept {
            set_block_size(32u);
            commands.set_kernel(
                0u, make_uint3(32u, 1u, 1u),
                make_uint3(4u, 1u, 1u));
            commands.set_dispatch_count(1u);
        };
        auto indirect_writer = device.compile(write_indirect);
        stream << indirect_writer(indirect).dispatch(1u)
               << shader(destination, 41u).dispatch(indirect, 0u, 1u)
               << synchronize();
        expect(state->indirect_buffer_created.load(
            std::memory_order_acquire));
        expect(state->indirect_writer_remapped.load(
            std::memory_order_acquire));
        expect(state->indirect_dispatch_remapped.load(
            std::memory_order_acquire));

        auto bound_shader = device.compile<1>([&] {
            auto index = dispatch_id().x;
            destination->write(index, index);
        });
        expect(static_cast<bool>(bound_shader));
        expect(state->ast_binding_remapped.load(std::memory_order_acquire));

        auto event = device.create_timeline_event();
        stream << event.signal(7u);
        expect(event.is_completed(7u));
        event.synchronize(7u);

        auto vertices = device.create_buffer<float3>(3u);
        auto triangles = device.create_buffer<Triangle>(1u);
        auto control_points = device.create_buffer<float4>(2u);
        auto segments = device.create_buffer<uint>(1u);
        auto aabbs = device.create_buffer<AABB>(1u);
        auto mesh = device.create_mesh(vertices, triangles);
        auto curve = device.create_curve(
            CurveBasis::PIECEWISE_LINEAR, control_points, segments);
        auto procedural = device.create_procedural_primitive(aabbs);
        AccelMotionOption motion_option;
        motion_option.keyframe_count = 2u;
        auto motion = device.create_motion_instance(mesh, motion_option);
        motion.set_keyframe(0u, make_float4x4(1.0f));
        motion.set_keyframe(1u, make_float4x4(1.0f));
        auto accel = device.create_accel();
        accel.emplace_back(mesh);
        accel.emplace_back(curve);
        accel.emplace_back(procedural);
        accel.emplace_back(motion);
        stream << mesh.build()
               << curve.build()
               << procedural.build()
               << motion.build()
               << accel.build()
               << synchronize();
        expect(state->mesh_build_remapped.load(std::memory_order_acquire));
        expect(state->curve_build_remapped.load(std::memory_order_acquire));
        expect(state->procedural_build_remapped.load(std::memory_order_acquire));
        expect(state->motion_build_remapped.load(std::memory_order_acquire));
        expect(state->accel_build_remapped.load(std::memory_order_acquire));
    }

    server.stop();
    server_thread.join();
}

void test_remote_backend_inline_fallback(const char *program_path) {
    auto state = make_shared<MockState>();
    auto native = make_shared<MockDevice>(Context{program_path}, state);
    ServerOptions server_options;
    server_options.port = 0u;
    server_options.token = "fallback-token";
    server_options.max_blob_cache_bytes = 0u;
    server_options.protocol_limits.max_frame_payload = 1024u;
    server_options.protocol_limits.max_string_size = 256u;
    server_options.protocol_limits.max_array_size = 64u;
    Server server{native, std::move(server_options)};
    std::thread server_thread{[&] { server.run(); }};

    {
        Context context{program_path};
        DeviceConfig config;
        config.extension = make_unique<RemoteDeviceConfigExt>(
            "127.0.0.1", server.port(), "fallback-token",
            2'000u, 5'000u, 1u * 1024u * 1024u,
            true, 1u);
        auto device = context.create_device("remote", &config, false);
        expect(device.query("remote.blob_cache.enabled") == "false");
        expect(device.query("remote.protocol.max_frame_payload") == "1024");
        expect(device.query("remote.protocol.max_string_size") == "256");
        expect(device.query("remote.protocol.max_array_size") == "64");
        constexpr auto count = 8u;
        vector<uint> upload(count);
        vector<uint> download(count, 0u);
        for (auto i = 0u; i < count; i++) { upload[i] = i + 17u; }
        auto buffer = device.create_buffer<uint>(count);
        auto stream = device.create_stream();
        stream << buffer.copy_from(span{upload})
               << buffer.copy_to(span{download})
               << synchronize();
        expect(download == upload);
    }

    server.stop();
    server_thread.join();
}

void test_remote_blob_protocol_rejects_bad_bodies(
    const char *program_path) {
    auto state = make_shared<MockState>();
    auto native = make_shared<MockDevice>(Context{program_path}, state);
    ServerOptions server_options;
    server_options.port = 0u;
    server_options.token = "blob-protocol-token";
    server_options.max_blob_cache_bytes = 1024u;
    server_options.max_blob_entry_size = 512u;
    server_options.blob_cache_min_size = 1u;
    Server server{native, std::move(server_options)};
    std::thread server_thread{[&] { server.run(); }};

    Connection connection;
    string error;
    expect(connection.connect(
        "127.0.0.1", server.port(), 2s, error));
    Writer hello;
    hello.write_u32(0x01020304u);
    hello.write_u8(sizeof(void *));
    hello.write_u8(1u);
    hello.write_u16(0u);
    hello.write_string("blob-protocol-token");
    expect(static_cast<bool>(connection.request(
        MessageKind::HELLO, hello.bytes(), 2s)));
    expect(static_cast<bool>(connection.request(
        MessageKind::BLOB_CACHE_INFO, {}, 2s)));

    vector<std::byte> good(4u, std::byte{0x11u});
    vector<std::byte> tampered(4u, std::byte{0x22u});
    auto key = compute_blob_key(good);
    Writer duplicate_prepare;
    duplicate_prepare.write_u64(1u);
    duplicate_prepare.write_u64(2u);
    write_blob_key(duplicate_prepare, key);
    write_blob_key(duplicate_prepare, key);
    auto duplicate = connection.request(
        MessageKind::PREPARE_BLOBS,
        duplicate_prepare.bytes(), 2s);
    expect(!static_cast<bool>(duplicate));
    expect(duplicate.status == Status::INVALID_REQUEST);

    Writer prepare;
    prepare.write_u64(2u);
    prepare.write_u64(1u);
    write_blob_key(prepare, key);
    auto prepared = connection.request(
        MessageKind::PREPARE_BLOBS, prepare.bytes(), 2s);
    expect(static_cast<bool>(prepared));
    Reader prepared_reader{prepared.body};
    expect(prepared_reader.read_u64() == 2u);
    expect(prepared_reader.read_u64() == 1u);
    expect(prepared_reader.read_u32() == 0u);
    expect(prepared_reader.finish());

    auto make_upload = [&](span<const std::byte> body) {
        Writer upload;
        upload.write_u64(2u);
        upload.write_u64(1u);
        upload.write_u32(0u);
        write_blob_key(upload, key);
        upload.write_blob(body);
        return upload;
    };
    auto bad_upload = make_upload(tampered);
    auto bad_response = connection.request(
        MessageKind::UPLOAD_BLOBS,
        bad_upload.bytes(), 2s);
    expect(!static_cast<bool>(bad_response));
    expect(bad_response.status == Status::INVALID_REQUEST);

    auto good_upload = make_upload(good);
    auto good_response = connection.request(
        MessageKind::UPLOAD_BLOBS,
        good_upload.bytes(), 2s);
    expect(static_cast<bool>(good_response));
    Reader good_reader{good_response.body};
    expect(good_reader.read_u64() == 2u);
    expect(good_reader.finish());

    expect(static_cast<bool>(connection.request(
        MessageKind::GOODBYE, {}, 2s)));
    connection.close();
    server.stop();
    server_thread.join();
}

void test_remote_server_survives_abrupt_disconnect(
    const char *program_path) {
    auto state = make_shared<MockState>();
    auto native = make_shared<MockDevice>(
        Context{program_path}, state);
    ServerOptions server_options;
    server_options.port = 0u;
    server_options.token = "reconnect-token";
    Server server{native, std::move(server_options)};
    std::thread server_thread{[&] { server.run(); }};

    // Protocol 1.0 peers omit device-selection fields. The service accepts
    // that payload and replies using the peer's minor version.
    {
        asio::io_context io;
        asio::ip::tcp::socket socket{io};
        socket.connect({asio::ip::make_address("127.0.0.1"), server.port()});
        Writer hello;
        hello.write_u32(0x01020304u);
        hello.write_u8(sizeof(void *));
        hello.write_u8(1u);
        hello.write_u16(0u);
        hello.write_string("reconnect-token");
        auto header = encode_frame_header(FrameHeader{
            .kind = MessageKind::HELLO,
            .request_id = 1u,
            .payload_size = hello.bytes().size(),
            .payload_checksum = payload_checksum(hello.bytes()),
            .wire_major = protocol_major,
            .wire_minor = 0u});
        std::array<asio::const_buffer, 2u> buffers{
            asio::buffer(header), asio::buffer(hello.bytes())};
        asio::write(socket, buffers);
        std::array<std::byte, frame_header_size> response_header_bytes{};
        asio::read(socket, asio::buffer(response_header_bytes));
        FrameHeader response_header;
        string response_error;
        expect(decode_frame_header(
            response_header_bytes, response_header, response_error));
        expect(response_header.wire_minor == 0u);
        vector<std::byte> response_payload(
            static_cast<size_t>(response_header.payload_size));
        asio::read(socket, asio::buffer(response_payload));
        expect(payload_checksum(response_payload) ==
               response_header.payload_checksum);
        ResponseView response;
        expect(decode_response_payload(
            response_payload, response, response_error));
        expect(response.status == Status::OK);
        asio::error_code ignored;
        socket.close(ignored);
    }

    auto connect_and_hello = [&](Connection &connection) {
        string error;
        expect(connection.connect(
            "127.0.0.1", server.port(), 2s, error));
        Writer hello;
        hello.write_u32(0x01020304u);
        hello.write_u8(sizeof(void *));
        hello.write_u8(1u);
        hello.write_u16(0u);
        hello.write_string("reconnect-token");
        return connection.request(
            MessageKind::HELLO, hello.bytes(), 2s);
    };

    Connection first;
    expect(static_cast<bool>(connect_and_hello(first)));
    Writer create_buffer;
    create_buffer.write_u8(static_cast<uint8_t>(BufferKind::BYTE));
    create_buffer.write_u64(4096u);
    expect(static_cast<bool>(first.request(
        MessageKind::CREATE_BUFFER, create_buffer.bytes(), 2s)));
    expect(state->live_buffers.load(std::memory_order_acquire) == 1u);
    // Simulate a process/network loss: no GOODBYE and no explicit resource
    // destruction. The session must reclaim its native resources.
    first.close();
    auto deadline = std::chrono::steady_clock::now() + 2s;
    while (state->live_buffers.load(std::memory_order_acquire) != 0u &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    expect(state->live_buffers.load(std::memory_order_acquire) == 0u);

    Connection second;
    expect(static_cast<bool>(connect_and_hello(second)));
    expect(static_cast<bool>(second.request(
        MessageKind::GOODBYE, {}, 2s)));
    second.close();

    server.stop();
    server_thread.join();
}

void test_remote_shader_usage_bounds(
    const char *program_path) {
    auto state = make_shared<MockState>();
    auto native = make_shared<MockDevice>(
        Context{program_path}, state);
    ServerOptions server_options;
    server_options.port = 0u;
    server_options.token = "shader-usage-token";
    Server server{native, std::move(server_options)};
    std::thread server_thread{[&] { server.run(); }};

    Connection connection;
    string error;
    expect(connection.connect(
        "127.0.0.1", server.port(), 2s, error));
    Writer hello;
    hello.write_u32(0x01020304u);
    hello.write_u8(sizeof(void *));
    hello.write_u8(1u);
    hello.write_u16(0u);
    hello.write_string("shader-usage-token");
    expect(static_cast<bool>(connection.request(
        MessageKind::HELLO, hello.bytes(), 2s)));

    auto kernel = luisa::compute::detail::FunctionBuilder::define_kernel([] {
        auto builder = luisa::compute::detail::FunctionBuilder::current();
        builder->set_block_size(uint3{1u, 1u, 1u});
        static_cast<void>(builder->buffer(
            Type::buffer(Type::of<uint>())));
    });
    auto ast = try_to_json(Function{kernel.get()});
    expect(static_cast<bool>(ast)) << ast.error;
    Writer create_shader;
    write_test_shader_option(create_shader, ShaderOption{});
    create_shader.write_blob({reinterpret_cast<const std::byte *>(ast.json.data()),
                              ast.json.size()});
    auto created = connection.request(
        MessageKind::CREATE_SHADER, create_shader.bytes(), 2s);
    expect(static_cast<bool>(created)) << created.message;
    Reader created_reader{created.body};
    auto shader = created_reader.read_u64();
    static_cast<void>(created_reader.read_u32());
    static_cast<void>(created_reader.read_u32());
    static_cast<void>(created_reader.read_u32());
    expect(created_reader.finish());
    expect(shader != invalid_resource_handle);

    Writer bad_usage;
    bad_usage.write_u64(shader);
    bad_usage.write_u64(1u);
    auto rejected = connection.request(
        MessageKind::SHADER_ARGUMENT_USAGE,
        bad_usage.bytes(), 2s);
    expect(!static_cast<bool>(rejected));
    expect(rejected.status == Status::INVALID_REQUEST);

    Writer query;
    query.write_string("mock.property");
    auto queried = connection.request(
        MessageKind::QUERY, query.bytes(), 2s);
    expect(static_cast<bool>(queried));
    Reader queried_reader{queried.body};
    expect(queried_reader.read_string() == "mock.value");
    expect(queried_reader.finish());

    Writer destroy_shader;
    destroy_shader.write_u64(shader);
    expect(static_cast<bool>(connection.request(
        MessageKind::DESTROY_SHADER,
        destroy_shader.bytes(), 2s)));
    expect(static_cast<bool>(connection.request(
        MessageKind::GOODBYE, {}, 2s)));
    connection.close();
    server.stop();
    server_thread.join();
}

void test_remote_device_disconnect_cleanup(
    const char *program_path) {
    auto state = make_shared<MockState>();
    auto native = make_shared<MockDevice>(
        Context{program_path}, state);
    ServerOptions server_options;
    server_options.port = 0u;
    server_options.token = "device-disconnect-token";
    Server server{native, std::move(server_options)};
    std::thread server_thread{[&] { server.run(); }};

    {
        Context context{program_path};
        DeviceConfig config;
        config.extension = make_unique<RemoteDeviceConfigExt>(
            "127.0.0.1", server.port(), "device-disconnect-token",
            2'000u, 5'000u, 1u * 1024u * 1024u,
            false, 1u);
        auto device = context.create_device("remote", &config, false);
        auto buffer = device.create_buffer<uint>(16u);
        auto stream = device.create_stream();
        expect(device.query("remote.connected") == "true");
        expect(static_cast<bool>(buffer));
        expect(static_cast<bool>(stream));

        server.stop();
        server_thread.join();
        auto deadline = std::chrono::steady_clock::now() + 2s;
        while (device.query("remote.connected") == "true" &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        expect(device.query("remote.connected") == "false");
    }
    expect(state->live_buffers.load(std::memory_order_acquire) == 0u);
}

void test_remote_service_device_selection(
    const char *program_path) {
    auto state = make_shared<MockState>();
    std::mutex requests_mutex;
    vector<DeviceRequest> requests;
    DeviceFactory factory =
        [program_path, state, &requests_mutex, &requests](
            const DeviceRequest &request,
            string &error) -> shared_ptr<DeviceInterface> {
        if (request.backend != "mock_a" &&
            request.backend != "mock_b") {
            error = "backend is not allowlisted by the test service";
            return nullptr;
        }
        {
            std::scoped_lock lock{requests_mutex};
            requests.emplace_back(request);
        }
        return make_shared<MockDevice>(
            Context{program_path}, state, request.backend);
    };
    ServerOptions server_options;
    server_options.port = 0u;
    server_options.token = "service-token";
    Server server{std::move(factory), std::move(server_options)};
    std::thread server_thread{[&] { server.run(); }};

    // A rejected device selection must only close that session.
    Connection rejected;
    string connection_error;
    expect(rejected.connect(
        "127.0.0.1", server.port(), 2s, connection_error));
    Writer rejected_hello;
    rejected_hello.write_u32(0x01020304u);
    rejected_hello.write_u8(sizeof(void *));
    rejected_hello.write_u8(1u);
    rejected_hello.write_u16(0u);
    rejected_hello.write_string("service-token");
    rejected_hello.write_string("forbidden");
    rejected_hello.write_u64(std::numeric_limits<size_t>::max());
    rejected_hello.write_bool(false);
    auto rejected_response = rejected.request(
        MessageKind::HELLO, rejected_hello.bytes(), 2s);
    expect(!static_cast<bool>(rejected_response));
    expect(rejected_response.status == Status::UNSUPPORTED);
    rejected.close();

    {
        Context context{program_path};
        DeviceConfig config_a;
        config_a.extension = make_unique<RemoteDeviceConfigExt>(
            "127.0.0.1", server.port(), "service-token",
            2'000u, 5'000u, 1u * 1024u * 1024u,
            false, 1u, string{}, "mock_a", 0u, false);
        auto device_a = context.create_device("remote", &config_a, false);
        expect(device_a.query("remote.native_backend") == "mock_a");
        expect(device_a.query("remote.device_selection") == "true");

        DeviceConfig config_b;
        config_b.extension = make_unique<RemoteDeviceConfigExt>(
            "127.0.0.1", server.port(), "service-token",
            2'000u, 5'000u, 1u * 1024u * 1024u,
            false, 1u, string{}, "mock_b", 1u, true);
        auto device_b = context.create_device("remote", &config_b, false);
        expect(device_b.query("remote.native_backend") == "mock_b");
        expect(device_b.query("remote.device_selection") == "true");

        auto buffer_a = device_a.create_buffer<uint>(4u);
        auto buffer_b = device_b.create_buffer<uint>(4u);
        expect(static_cast<bool>(buffer_a));
        expect(static_cast<bool>(buffer_b));
        {
            std::scoped_lock lock{requests_mutex};
            expect(requests.size() == 2u);
            if (requests.size() == 2u) {
                expect(requests[0].backend == "mock_a");
                expect(requests[0].device_index == 0u);
                expect(!requests[0].enable_validation);
                expect(requests[1].backend == "mock_b");
                expect(requests[1].device_index == 1u);
                expect(requests[1].enable_validation);
            }
        }
    }

    server.stop();
    server_thread.join();
}

void test_remote_service_survives_device_factory_exception(
    const char *program_path) {
    auto state = make_shared<MockState>();
    DeviceFactory factory =
        [program_path, state](
            const DeviceRequest &request,
            string &) -> shared_ptr<DeviceInterface> {
        if (request.backend == "throw") {
            throw std::runtime_error{"test factory failure"};
        }
        return make_shared<MockDevice>(
            Context{program_path}, state, request.backend);
    };
    ServerOptions server_options;
    server_options.port = 0u;
    server_options.token = "factory-exception-token";
    Server server{std::move(factory), std::move(server_options)};
    std::thread server_thread{[&] { server.run(); }};

    Connection rejected;
    string connection_error;
    expect(rejected.connect(
        "127.0.0.1", server.port(), 2s, connection_error));
    Writer hello;
    hello.write_u32(0x01020304u);
    hello.write_u8(sizeof(void *));
    hello.write_u8(1u);
    hello.write_u16(0u);
    hello.write_string("factory-exception-token");
    hello.write_string("throw");
    hello.write_u64(std::numeric_limits<size_t>::max());
    hello.write_bool(false);
    auto response = rejected.request(
        MessageKind::HELLO, hello.bytes(), 2s);
    expect(!static_cast<bool>(response));
    expect(response.status == Status::BACKEND_ERROR);
    expect(response.message.find("test factory failure") != string::npos);
    rejected.close();

    // A failure in one session must not prevent the service from accepting
    // and provisioning another independent session.
    Context context{program_path};
    DeviceConfig config;
    config.extension = make_unique<RemoteDeviceConfigExt>(
        "127.0.0.1", server.port(), "factory-exception-token",
        2'000u, 5'000u, 1u * 1024u * 1024u,
        false, 1u, string{}, "mock", 0u, false);
    auto device = context.create_device("remote", &config, false);
    expect(device.query("remote.connected") == "true");
    expect(device.query("remote.native_backend") == "mock");

    server.stop();
    server_thread.join();
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    "remote_backend_e2e"_test = [&] {
        test_remote_backend_e2e(argv[0]);
    };
    "remote_backend_inline_fallback"_test = [&] {
        test_remote_backend_inline_fallback(argv[0]);
    };
    "remote_blob_protocol_rejects_bad_bodies"_test = [&] {
        test_remote_blob_protocol_rejects_bad_bodies(argv[0]);
    };
    "remote_server_survives_abrupt_disconnect"_test = [&] {
        test_remote_server_survives_abrupt_disconnect(argv[0]);
    };
    "remote_shader_usage_bounds"_test = [&] {
        test_remote_shader_usage_bounds(argv[0]);
    };
    "remote_device_disconnect_cleanup"_test = [&] {
        test_remote_device_disconnect_cleanup(argv[0]);
    };
    "remote_service_device_selection"_test = [&] {
        test_remote_service_device_selection(argv[0]);
    };
    "remote_service_survives_device_factory_exception"_test = [&] {
        test_remote_service_survives_device_factory_exception(argv[0]);
    };
}
