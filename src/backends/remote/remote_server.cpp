#include "remote_server.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <asio.hpp>

#include <luisa/ast/ast2json.h>
#include <luisa/ast/type.h>
#include <luisa/core/logging.h>
#include <luisa/core/mathematics.h>
#include <luisa/core/stl/format.h>

#include "remote_command_codec.h"

namespace luisa::compute::remote {

namespace {

using Tcp = asio::ip::tcp;
constexpr luisa::string_view indirect_dispatch_element_type_name =
    "LC_IndirectKernelDispatch";

struct Reply {
    Status status{Status::OK};
    luisa::string message;
    luisa::vector<std::byte> body;
};

[[nodiscard]] Reply invalid(luisa::string message) noexcept {
    return Reply{.status = Status::INVALID_REQUEST,
                 .message = std::move(message)};
}

[[nodiscard]] Reply unsupported(luisa::string message) noexcept {
    return Reply{.status = Status::UNSUPPORTED,
                 .message = std::move(message)};
}

[[nodiscard]] Reply not_found(luisa::string message) noexcept {
    return Reply{.status = Status::NOT_FOUND,
                 .message = std::move(message)};
}

[[nodiscard]] bool secure_equal(
    luisa::string_view lhs, luisa::string_view rhs) noexcept {
    auto difference = static_cast<uint8_t>(lhs.size() != rhs.size());
    auto count = std::max(lhs.size(), rhs.size());
    for (size_t i = 0u; i < count; i++) {
        auto lhs_byte = i < lhs.size() ? static_cast<uint8_t>(lhs[i]) : 0u;
        auto rhs_byte = i < rhs.size() ? static_cast<uint8_t>(rhs[i]) : 0u;
        difference |= lhs_byte ^ rhs_byte;
    }
    return difference == 0u;
}

[[nodiscard]] bool valid_range(
    size_t offset, size_t size, size_t total) noexcept {
    return offset <= total && size <= total - offset;
}

[[nodiscard]] bool valid_backend_name(
    luisa::string_view name) noexcept {
    if (name.size() > 128u) { return false; }
    return std::all_of(
        name.begin(), name.end(), [](char c) noexcept {
            auto u = static_cast<unsigned char>(c);
            return (u >= 'a' && u <= 'z') ||
                   (u >= 'A' && u <= 'Z') ||
                   (u >= '0' && u <= '9') ||
                   c == '_' || c == '-' || c == '.';
        });
}

}// namespace

class Session final : public luisa::enable_shared_from_this<Session>,
                      public CommandHandleResolver,
                      public ASTJsonBindingResolver {

private:
    struct BufferRecord {
        BufferCreationInfo native;
        size_t size_bytes{};
        bool is_indirect_dispatch{};
        size_t indirect_dispatch_capacity{};
    };
    struct TextureRecord {
        ResourceCreationInfo native;
        PixelStorage storage{};
        uint3 size{};
        uint mip_levels{};
    };
    struct ResourceRecord {
        ResourceCreationInfo native;
    };
    struct StreamRecord {
        ResourceCreationInfo native;
        bool log_callback_installed{};
    };
    struct BindlessRecord {
        ResourceCreationInfo native;
        size_t slot_count{};
        BindlessSlotType slot_type{};
    };
    struct ShaderRecord {
        ShaderCreationInfo native;
        luisa::vector<ShaderArgumentDesc> arguments;
    };
    struct MotionInstanceRecord {
        ResourceCreationInfo native;
        size_t keyframe_count{};
    };
    struct PreparedBlobBatch {
        luisa::vector<BlobKey> keys;
        luisa::vector<BlobCache::BlobPtr> blobs;
    };

    Tcp::socket _socket;
    luisa::shared_ptr<DeviceInterface> _native;
    const DeviceFactory &_device_factory;
    luisa::shared_ptr<BlobCache> _blob_cache;
    const ServerOptions &_options;
    bool _device_selection_enabled{};
    std::mutex _write_mutex;
    std::mutex _pending_mutex;
    std::unordered_set<uint64_t> _pending_submissions;
    std::unordered_map<uint64_t, BufferRecord> _buffers;
    std::unordered_map<uint64_t, TextureRecord> _textures;
    std::unordered_map<uint64_t, BindlessRecord> _bindless_arrays;
    std::unordered_map<uint64_t, StreamRecord> _streams;
    std::unordered_map<uint64_t, ShaderRecord> _shaders;
    std::unordered_map<uint64_t, ResourceRecord> _events;
    std::unordered_map<uint64_t, ResourceRecord> _meshes;
    std::unordered_map<uint64_t, ResourceRecord> _curves;
    std::unordered_map<uint64_t, ResourceRecord> _procedural_primitives;
    std::unordered_map<uint64_t, MotionInstanceRecord> _motion_instances;
    std::unordered_map<uint64_t, ResourceRecord> _accels;
    std::unordered_map<uint64_t, PreparedBlobBatch> _blob_batches;
    std::atomic_bool _alive{true};
    uint64_t _next_resource_index{1u};
    bool _handshake_complete{false};
    bool _close_after_reply{false};
    uint16_t _wire_minor{protocol_minor};
    bool _wire_version_set{false};

private:
    [[nodiscard]] size_t _resource_count() const noexcept {
        return _buffers.size() + _textures.size() + _bindless_arrays.size() +
               _streams.size() + _shaders.size() + _events.size() +
               _meshes.size() + _curves.size() +
               _procedural_primitives.size() + _motion_instances.size() +
               _accels.size();
    }

    [[nodiscard]] uint64_t _allocate_id(ResourceKind kind) noexcept {
        if (_next_resource_index > resource_index_mask) { return invalid_resource_handle; }
        return make_resource_id(kind, _next_resource_index++);
    }

    [[nodiscard]] bool _can_create_resource(Reply &reply) const noexcept {
        if (_resource_count() >= _options.max_resources) {
            reply.status = Status::RESOURCE_LIMIT;
            reply.message = "Remote session resource-count limit reached.";
            return false;
        }
        if (_next_resource_index > resource_index_mask) {
            reply.status = Status::RESOURCE_LIMIT;
            reply.message = "Remote resource-ID space exhausted.";
            return false;
        }
        return true;
    }

    [[nodiscard]] bool _send_frame(
        MessageKind kind, uint64_t request_id,
        luisa::span<const std::byte> payload) noexcept {
        if (!_alive.load(std::memory_order_acquire) ||
            payload.size() > _options.protocol_limits.max_frame_payload) {
            return false;
        }
        auto header = encode_frame_header(FrameHeader{
            .kind = kind,
            .request_id = request_id,
            .payload_size = payload.size(),
            .payload_checksum = payload_checksum(payload),
            .wire_major = protocol_major,
            .wire_minor = _wire_minor});
        asio::error_code error;
        {
            std::scoped_lock lock{_write_mutex};
            std::array<asio::const_buffer, 2u> buffers{
                asio::buffer(static_cast<const void *>(header.data()), header.size()),
                asio::buffer(payload.data(), payload.size())};
            asio::write(_socket, buffers, error);
        }
        if (error) { _alive.store(false, std::memory_order_release); }
        return !error;
    }

    [[nodiscard]] bool _send_reply(
        MessageKind request_kind, uint64_t request_id,
        Reply reply) noexcept {
        auto payload = make_response_payload(
            request_kind, reply.status, reply.message, reply.body);
        auto response_kind = reply.status == Status::OK ?
                                 MessageKind::RESPONSE :
                                 MessageKind::ERROR;
        return _send_frame(response_kind, request_id, payload);
    }

    [[nodiscard]] bool _read_frame(
        FrameHeader &header, luisa::vector<std::byte> &payload) noexcept {
        std::array<std::byte, frame_header_size> header_bytes{};
        asio::error_code error;
        asio::read(_socket, asio::buffer(header_bytes), error);
        if (error) { return false; }
        luisa::string decode_error;
        if (!decode_frame_header(
                header_bytes, header, decode_error,
                _options.protocol_limits) ||
            header.request_id == 0u) {
            return false;
        }
        if (!_wire_version_set) {
            _wire_minor = header.wire_minor;
            _wire_version_set = true;
        } else if (header.wire_major != protocol_major ||
                   header.wire_minor != _wire_minor) {
            return false;
        }
        payload.resize(static_cast<size_t>(header.payload_size));
        if (!payload.empty()) {
            asio::read(_socket, asio::buffer(payload), error);
            if (error) { return false; }
        }
        return payload_checksum(payload) == header.payload_checksum;
    }

    [[nodiscard]] Reply _hello(luisa::span<const std::byte> payload) noexcept {
        if (_handshake_complete) {
            return invalid("Remote HELLO may only be sent once per session.");
        }
        Reader reader{payload, _options.protocol_limits};
        auto endian_marker = reader.read_u32();
        auto pointer_width = reader.read_u8();
        auto ieee754 = reader.read_u8();
        auto reserved = reader.read_u16();
        auto token = reader.read_string();
        DeviceRequest device_request;
        if (reader.ok() && reader.remaining() != 0u) {
            device_request.backend = reader.read_string();
            auto device_index = reader.read_u64();
            device_request.enable_validation = reader.read_bool();
            if (device_index > std::numeric_limits<size_t>::max()) {
                return invalid("Remote device index exceeds the host size limit.");
            }
            device_request.device_index = static_cast<size_t>(device_index);
        }
        if (!reader.finish()) { return invalid(luisa::string{reader.error()}); }
        if (std::endian::native != std::endian::little ||
            endian_marker != 0x01020304u || pointer_width != 8u ||
            ieee754 != 1u || reserved != 0u) {
            return Reply{
                .status = Status::VERSION_MISMATCH,
                .message = "Remote protocol v1 requires little-endian, 64-bit, IEEE-754 hosts."};
        }
        if (!secure_equal(token, _options.token)) {
            _close_after_reply = true;
            return Reply{
                .status = Status::AUTHENTICATION_FAILED,
                .message = "Remote authentication failed."};
        }
        if (!valid_backend_name(device_request.backend)) {
            _close_after_reply = true;
            return invalid("Remote backend name contains invalid characters or is too long.");
        }
        luisa::string device_error;
        auto factory_failed = false;
        try {
            _native = _device_factory(device_request, device_error);
        } catch (const std::exception &exception) {
            factory_failed = true;
            device_error = luisa::format(
                "Remote device factory threw an exception: {}", exception.what());
        } catch (...) {
            factory_failed = true;
            device_error = "Remote device factory threw an unknown exception.";
        }
        if (_native == nullptr) {
            _close_after_reply = true;
            return Reply{
                .status = factory_failed ?
                              Status::BACKEND_ERROR :
                              Status::UNSUPPORTED,
                .message = device_error.empty() ?
                               "Remote device request was rejected by the service." :
                               std::move(device_error)};
        }
        _handshake_complete = true;
        Writer writer;
        writer.write_string(_native->backend_name());
        writer.write_u32(_native->compute_warp_size());
        writer.write_u64(_native->compute_max_shared_memory_size());
        writer.write_u64(_native->memory_granularity());
        auto features =
            static_cast<uint64_t>(Feature::BUFFER) |
            static_cast<uint64_t>(Feature::TEXTURE) |
            static_cast<uint64_t>(Feature::STREAM) |
            static_cast<uint64_t>(Feature::AST_SHADER) |
            static_cast<uint64_t>(Feature::EVENT) |
            static_cast<uint64_t>(Feature::ASYNC_DISPATCH) |
            static_cast<uint64_t>(Feature::STREAM_LOG) |
            static_cast<uint64_t>(Feature::BINDLESS_ARRAY) |
            static_cast<uint64_t>(Feature::RAY_TRACING) |
            static_cast<uint64_t>(Feature::LIMIT_NEGOTIATION);
        if (_device_selection_enabled) {
            features |= static_cast<uint64_t>(Feature::DEVICE_SELECTION);
        }
        auto blob_cache_enabled =
            _blob_cache != nullptr &&
            _options.max_blob_entry_size != 0u &&
            _options.max_blobs_per_batch != 0u &&
            _options.max_prepared_blob_batches != 0u;
        if (blob_cache_enabled) {
            features |= static_cast<uint64_t>(Feature::BLOB_CACHE);
        }
        writer.write_u64(features);
        return Reply{.body = std::move(writer).take()};
    }

    [[nodiscard]] Reply _create_buffer(
        luisa::span<const std::byte> payload) noexcept {
        Reader reader{payload, _options.protocol_limits};
        auto kind_value = reader.read_u8();
        auto value = reader.read_u64();
        if (!reader.finish()) { return invalid(luisa::string{reader.error()}); }
        if (kind_value > static_cast<uint8_t>(BufferKind::INDIRECT_DISPATCH) ||
            value > std::numeric_limits<size_t>::max()) {
            return invalid("Remote buffer descriptor is invalid.");
        }
        auto kind = static_cast<BufferKind>(kind_value);
        if ((kind == BufferKind::BYTE &&
             value > _options.max_resource_size) ||
            (kind == BufferKind::INDIRECT_DISPATCH &&
             value > _options.protocol_limits.max_array_size)) {
            return Reply{.status = Status::RESOURCE_LIMIT,
                         .message = "Remote buffer exceeds the resource-size limit."};
        }
        Reply reply;
        if (!_can_create_resource(reply)) { return reply; }
        auto count = static_cast<size_t>(value);
        auto element = kind == BufferKind::BYTE ?
                           Type::of<void>() :
                           Type::custom(indirect_dispatch_element_type_name);
        auto native = _native->create_buffer(
            element, count, nullptr);
        if (!native.valid() ||
            native.total_size_bytes > _options.max_resource_size) {
            if (native.valid()) { _native->destroy_buffer(native.handle); }
            return Reply{.status = Status::BACKEND_ERROR,
                         .message = "Native backend failed to create a bounded buffer."};
        }
        auto id = _allocate_id(ResourceKind::BUFFER);
        if (id == invalid_resource_handle) {
            _native->destroy_buffer(native.handle);
            return Reply{.status = Status::RESOURCE_LIMIT,
                         .message = "Remote resource-ID space exhausted."};
        }
        _buffers.emplace(
            id, BufferRecord{
                    .native = native,
                    .size_bytes = native.total_size_bytes,
                    .is_indirect_dispatch =
                        kind == BufferKind::INDIRECT_DISPATCH,
                    .indirect_dispatch_capacity =
                        kind == BufferKind::INDIRECT_DISPATCH ? count : 0u});
        Writer writer;
        writer.write_u64(id);
        writer.write_u64(native.element_stride);
        writer.write_u64(native.total_size_bytes);
        return Reply{.body = std::move(writer).take()};
    }

    [[nodiscard]] Reply _destroy_buffer(
        luisa::span<const std::byte> payload) noexcept {
        Reader reader{payload, _options.protocol_limits};
        auto id = reader.read_u64();
        if (!reader.finish()) { return invalid(luisa::string{reader.error()}); }
        auto iter = _buffers.find(id);
        if (iter == _buffers.end()) { return not_found("Remote buffer was not found."); }
        _native->destroy_buffer(iter->second.native.handle);
        _buffers.erase(iter);
        return {};
    }

    [[nodiscard]] Reply _create_texture(
        luisa::span<const std::byte> payload) noexcept {
        Reader reader{payload, _options.protocol_limits};
        auto format_value = reader.read_u32();
        auto dimension = reader.read_u32();
        auto width = reader.read_u32();
        auto height = reader.read_u32();
        auto depth = reader.read_u32();
        auto mip_levels = reader.read_u32();
        auto simultaneous_access = reader.read_bool();
        auto allow_raster_target = reader.read_bool();
        if (!reader.finish()) { return invalid(luisa::string{reader.error()}); }
        if (format_value >= pixel_format_count ||
            (dimension != 2u && dimension != 3u) ||
            width == 0u || height == 0u || depth == 0u || mip_levels == 0u ||
            (dimension == 2u && depth != 1u)) {
            return invalid("Remote texture descriptor is invalid.");
        }
        auto storage = pixel_format_to_storage(
            static_cast<PixelFormat>(format_value));
        auto base_size = uint3{width, height, depth};
        auto max_dimension = std::max({width, height, depth});
        auto max_mip_levels = 1u;
        while (max_dimension > 1u) {
            max_dimension >>= 1u;
            max_mip_levels++;
        }
        if (mip_levels > max_mip_levels) {
            return invalid("Remote texture mip-level count is invalid.");
        }
        uint64_t total_size{};
        for (auto level = 0u; level < mip_levels; level++) {
            auto level_size = luisa::max(base_size >> level, 1u);
            auto size_result = checked_pixel_storage_size(storage, level_size);
            if (!size_result ||
                size_result.size > _options.max_resource_size -
                                       std::min(total_size, _options.max_resource_size)) {
                return Reply{.status = Status::RESOURCE_LIMIT,
                             .message = "Remote texture exceeds the resource-size limit."};
            }
            total_size += size_result.size;
        }
        if (total_size > _options.max_resource_size) {
            return Reply{.status = Status::RESOURCE_LIMIT,
                         .message = "Remote texture exceeds the resource-size limit."};
        }
        Reply reply;
        if (!_can_create_resource(reply)) { return reply; }
        auto native = _native->create_texture(
            static_cast<PixelFormat>(format_value), dimension,
            width, height, depth, mip_levels, nullptr,
            simultaneous_access, allow_raster_target);
        if (!native.valid()) {
            return Reply{.status = Status::BACKEND_ERROR,
                         .message = "Native backend failed to create the texture."};
        }
        auto id = _allocate_id(ResourceKind::TEXTURE);
        _textures.emplace(
            id, TextureRecord{native, storage, base_size, mip_levels});
        Writer writer;
        writer.write_u64(id);
        return Reply{.body = std::move(writer).take()};
    }

    [[nodiscard]] Reply _destroy_texture(
        luisa::span<const std::byte> payload) noexcept {
        Reader reader{payload, _options.protocol_limits};
        auto id = reader.read_u64();
        if (!reader.finish()) { return invalid(luisa::string{reader.error()}); }
        auto iter = _textures.find(id);
        if (iter == _textures.end()) { return not_found("Remote texture was not found."); }
        _native->destroy_texture(iter->second.native.handle);
        _textures.erase(iter);
        return {};
    }

    [[nodiscard]] Reply _create_bindless_array(
        luisa::span<const std::byte> payload) noexcept {
        Reader reader{payload, _options.protocol_limits};
        auto slot_count = reader.read_u64();
        auto slot_type_value = reader.read_u32();
        if (!reader.finish()) { return invalid(luisa::string{reader.error()}); }
        if (slot_count == 0u || slot_count > _options.protocol_limits.max_array_size ||
            slot_count > std::numeric_limits<size_t>::max() ||
            slot_type_value > static_cast<uint32_t>(BindlessSlotType::TEXTURE3D_ONLY)) {
            return invalid("Remote bindless-array descriptor is invalid.");
        }
        Reply reply;
        if (!_can_create_resource(reply)) { return reply; }
        auto slot_type = static_cast<BindlessSlotType>(slot_type_value);
        auto native = _native->create_bindless_array(
            static_cast<size_t>(slot_count), slot_type);
        if (!native.valid()) {
            return Reply{.status = Status::BACKEND_ERROR,
                         .message = "Native backend failed to create the bindless array."};
        }
        auto id = _allocate_id(ResourceKind::BINDLESS_ARRAY);
        _bindless_arrays.emplace(
            id, BindlessRecord{native, static_cast<size_t>(slot_count), slot_type});
        Writer writer;
        writer.write_u64(id);
        return Reply{.body = std::move(writer).take()};
    }

    [[nodiscard]] Reply _destroy_bindless_array(
        luisa::span<const std::byte> payload) noexcept {
        Reader reader{payload, _options.protocol_limits};
        auto id = reader.read_u64();
        if (!reader.finish()) { return invalid(luisa::string{reader.error()}); }
        auto iter = _bindless_arrays.find(id);
        if (iter == _bindless_arrays.end()) {
            return not_found("Remote bindless array was not found.");
        }
        _native->destroy_bindless_array(iter->second.native.handle);
        _bindless_arrays.erase(iter);
        return {};
    }

    [[nodiscard]] Reply _create_stream(
        luisa::span<const std::byte> payload) noexcept {
        Reader reader{payload, _options.protocol_limits};
        auto tag_value = reader.read_u32();
        if (!reader.finish()) { return invalid(luisa::string{reader.error()}); }
        if (tag_value > static_cast<uint32_t>(StreamTag::COPY)) {
            return unsupported("Remote protocol v1 does not support custom streams.");
        }
        Reply reply;
        if (!_can_create_resource(reply)) { return reply; }
        auto native = _native->create_stream(static_cast<StreamTag>(tag_value));
        if (!native.valid()) {
            return Reply{.status = Status::BACKEND_ERROR,
                         .message = "Native backend failed to create the stream."};
        }
        auto id = _allocate_id(ResourceKind::STREAM);
        _streams.emplace(id, StreamRecord{native});
        Writer writer;
        writer.write_u64(id);
        return Reply{.body = std::move(writer).take()};
    }

    [[nodiscard]] Reply _destroy_stream(
        luisa::span<const std::byte> payload) noexcept {
        Reader reader{payload, _options.protocol_limits};
        auto id = reader.read_u64();
        if (!reader.finish()) { return invalid(luisa::string{reader.error()}); }
        auto iter = _streams.find(id);
        if (iter == _streams.end()) { return not_found("Remote stream was not found."); }
        _native->synchronize_stream(iter->second.native.handle);
        if (iter->second.log_callback_installed) {
            _native->set_stream_log_callback(iter->second.native.handle, {});
        }
        _native->destroy_stream(iter->second.native.handle);
        _streams.erase(iter);
        return {};
    }

    [[nodiscard]] Reply _synchronize_stream(
        luisa::span<const std::byte> payload) noexcept {
        Reader reader{payload, _options.protocol_limits};
        auto id = reader.read_u64();
        if (!reader.finish()) { return invalid(luisa::string{reader.error()}); }
        auto iter = _streams.find(id);
        if (iter == _streams.end()) { return not_found("Remote stream was not found."); }
        _native->synchronize_stream(iter->second.native.handle);
        return {};
    }

    [[nodiscard]] static ShaderOption _read_shader_option(
        Reader &reader) noexcept {
        ShaderOption option;
        option.enable_cache = reader.read_bool();
        option.enable_fast_math = reader.read_bool();
        option.enable_debug_info = reader.read_bool();
        option.compile_only = reader.read_bool();
        option.max_registers = reader.read_u32();
        option.time_trace = reader.read_bool();
        option.enable_extended_accel_limits = reader.read_bool();
        option.enable_scalarizer = reader.read_bool();
        option.enable_ray_query_pipeline = reader.read_bool();
        option.force_ray_query_pipeline = reader.read_bool();
        option.enable_driver_optimization = reader.read_bool();
        option.name = reader.read_string();
        return option;
    }

    [[nodiscard]] Reply _create_shader(
        luisa::span<const std::byte> payload) noexcept {
        Reader reader{payload, _options.protocol_limits};
        auto option = _read_shader_option(reader);
        auto ast_bytes = reader.read_blob();
        if (!reader.finish()) { return invalid(luisa::string{reader.error()}); }
        if (ast_bytes.size() > ASTJsonLimits{}.max_document_bytes) {
            return Reply{.status = Status::RESOURCE_LIMIT,
                         .message = "Remote shader AST exceeds the document limit."};
        }
        auto ast = from_json(
            luisa::string_view{reinterpret_cast<const char *>(ast_bytes.data()),
                               ast_bytes.size()},
            ASTJsonLimits{}, this);
        if (!ast) {
            return invalid(luisa::format(
                "Remote shader AST is invalid: {}", ast.error));
        }
        auto function = Function{ast.function.get()};
        if (function.tag() != Function::Tag::KERNEL) {
            return invalid("Remote shader AST root must be a kernel.");
        }
        luisa::vector<ShaderArgumentDesc> arguments;
        arguments.reserve(function.unbound_arguments().size());
        for (auto variable : function.unbound_arguments()) {
            ShaderArgumentDesc argument;
            auto type = variable.type();
            switch (variable.tag()) {
                case Variable::Tag::BUFFER:
                    if (!(type->is_buffer() ||
                          (type->is_custom() &&
                           type->description() ==
                               ast_json_indirect_dispatch_buffer_type_name))) {
                        return invalid("Remote shader buffer argument type is invalid.");
                    }
                    argument.tag = Argument::Tag::BUFFER;
                    argument.indirect_dispatch_buffer = type->is_custom();
                    break;
                case Variable::Tag::TEXTURE:
                    if (!type->is_texture()) {
                        return invalid("Remote shader texture argument type is invalid.");
                    }
                    argument.tag = Argument::Tag::TEXTURE;
                    break;
                case Variable::Tag::BINDLESS_ARRAY:
                    if (!type->is_bindless_array()) {
                        return invalid("Remote shader bindless argument type is invalid.");
                    }
                    argument.tag = Argument::Tag::BINDLESS_ARRAY;
                    break;
                case Variable::Tag::ACCEL:
                    if (!type->is_accel()) {
                        return invalid("Remote shader accel argument type is invalid.");
                    }
                    argument.tag = Argument::Tag::ACCEL;
                    break;
                case Variable::Tag::LOCAL:
                    if (type->is_resource() || type->is_custom()) {
                        return invalid("Remote shader uniform argument type is invalid.");
                    }
                    argument.tag = Argument::Tag::UNIFORM;
                    argument.uniform_size = type->size();
                    argument.uniform_alignment = type->alignment();
                    break;
                default:
                    return invalid("Remote shader contains an unsupported argument kind.");
            }
            arguments.emplace_back(argument);
        }
        Reply reply;
        if (!option.compile_only && !_can_create_resource(reply)) { return reply; }
        auto native = _native->create_shader(option, function);
        if (!native.valid() && !option.compile_only) {
            return Reply{.status = Status::BACKEND_ERROR,
                         .message = "Native backend failed to create the shader."};
        }
        auto id = invalid_resource_handle;
        if (native.valid()) {
            id = _allocate_id(ResourceKind::SHADER);
            _shaders.emplace(
                id, ShaderRecord{native, std::move(arguments)});
        }
        Writer writer;
        writer.write_u64(id);
        writer.write_u32(native.block_size.x);
        writer.write_u32(native.block_size.y);
        writer.write_u32(native.block_size.z);
        return Reply{.body = std::move(writer).take()};
    }

    [[nodiscard]] Reply _destroy_shader(
        luisa::span<const std::byte> payload) noexcept {
        Reader reader{payload, _options.protocol_limits};
        auto id = reader.read_u64();
        if (!reader.finish()) { return invalid(luisa::string{reader.error()}); }
        auto iter = _shaders.find(id);
        if (iter == _shaders.end()) { return not_found("Remote shader was not found."); }
        _native->destroy_shader(iter->second.native.handle);
        _shaders.erase(iter);
        return {};
    }

    [[nodiscard]] Reply _shader_argument_usage(
        luisa::span<const std::byte> payload) noexcept {
        Reader reader{payload, _options.protocol_limits};
        auto id = reader.read_u64();
        auto index = reader.read_u64();
        if (!reader.finish()) { return invalid(luisa::string{reader.error()}); }
        auto iter = _shaders.find(id);
        if (iter == _shaders.end()) { return not_found("Remote shader was not found."); }
        if (index > std::numeric_limits<size_t>::max() ||
            index >= iter->second.arguments.size()) {
            return invalid("Remote shader argument index is invalid.");
        }
        auto usage = _native->shader_argument_usage(
            iter->second.native.handle, static_cast<size_t>(index));
        Writer writer;
        writer.write_u32(static_cast<uint32_t>(usage));
        return Reply{.body = std::move(writer).take()};
    }

    [[nodiscard]] Reply _create_event(
        luisa::span<const std::byte> payload) noexcept {
        if (!payload.empty()) { return invalid("Remote create-event payload must be empty."); }
        Reply reply;
        if (!_can_create_resource(reply)) { return reply; }
        auto native = _native->create_event();
        if (!native.valid()) {
            return Reply{.status = Status::BACKEND_ERROR,
                         .message = "Native backend failed to create the event."};
        }
        auto id = _allocate_id(ResourceKind::EVENT);
        _events.emplace(id, ResourceRecord{native});
        Writer writer;
        writer.write_u64(id);
        return Reply{.body = std::move(writer).take()};
    }

    [[nodiscard]] Reply _destroy_event(
        luisa::span<const std::byte> payload) noexcept {
        Reader reader{payload, _options.protocol_limits};
        auto id = reader.read_u64();
        if (!reader.finish()) { return invalid(luisa::string{reader.error()}); }
        auto iter = _events.find(id);
        if (iter == _events.end()) { return not_found("Remote event was not found."); }
        _native->destroy_event(iter->second.native.handle);
        _events.erase(iter);
        return {};
    }

    [[nodiscard]] Reply _event_stream_operation(
        MessageKind kind, luisa::span<const std::byte> payload) noexcept {
        Reader reader{payload, _options.protocol_limits};
        auto event_id = reader.read_u64();
        auto stream_id = reader.read_u64();
        auto value = reader.read_u64();
        if (!reader.finish()) { return invalid(luisa::string{reader.error()}); }
        auto event = _events.find(event_id);
        auto stream = _streams.find(stream_id);
        if (event == _events.end() || stream == _streams.end()) {
            return not_found("Remote event or stream was not found.");
        }
        if (kind == MessageKind::SIGNAL_EVENT) {
            _native->signal_event(
                event->second.native.handle, stream->second.native.handle, value);
        } else {
            _native->wait_event(
                event->second.native.handle, stream->second.native.handle, value);
        }
        return {};
    }

    [[nodiscard]] Reply _event_query(
        MessageKind kind, luisa::span<const std::byte> payload) noexcept {
        Reader reader{payload, _options.protocol_limits};
        auto id = reader.read_u64();
        auto value = reader.read_u64();
        if (!reader.finish()) { return invalid(luisa::string{reader.error()}); }
        auto event = _events.find(id);
        if (event == _events.end()) { return not_found("Remote event was not found."); }
        if (kind == MessageKind::SYNCHRONIZE_EVENT) {
            _native->synchronize_event(event->second.native.handle, value);
            return {};
        }
        Writer writer;
        writer.write_bool(_native->is_event_completed(
            event->second.native.handle, value));
        return Reply{.body = std::move(writer).take()};
    }

    [[nodiscard]] bool _decode_accel_option(
        luisa::span<const std::byte> payload,
        AccelOption &option, luisa::string &error) noexcept {
        Reader reader{payload, _options.protocol_limits};
        auto hint = reader.read_u32();
        option.allow_compaction = reader.read_bool();
        option.allow_update = reader.read_bool();
        option.motion.keyframe_count = reader.read_u32();
        option.motion.time_start = reader.read_f32();
        option.motion.time_end = reader.read_f32();
        option.motion.should_vanish_start = reader.read_bool();
        option.motion.should_vanish_end = reader.read_bool();
        auto mode = reader.read_u8();
        if (!reader.finish()) {
            error = reader.error();
            return false;
        }
        if (hint > static_cast<uint32_t>(AccelOption::UsageHint::FAST_BUILD) ||
            mode > static_cast<uint8_t>(AccelMotionMode::SRT) ||
            option.motion.keyframe_count >
                _options.protocol_limits.max_array_size ||
            !std::isfinite(option.motion.time_start) ||
            !std::isfinite(option.motion.time_end) ||
            option.motion.time_start > option.motion.time_end) {
            error = "Remote acceleration-structure option is invalid.";
            return false;
        }
        option.hint = static_cast<AccelOption::UsageHint>(hint);
        option.motion.mode = static_cast<AccelMotionMode>(mode);
        return true;
    }

    [[nodiscard]] Reply _create_accel_resource(
        MessageKind kind, luisa::span<const std::byte> payload) noexcept {
        AccelOption option;
        luisa::string error;
        if (!_decode_accel_option(payload, option, error)) {
            return invalid(std::move(error));
        }
        if (kind == MessageKind::CREATE_MOTION_INSTANCE &&
            option.motion.keyframe_count < 2u) {
            return invalid("Remote motion instances require at least two keyframes.");
        }
        Reply reply;
        if (!_can_create_resource(reply)) { return reply; }
        ResourceCreationInfo native;
        ResourceKind resource_kind_value{};
        switch (kind) {
            case MessageKind::CREATE_MESH:
                native = _native->create_mesh(option);
                resource_kind_value = ResourceKind::MESH;
                break;
            case MessageKind::CREATE_CURVE:
                native = _native->create_curve(option);
                resource_kind_value = ResourceKind::CURVE;
                break;
            case MessageKind::CREATE_PROCEDURAL_PRIMITIVE:
                native = _native->create_procedural_primitive(option);
                resource_kind_value = ResourceKind::PROCEDURAL_PRIMITIVE;
                break;
            case MessageKind::CREATE_MOTION_INSTANCE:
                native = _native->create_motion_instance(option.motion);
                resource_kind_value = ResourceKind::MOTION_INSTANCE;
                break;
            case MessageKind::CREATE_ACCEL:
                native = _native->create_accel(option);
                resource_kind_value = ResourceKind::ACCEL;
                break;
            default: return invalid("Invalid remote acceleration-structure request.");
        }
        if (!native.valid()) {
            return Reply{.status = Status::BACKEND_ERROR,
                         .message = "Native backend failed to create the acceleration-structure resource."};
        }
        auto id = _allocate_id(resource_kind_value);
        switch (resource_kind_value) {
            case ResourceKind::MESH:
                _meshes.emplace(id, ResourceRecord{native});
                break;
            case ResourceKind::CURVE:
                _curves.emplace(id, ResourceRecord{native});
                break;
            case ResourceKind::PROCEDURAL_PRIMITIVE:
                _procedural_primitives.emplace(id, ResourceRecord{native});
                break;
            case ResourceKind::MOTION_INSTANCE:
                _motion_instances.emplace(
                    id, MotionInstanceRecord{
                            native, option.motion.keyframe_count});
                break;
            case ResourceKind::ACCEL:
                _accels.emplace(id, ResourceRecord{native});
                break;
            default: break;
        }
        Writer writer;
        writer.write_u64(id);
        return Reply{.body = std::move(writer).take()};
    }

    [[nodiscard]] Reply _destroy_accel_resource(
        MessageKind kind, luisa::span<const std::byte> payload) noexcept {
        Reader reader{payload, _options.protocol_limits};
        auto id = reader.read_u64();
        if (!reader.finish()) { return invalid(luisa::string{reader.error()}); }
        switch (kind) {
            case MessageKind::DESTROY_MESH:
                if (auto iter = _meshes.find(id); iter != _meshes.end()) {
                    _native->destroy_mesh(iter->second.native.handle);
                    _meshes.erase(iter);
                    return {};
                }
                break;
            case MessageKind::DESTROY_CURVE:
                if (auto iter = _curves.find(id); iter != _curves.end()) {
                    _native->destroy_curve(iter->second.native.handle);
                    _curves.erase(iter);
                    return {};
                }
                break;
            case MessageKind::DESTROY_PROCEDURAL_PRIMITIVE:
                if (auto iter = _procedural_primitives.find(id);
                    iter != _procedural_primitives.end()) {
                    _native->destroy_procedural_primitive(iter->second.native.handle);
                    _procedural_primitives.erase(iter);
                    return {};
                }
                break;
            case MessageKind::DESTROY_MOTION_INSTANCE:
                if (auto iter = _motion_instances.find(id);
                    iter != _motion_instances.end()) {
                    _native->destroy_motion_instance(iter->second.native.handle);
                    _motion_instances.erase(iter);
                    return {};
                }
                break;
            case MessageKind::DESTROY_ACCEL:
                if (auto iter = _accels.find(id); iter != _accels.end()) {
                    _native->destroy_accel(iter->second.native.handle);
                    _accels.erase(iter);
                    return {};
                }
                break;
            default: return invalid("Invalid remote acceleration-structure destroy request.");
        }
        return not_found("Remote acceleration-structure resource was not found.");
    }

    [[nodiscard]] Reply _prepare_blobs(
        luisa::span<const std::byte> payload) noexcept {
        if (_blob_cache == nullptr) {
            return unsupported("Remote blob caching is disabled on this server.");
        }
        Reader reader{payload, _options.protocol_limits};
        auto submission_id = reader.read_u64();
        auto count = reader.read_u64();
        if (!reader.ok()) { return invalid(luisa::string{reader.error()}); }
        if (submission_id == 0u || count == 0u ||
            count > _options.max_blobs_per_batch ||
            count > _options.protocol_limits.max_array_size ||
            count > std::numeric_limits<size_t>::max()) {
            return invalid("Remote blob-prepare descriptor count or submission ID is invalid.");
        }
        if (_blob_batches.size() >= _options.max_prepared_blob_batches) {
            return Reply{
                .status = Status::RESOURCE_LIMIT,
                .message = "Remote prepared-blob batch limit reached."};
        }
        if (_blob_batches.contains(submission_id)) {
            return invalid("A remote blob batch already exists for this submission ID.");
        }
        {
            std::scoped_lock lock{_pending_mutex};
            if (_pending_submissions.contains(submission_id)) {
                return invalid("Remote submission ID is already pending.");
            }
        }

        PreparedBlobBatch batch;
        batch.keys.reserve(static_cast<size_t>(count));
        batch.blobs.reserve(static_cast<size_t>(count));
        std::unordered_set<BlobKey, BlobKeyHash> unique;
        uint64_t total_size{};
        luisa::vector<uint32_t> misses;
        for (uint64_t i = 0u; i < count; i++) {
            BlobKey key;
            if (!read_blob_key(reader, key)) {
                return invalid(luisa::string{reader.error()});
            }
            if (key.size == 0u ||
                key.size > _options.max_blob_entry_size ||
                key.size > _blob_cache->capacity_bytes() ||
                total_size > _blob_cache->capacity_bytes() - key.size) {
                return Reply{
                    .status = Status::RESOURCE_LIMIT,
                    .message = "Remote prepared blobs exceed the cache limits."};
            }
            if (!unique.emplace(key).second) {
                return invalid("Remote blob descriptors must be unique within a batch.");
            }
            total_size += key.size;
            batch.keys.emplace_back(key);
            auto blob = _blob_cache->find(key);
            if (blob == nullptr) {
                misses.emplace_back(static_cast<uint32_t>(i));
            }
            batch.blobs.emplace_back(std::move(blob));
        }
        if (!reader.finish()) {
            return invalid(luisa::string{reader.error()});
        }
        _blob_batches.emplace(submission_id, std::move(batch));
        Writer writer;
        writer.write_u64(submission_id);
        writer.write_u64(misses.size());
        for (auto index : misses) { writer.write_u32(index); }
        return Reply{.body = std::move(writer).take()};
    }

    [[nodiscard]] Reply _blob_cache_info(
        luisa::span<const std::byte> payload) noexcept {
        Reader reader{payload, _options.protocol_limits};
        if (!reader.finish()) {
            return invalid(luisa::string{reader.error()});
        }
        if (_blob_cache == nullptr) {
            return unsupported("Remote blob caching is disabled on this server.");
        }
        Writer writer;
        writer.write_u64(_options.max_blob_entry_size);
        writer.write_u64(_options.blob_cache_min_size);
        writer.write_u64(_options.max_blobs_per_batch);
        return Reply{.body = std::move(writer).take()};
    }

    [[nodiscard]] Reply _protocol_info(
        luisa::span<const std::byte> payload) noexcept {
        Reader reader{payload, _options.protocol_limits};
        if (!reader.finish()) {
            return invalid(luisa::string{reader.error()});
        }
        Writer writer;
        writer.write_u64(_options.protocol_limits.max_frame_payload);
        writer.write_u64(_options.protocol_limits.max_string_size);
        writer.write_u64(_options.protocol_limits.max_array_size);
        return Reply{.body = std::move(writer).take()};
    }

    [[nodiscard]] Reply _upload_blobs(
        luisa::span<const std::byte> payload) noexcept {
        if (_blob_cache == nullptr) {
            return unsupported("Remote blob caching is disabled on this server.");
        }
        struct PendingUpload {
            uint32_t index{};
            BlobKey key;
            luisa::span<const std::byte> bytes;
        };
        Reader reader{payload, _options.protocol_limits};
        auto submission_id = reader.read_u64();
        auto count = reader.read_u64();
        auto batch_iterator = _blob_batches.find(submission_id);
        if (!reader.ok()) { return invalid(luisa::string{reader.error()}); }
        if (submission_id == 0u ||
            batch_iterator == _blob_batches.end() ||
            count == 0u ||
            count > batch_iterator->second.keys.size() ||
            count > std::numeric_limits<size_t>::max()) {
            return invalid("Remote blob-upload batch or count is invalid.");
        }
        auto &batch = batch_iterator->second;
        luisa::vector<PendingUpload> pending;
        pending.reserve(static_cast<size_t>(count));
        luisa::vector<uint8_t> seen(batch.keys.size(), 0u);
        for (uint64_t i = 0u; i < count; i++) {
            auto index = reader.read_u32();
            BlobKey key;
            if (!read_blob_key(reader, key)) {
                return invalid(luisa::string{reader.error()});
            }
            auto bytes = reader.read_blob();
            if (!reader.ok()) {
                return invalid(luisa::string{reader.error()});
            }
            if (index >= batch.keys.size() || seen[index] != 0u ||
                batch.blobs[index] != nullptr ||
                key != batch.keys[index] || bytes.size() != key.size) {
                return invalid("Remote blob-upload descriptor does not match its prepared slot.");
            }
            seen[index] = 1u;
            pending.emplace_back(PendingUpload{index, key, bytes});
        }
        if (!reader.finish()) {
            return invalid(luisa::string{reader.error()});
        }

        luisa::vector<BlobCache::BlobPtr> published;
        published.reserve(pending.size());
        for (auto &&upload : pending) {
            luisa::string error;
            BlobCacheError cache_error{};
            auto blob = _blob_cache->publish(
                upload.key, upload.bytes, cache_error, error);
            if (blob == nullptr) {
                auto status = cache_error == BlobCacheError::CAPACITY ?
                                  Status::RESOURCE_LIMIT :
                                  Status::INVALID_REQUEST;
                return Reply{.status = status, .message = std::move(error)};
            }
            published.emplace_back(std::move(blob));
        }
        for (size_t i = 0u; i < pending.size(); i++) {
            batch.blobs[pending[i].index] = std::move(published[i]);
        }
        Writer writer;
        writer.write_u64(submission_id);
        return Reply{.body = std::move(writer).take()};
    }

    void _send_completion(
        uint64_t submission_id,
        luisa::vector<ServerDownload> downloads) noexcept {
        {
            std::scoped_lock lock{_pending_mutex};
            _pending_submissions.erase(submission_id);
        }
        Writer writer;
        writer.write_u64(submission_id);
        writer.write_u16(static_cast<uint16_t>(Status::OK));
        writer.write_string({});
        writer.write_u64(downloads.size());
        for (auto &&download : downloads) {
            writer.write_u32(download.index);
            writer.write_blob(*download.storage);
        }
        static_cast<void>(_send_frame(
            MessageKind::DISPATCH_COMPLETE, submission_id,
            writer.bytes()));
    }

    [[nodiscard]] Reply _dispatch(
        luisa::span<const std::byte> payload) noexcept {
        auto decoded = decode_submission(payload, *this, _options.protocol_limits);
        auto blob_batch = _blob_batches.find(decoded.submission_id);
        if (!decoded) {
            if (blob_batch != _blob_batches.end()) {
                _blob_batches.erase(blob_batch);
            }
            return invalid(std::move(decoded.error));
        }
        if (blob_batch != _blob_batches.end()) {
            luisa::vector<uint8_t> used(
                blob_batch->second.keys.size(), 0u);
            for (auto index : decoded.upload_blob_indices) {
                if (index >= used.size()) {
                    _blob_batches.erase(blob_batch);
                    decoded.command_list.clear();
                    return invalid("Remote dispatch references an invalid prepared blob index.");
                }
                used[index] = 1u;
            }
            auto complete = std::all_of(
                blob_batch->second.blobs.begin(),
                blob_batch->second.blobs.end(),
                [](auto &&blob) noexcept { return blob != nullptr; });
            auto all_used = std::all_of(
                used.begin(), used.end(),
                [](auto value) noexcept { return value != 0u; });
            auto filled_count = static_cast<size_t>(std::count_if(
                blob_batch->second.blobs.begin(),
                blob_batch->second.blobs.end(),
                [](auto &&blob) noexcept { return blob != nullptr; }));
            auto used_count = static_cast<size_t>(std::count_if(
                used.begin(), used.end(),
                [](auto value) noexcept { return value != 0u; }));
            auto descriptor_count = blob_batch->second.keys.size();
            _blob_batches.erase(blob_batch);
            if (!complete || !all_used) {
                decoded.command_list.clear();
                return invalid(luisa::format(
                    "Remote dispatch blob batch is incomplete or contains unused descriptors "
                    "({} descriptors, {} filled, {} referenced).",
                    descriptor_count, filled_count, used_count));
            }
        } else if (!decoded.upload_blob_indices.empty()) {
            decoded.command_list.clear();
            return invalid("Remote dispatch references a missing prepared blob batch.");
        }
        auto stream = _streams.find(decoded.stream_handle);
        if (stream == _streams.end()) {
            decoded.command_list.clear();
            return not_found("Remote dispatch stream was not found.");
        }
        {
            std::scoped_lock lock{_pending_mutex};
            if (_pending_submissions.size() >= _options.max_pending_submissions) {
                decoded.command_list.clear();
                return Reply{.status = Status::RESOURCE_LIMIT,
                             .message = "Remote pending-submission limit reached."};
            }
            if (!_pending_submissions.emplace(decoded.submission_id).second) {
                decoded.command_list.clear();
                return invalid("Remote submission ID is already pending.");
            }
        }
        auto submission_id = decoded.submission_id;
        auto downloads = std::move(decoded.downloads);
        auto upload_blobs = std::move(decoded.upload_blobs);
        auto self = shared_from_this();
        decoded.command_list.add_callback(
            [self = std::move(self), submission_id,
             downloads = std::move(downloads),
             upload_blobs = std::move(upload_blobs)]() mutable noexcept {
                static_cast<void>(upload_blobs);
                self->_send_completion(submission_id, std::move(downloads));
            });
        _native->dispatch(
            stream->second.native.handle,
            std::move(decoded.command_list));
        Writer writer;
        writer.write_u64(submission_id);
        return Reply{.body = std::move(writer).take()};
    }

    [[nodiscard]] Reply _query(
        luisa::span<const std::byte> payload) noexcept {
        Reader reader{payload, _options.protocol_limits};
        auto property = reader.read_string();
        if (!reader.finish()) { return invalid(luisa::string{reader.error()}); }
        luisa::string value;
        if (property == "remote.blob_cache.enabled") {
            value = _blob_cache == nullptr ? "false" : "true";
        } else if (property == "remote.blob_cache.capacity_bytes") {
            value = luisa::format(
                "{}", _blob_cache == nullptr ?
                          0u :
                          _blob_cache->capacity_bytes());
        } else if (property.starts_with("remote.blob_cache.") &&
                   _blob_cache != nullptr) {
            auto stats = _blob_cache->stats();
            if (property == "remote.blob_cache.hits") {
                value = luisa::format("{}", stats.hits);
            } else if (property == "remote.blob_cache.misses") {
                value = luisa::format("{}", stats.misses);
            } else if (property == "remote.blob_cache.stores") {
                value = luisa::format("{}", stats.stores);
            } else if (property == "remote.blob_cache.evictions") {
                value = luisa::format("{}", stats.evictions);
            } else if (property == "remote.blob_cache.uploaded_bytes") {
                value = luisa::format("{}", stats.uploaded_bytes);
            } else if (property == "remote.blob_cache.resident_bytes") {
                value = luisa::format("{}", stats.resident_bytes);
            } else if (property == "remote.blob_cache.resident_entries") {
                value = luisa::format("{}", stats.resident_entries);
            }
        } else {
            value = _native->query(property);
        }
        Writer writer;
        writer.write_string(value);
        return Reply{.body = std::move(writer).take()};
    }

    [[nodiscard]] bool _resolve_resource_for_name(
        Resource::Tag tag, uint64_t remote, uint64_t &native) const noexcept {
        switch (tag) {
            case Resource::Tag::BUFFER:
                if (auto i = _buffers.find(remote); i != _buffers.end()) {
                    native = i->second.native.handle;
                    return true;
                }
                break;
            case Resource::Tag::TEXTURE:
                if (auto i = _textures.find(remote); i != _textures.end()) {
                    native = i->second.native.handle;
                    return true;
                }
                break;
            case Resource::Tag::BINDLESS_ARRAY:
                if (auto i = _bindless_arrays.find(remote);
                    i != _bindless_arrays.end()) {
                    native = i->second.native.handle;
                    return true;
                }
                break;
            case Resource::Tag::STREAM:
                if (auto i = _streams.find(remote); i != _streams.end()) {
                    native = i->second.native.handle;
                    return true;
                }
                break;
            case Resource::Tag::EVENT:
                if (auto i = _events.find(remote); i != _events.end()) {
                    native = i->second.native.handle;
                    return true;
                }
                break;
            case Resource::Tag::SHADER:
                if (auto i = _shaders.find(remote); i != _shaders.end()) {
                    native = i->second.native.handle;
                    return true;
                }
                break;
            case Resource::Tag::MESH:
                if (auto i = _meshes.find(remote); i != _meshes.end()) {
                    native = i->second.native.handle;
                    return true;
                }
                break;
            case Resource::Tag::CURVE:
                if (auto i = _curves.find(remote); i != _curves.end()) {
                    native = i->second.native.handle;
                    return true;
                }
                break;
            case Resource::Tag::PROCEDURAL_PRIMITIVE:
                if (auto i = _procedural_primitives.find(remote);
                    i != _procedural_primitives.end()) {
                    native = i->second.native.handle;
                    return true;
                }
                break;
            case Resource::Tag::MOTION_INSTANCE:
                if (auto i = _motion_instances.find(remote);
                    i != _motion_instances.end()) {
                    native = i->second.native.handle;
                    return true;
                }
                break;
            case Resource::Tag::ACCEL:
                if (auto i = _accels.find(remote); i != _accels.end()) {
                    native = i->second.native.handle;
                    return true;
                }
                break;
            default: break;
        }
        return false;
    }

    [[nodiscard]] Reply _set_name(
        luisa::span<const std::byte> payload) noexcept {
        Reader reader{payload, _options.protocol_limits};
        auto tag_value = reader.read_u32();
        auto remote = reader.read_u64();
        auto name = reader.read_string();
        if (!reader.finish()) { return invalid(luisa::string{reader.error()}); }
        if (tag_value > static_cast<uint32_t>(Resource::Tag::TENSOR_GRAPH)) {
            return invalid("Remote resource tag is invalid.");
        }
        auto tag = static_cast<Resource::Tag>(tag_value);
        uint64_t native{};
        if (!_resolve_resource_for_name(tag, remote, native)) {
            return not_found("Remote named resource was not found or unsupported.");
        }
        _native->set_name(tag, native, name);
        return {};
    }

    [[nodiscard]] Reply _set_stream_log_callback(
        luisa::span<const std::byte> payload) noexcept {
        Reader reader{payload, _options.protocol_limits};
        auto remote_stream = reader.read_u64();
        auto enabled = reader.read_bool();
        if (!reader.finish()) { return invalid(luisa::string{reader.error()}); }
        auto stream = _streams.find(remote_stream);
        if (stream == _streams.end()) { return not_found("Remote stream was not found."); }
        if (!enabled) {
            if (stream->second.log_callback_installed) {
                _native->set_stream_log_callback(stream->second.native.handle, {});
                stream->second.log_callback_installed = false;
            }
            return {};
        }
        auto weak = luisa::weak_ptr<Session>{shared_from_this()};
        auto max_string_size = _options.protocol_limits.max_string_size;
        _native->set_stream_log_callback(
            stream->second.native.handle,
            [weak = std::move(weak), remote_stream,
             max_string_size](luisa::string_view message) noexcept {
                if (auto self = weak.lock()) {
                    if (message.size() > max_string_size) {
                        message = message.substr(0u, static_cast<size_t>(max_string_size));
                    }
                    Writer writer;
                    writer.write_u64(remote_stream);
                    writer.write_string(message);
                    static_cast<void>(self->_send_frame(
                        MessageKind::STREAM_LOG, remote_stream, writer.bytes()));
                }
            });
        stream->second.log_callback_installed = true;
        return {};
    }

    [[nodiscard]] Reply _handle(
        MessageKind kind, luisa::span<const std::byte> payload) noexcept {
        if (kind == MessageKind::HELLO) { return _hello(payload); }
        if (!_handshake_complete) {
            return Reply{.status = Status::AUTHENTICATION_FAILED,
                         .message = "Remote HELLO is required before other requests."};
        }
        switch (kind) {
            case MessageKind::CREATE_BUFFER: return _create_buffer(payload);
            case MessageKind::DESTROY_BUFFER: return _destroy_buffer(payload);
            case MessageKind::CREATE_TEXTURE: return _create_texture(payload);
            case MessageKind::DESTROY_TEXTURE: return _destroy_texture(payload);
            case MessageKind::CREATE_BINDLESS_ARRAY: return _create_bindless_array(payload);
            case MessageKind::DESTROY_BINDLESS_ARRAY: return _destroy_bindless_array(payload);
            case MessageKind::CREATE_STREAM: return _create_stream(payload);
            case MessageKind::DESTROY_STREAM: return _destroy_stream(payload);
            case MessageKind::SYNCHRONIZE_STREAM: return _synchronize_stream(payload);
            case MessageKind::CREATE_SHADER: return _create_shader(payload);
            case MessageKind::DESTROY_SHADER: return _destroy_shader(payload);
            case MessageKind::SHADER_ARGUMENT_USAGE: return _shader_argument_usage(payload);
            case MessageKind::CREATE_EVENT: return _create_event(payload);
            case MessageKind::DESTROY_EVENT: return _destroy_event(payload);
            case MessageKind::SIGNAL_EVENT:
            case MessageKind::WAIT_EVENT: return _event_stream_operation(kind, payload);
            case MessageKind::IS_EVENT_COMPLETED:
            case MessageKind::SYNCHRONIZE_EVENT: return _event_query(kind, payload);
            case MessageKind::DISPATCH: return _dispatch(payload);
            case MessageKind::PREPARE_BLOBS: return _prepare_blobs(payload);
            case MessageKind::UPLOAD_BLOBS: return _upload_blobs(payload);
            case MessageKind::BLOB_CACHE_INFO: return _blob_cache_info(payload);
            case MessageKind::PROTOCOL_INFO: return _protocol_info(payload);
            case MessageKind::QUERY: return _query(payload);
            case MessageKind::SET_NAME: return _set_name(payload);
            case MessageKind::SET_STREAM_LOG_CALLBACK:
                return _set_stream_log_callback(payload);
            case MessageKind::CREATE_MESH:
            case MessageKind::CREATE_CURVE:
            case MessageKind::CREATE_PROCEDURAL_PRIMITIVE:
            case MessageKind::CREATE_MOTION_INSTANCE:
            case MessageKind::CREATE_ACCEL:
                return _create_accel_resource(kind, payload);
            case MessageKind::DESTROY_MESH:
            case MessageKind::DESTROY_CURVE:
            case MessageKind::DESTROY_PROCEDURAL_PRIMITIVE:
            case MessageKind::DESTROY_MOTION_INSTANCE:
            case MessageKind::DESTROY_ACCEL:
                return _destroy_accel_resource(kind, payload);
            case MessageKind::GOODBYE:
                _close_after_reply = true;
                return {};
            case MessageKind::LOAD_SHADER:
                return unsupported("Remote protocol v1 does not implement this resource yet.");
            default:
                return unsupported("Remote request kind is unsupported.");
        }
    }

    void _cleanup() noexcept {
        if (_native == nullptr) { return; }
        _blob_batches.clear();
        for (auto &[id, stream] : _streams) {
            static_cast<void>(id);
            _native->synchronize_stream(stream.native.handle);
            if (stream.log_callback_installed) {
                _native->set_stream_log_callback(stream.native.handle, {});
            }
        }
        for (auto &[id, shader] : _shaders) {
            static_cast<void>(id);
            _native->destroy_shader(shader.native.handle);
        }
        for (auto &[id, event] : _events) {
            static_cast<void>(id);
            _native->destroy_event(event.native.handle);
        }
        for (auto &[id, accel] : _accels) {
            static_cast<void>(id);
            _native->destroy_accel(accel.native.handle);
        }
        for (auto &[id, instance] : _motion_instances) {
            static_cast<void>(id);
            _native->destroy_motion_instance(instance.native.handle);
        }
        for (auto &[id, mesh] : _meshes) {
            static_cast<void>(id);
            _native->destroy_mesh(mesh.native.handle);
        }
        for (auto &[id, curve] : _curves) {
            static_cast<void>(id);
            _native->destroy_curve(curve.native.handle);
        }
        for (auto &[id, primitive] : _procedural_primitives) {
            static_cast<void>(id);
            _native->destroy_procedural_primitive(primitive.native.handle);
        }
        for (auto &[id, array] : _bindless_arrays) {
            static_cast<void>(id);
            _native->destroy_bindless_array(array.native.handle);
        }
        for (auto &[id, texture] : _textures) {
            static_cast<void>(id);
            _native->destroy_texture(texture.native.handle);
        }
        for (auto &[id, buffer] : _buffers) {
            static_cast<void>(id);
            _native->destroy_buffer(buffer.native.handle);
        }
        for (auto &[id, stream] : _streams) {
            static_cast<void>(id);
            _native->destroy_stream(stream.native.handle);
        }
        _shaders.clear();
        _events.clear();
        _accels.clear();
        _motion_instances.clear();
        _meshes.clear();
        _curves.clear();
        _procedural_primitives.clear();
        _bindless_arrays.clear();
        _textures.clear();
        _buffers.clear();
        _streams.clear();
    }

public:
    Session(Tcp::socket socket,
            const DeviceFactory &device_factory,
            luisa::shared_ptr<BlobCache> blob_cache,
            const ServerOptions &options,
            bool device_selection_enabled) noexcept
        : _socket{std::move(socket)},
          _device_factory{device_factory},
          _blob_cache{std::move(blob_cache)},
          _options{options},
          _device_selection_enabled{device_selection_enabled} {}

    ~Session() noexcept override = default;

    void stop() noexcept {
        _alive.store(false, std::memory_order_release);
        asio::error_code ignored;
        _socket.cancel(ignored);
        _socket.shutdown(Tcp::socket::shutdown_both, ignored);
        _socket.close(ignored);
    }

    void run() noexcept {
        while (_alive.load(std::memory_order_acquire)) {
            FrameHeader header;
            luisa::vector<std::byte> payload;
            if (!_read_frame(header, payload)) { break; }
            auto reply = _handle(header.kind, payload);
            if (!_send_reply(header.kind, header.request_id, std::move(reply))) {
                break;
            }
            if (_close_after_reply) { break; }
        }
        _alive.store(false, std::memory_order_release);
        asio::error_code ignored;
        _socket.shutdown(Tcp::socket::shutdown_both, ignored);
        _socket.close(ignored);
        _cleanup();
        _native.reset();
    }

    [[nodiscard]] bool resolve_buffer(
        uint64_t remote, uint64_t &native, size_t &size_bytes,
        luisa::string &error) const noexcept override {
        auto iter = _buffers.find(remote);
        if (iter == _buffers.end() || resource_kind(remote) != ResourceKind::BUFFER) {
            error = "Remote buffer handle was not found.";
            return false;
        }
        if (iter->second.is_indirect_dispatch) {
            error = "An indirect-dispatch buffer cannot be used as a byte buffer.";
            return false;
        }
        native = iter->second.native.handle;
        size_bytes = iter->second.size_bytes;
        return true;
    }

    [[nodiscard]] bool resolve_buffer_descriptor(
        uint64_t remote, ResolvedBuffer &buffer,
        luisa::string &error) const noexcept override {
        auto iter = _buffers.find(remote);
        if (iter == _buffers.end() ||
            resource_kind(remote) != ResourceKind::BUFFER) {
            error = "Remote buffer handle was not found.";
            return false;
        }
        buffer = ResolvedBuffer{
            .handle = iter->second.native.handle,
            .size_bytes = iter->second.size_bytes,
            .is_indirect_dispatch = iter->second.is_indirect_dispatch,
            .indirect_dispatch_capacity =
                iter->second.indirect_dispatch_capacity};
        return true;
    }

    [[nodiscard]] bool resolve_texture(
        uint64_t remote, uint64_t &native,
        luisa::string &error) const noexcept override {
        auto iter = _textures.find(remote);
        if (iter == _textures.end() || resource_kind(remote) != ResourceKind::TEXTURE) {
            error = "Remote texture handle was not found.";
            return false;
        }
        native = iter->second.native.handle;
        return true;
    }

    [[nodiscard]] bool resolve_texture_descriptor(
        uint64_t remote, ResolvedTexture &texture,
        luisa::string &error) const noexcept override {
        auto iter = _textures.find(remote);
        if (iter == _textures.end() || resource_kind(remote) != ResourceKind::TEXTURE) {
            error = "Remote texture handle was not found.";
            return false;
        }
        texture = ResolvedTexture{
            .handle = iter->second.native.handle,
            .storage = iter->second.storage,
            .size = iter->second.size,
            .mip_levels = iter->second.mip_levels};
        return true;
    }

    [[nodiscard]] bool resolve_bindless_array(
        uint64_t remote, uint64_t &native,
        luisa::string &error) const noexcept override {
        auto iter = _bindless_arrays.find(remote);
        if (iter == _bindless_arrays.end() ||
            resource_kind(remote) != ResourceKind::BINDLESS_ARRAY) {
            error = "Remote bindless-array handle was not found.";
            return false;
        }
        native = iter->second.native.handle;
        return true;
    }

    [[nodiscard]] bool resolve_bindless_array_for_update(
        uint64_t remote, uint64_t &native, size_t &slot_count,
        BindlessSlotType &slot_type,
        luisa::string &error) const noexcept override {
        auto iter = _bindless_arrays.find(remote);
        if (iter == _bindless_arrays.end() ||
            resource_kind(remote) != ResourceKind::BINDLESS_ARRAY) {
            error = "Remote bindless-array handle was not found.";
            return false;
        }
        native = iter->second.native.handle;
        slot_count = iter->second.slot_count;
        slot_type = iter->second.slot_type;
        return true;
    }

    [[nodiscard]] bool resolve_accel(
        uint64_t remote, uint64_t &native,
        luisa::string &error) const noexcept override {
        auto iter = _accels.find(remote);
        if (iter == _accels.end() || resource_kind(remote) != ResourceKind::ACCEL) {
            error = "Remote acceleration-structure handle was not found.";
            return false;
        }
        native = iter->second.native.handle;
        return true;
    }

    [[nodiscard]] bool resolve_mesh(
        uint64_t remote, uint64_t &native,
        luisa::string &error) const noexcept override {
        auto iter = _meshes.find(remote);
        if (iter == _meshes.end() || resource_kind(remote) != ResourceKind::MESH) {
            error = "Remote mesh handle was not found.";
            return false;
        }
        native = iter->second.native.handle;
        return true;
    }

    [[nodiscard]] bool resolve_curve(
        uint64_t remote, uint64_t &native,
        luisa::string &error) const noexcept override {
        auto iter = _curves.find(remote);
        if (iter == _curves.end() || resource_kind(remote) != ResourceKind::CURVE) {
            error = "Remote curve handle was not found.";
            return false;
        }
        native = iter->second.native.handle;
        return true;
    }

    [[nodiscard]] bool resolve_procedural_primitive(
        uint64_t remote, uint64_t &native,
        luisa::string &error) const noexcept override {
        auto iter = _procedural_primitives.find(remote);
        if (iter == _procedural_primitives.end() ||
            resource_kind(remote) != ResourceKind::PROCEDURAL_PRIMITIVE) {
            error = "Remote procedural-primitive handle was not found.";
            return false;
        }
        native = iter->second.native.handle;
        return true;
    }

    [[nodiscard]] bool resolve_motion_instance(
        uint64_t remote, uint64_t &native, size_t &keyframe_count,
        luisa::string &error) const noexcept override {
        auto iter = _motion_instances.find(remote);
        if (iter == _motion_instances.end() ||
            resource_kind(remote) != ResourceKind::MOTION_INSTANCE) {
            error = "Remote motion-instance handle was not found.";
            return false;
        }
        native = iter->second.native.handle;
        keyframe_count = iter->second.keyframe_count;
        return true;
    }

    [[nodiscard]] bool resolve_accel_resource(
        uint64_t remote, uint64_t &native,
        luisa::string &error) const noexcept override {
        return resolve_accel(remote, native, error);
    }

    [[nodiscard]] bool resolve_primitive(
        uint64_t remote, uint64_t &native,
        luisa::string &error) const noexcept override {
        switch (resource_kind(remote)) {
            case ResourceKind::MESH:
                return resolve_mesh(remote, native, error);
            case ResourceKind::CURVE:
                return resolve_curve(remote, native, error);
            case ResourceKind::PROCEDURAL_PRIMITIVE:
                return resolve_procedural_primitive(remote, native, error);
            case ResourceKind::MOTION_INSTANCE: {
                size_t unused{};
                return resolve_motion_instance(remote, native, unused, error);
            }
            default:
                error = "Remote primitive handle has an invalid resource kind.";
                return false;
        }
    }

    [[nodiscard]] bool resolve_upload_blob(
        uint64_t submission_id, uint32_t blob_index,
        size_t expected_size, BlobCache::BlobPtr &blob,
        luisa::string &error) const noexcept override {
        auto batch = _blob_batches.find(submission_id);
        if (batch == _blob_batches.end() ||
            blob_index >= batch->second.blobs.size()) {
            error = "Remote cached upload references an unknown blob batch or index.";
            return false;
        }
        auto &&key = batch->second.keys[blob_index];
        auto &&cached = batch->second.blobs[blob_index];
        if (cached == nullptr) {
            error = "Remote cached upload references a blob body that was not uploaded.";
            return false;
        }
        if (key.size != expected_size ||
            cached->size() != expected_size) {
            error = "Remote cached upload failed size verification.";
            return false;
        }
        blob = cached;
        return true;
    }

    [[nodiscard]] bool resolve_shader(
        uint64_t remote, ResolvedShader &shader,
        luisa::string &error) const noexcept override {
        auto iter = _shaders.find(remote);
        if (iter == _shaders.end() || resource_kind(remote) != ResourceKind::SHADER) {
            error = "Remote shader handle was not found.";
            return false;
        }
        shader = ResolvedShader{
            .handle = iter->second.native.handle,
            .arguments = iter->second.arguments};
        return true;
    }

    [[nodiscard]] bool resolve_buffer(
        const Type *serialized_type, uint64_t serialized_handle,
        size_t serialized_offset,
        size_t serialized_size, Function::BufferBinding &binding,
        luisa::string &error) const noexcept override {
        ResolvedBuffer buffer;
        if (!resolve_buffer_descriptor(
                serialized_handle, buffer, error)) {
            return false;
        }
        auto expected_indirect = serialized_type->is_custom() &&
                                 serialized_type->description() ==
                                     ast_json_indirect_dispatch_buffer_type_name;
        if (buffer.is_indirect_dispatch != expected_indirect) {
            error = "Remote AST buffer kind does not match its serialized type.";
            return false;
        }
        auto total = expected_indirect ?
                         buffer.indirect_dispatch_capacity :
                         buffer.size_bytes;
        if (!valid_range(serialized_offset, serialized_size, total)) {
            error = "Remote AST buffer binding is out of bounds.";
            return false;
        }
        binding = Function::BufferBinding{
            buffer.handle, serialized_offset, serialized_size};
        return true;
    }

    [[nodiscard]] bool resolve_texture(
        uint64_t serialized_handle, uint32_t serialized_level,
        Function::TextureBinding &binding,
        luisa::string &error) const noexcept override {
        auto iter = _textures.find(serialized_handle);
        if (iter == _textures.end() || serialized_level >= iter->second.mip_levels) {
            error = "Remote AST texture binding is invalid.";
            return false;
        }
        binding = Function::TextureBinding{
            iter->second.native.handle, serialized_level};
        return true;
    }

    [[nodiscard]] bool resolve_bindless_array(
        uint64_t serialized_handle,
        Function::BindlessArrayBinding &binding,
        luisa::string &error) const noexcept override {
        uint64_t native{};
        if (!resolve_bindless_array(serialized_handle, native, error)) { return false; }
        binding = Function::BindlessArrayBinding{native};
        return true;
    }

    [[nodiscard]] bool resolve_accel(
        uint64_t serialized_handle, Function::AccelBinding &binding,
        luisa::string &error) const noexcept override {
        uint64_t native{};
        if (!resolve_accel(serialized_handle, native, error)) { return false; }
        binding = Function::AccelBinding{native};
        return true;
    }
};

class Server::Impl {

private:
    struct SessionWorker {
        luisa::shared_ptr<Session> session;
        luisa::shared_ptr<std::atomic_bool> done;
        std::thread thread;
    };

    asio::io_context _io;
    Tcp::acceptor _acceptor;
    DeviceFactory _device_factory;
    ServerOptions _options;
    luisa::shared_ptr<BlobCache> _blob_cache;
    std::atomic_bool _stopping{false};
    std::atomic_bool _running{false};
    std::mutex _sessions_mutex;
    luisa::vector<luisa::unique_ptr<SessionWorker>> _sessions;
    uint16_t _port{};
    bool _ipv6{};
    bool _device_selection_enabled{};

private:
    void _reap_finished_sessions() noexcept {
        luisa::vector<luisa::unique_ptr<SessionWorker>> finished;
        {
            std::scoped_lock lock{_sessions_mutex};
            for (auto i = _sessions.size(); i != 0u;) {
                i--;
                if (_sessions[i]->done->load(std::memory_order_acquire)) {
                    finished.emplace_back(std::move(_sessions[i]));
                    _sessions.erase(_sessions.begin() + i);
                }
            }
        }
        for (auto &worker : finished) {
            if (worker->thread.joinable()) { worker->thread.join(); }
        }
    }

    void _stop_and_join_sessions() noexcept {
        luisa::vector<luisa::unique_ptr<SessionWorker>> sessions;
        {
            std::scoped_lock lock{_sessions_mutex};
            sessions = std::move(_sessions);
            _sessions.clear();
        }
        for (auto &worker : sessions) { worker->session->stop(); }
        for (auto &worker : sessions) {
            if (worker->thread.joinable()) { worker->thread.join(); }
        }
    }

public:
    Impl(DeviceFactory device_factory, ServerOptions options,
         bool device_selection_enabled)
        : _acceptor{_io},
          _device_factory{std::move(device_factory)},
          _options{std::move(options)},
          _device_selection_enabled{device_selection_enabled} {
        if (!_device_factory) {
            throw std::invalid_argument{"Remote server requires a device factory."};
        }
        if (_options.protocol_limits.max_frame_payload < 256u ||
            _options.protocol_limits.max_string_size == 0u ||
            _options.protocol_limits.max_array_size == 0u ||
            _options.max_concurrent_sessions == 0u ||
            _options.max_concurrent_sessions > 4096u) {
            throw std::invalid_argument{
                "Remote server protocol or session limits are invalid."};
        }
        constexpr uint64_t blob_upload_frame_overhead = 68u;
        constexpr uint64_t blob_prepare_header_size = 16u;
        constexpr uint64_t blob_descriptor_size =
            sizeof(uint64_t) + blob_digest_size;
        auto frame_blob_limit =
            _options.protocol_limits.max_frame_payload >
                    blob_upload_frame_overhead ?
                _options.protocol_limits.max_frame_payload -
                    blob_upload_frame_overhead :
                0u;
        _options.max_blob_entry_size = std::min(
            {_options.max_blob_entry_size,
             _options.max_blob_cache_bytes,
             _options.max_resource_size,
             frame_blob_limit});
        auto frame_blob_count_limit =
            (_options.protocol_limits.max_frame_payload -
             std::min(blob_prepare_header_size,
                      _options.protocol_limits.max_frame_payload)) /
            blob_descriptor_size;
        _options.max_blobs_per_batch = std::min(
            {_options.max_blobs_per_batch,
             _options.protocol_limits.max_array_size,
             frame_blob_count_limit,
             static_cast<uint64_t>(
                 std::numeric_limits<uint32_t>::max())});
        if (_options.max_blob_cache_bytes == 0u ||
            _options.max_blob_entry_size == 0u ||
            _options.blob_cache_min_size >
                _options.max_blob_entry_size ||
            _options.max_blobs_per_batch == 0u ||
            _options.max_prepared_blob_batches == 0u) {
            _options.max_blob_entry_size = 0u;
        } else {
            _blob_cache = luisa::make_shared<BlobCache>(
                _options.max_blob_cache_bytes);
        }
        asio::error_code error;
        auto address = asio::ip::make_address(_options.listen_address, error);
        if (error) {
            throw std::invalid_argument{luisa::format(
                "Invalid remote listen address '{}': {}",
                _options.listen_address, error.message())};
        }
        _ipv6 = address.is_v6();
        auto endpoint = Tcp::endpoint{address, _options.port};
        _acceptor.open(endpoint.protocol());
        _acceptor.set_option(Tcp::acceptor::reuse_address{true});
        _acceptor.bind(endpoint);
        _acceptor.listen();
        _port = _acceptor.local_endpoint().port();
    }

    ~Impl() noexcept { stop(); }

    void run() {
        if (_running.exchange(true, std::memory_order_acq_rel)) {
            throw std::logic_error{"Remote server is already running."};
        }
        while (!_stopping.load(std::memory_order_acquire)) {
            Tcp::socket socket{_io};
            asio::error_code error;
            _acceptor.accept(socket, error);
            if (error) {
                if (_stopping.load(std::memory_order_acquire)) { break; }
                if (error == asio::error::interrupted) { continue; }
                LUISA_WARNING("Remote accept failed: {}", error.message());
                continue;
            }
            if (_stopping.load(std::memory_order_acquire)) {
                socket.close(error);
                break;
            }
            _reap_finished_sessions();
            {
                std::scoped_lock lock{_sessions_mutex};
                if (_sessions.size() >= _options.max_concurrent_sessions) {
                    LUISA_WARNING(
                        "Rejecting remote connection: concurrent-session limit ({}) reached.",
                        _options.max_concurrent_sessions);
                    socket.shutdown(Tcp::socket::shutdown_both, error);
                    socket.close(error);
                    continue;
                }
            }
            auto session = luisa::make_shared<Session>(
                std::move(socket), _device_factory, _blob_cache, _options,
                _device_selection_enabled);
            auto worker = luisa::make_unique<SessionWorker>();
            worker->session = session;
            worker->done = luisa::make_shared<std::atomic_bool>(false);
            try {
                worker->thread = std::thread{
                    [session = std::move(session), done = worker->done]() noexcept {
                        session->run();
                        done->store(true, std::memory_order_release);
                    }};
            } catch (const std::exception &exception) {
                LUISA_WARNING(
                    "Failed to start remote session worker: {}", exception.what());
                worker->session->stop();
                continue;
            } catch (...) {
                LUISA_WARNING("Failed to start remote session worker.");
                worker->session->stop();
                continue;
            }
            {
                std::scoped_lock lock{_sessions_mutex};
                _sessions.emplace_back(std::move(worker));
            }
        }
        _stop_and_join_sessions();
        _running.store(false, std::memory_order_release);
    }

    void stop() noexcept {
        if (_stopping.exchange(true, std::memory_order_acq_rel)) { return; }
        // Wake a blocking accept without depending on platform-specific
        // acceptor cancellation behavior.
        asio::io_context wake_io;
        Tcp::socket wake_socket{wake_io};
        asio::error_code ignored;
        auto loopback = _ipv6 ?
                            asio::ip::address{asio::ip::address_v6::loopback()} :
                            asio::ip::address{asio::ip::address_v4::loopback()};
        wake_socket.connect(Tcp::endpoint{loopback, _port}, ignored);
        wake_socket.close(ignored);
        _acceptor.close(ignored);
        luisa::vector<luisa::shared_ptr<Session>> sessions;
        {
            std::scoped_lock lock{_sessions_mutex};
            sessions.reserve(_sessions.size());
            for (auto &worker : _sessions) {
                sessions.emplace_back(worker->session);
            }
        }
        for (auto &&session : sessions) { session->stop(); }
    }

    [[nodiscard]] uint16_t port() const noexcept { return _port; }
};

Server::Server(luisa::shared_ptr<DeviceInterface> native_device,
               ServerOptions options)
    : _impl{std::make_unique<Impl>(
          [native = std::move(native_device)](
              const DeviceRequest &request,
              luisa::string &error) -> luisa::shared_ptr<DeviceInterface> {
              if (native == nullptr) {
                  error = "Remote server has no native device.";
                  return nullptr;
              }
              if ((!request.backend.empty() &&
                   request.backend != native->backend_name()) ||
                  request.device_index !=
                      std::numeric_limits<size_t>::max() ||
                  request.enable_validation) {
                  error = "This remote server uses a fixed native device and cannot honor device-selection parameters.";
                  return nullptr;
              }
              return native;
          },
          std::move(options), false)} {}

Server::Server(DeviceFactory device_factory, ServerOptions options)
    : _impl{std::make_unique<Impl>(
          std::move(device_factory), std::move(options), true)} {}

Server::~Server() noexcept = default;

void Server::run() { _impl->run(); }

void Server::stop() noexcept { _impl->stop(); }

uint16_t Server::port() const noexcept { return _impl->port(); }

}// namespace luisa::compute::remote
