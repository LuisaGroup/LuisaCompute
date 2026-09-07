#include "remote_device.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <luisa/ast/ast2json.h>
#include <luisa/backends/ext/remote_config_ext.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/dispatch_buffer.h>
#include <luisa/runtime/swapchain.h>

#include "remote_command_codec.h"
#include "remote_transport.h"

namespace luisa::compute::remote {

namespace {

using namespace std::chrono_literals;

struct ClientOptions {
    luisa::string host{"127.0.0.1"};
    uint16_t port{18080u};
    luisa::string token;
    std::chrono::milliseconds connect_timeout{10s};
    std::chrono::milliseconds request_timeout{60s};
    uint64_t max_in_flight_bytes{256ull * 1024ull * 1024ull};
    bool enable_blob_cache{true};
    uint64_t blob_cache_min_size{64ull * 1024ull};
    luisa::string local_present_backend;
    luisa::string server_backend;
    size_t server_device_index{std::numeric_limits<size_t>::max()};
    bool server_enable_validation{false};
};

[[nodiscard]] ClientOptions client_options(
    const DeviceConfig *config) noexcept {
    ClientOptions options;
    if (config != nullptr && config->extension != nullptr) {
        auto extension = static_cast<const RemoteDeviceConfigExt *>(
            config->extension.get());
        options.host = extension->host();
        options.port = extension->port();
        options.token = extension->token();
        options.connect_timeout = std::chrono::milliseconds{
            extension->connect_timeout_ms()};
        options.request_timeout = std::chrono::milliseconds{
            extension->request_timeout_ms()};
        options.max_in_flight_bytes = extension->max_in_flight_bytes();
        options.enable_blob_cache = extension->enable_blob_cache();
        options.blob_cache_min_size = extension->blob_cache_min_size();
        options.local_present_backend = extension->local_present_backend();
        options.server_backend = extension->server_backend();
        options.server_device_index = extension->server_device_index();
        options.server_enable_validation = extension->server_enable_validation();
    }
    return options;
}

void write_shader_option(Writer &writer,
                         const ShaderOption &option) noexcept {
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

void write_accel_option(Writer &writer,
                        const AccelOption &option) noexcept {
    writer.write_u32(static_cast<uint32_t>(option.hint));
    writer.write_bool(option.allow_compaction);
    writer.write_bool(option.allow_update);
    writer.write_u32(option.motion.keyframe_count);
    writer.write_f32(option.motion.time_start);
    writer.write_f32(option.motion.time_end);
    writer.write_bool(option.motion.should_vanish_start);
    writer.write_bool(option.motion.should_vanish_end);
    writer.write_u8(static_cast<uint8_t>(option.motion.mode));
}

}// namespace

class RemoteDevice::Impl {

private:
    struct TextureDesc {
        PixelStorage storage{};
        uint3 size{};
        uint32_t mip_levels{};
    };

    struct LocalSwapchain {
        uint64_t remote_stream{};
        luisa::shared_ptr<DeviceInterface> device;
        ResourceCreationInfo stream{};
        SwapchainCreationInfo swapchain{};
        ResourceCreationInfo image{};
        uint2 size{};
        luisa::vector<std::byte> staging;
        bool submission_pending{};
    };

    struct PendingSubmission {
        uint64_t stream_handle{};
        luisa::vector<DownloadTarget> downloads;
        CommandList::CallbackContainer callbacks;
        uint64_t footprint{};
    };

    struct WorkItem {
        MessageKind kind{};
        uint64_t id{};
        luisa::vector<std::byte> payload;
    };

    struct CompletionDownload {
        uint32_t index{};
        luisa::span<const std::byte> bytes;
    };

    ProtocolLimits _limits;
    Connection _connection{_limits};
    Context _context;
    ClientOptions _options;
    std::chrono::milliseconds _request_timeout;
    uint64_t _max_in_flight_bytes{};
    uint _warp_size{};
    size_t _max_shared_memory{};
    uint64_t _memory_granularity{};
    uint64_t _features{};
    uint64_t _max_blob_entry_size{};
    uint64_t _blob_cache_min_size{};
    uint64_t _max_blobs_per_batch{};
    luisa::string _native_backend;
    std::atomic_uint64_t _next_submission_id{1u};

    std::mutex _state_mutex;
    std::condition_variable _state_cv;
    std::unordered_map<uint64_t, PendingSubmission> _pending;
    std::unordered_map<uint64_t, luisa::vector<Usage>> _shader_usages;
    std::unordered_map<uint64_t, DeviceInterface::StreamLogCallback> _log_callbacks;
    std::unordered_map<uint64_t, TextureDesc> _textures;
    uint64_t _pending_bytes{};
    bool _closed{false};

    std::mutex _present_mutex;
    luisa::shared_ptr<DeviceInterface> _present_device;
    luisa::string _present_backend;
    std::unordered_map<uint64_t, luisa::unique_ptr<LocalSwapchain>> _swapchains;
    uint64_t _next_swapchain_id{1u};

    std::mutex _work_mutex;
    std::condition_variable _work_cv;
    std::deque<WorkItem> _work;
    std::thread _worker;
    bool _worker_stop{false};

private:
    [[nodiscard]] luisa::string _select_present_backend() const noexcept {
        if (!_options.local_present_backend.empty()) {
            if (_options.local_present_backend == "remote") {
                LUISA_ERROR("The local presentation backend cannot be 'remote'.");
            }
            return _options.local_present_backend;
        }
        auto installed = _context.installed_backends();
        auto has_backend = [installed](luisa::string_view name) noexcept {
            return std::find(installed.begin(), installed.end(), name) != installed.end();
        };
#if defined(__APPLE__)
        constexpr std::array candidates{"metal", "metal4", "vk", "fallback"};
#elif defined(_WIN32)
        constexpr std::array candidates{"dx", "vk", "fallback"};
#else
        constexpr std::array candidates{"vk", "fallback"};
#endif
        for (auto candidate : candidates) {
            if (has_backend(candidate)) { return candidate; }
        }
        LUISA_ERROR("No local presentation backend is installed. Configure RemoteDeviceConfigExt::local_present_backend explicitly.");
    }

    void _ensure_present_device_locked() noexcept {
        if (_present_device != nullptr) { return; }
        _present_backend = _select_present_backend();
        DeviceConfig config{};
        config.headless = false;
        auto device = _context.create_device(_present_backend, &config, false);
        _present_device = device.impl_shared();
        if (_present_device == nullptr) {
            LUISA_ERROR("Failed to create local '{}' presentation device.", _present_backend);
        }
        LUISA_INFO("Remote client presentation uses the local '{}' backend.", _present_backend);
    }

    static void _destroy_local_swapchain(LocalSwapchain &swapchain) noexcept {
        if (swapchain.device == nullptr) { return; }
        if (swapchain.submission_pending && swapchain.stream.valid()) {
            swapchain.device->synchronize_stream(swapchain.stream.handle);
        }
        if (swapchain.image.valid()) {
            swapchain.device->destroy_texture(swapchain.image.handle);
            swapchain.image.invalidate();
        }
        if (swapchain.swapchain.valid()) {
            swapchain.device->destroy_swapchain(swapchain.swapchain.handle);
            swapchain.swapchain.invalidate();
        }
        if (swapchain.stream.valid()) {
            swapchain.device->destroy_stream(swapchain.stream.handle);
            swapchain.stream.invalidate();
        }
    }

    [[nodiscard]] Response _request(
        MessageKind kind,
        luisa::span<const std::byte> payload = {}) noexcept {
        if (payload.size() > _limits.max_frame_payload) {
            LUISA_ERROR(
                "Remote request {} requires {} payload bytes, exceeding the negotiated limit {}.",
                static_cast<uint16_t>(kind), payload.size(),
                _limits.max_frame_payload);
        }
        auto response = _connection.request(kind, payload, _request_timeout);
        if (!response) {
            LUISA_ERROR(
                "Remote request {} failed (status {}): {}",
                static_cast<uint16_t>(kind),
                static_cast<uint16_t>(response.status),
                response.message);
        }
        return response;
    }

    void _protocol_failure(luisa::string_view message) noexcept {
        LUISA_WARNING("Remote protocol failure: {}", message);
        _connection.close();
    }

    void _enqueue_work(
        MessageKind kind, uint64_t id,
        luisa::span<const std::byte> payload) noexcept {
        WorkItem item{.kind = kind, .id = id};
        item.payload.assign(payload.begin(), payload.end());
        {
            std::scoped_lock lock{_work_mutex};
            if (_worker_stop) { return; }
            _work.emplace_back(std::move(item));
        }
        _work_cv.notify_one();
    }

    void _process_completion(const WorkItem &item) noexcept {
        Reader reader{item.payload, _limits};
        auto submission_id = reader.read_u64();
        auto status = static_cast<Status>(reader.read_u16());
        auto message = reader.read_string();
        auto download_count = reader.read_u64();
        if (!reader.ok() || submission_id != item.id ||
            submission_id == 0u ||
            download_count > _limits.max_array_size ||
            download_count > std::numeric_limits<size_t>::max()) {
            _protocol_failure(reader.ok() ?
                                  "Malformed remote completion header." :
                                  reader.error());
            return;
        }
        luisa::vector<CompletionDownload> downloads;
        downloads.reserve(static_cast<size_t>(download_count));
        for (uint64_t i = 0u; i < download_count; i++) {
            auto index = reader.read_u32();
            auto bytes = reader.read_blob();
            if (!reader.ok()) {
                _protocol_failure(reader.error());
                return;
            }
            downloads.emplace_back(CompletionDownload{index, bytes});
        }
        if (!reader.finish()) {
            _protocol_failure(reader.error());
            return;
        }
        if (status != Status::OK) {
            _protocol_failure(luisa::format(
                "Remote submission {} failed: {}", submission_id, message));
            return;
        }

        PendingSubmission pending;
        {
            std::unique_lock lock{_state_mutex};
            auto iter = _pending.find(submission_id);
            if (iter == _pending.end()) {
                lock.unlock();
                _protocol_failure("Completion references an unknown submission ID.");
                return;
            }
            if (downloads.size() != iter->second.downloads.size()) {
                lock.unlock();
                _protocol_failure("Completion download count does not match the submission.");
                return;
            }
            for (size_t i = 0u; i < downloads.size(); i++) {
                if (downloads[i].index != i ||
                    downloads[i].bytes.size() != iter->second.downloads[i].size) {
                    lock.unlock();
                    _protocol_failure("Completion download descriptor does not match the submission.");
                    return;
                }
            }
            pending = std::move(iter->second);
            _pending_bytes -= pending.footprint;
            _pending.erase(iter);
        }
        _state_cv.notify_all();
        for (size_t i = 0u; i < downloads.size(); i++) {
            if (!downloads[i].bytes.empty()) {
                std::memcpy(pending.downloads[i].data,
                            downloads[i].bytes.data(),
                            downloads[i].bytes.size());
            }
        }
        for (auto &callback : pending.callbacks) { callback(); }
    }

    void _process_stream_log(const WorkItem &item) noexcept {
        Reader reader{item.payload, _limits};
        auto stream = reader.read_u64();
        auto message = reader.read_string();
        if (!reader.finish() || stream != item.id) {
            _protocol_failure(reader.ok() ?
                                  "Malformed remote stream-log notification." :
                                  reader.error());
            return;
        }
        DeviceInterface::StreamLogCallback callback;
        {
            std::scoped_lock lock{_state_mutex};
            if (auto iter = _log_callbacks.find(stream);
                iter != _log_callbacks.end()) {
                callback = iter->second;
            }
        }
        if (callback) { callback(message); }
    }

    void _worker_loop() noexcept {
        while (true) {
            WorkItem item;
            {
                std::unique_lock lock{_work_mutex};
                _work_cv.wait(lock, [&]() noexcept {
                    return _worker_stop || !_work.empty();
                });
                if (_work.empty()) {
                    if (_worker_stop) { return; }
                    continue;
                }
                item = std::move(_work.front());
                _work.pop_front();
            }
            if (item.kind == MessageKind::DISPATCH_COMPLETE) {
                _process_completion(item);
            } else if (item.kind == MessageKind::STREAM_LOG) {
                _process_stream_log(item);
            }
        }
    }

    void _handshake() noexcept {
        if constexpr (std::endian::native != std::endian::little) {
            LUISA_ERROR(
                "Remote protocol v1 requires a little-endian client because AST constants use native byte order.");
        }
        Writer writer;
        writer.write_u32(0x01020304u);
        writer.write_u8(sizeof(void *));
        writer.write_u8(1u);
        writer.write_u16(0u);
        writer.write_string(_options.token);
        writer.write_string(_options.server_backend);
        writer.write_u64(_options.server_device_index);
        writer.write_bool(_options.server_enable_validation);
        auto response = _request(MessageKind::HELLO, writer.bytes());
        Reader reader{response.body, _limits};
        _native_backend = reader.read_string();
        _warp_size = reader.read_u32();
        auto max_shared_memory = reader.read_u64();
        _memory_granularity = reader.read_u64();
        _features = reader.read_u64();
        if (!reader.finish() ||
            max_shared_memory > std::numeric_limits<size_t>::max() ||
            _warp_size == 0u || _memory_granularity == 0u) {
            LUISA_ERROR("Remote HELLO response is malformed: {}",
                        reader.error());
        }
        _max_shared_memory = static_cast<size_t>(max_shared_memory);
        if ((_features & static_cast<uint64_t>(
                             Feature::LIMIT_NEGOTIATION)) != 0u) {
            auto protocol_info = _request(MessageKind::PROTOCOL_INFO);
            Reader protocol_reader{protocol_info.body, _limits};
            auto max_frame_payload = protocol_reader.read_u64();
            auto max_string_size = protocol_reader.read_u64();
            auto max_array_size = protocol_reader.read_u64();
            if (!protocol_reader.finish() || max_frame_payload == 0u ||
                max_string_size == 0u || max_array_size == 0u) {
                LUISA_ERROR("Remote protocol-limit response is malformed: {}",
                            protocol_reader.error());
            }
            _limits.max_frame_payload = std::min(
                _limits.max_frame_payload, max_frame_payload);
            _limits.max_string_size = std::min(
                _limits.max_string_size, max_string_size);
            _limits.max_array_size = std::min(
                _limits.max_array_size, max_array_size);
        }
        if (!_options.enable_blob_cache ||
            (_features & static_cast<uint64_t>(Feature::BLOB_CACHE)) == 0u) {
            _features &= ~static_cast<uint64_t>(Feature::BLOB_CACHE);
            return;
        }
        auto cache_info = _request(MessageKind::BLOB_CACHE_INFO);
        Reader cache_reader{cache_info.body, _limits};
        _max_blob_entry_size = cache_reader.read_u64();
        _blob_cache_min_size = cache_reader.read_u64();
        _max_blobs_per_batch = cache_reader.read_u64();
        if (!cache_reader.finish() || _max_blob_entry_size == 0u ||
            _max_blobs_per_batch == 0u ||
            _blob_cache_min_size > _max_blob_entry_size) {
            LUISA_ERROR("Remote blob-cache capability response is malformed: {}",
                        cache_reader.error());
        }
        _blob_cache_min_size = std::max(
            _blob_cache_min_size,
            _options.blob_cache_min_size);
        if (_blob_cache_min_size > _max_blob_entry_size) {
            _features &= ~static_cast<uint64_t>(Feature::BLOB_CACHE);
            _max_blob_entry_size = 0u;
        }
    }

    void _prepare_blobs(
        uint64_t submission_id,
        const UploadBlobPlan &plan) noexcept {
        if (plan.blobs.empty()) { return; }
        Writer prepare;
        prepare.write_u64(submission_id);
        prepare.write_u64(plan.blobs.size());
        for (auto &&blob : plan.blobs) {
            write_blob_key(prepare, blob.key);
        }
        auto response = _request(
            MessageKind::PREPARE_BLOBS, prepare.bytes());
        Reader reader{response.body, _limits};
        if (reader.read_u64() != submission_id) {
            LUISA_ERROR("Remote blob-prepare acknowledgement has the wrong submission ID.");
        }
        auto miss_count = reader.read_u64();
        if (!reader.ok() || miss_count > plan.blobs.size() ||
            miss_count > std::numeric_limits<size_t>::max()) {
            LUISA_ERROR("Remote blob-prepare acknowledgement is malformed: {}",
                        reader.error());
        }
        luisa::vector<uint32_t> misses;
        misses.reserve(static_cast<size_t>(miss_count));
        luisa::vector<bool> seen(plan.blobs.size(), false);
        for (uint64_t i = 0u; i < miss_count; i++) {
            auto index = reader.read_u32();
            if (!reader.ok() || index >= plan.blobs.size() || seen[index]) {
                LUISA_ERROR("Remote blob-prepare miss list is malformed.");
            }
            seen[index] = true;
            misses.emplace_back(index);
        }
        if (!reader.finish()) {
            LUISA_ERROR("Remote blob-prepare acknowledgement is malformed: {}",
                        reader.error());
        }

        constexpr uint64_t upload_header_size = 16u;
        constexpr uint64_t upload_record_overhead =
            sizeof(uint32_t) + sizeof(uint64_t) +
            blob_digest_size + sizeof(uint64_t);
        for (size_t begin = 0u; begin < misses.size();) {
            auto end = begin;
            auto payload_size = upload_header_size;
            while (end < misses.size()) {
                auto blob_size = plan.blobs[misses[end]].key.size;
                if (blob_size > _limits.max_frame_payload ||
                    upload_record_overhead >
                        _limits.max_frame_payload - blob_size ||
                    payload_size >
                        _limits.max_frame_payload -
                            upload_record_overhead - blob_size) {
                    break;
                }
                payload_size += upload_record_overhead + blob_size;
                end++;
            }
            if (end == begin) {
                LUISA_ERROR("A remote blob upload exceeds the frame limit.");
            }
            Writer upload{static_cast<size_t>(payload_size)};
            upload.write_u64(submission_id);
            upload.write_u64(end - begin);
            for (auto i = begin; i < end; i++) {
                auto index = misses[i];
                auto &&blob = plan.blobs[index];
                upload.write_u32(index);
                write_blob_key(upload, blob.key);
                upload.write_blob(blob.bytes);
            }
            auto uploaded = _request(
                MessageKind::UPLOAD_BLOBS, upload.bytes());
            Reader uploaded_reader{uploaded.body, _limits};
            if (uploaded_reader.read_u64() != submission_id ||
                !uploaded_reader.finish()) {
                LUISA_ERROR("Remote blob-upload acknowledgement is malformed.");
            }
            begin = end;
        }
    }

    [[nodiscard]] uint64_t _create_handle(
        MessageKind kind, const Writer &writer) noexcept {
        auto response = _request(kind, writer.bytes());
        Reader reader{response.body, _limits};
        auto handle = reader.read_u64();
        if (!reader.finish() || handle == invalid_resource_handle) {
            LUISA_ERROR("Remote create response is malformed: {}", reader.error());
        }
        return handle;
    }

    [[nodiscard]] bool _destroy_handle(
        MessageKind kind, uint64_t handle) noexcept {
        if (!_connection.connected()) { return false; }
        Writer writer;
        writer.write_u64(handle);
        auto response = _connection.request(
            kind, writer.bytes(), _request_timeout);
        if (!response) {
            if (response.status != Status::CONNECTION_CLOSED) {
                LUISA_WARNING(
                    "Failed to destroy remote resource {} with request {}: {}",
                    handle, static_cast<uint16_t>(kind), response.message);
            }
            return false;
        }
        return true;
    }

    [[nodiscard]] bool _try_wait_stream_idle(
        uint64_t stream_handle) noexcept {
        std::unique_lock lock{_state_mutex};
        auto idle = _state_cv.wait_for(
            lock, _request_timeout, [&]() noexcept {
                if (_closed) { return true; }
                for (auto &&[id, submission] : _pending) {
                    static_cast<void>(id);
                    if (submission.stream_handle == stream_handle) {
                        return false;
                    }
                }
                return true;
            });
        return idle && !_closed;
    }

    void _wait_stream_idle(uint64_t stream_handle) noexcept {
        if (!_try_wait_stream_idle(stream_handle)) {
            LUISA_ERROR("Timed out draining remote stream completions.");
        }
    }

    void _synchronize_remote_stream(uint64_t handle) noexcept {
        Writer writer;
        writer.write_u64(handle);
        static_cast<void>(_request(
            MessageKind::SYNCHRONIZE_STREAM, writer.bytes()));
        _wait_stream_idle(handle);
    }

public:
    explicit Impl(Context context, ClientOptions options) noexcept
        : _context{std::move(context)},
          _options{std::move(options)},
          _request_timeout{_options.request_timeout},
          _max_in_flight_bytes{_options.max_in_flight_bytes} {
        _connection.set_notification_handler(
            [this](MessageKind kind, uint64_t id,
                   luisa::span<const std::byte> payload) noexcept {
                if (kind != MessageKind::DISPATCH_COMPLETE &&
                    kind != MessageKind::STREAM_LOG) {
                    _protocol_failure("Received an unknown remote notification.");
                    return;
                }
                _enqueue_work(kind, id, payload);
            });
        _connection.set_close_handler(
            [this](luisa::string_view) noexcept {
                {
                    std::scoped_lock lock{_state_mutex};
                    _closed = true;
                }
                _state_cv.notify_all();
            });
        luisa::string error;
        if (!_connection.connect(
                _options.host, _options.port,
                _options.connect_timeout, error)) {
            LUISA_ERROR("Failed to connect to remote backend at {}:{}: {}",
                        _options.host, _options.port, error);
        }
        _handshake();
        _worker = std::thread{[this]() noexcept { _worker_loop(); }};
    }

    ~Impl() noexcept {
        luisa::vector<luisa::unique_ptr<LocalSwapchain>> swapchains;
        {
            std::scoped_lock lock{_present_mutex};
            swapchains.reserve(_swapchains.size());
            for (auto &[handle, swapchain] : _swapchains) {
                static_cast<void>(handle);
                swapchains.emplace_back(std::move(swapchain));
            }
            _swapchains.clear();
        }
        for (auto &swapchain : swapchains) {
            _destroy_local_swapchain(*swapchain);
        }
        if (_connection.connected()) {
            static_cast<void>(_connection.request(
                MessageKind::GOODBYE, {}, _request_timeout));
        }
        _connection.close();
        {
            std::scoped_lock lock{_work_mutex};
            _worker_stop = true;
        }
        _work_cv.notify_all();
        if (_worker.joinable()) { _worker.join(); }
        _connection.set_notification_handler({});
        _connection.set_close_handler({});
        std::scoped_lock lock{_state_mutex};
        if (!_pending.empty()) {
            LUISA_WARNING("Remote device closed with {} incomplete submissions.",
                          _pending.size());
        }
    }

    [[nodiscard]] uint warp_size() const noexcept { return _warp_size; }
    [[nodiscard]] size_t max_shared_memory() const noexcept {
        return _max_shared_memory;
    }
    [[nodiscard]] uint64_t memory_granularity() const noexcept {
        return _memory_granularity;
    }

    [[nodiscard]] BufferCreationInfo create_buffer(
        const Type *element, size_t elem_count) noexcept {
        auto indirect = element == Type::of<IndirectKernelDispatch>();
        if (element->is_custom() && !indirect) {
            LUISA_ERROR(
                "Remote protocol v1 does not support buffer element type '{}'.",
                element->description());
        }
        size_t stride{};
        if (!indirect) {
            stride = element == Type::of<void>() ? 1u : element->size();
            if (elem_count != 0u &&
                stride > std::numeric_limits<size_t>::max() / elem_count) {
                LUISA_ERROR("Remote buffer size overflow.");
            }
        }
        auto requested_size = indirect ? 0u : stride * elem_count;
        Writer writer;
        writer.write_u8(static_cast<uint8_t>(
            indirect ? BufferKind::INDIRECT_DISPATCH : BufferKind::BYTE));
        writer.write_u64(indirect ? elem_count : requested_size);
        auto response = _request(MessageKind::CREATE_BUFFER, writer.bytes());
        Reader reader{response.body, _limits};
        BufferCreationInfo info{};
        info.handle = reader.read_u64();
        auto native_stride = reader.read_u64();
        auto native_size = reader.read_u64();
        if (!reader.finish() ||
            info.handle == invalid_resource_handle ||
            native_stride == 0u ||
            native_stride > std::numeric_limits<size_t>::max() ||
            native_size > std::numeric_limits<size_t>::max() ||
            (!indirect && native_size < requested_size)) {
            LUISA_ERROR("Remote buffer-create response is malformed: {}",
                        reader.error());
        }
        info.native_handle = nullptr;
        info.element_stride = indirect ?
                                  static_cast<size_t>(native_stride) :
                                  stride;
        info.total_size_bytes = static_cast<size_t>(native_size);
        return info;
    }

    void destroy_buffer(uint64_t handle) noexcept {
        static_cast<void>(_destroy_handle(
            MessageKind::DESTROY_BUFFER, handle));
    }

    [[nodiscard]] uint64_t create_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth, uint mip_levels,
        bool simultaneous_access, bool allow_raster_target) noexcept {
        Writer writer;
        writer.write_u32(static_cast<uint32_t>(format));
        writer.write_u32(dimension);
        writer.write_u32(width);
        writer.write_u32(height);
        writer.write_u32(depth);
        writer.write_u32(mip_levels);
        writer.write_bool(simultaneous_access);
        writer.write_bool(allow_raster_target);
        auto handle = _create_handle(MessageKind::CREATE_TEXTURE, writer);
        {
            std::scoped_lock lock{_state_mutex};
            _textures.emplace(handle, TextureDesc{
                                          .storage = pixel_format_to_storage(format),
                                          .size = uint3{width, height, depth},
                                          .mip_levels = mip_levels});
        }
        return handle;
    }

    void destroy_texture(uint64_t handle) noexcept {
        static_cast<void>(_destroy_handle(
            MessageKind::DESTROY_TEXTURE, handle));
        std::scoped_lock lock{_state_mutex};
        _textures.erase(handle);
    }

    [[nodiscard]] uint64_t create_bindless_array(
        size_t slot_count, BindlessSlotType slot_type) noexcept {
        Writer writer;
        writer.write_u64(slot_count);
        writer.write_u32(static_cast<uint32_t>(slot_type));
        return _create_handle(MessageKind::CREATE_BINDLESS_ARRAY, writer);
    }

    void destroy_bindless_array(uint64_t handle) noexcept {
        static_cast<void>(_destroy_handle(
            MessageKind::DESTROY_BINDLESS_ARRAY, handle));
    }

    [[nodiscard]] uint64_t create_stream(StreamTag tag) noexcept {
        Writer writer;
        writer.write_u32(static_cast<uint32_t>(tag));
        return _create_handle(MessageKind::CREATE_STREAM, writer);
    }

    void destroy_stream(uint64_t handle) noexcept {
        if (_destroy_handle(MessageKind::DESTROY_STREAM, handle) &&
            !_try_wait_stream_idle(handle)) {
            LUISA_WARNING(
                "Remote stream {} disconnected while draining completions during destruction.",
                handle);
        }
        std::scoped_lock lock{_state_mutex};
        _log_callbacks.erase(handle);
    }

    void synchronize_stream(uint64_t handle) noexcept {
        _synchronize_remote_stream(handle);
        std::scoped_lock lock{_present_mutex};
        for (auto &[swapchain_handle, swapchain] : _swapchains) {
            static_cast<void>(swapchain_handle);
            if (swapchain->remote_stream == handle &&
                swapchain->submission_pending) {
                swapchain->device->synchronize_stream(swapchain->stream.handle);
                swapchain->submission_pending = false;
            }
        }
    }

    void dispatch(uint64_t stream_handle, CommandList &&list) noexcept {
        auto submission_id = _next_submission_id.fetch_add(
            1u, std::memory_order_relaxed);
        if (submission_id == 0u) {
            LUISA_ERROR("Remote submission-ID space exhausted.");
        }
        UploadBlobPlan blob_plan;
        if ((_features & static_cast<uint64_t>(Feature::BLOB_CACHE)) != 0u) {
            auto blob_limits = _limits;
            blob_limits.max_array_size = std::min(
                blob_limits.max_array_size,
                _max_blobs_per_batch);
            blob_plan = plan_upload_blobs(
                list.commands(), _blob_cache_min_size,
                _max_blob_entry_size, blob_limits);
            if (!blob_plan) {
                LUISA_ERROR("Failed to plan remote upload blobs: {}",
                            blob_plan.error);
            }
        }
        auto encoded = encode_submission(
            stream_handle, submission_id, list.commands(), _limits,
            blob_plan.blobs.empty() ? nullptr : &blob_plan);
        if (!encoded) {
            LUISA_ERROR("Failed to encode remote submission: {}", encoded.error);
        }
        uint64_t footprint = encoded.payload.size();
        for (auto download : encoded.downloads) {
            if (download.size > std::numeric_limits<uint64_t>::max() - footprint) {
                LUISA_ERROR("Remote submission footprint overflow.");
            }
            footprint += download.size;
        }
        if (footprint > _max_in_flight_bytes) {
            LUISA_ERROR(
                "Remote submission requires {} in-flight bytes, exceeding the configured limit {}.",
                footprint, _max_in_flight_bytes);
        }
        PendingSubmission pending{
            .stream_handle = stream_handle,
            .downloads = std::move(encoded.downloads),
            .callbacks = list.steal_callbacks(),
            .footprint = footprint};
        {
            std::unique_lock lock{_state_mutex};
            auto ready = _state_cv.wait_for(
                lock, _request_timeout, [&]() noexcept {
                    return _closed ||
                           footprint <= _max_in_flight_bytes - _pending_bytes;
                });
            if (!ready || _closed) {
                LUISA_ERROR("Timed out waiting for remote in-flight capacity.");
            }
            _pending_bytes += footprint;
            _pending.emplace(submission_id, std::move(pending));
        }
        _prepare_blobs(submission_id, blob_plan);
        auto response = _connection.request(
            MessageKind::DISPATCH, encoded.payload, _request_timeout);
        if (!response) {
            {
                std::scoped_lock lock{_state_mutex};
                if (auto iter = _pending.find(submission_id);
                    iter != _pending.end()) {
                    _pending_bytes -= iter->second.footprint;
                    _pending.erase(iter);
                }
            }
            _state_cv.notify_all();
            LUISA_ERROR("Remote dispatch was rejected (status {}): {}",
                        static_cast<uint16_t>(response.status),
                        response.message);
        }
        Reader reader{response.body, _limits};
        if (reader.read_u64() != submission_id || !reader.finish()) {
            LUISA_ERROR("Remote dispatch acknowledgement is malformed.");
        }
    }

    void set_stream_log_callback(
        uint64_t stream_handle,
        const DeviceInterface::StreamLogCallback &callback) noexcept {
        {
            std::scoped_lock lock{_state_mutex};
            if (callback) {
                _log_callbacks[stream_handle] = callback;
            } else {
                _log_callbacks.erase(stream_handle);
            }
        }
        Writer writer;
        writer.write_u64(stream_handle);
        writer.write_bool(static_cast<bool>(callback));
        static_cast<void>(_request(
            MessageKind::SET_STREAM_LOG_CALLBACK, writer.bytes()));
    }

    [[nodiscard]] SwapchainCreationInfo create_swapchain(
        const SwapchainOption &option,
        uint64_t remote_stream) noexcept {
        std::scoped_lock lock{_present_mutex};
        _ensure_present_device_locked();
        auto local = luisa::make_unique<LocalSwapchain>();
        local->remote_stream = remote_stream;
        local->device = _present_device;
        local->stream = local->device->create_stream(StreamTag::GRAPHICS);
        if (!local->stream.valid()) {
            LUISA_ERROR("The local '{}' backend failed to create a graphics stream for remote presentation.", _present_backend);
        }
        local->swapchain = local->device->create_swapchain(
            option, local->stream.handle);
        if (!local->swapchain.valid()) {
            LUISA_ERROR("The local '{}' backend failed to create a swapchain for the remote client.", _present_backend);
        }
        local->size = option.size;
        auto storage_size = checked_pixel_storage_size(
            local->swapchain.storage,
            uint3{option.size.x, option.size.y, 1u});
        if (!storage_size) {
            LUISA_ERROR("The local swapchain returned an invalid pixel storage.");
        }
        local->image = local->device->create_texture(
            pixel_storage_to_format<float>(local->swapchain.storage),
            2u, option.size.x, option.size.y, 1u, 1u,
            nullptr, false, false);
        if (!local->image.valid()) {
            LUISA_ERROR("The local '{}' backend failed to create the presentation mirror image.", _present_backend);
        }
        local->staging.resize(storage_size.size);
        auto handle = _next_swapchain_id++;
        if (handle == 0u || handle == invalid_resource_handle ||
            _next_swapchain_id == 0u) {
            LUISA_ERROR("The local remote-swapchain handle space is exhausted.");
        }
        auto native_handle = local->swapchain.native_handle;
        auto storage = local->swapchain.storage;
        _swapchains.emplace(handle, std::move(local));
        return SwapchainCreationInfo{
            ResourceCreationInfo{
                .handle = handle,
                .native_handle = native_handle},
            storage};
    }

    void destroy_swapchain(uint64_t handle) noexcept {
        luisa::unique_ptr<LocalSwapchain> swapchain;
        {
            std::scoped_lock lock{_present_mutex};
            auto iter = _swapchains.find(handle);
            if (iter == _swapchains.end()) {
                LUISA_WARNING("Ignoring destruction of an unknown remote-client swapchain {}.", handle);
                return;
            }
            swapchain = std::move(iter->second);
            _swapchains.erase(iter);
        }
        _destroy_local_swapchain(*swapchain);
    }

    void present(uint64_t remote_stream, uint64_t swapchain_handle,
                 uint64_t texture_handle) noexcept {
        TextureDesc texture;
        {
            std::scoped_lock lock{_state_mutex};
            auto iter = _textures.find(texture_handle);
            if (iter == _textures.end()) {
                LUISA_ERROR("Remote presentation references an unknown texture {}.", texture_handle);
            }
            texture = iter->second;
        }
        std::scoped_lock lock{_present_mutex};
        auto iter = _swapchains.find(swapchain_handle);
        if (iter == _swapchains.end()) {
            LUISA_ERROR("Remote presentation references an unknown local swapchain {}.", swapchain_handle);
        }
        auto &swapchain = *iter->second;
        if (swapchain.remote_stream != remote_stream) {
            LUISA_ERROR("A remote-client swapchain must be presented on the stream that created it.");
        }
        if (texture.storage != swapchain.swapchain.storage ||
            any(texture.size !=
                uint3{swapchain.size.x, swapchain.size.y, 1u})) {
            LUISA_ERROR(
                "Remote presentation image does not match the local swapchain: texture {}x{}x{} storage {}, swapchain {}x{} storage {}.",
                texture.size.x, texture.size.y, texture.size.z,
                static_cast<uint32_t>(texture.storage),
                swapchain.size.x, swapchain.size.y,
                static_cast<uint32_t>(swapchain.swapchain.storage));
        }
        if (swapchain.submission_pending) {
            swapchain.device->synchronize_stream(swapchain.stream.handle);
            swapchain.submission_pending = false;
        }

        auto download = CommandList::create(1u);
        download.append(luisa::make_unique<TextureDownloadCommand>(
            texture_handle, texture.storage, 0u, texture.size,
            swapchain.staging.data()));
        auto download_commit = download.commit();
        dispatch(remote_stream,
                 std::move(download_commit).command_list());
        _synchronize_remote_stream(remote_stream);

        auto upload = CommandList::create(1u);
        upload.append(luisa::make_unique<TextureUploadCommand>(
            swapchain.image.handle, swapchain.swapchain.storage,
            0u, uint3{swapchain.size.x, swapchain.size.y, 1u},
            swapchain.staging.data()));
        auto upload_commit = upload.commit();
        swapchain.device->dispatch(
            swapchain.stream.handle,
            std::move(upload_commit).command_list());
        swapchain.device->present_display_in_stream(
            swapchain.stream.handle,
            swapchain.swapchain.handle,
            swapchain.image.handle);
        swapchain.submission_pending = true;
    }

    [[nodiscard]] ShaderCreationInfo create_shader(
        const ShaderOption &option, Function kernel) noexcept {
        if (!option.native_include.empty()) {
            LUISA_ERROR(
                "Remote AST shaders reject backend-native include text in protocol v1.");
        }
        auto ast = try_to_json(kernel);
        if (!ast) {
            LUISA_ERROR("Failed to serialize remote shader AST: {}", ast.error);
        }
        Writer writer;
        write_shader_option(writer, option);
        writer.write_blob({reinterpret_cast<const std::byte *>(ast.json.data()),
                           ast.json.size()});
        auto response = _request(MessageKind::CREATE_SHADER, writer.bytes());
        Reader reader{response.body, _limits};
        ShaderCreationInfo info{};
        info.handle = reader.read_u64();
        info.native_handle = nullptr;
        info.block_size = uint3{
            reader.read_u32(), reader.read_u32(), reader.read_u32()};
        if (!reader.finish() ||
            (!option.compile_only && !info.valid())) {
            LUISA_ERROR("Remote shader-create response is malformed: {}",
                        reader.error());
        }
        if (info.valid()) {
            luisa::vector<Usage> usages;
            usages.reserve(kernel.unbound_arguments().size());
            for (auto argument : kernel.unbound_arguments()) {
                usages.emplace_back(kernel.variable_usage(argument.uid()));
            }
            std::scoped_lock lock{_state_mutex};
            _shader_usages.emplace(info.handle, std::move(usages));
        }
        return info;
    }

    [[nodiscard]] Usage shader_argument_usage(
        uint64_t handle, size_t index) noexcept {
        {
            std::scoped_lock lock{_state_mutex};
            if (auto iter = _shader_usages.find(handle);
                iter != _shader_usages.end()) {
                if (index >= iter->second.size()) {
                    LUISA_ERROR("Remote shader argument index {} is out of range.", index);
                }
                return iter->second[index];
            }
        }
        Writer writer;
        writer.write_u64(handle);
        writer.write_u64(index);
        auto response = _request(
            MessageKind::SHADER_ARGUMENT_USAGE, writer.bytes());
        Reader reader{response.body, _limits};
        auto usage_value = reader.read_u32();
        if (!reader.finish() ||
            usage_value > static_cast<uint32_t>(Usage::READ_WRITE)) {
            LUISA_ERROR("Remote shader-usage response is malformed.");
        }
        return static_cast<Usage>(usage_value);
    }

    void destroy_shader(uint64_t handle) noexcept {
        static_cast<void>(_destroy_handle(
            MessageKind::DESTROY_SHADER, handle));
        std::scoped_lock lock{_state_mutex};
        _shader_usages.erase(handle);
    }

    [[nodiscard]] uint64_t create_event() noexcept {
        Writer writer;
        return _create_handle(MessageKind::CREATE_EVENT, writer);
    }

    void destroy_event(uint64_t handle) noexcept {
        static_cast<void>(_destroy_handle(
            MessageKind::DESTROY_EVENT, handle));
    }

    void event_stream_operation(
        MessageKind kind, uint64_t event, uint64_t stream,
        uint64_t value) noexcept {
        Writer writer;
        writer.write_u64(event);
        writer.write_u64(stream);
        writer.write_u64(value);
        static_cast<void>(_request(kind, writer.bytes()));
    }

    [[nodiscard]] bool is_event_completed(
        uint64_t event, uint64_t value) noexcept {
        Writer writer;
        writer.write_u64(event);
        writer.write_u64(value);
        auto response = _request(
            MessageKind::IS_EVENT_COMPLETED, writer.bytes());
        Reader reader{response.body, _limits};
        auto completed = reader.read_bool();
        if (!reader.finish()) {
            LUISA_ERROR("Remote event-query response is malformed: {}",
                        reader.error());
        }
        return completed;
    }

    void synchronize_event(uint64_t event, uint64_t value) noexcept {
        Writer writer;
        writer.write_u64(event);
        writer.write_u64(value);
        static_cast<void>(_request(
            MessageKind::SYNCHRONIZE_EVENT, writer.bytes()));
    }

    [[nodiscard]] uint64_t create_accel_resource(
        MessageKind kind, const AccelOption &option) noexcept {
        if ((_features & static_cast<uint64_t>(Feature::RAY_TRACING)) == 0u) {
            LUISA_ERROR("The remote server does not advertise ray-tracing support.");
        }
        Writer writer;
        write_accel_option(writer, option);
        return _create_handle(kind, writer);
    }

    void destroy_accel_resource(
        MessageKind kind, uint64_t handle) noexcept {
        static_cast<void>(_destroy_handle(kind, handle));
    }

    [[nodiscard]] luisa::string query(
        luisa::string_view property) noexcept {
        if (property == "remote.connected") {
            return _connection.connected() ? "true" : "false";
        }
        if (property == "remote.native_backend") { return _native_backend; }
        if (property == "remote.device_selection") {
            return (_features & static_cast<uint64_t>(
                                    Feature::DEVICE_SELECTION)) != 0u ?
                       "true" :
                       "false";
        }
        if (property == "remote.protocol.max_frame_payload") {
            return luisa::format("{}", _limits.max_frame_payload);
        }
        if (property == "remote.protocol.max_string_size") {
            return luisa::format("{}", _limits.max_string_size);
        }
        if (property == "remote.protocol.max_array_size") {
            return luisa::format("{}", _limits.max_array_size);
        }
        if (property == "remote.local_present_backend") {
            std::scoped_lock lock{_present_mutex};
            return _present_backend.empty() ?
                       _select_present_backend() :
                       _present_backend;
        }
        Writer writer;
        writer.write_string(property);
        auto response = _request(MessageKind::QUERY, writer.bytes());
        Reader reader{response.body, _limits};
        auto value = reader.read_string();
        if (!reader.finish()) {
            LUISA_ERROR("Remote query response is malformed: {}", reader.error());
        }
        return value;
    }

    void set_name(Resource::Tag tag, uint64_t handle,
                  luisa::string_view name) noexcept {
        if (tag == Resource::Tag::SWAP_CHAIN) {
            std::scoped_lock lock{_present_mutex};
            if (auto iter = _swapchains.find(handle);
                iter != _swapchains.end()) {
                iter->second->device->set_name(
                    tag, iter->second->swapchain.handle, name);
            }
            return;
        }
        Writer writer;
        writer.write_u32(static_cast<uint32_t>(tag));
        writer.write_u64(handle);
        writer.write_string(name);
        static_cast<void>(_request(MessageKind::SET_NAME, writer.bytes()));
    }
};

RemoteDevice::RemoteDevice(Context &&context,
                           const DeviceConfig *config) noexcept
    : DeviceInterface{Context{context}},
      _impl{std::make_unique<Impl>(
          std::move(context), client_options(config))} {}

RemoteDevice::~RemoteDevice() noexcept = default;

void *RemoteDevice::native_handle() const noexcept { return nullptr; }

uint RemoteDevice::compute_warp_size() const noexcept {
    return _impl->warp_size();
}

size_t RemoteDevice::compute_max_shared_memory_size() const noexcept {
    return _impl->max_shared_memory();
}

uint64_t RemoteDevice::memory_granularity() const noexcept {
    return _impl->memory_granularity();
}

BufferCreationInfo RemoteDevice::create_buffer(
    const Type *element, size_t elem_count,
    void *external_memory) noexcept {
    if (external_memory != nullptr) {
        LUISA_WARNING("Remote protocol v1 cannot import process-local buffer memory.");
        return BufferCreationInfo::make_invalid();
    }
    return _impl->create_buffer(element, elem_count);
}

void RemoteDevice::destroy_buffer(uint64_t handle) noexcept {
    _impl->destroy_buffer(handle);
}

ResourceCreationInfo RemoteDevice::create_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels, void *external_native_handle,
    bool simultaneous_access,
    bool allow_raster_target) noexcept {
    if (external_native_handle != nullptr) {
        LUISA_WARNING("Remote protocol v1 cannot import process-local textures.");
        return ResourceCreationInfo::make_invalid();
    }
    return ResourceCreationInfo{
        .handle = _impl->create_texture(
            format, dimension, width, height, depth, mipmap_levels,
            simultaneous_access, allow_raster_target),
        .native_handle = nullptr};
}

void RemoteDevice::destroy_texture(uint64_t handle) noexcept {
    _impl->destroy_texture(handle);
}

ResourceCreationInfo RemoteDevice::create_bindless_array(
    size_t size, BindlessSlotType type) noexcept {
    return ResourceCreationInfo{
        .handle = _impl->create_bindless_array(size, type),
        .native_handle = nullptr};
}

void RemoteDevice::destroy_bindless_array(uint64_t handle) noexcept {
    _impl->destroy_bindless_array(handle);
}

ResourceCreationInfo RemoteDevice::create_stream(StreamTag stream_tag) noexcept {
    return ResourceCreationInfo{
        .handle = _impl->create_stream(stream_tag),
        .native_handle = nullptr};
}

void RemoteDevice::destroy_stream(uint64_t handle) noexcept {
    _impl->destroy_stream(handle);
}

void RemoteDevice::synchronize_stream(uint64_t stream_handle) noexcept {
    _impl->synchronize_stream(stream_handle);
}

void RemoteDevice::dispatch(
    uint64_t stream_handle, CommandList &&list) noexcept {
    struct PresentDesc {
        uint64_t swapchain{};
        uint64_t texture{};
    };
    luisa::vector<PresentDesc> presents;
    presents.reserve(list.presents().size());
    for (auto &&present : list.presents()) {
        if (present.chain == nullptr) {
            LUISA_ERROR("Remote command list contains a null swapchain presentation.");
        }
        presents.emplace_back(PresentDesc{
            .swapchain = present.chain->handle(),
            .texture = present.frame.handle()});
    }
    _impl->dispatch(stream_handle, std::move(list));
    for (auto present : presents) {
        _impl->present(
            stream_handle, present.swapchain, present.texture);
    }
}

void RemoteDevice::set_stream_log_callback(
    uint64_t stream_handle,
    const StreamLogCallback &callback) noexcept {
    _impl->set_stream_log_callback(stream_handle, callback);
}

SwapchainCreationInfo RemoteDevice::create_swapchain(
    const SwapchainOption &option, uint64_t stream_handle) noexcept {
    return _impl->create_swapchain(option, stream_handle);
}

void RemoteDevice::destroy_swapchain(uint64_t handle) noexcept {
    _impl->destroy_swapchain(handle);
}

void RemoteDevice::present_display_in_stream(
    uint64_t stream_handle, uint64_t swapchain_handle,
    uint64_t image_handle) noexcept {
    _impl->present(stream_handle, swapchain_handle, image_handle);
}

ShaderCreationInfo RemoteDevice::create_shader(
    const ShaderOption &option, Function kernel) noexcept {
    return _impl->create_shader(option, kernel);
}

ShaderCreationInfo RemoteDevice::load_shader(
    luisa::string_view,
    luisa::span<const Type *const>) noexcept {
    LUISA_WARNING("Remote AOT shader loading is not implemented in protocol v1 yet; use AST JIT creation.");
    return ShaderCreationInfo::make_invalid();
}

Usage RemoteDevice::shader_argument_usage(
    uint64_t handle, size_t index) noexcept {
    return _impl->shader_argument_usage(handle, index);
}

void RemoteDevice::destroy_shader(uint64_t handle) noexcept {
    _impl->destroy_shader(handle);
}

ResourceCreationInfo RemoteDevice::create_event() noexcept {
    return ResourceCreationInfo{
        .handle = _impl->create_event(),
        .native_handle = nullptr};
}

void RemoteDevice::destroy_event(uint64_t handle) noexcept {
    _impl->destroy_event(handle);
}

void RemoteDevice::signal_event(
    uint64_t handle, uint64_t stream_handle,
    uint64_t fence_value) noexcept {
    _impl->event_stream_operation(
        MessageKind::SIGNAL_EVENT, handle, stream_handle, fence_value);
}

void RemoteDevice::wait_event(
    uint64_t handle, uint64_t stream_handle,
    uint64_t fence_value) noexcept {
    _impl->event_stream_operation(
        MessageKind::WAIT_EVENT, handle, stream_handle, fence_value);
}

bool RemoteDevice::is_event_completed(
    uint64_t handle, uint64_t fence_value) const noexcept {
    return _impl->is_event_completed(handle, fence_value);
}

void RemoteDevice::synchronize_event(
    uint64_t handle, uint64_t fence_value) noexcept {
    _impl->synchronize_event(handle, fence_value);
}

ResourceCreationInfo RemoteDevice::create_mesh(
    const AccelOption &option) noexcept {
    return ResourceCreationInfo{
        .handle = _impl->create_accel_resource(
            MessageKind::CREATE_MESH, option),
        .native_handle = nullptr};
}

void RemoteDevice::destroy_mesh(uint64_t handle) noexcept {
    _impl->destroy_accel_resource(MessageKind::DESTROY_MESH, handle);
}

ResourceCreationInfo RemoteDevice::create_procedural_primitive(
    const AccelOption &option) noexcept {
    return ResourceCreationInfo{
        .handle = _impl->create_accel_resource(
            MessageKind::CREATE_PROCEDURAL_PRIMITIVE, option),
        .native_handle = nullptr};
}

void RemoteDevice::destroy_procedural_primitive(uint64_t handle) noexcept {
    _impl->destroy_accel_resource(
        MessageKind::DESTROY_PROCEDURAL_PRIMITIVE, handle);
}

ResourceCreationInfo RemoteDevice::create_curve(
    const AccelOption &option) noexcept {
    return ResourceCreationInfo{
        .handle = _impl->create_accel_resource(
            MessageKind::CREATE_CURVE, option),
        .native_handle = nullptr};
}

void RemoteDevice::destroy_curve(uint64_t handle) noexcept {
    _impl->destroy_accel_resource(MessageKind::DESTROY_CURVE, handle);
}

ResourceCreationInfo RemoteDevice::create_motion_instance(
    const AccelMotionOption &option) noexcept {
    AccelOption accel_option;
    accel_option.motion = option;
    return ResourceCreationInfo{
        .handle = _impl->create_accel_resource(
            MessageKind::CREATE_MOTION_INSTANCE, accel_option),
        .native_handle = nullptr};
}

void RemoteDevice::destroy_motion_instance(uint64_t handle) noexcept {
    _impl->destroy_accel_resource(
        MessageKind::DESTROY_MOTION_INSTANCE, handle);
}

ResourceCreationInfo RemoteDevice::create_accel(
    const AccelOption &option) noexcept {
    return ResourceCreationInfo{
        .handle = _impl->create_accel_resource(
            MessageKind::CREATE_ACCEL, option),
        .native_handle = nullptr};
}

void RemoteDevice::destroy_accel(uint64_t handle) noexcept {
    _impl->destroy_accel_resource(MessageKind::DESTROY_ACCEL, handle);
}

luisa::string RemoteDevice::query(
    luisa::string_view property) noexcept {
    return _impl->query(property);
}

DeviceExtension *RemoteDevice::extension(luisa::string_view) noexcept {
    return nullptr;
}

void RemoteDevice::set_name(
    Resource::Tag resource_tag, uint64_t resource_handle,
    luisa::string_view name) noexcept {
    _impl->set_name(resource_tag, resource_handle, name);
}

}// namespace luisa::compute::remote

LUISA_EXPORT_API luisa::compute::DeviceInterface *create(
    luisa::compute::Context &&context,
    const luisa::compute::DeviceConfig *config) noexcept {
    return luisa::new_with_allocator<
        luisa::compute::remote::RemoteDevice>(
        std::move(context), config);
}

LUISA_EXPORT_API void destroy(
    luisa::compute::DeviceInterface *device) noexcept {
    luisa::delete_with_allocator(device);
}

LUISA_EXPORT_API void backend_device_names(
    luisa::vector<luisa::string> &names) noexcept {
    names.clear();
    names.emplace_back("Remote LuisaCompute device");
}

#include "../common/export_version.inl.h"
