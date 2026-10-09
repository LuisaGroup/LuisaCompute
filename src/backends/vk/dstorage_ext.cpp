#include "dstorage_ext.h"
#include <algorithm>
#include <cstring>
#include <new>
#include <luisa/core/magic_enum.h>
#include <luisa/runtime/rhi/pixel.h>

namespace lc::vk {

VkDStorageFile::VkDStorageFile(luisa::string_view path) noexcept {
    _path = luisa::string{path};
    _file = std::fopen(_path.c_str(), "rb");
    if (_file == nullptr) { return; }
    // Binary-safe size probe; does not disturb the read cursor on success.
    // Binary-safe size probe using the platform's 64-bit seek/tell
    // (`_fseeki64`/`_ftelli64` on Windows, `fseeko`/`ftello` elsewhere).  The
    // fresh handle is positioned at 0 so the branch never depends on `long`
    // being 64-bit.
#if defined(_WIN32)
    _fseeki64(_file, 0, SEEK_END);
    auto end = _ftelli64(_file);
    auto repositioned = _fseeki64(_file, 0, SEEK_SET) == 0;
#else
    fseeko(_file, 0, SEEK_END);
    auto end = ftello(_file);
    auto repositioned = fseeko(_file, 0, SEEK_SET) == 0;
#endif
    _size_bytes = end < 0 ? 0u : static_cast<size_t>(end);
    if (!repositioned) {
        std::fclose(_file);
        _file = nullptr;
        _size_bytes = 0u;
    }
}

VkDStorageFile::~VkDStorageFile() noexcept {
    if (_file != nullptr) {
        std::fclose(_file);
        _file = nullptr;
    }
}

bool VkDStorageFile::read(uint64_t offset, size_t size, void *dst) noexcept {
    if (_file == nullptr) { return false; }
    if (offset > _size_bytes || size > _size_bytes - offset) { return false; }
#if defined(_WIN32)
    if (_fseeki64(_file, static_cast<__int64>(offset), SEEK_SET) != 0) { return false; }
#else
    if (fseeko(_file, static_cast<off_t>(offset), SEEK_SET) != 0) { return false; }
#endif
    return size == 0u || std::fread(dst, 1u, size, _file) == size;
}

VkDStorageStream::VkDStorageStream(Device *device, VkDStorageExt *ext) noexcept
    : _device{device}, _ext{ext} {
    // A real COPY stream: `StreamTag::COPY` uses the copy queue family and owns
    // its own timeline event, so `synchronize()`/`signal()`/`wait()` behave
    // exactly like on a user-created COPY stream.
    auto info = device->create_stream(StreamTag::COPY);
    if (info.handle != luisa::compute::invalid_resource_handle) {
        _copy_stream_handle = info.handle;
        _copy_stream = reinterpret_cast<Stream *>(info.handle);
    } else {
        LUISA_WARNING("Vulkan DirectStorage fallback failed to create its internal COPY stream.");
    }
}

VkDStorageStream::~VkDStorageStream() noexcept {
    if (_copy_stream_handle != 0u && _device != nullptr) {
        _device->synchronize_stream(_copy_stream_handle);
        _device->destroy_stream(_copy_stream_handle);
        _copy_stream_handle = 0u;
        _copy_stream = nullptr;
    }
}

void VkDStorageStream::synchronize() noexcept {
    if (_copy_stream_handle != 0u) {
        _device->synchronize_stream(_copy_stream_handle);
    }
}

void VkDStorageStream::signal(Event *event, uint64_t fence) noexcept {
    if (_copy_stream != nullptr) {
        _copy_stream->signal(event, fence);
    }
}

void VkDStorageStream::wait(Event *event, uint64_t fence) noexcept {
    if (_copy_stream != nullptr) {
        _copy_stream->wait(event, fence);
    }
}

void VkDStorageStream::set_log_callback(const DeviceInterface::StreamLogCallback &callback) noexcept {
    if (_copy_stream != nullptr) {
        _copy_stream->logger = callback;
    }
}

void VkDStorageStream::dispatch(CommandList &&list) noexcept {
    std::lock_guard lck{_mtx};
    if (_copy_stream == nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Vulkan DirectStorage fallback stream is invalid "
            "(its internal COPY stream could not be created).");
    }
    // Take ownership of the batch: the incoming list must end up empty
    // (`CommandList::~CommandList` asserts that it was committed or emptied),
    // and the commands must stay alive until the internal COPY stream has
    // recorded them (which `_device->dispatch` does synchronously, below).
    auto commands = list.steal_commands();
    auto callbacks = list.steal_callbacks();
    auto presents = list.steal_presents();
    if (!presents.empty()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Present commands are not allowed on a DirectStorage stream.");
    }
    // ---- validation + scratch sizing (pass 1) ------------------------------
    size_t total_raw = 0u;
    size_t total_tight = 0u;
    for (auto &&command : commands) {
        if (command->tag() != Command::Tag::ECustomCommand ||
            static_cast<CustomCommand const *>(command.get())->custom_cmd_uuid() !=
                to_underlying(CustomCommandUUID::DSTORAGE_READ)) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Only DStorageReadCommand is allowed on a DirectStorage stream.");
        }
        auto *cmd = static_cast<DStorageReadCommand const *>(command.get());
        if (cmd->is_compressed()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "The Vulkan DirectStorage fallback cannot decompress (got {}). "
                "Use DStorageCompression::None, or decompress on the CPU "
                "(DStorageExt::supports_compression() reports the capability).",
                to_string(cmd->compression()));
        }
        if (luisa::holds_alternative<DStorageReadCommand::TextureRequest>(cmd->request()) &&
            cmd->source_size_bytes() < cmd->required_source_size_bytes()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "DStorage texture source is too small: {} byte(s) available, but the "
                "pitch-aligned layout needs {} byte(s). Lay out texture sources with "
                "dstorage_texture_row_pitch()/dstorage_texture_source_size(), or use a "
                "buffer destination.",
                cmd->source_size_bytes(), cmd->required_source_size_bytes());
        }
        auto transfer = cmd->effective_transfer_size();
        if (transfer == 0u) { continue; }
        total_raw += transfer;
        if (luisa::holds_alternative<DStorageReadCommand::TextureRequest>(cmd->request())) {
            auto const &tex = luisa::get<DStorageReadCommand::TextureRequest>(cmd->request());
            auto region = uint3{tex.size[0], tex.size[1], tex.size[2]};
            auto pitch = dstorage_texture_row_pitch(tex.storage, region.x);
            auto tight_row = pixel_storage_size(tex.storage, uint3{region.x, 1u, 1u});
            if (pitch != tight_row) {
                total_tight += pixel_storage_size(tex.storage, region);
            }
        }
    }
    // The scratch is shared with the submitted command list so it is released
    // only once the GPU copy has completed (the callback runs when the command
    // buffer retires), instead of relying on `Stream::dispatch` recording
    // synchronously.
    auto scratch = luisa::make_shared<luisa::vector<std::byte>>();
    auto tight_scratch = luisa::make_shared<luisa::vector<std::byte>>();
    scratch->resize(total_raw);
    tight_scratch->resize(total_tight);
    // ---- host acquire + command emission (pass 2) --------------------------
    CommandList uploads;
    size_t raw_offset = 0u;
    size_t tight_offset = 0u;
    for (auto &&command : commands) {
        auto *cmd = static_cast<DStorageReadCommand const *>(command.get());
        auto transfer = cmd->effective_transfer_size();
        if (transfer == 0u) { continue; }
        auto *raw_slot = scratch->data() + raw_offset;
        raw_offset += transfer;
        std::byte const *host_ptr = raw_slot;
        luisa::visit(
            [&]<typename T>(T const &src) {
                if constexpr (std::is_same_v<T, DStorageReadCommand::FileSource>) {
                    if (src.handle == luisa::compute::invalid_resource_handle) [[unlikely]] {
                        LUISA_ERROR_WITH_LOCATION(
                            "DStorage file source is invalid (the file failed to open).");
                    }
                    auto *file = reinterpret_cast<VkDStorageFile *>(src.handle);
                    if (!file->read(src.offset_bytes, transfer, raw_slot)) [[unlikely]] {
                        LUISA_ERROR_WITH_LOCATION(
                            "Failed to read {} byte(s) at offset {} from '{}'.",
                            transfer, src.offset_bytes, file->path());
                    }
                    host_ptr = raw_slot;
                } else {
                    auto base = reinterpret_cast<std::byte const *>(src.handle);
                    if (base == nullptr) [[unlikely]] {
                        LUISA_ERROR_WITH_LOCATION(
                            "DStorage memory source is null (the pinned-memory handle "
                            "is invalid).");
                    }
                    if (_ext != nullptr) {
                        auto pinned = _ext->pinned_memory_size(src.handle);
                        if (pinned != 0u) {
                            LUISA_ASSERT(src.offset_bytes <= pinned &&
                                             src.size_bytes <= pinned - src.offset_bytes,
                                         "DStorage memory source out of range: offset {} + "
                                         "size {} exceeds the pinned range {}.",
                                         src.offset_bytes, src.size_bytes, pinned);
                        }
                    }
                    host_ptr = base + src.offset_bytes;
                }
            },
            cmd->source());
        luisa::visit(
            [&]<typename T>(T const &dst) {
                if constexpr (std::is_same_v<T, DStorageReadCommand::BufferRequest>) {
                    uploads.append(luisa::make_unique<BufferUploadCommand>(
                        dst.handle, dst.offset_bytes, transfer, host_ptr));
                } else if constexpr (std::is_same_v<T, DStorageReadCommand::TextureRequest>) {
                    auto offset = uint3{dst.offset[0], dst.offset[1], dst.offset[2]};
                    auto region = uint3{dst.size[0], dst.size[1], dst.size[2]};
                    LUISA_VERBOSE("VK DStorage texture upload: level={}, offset=({},{},{}), size=({},{},{}), transfer={}.",
                                  dst.level, offset.x, offset.y, offset.z,
                                  region.x, region.y, region.z, transfer);
                    auto pitch = dstorage_texture_row_pitch(dst.storage, region.x);
                    auto tight_row = pixel_storage_size(dst.storage, uint3{region.x, 1u, 1u});
                    const void *data = host_ptr;
                    if (pitch != tight_row) {
                        // The VK upload path expects tightly packed rows; the
                        // direct-storage contract pads them to 256 bytes, so
                        // repack here.
                        auto *tight_slot = tight_scratch->data() + tight_offset;
                        tight_offset += pixel_storage_size(dst.storage, region);
                        for (uint32_t z = 0u; z < region.z; ++z) {
                            for (uint32_t y = 0u; y < region.y; ++y) {
                                auto row = static_cast<size_t>(z) * region.y + y;
                                std::memcpy(tight_slot + row * tight_row,
                                            host_ptr + row * pitch,
                                            tight_row);
                            }
                        }
                        data = tight_slot;
                    }
                    uploads.append(luisa::make_unique<TextureUploadCommand>(
                        dst.handle, dst.storage, dst.level, region, data, offset));
                } else {// MemoryRequest
                    // Host-side destination: no GPU command is needed.
                    std::memcpy(dst.data, host_ptr, transfer);
                }
            },
            cmd->request());
    }
    for (auto &&callback : callbacks) {
        uploads.add_callback(std::move(callback));
    }
    // Keeping the scratch alive until the batch retires also makes the
    // fallback independent of `Stream::dispatch`'s synchronous recording.
    uploads.add_callback([scratch = std::move(scratch),
                          tight_scratch = std::move(tight_scratch)]() noexcept {});
    _device->dispatch(_copy_stream_handle, std::move(uploads));
    // `Stream::dispatch` records synchronously (it copies upload data into the
    // per-command upload allocation and stores the callbacks), so the command
    // list can be released right away.  Clearing also satisfies
    // `CommandList::~CommandList`, which asserts it was committed or emptied.
    uploads.clear();
}

VkDStorageExt::~VkDStorageExt() noexcept {
    // Custom streams are normally destroyed by the runtime `Stream` destructor
    // (which calls `Device::destroy_stream`); ~Device has a safety net for any
    // that survive.  Nothing to release here.
}

ResourceCreationInfo VkDStorageExt::create_stream_handle(const DStorageStreamOption &option) noexcept {
    auto *stream = new VkDStorageStream{_device, this};
    if (!stream->valid()) [[unlikely]] {
        delete stream;
        return ResourceCreationInfo::make_invalid();
    }
    _device->add_custom_stream(reinterpret_cast<uint64_t>(stream), stream);
    LUISA_INFO(
        "Vulkan DirectStorage fallback stream created: host-staged reads on an "
        "internal COPY stream, staging_buffer_size = {} byte(s), source hint = {}.",
        option.staging_buffer_size, to_string(option.source));
    ResourceCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(stream);
    info.native_handle = stream->queue();
    return info;
}

DStorageExt::FileCreationInfo VkDStorageExt::open_file_handle(luisa::string_view path) noexcept {
    auto *file = new (std::nothrow) VkDStorageFile{path};
    if (file == nullptr || !file->valid() || file->size_bytes() == 0u) {
        LUISA_WARNING("Vulkan DirectStorage fallback could not open file '{}'.", path);
        delete file;
        return FileCreationInfo::make_invalid();
    }
    FileCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(file);
    info.native_handle = nullptr;
    info.size_bytes = file->size_bytes();
    return info;
}

void VkDStorageExt::close_file_handle(uint64_t handle) noexcept {
    delete reinterpret_cast<VkDStorageFile *>(handle);
}

DStorageExt::PinnedMemoryInfo VkDStorageExt::pin_host_memory(void *ptr, size_t size_bytes) noexcept {
    // The fallback reads host memory directly, so the raw pointer *is* the
    // handle (documented in dstorage_ext_interface.h).  The range is recorded
    // so memory-sourced requests can be validated.
    if (ptr == nullptr || size_bytes == 0u) [[unlikely]] {
        LUISA_WARNING("Cannot pin a null or empty host range "
                      "(pointer = {}, size = {} byte(s)).",
                      ptr, size_bytes);
        return PinnedMemoryInfo::make_invalid();
    }
    auto handle = reinterpret_cast<uint64_t>(ptr);
    {
        std::lock_guard lck{_mtx};
        auto iter = _pinned_ranges.try_emplace(handle, size_bytes);
        if (!iter.second) {
            iter.first->second = std::max(iter.first->second, size_bytes);
        }
    }
    PinnedMemoryInfo info{};
    info.handle = handle;
    info.native_handle = ptr;
    info.size_bytes = size_bytes;
    return info;
}

void VkDStorageExt::unpin_host_memory(uint64_t handle) noexcept {
    std::lock_guard lck{_mtx};
    _pinned_ranges.erase(handle);
}

size_t VkDStorageExt::pinned_memory_size(uint64_t handle) noexcept {
    std::lock_guard lck{_mtx};
    auto iter = _pinned_ranges.find(handle);
    return iter == _pinned_ranges.end() ? 0u : iter->second;
}

void VkDStorageExt::compress(const void *data, size_t size_bytes,
                             Compression algorithm, CompressionQuality quality,
                             luisa::vector<std::byte> &result) noexcept {
    static_cast<void>(quality);
    if (algorithm == Compression::None) {
        result.resize(size_bytes);
        if (size_bytes != 0u) {
            std::memcpy(result.data(), data, size_bytes);
        }
        return;
    }
    LUISA_ERROR_WITH_LOCATION(
        "The Vulkan DirectStorage fallback does not provide a {} compressor. "
        "Use DStorageCompression::None (a plain copy) or compress the data "
        "offline with a backend that supports it.",
        to_string(algorithm));
}

}// namespace lc::vk
