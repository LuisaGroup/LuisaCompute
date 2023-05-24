#pragma once
#include <runtime/rhi/command.h>
namespace luisa::compute {
class DStorageReadCommand : public CustomCommand {
public:
    struct BufferEnqueue {
        uint64_t buffer_handle;
        size_t buffer_offset;
    };
    struct ImageEnqueue {
        uint64_t image_handle;
        uint32_t mip_level;
    };
    struct MemoryEnqueue {
        void *dst_ptr;
    };
    struct FileSource {
        uint64_t file_handle;
        size_t file_offset;
    };
    struct MemorySource {
        void const *src_ptr;
    };
    enum class Compression : uint32_t {
        None,
        GDeflate,
    };
    luisa::variant<FileSource, MemorySource> src;
    size_t src_size;
    size_t dst_size;
    Compression compression;

    using EnqueueCommand = luisa::variant<
        BufferEnqueue,
        ImageEnqueue,
        MemoryEnqueue>;

private:
    EnqueueCommand _enqueue_cmd;

public:
    template<typename Arg>
        requires(std::is_constructible_v<EnqueueCommand, Arg &&>)
    explicit DStorageReadCommand(
        uint64_t file_handle,
        size_t file_offset,
        size_t src_size,
        size_t dst_size,
        Compression compression,
        Arg &&cmd)
        : src{FileSource{file_handle, file_offset}},
          src_size{src_size},
          dst_size{dst_size},
          compression{compression},
          _enqueue_cmd{std::forward<Arg>(cmd)} {}
    template<typename Arg>
        requires(std::is_constructible_v<EnqueueCommand, Arg &&>)
    explicit DStorageReadCommand(
        void const *src_ptr,
        size_t src_size,
        size_t dst_size,
        Compression compression,
        Arg &&cmd)
        : src{MemorySource{src_ptr}},
          src_size{src_size},
          dst_size{dst_size},
          compression{compression},
          _enqueue_cmd{std::forward<Arg>(cmd)} {}
    [[nodiscard]] auto const &enqueue_cmd() const { return _enqueue_cmd; }
    uint64_t uuid() const noexcept override { return dstorage_command_uuid; }
    LUISA_MAKE_COMMAND_COMMON(StreamTag::CUSTOM)
};
}// namespace luisa::compute