#include "stream.h"
#include "shader.h"
#include <luisa/core/fiber.h>
namespace lc::toy_c {
LCStream::LCStream(): _evt(luisa::fiber::event::Mode::Manual, true) {
}
void LCStream::dispatch(CommandList &&cmdlist) {
    luisa::fiber::event new_evt;
    luisa::fiber::schedule(
        luisa::SharedFunction<void()>{[this, _evt = std::move(_evt), new_evt, cmdlist = std::move(cmdlist)]() mutable {
            _evt.wait();
            auto sig = vstd::scope_exit([&]() {
                new_evt.signal();
            });
            luisa::vector<std::byte> arg_alloc;
            arg_alloc.reserve(1024);
            for (auto &base_cmd : cmdlist.commands()) {
                switch (base_cmd->tag()) {
                    case Command::Tag::EBufferUploadCommand: {
                        auto cmd = static_cast<BufferUploadCommand const *>(base_cmd.get());
                        std::memcpy(reinterpret_cast<void *>(cmd->handle() + cmd->offset()), cmd->data(), cmd->size());
                    } break;
                    case Command::Tag::EBufferDownloadCommand: {
                        auto cmd = static_cast<BufferDownloadCommand const *>(base_cmd.get());
                        std::memcpy(cmd->data(), reinterpret_cast<void *>(cmd->handle() + cmd->offset()), cmd->size());
                    } break;
                    case Command::Tag::EBufferCopyCommand: {
                        auto cmd = static_cast<BufferCopyCommand const *>(base_cmd.get());
                        std::memcpy(reinterpret_cast<void *>(cmd->dst_handle() + cmd->dst_offset()), reinterpret_cast<void *>(cmd->src_handle() + cmd->src_offset()), cmd->size());
                    } break;
                    case Command::Tag::EShaderDispatchCommand: {
                        auto cmd = static_cast<ShaderDispatchCommand const *>(base_cmd.get());
                        auto shader = reinterpret_cast<LCShader *>(cmd->handle());
                        auto arg_buffer = cmd->arguments();
                        if (cmd->is_indirect()) {
                            LUISA_ERROR("Backend do not support indirect dispatch.");
                        } else if (cmd->is_multiple_dispatch()) {
                            shader->dispatch(
                                cmd->dispatch_sizes(),
                                arg_buffer,
                                reinterpret_cast<std::byte const *>(arg_buffer.data()),
                                arg_alloc);
                        } else {
                            shader->dispatch(
                                cmd->dispatch_size(),
                                arg_buffer,
                                reinterpret_cast<std::byte const *>(arg_buffer.data()),
                                arg_alloc);
                        }
                    } break;
                    default:
                        LUISA_ERROR("Command {} not supported.", luisa::to_string(base_cmd->tag()));
                        break;
                }
            }
            for (auto &i : cmdlist.callbacks()) {
                i();
            }
        }});
    _evt = std::move(new_evt);
}

void LCStream::sync() {
    _evt.wait();
}
LCStream::~LCStream() {
    _evt.wait();
}
}// namespace lc::toy_c