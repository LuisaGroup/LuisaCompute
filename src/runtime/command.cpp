//
// Created by Mike Smith on 2021/2/15.
//


#include <runtime/command.h>
#include <runtime/device.h>
#include <runtime/stream.h>

namespace luisa::compute {

#define LUISA_MAKE_COMMAND_OVERRIDE_IMPL(Type) \
    void Type::commit(Stream &stream) { stream._dispatch(*this); } \
    void Type::recycle() { this->device()->command_pool().recycle(this); }

LUISA_MAKE_COMMAND_OVERRIDE_IMPL(BufferCopyCommand)
LUISA_MAKE_COMMAND_OVERRIDE_IMPL(BufferUploadCommand)
LUISA_MAKE_COMMAND_OVERRIDE_IMPL(BufferDownloadCommand)

void BufferUploadCommand::finalize(Stream &stream) {
    // TODO...
}



}
