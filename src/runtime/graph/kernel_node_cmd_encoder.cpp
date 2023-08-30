#include <luisa/runtime/graph/kernel_node_cmd_encoder.h>

using namespace luisa::compute::graph;

KernelNodeCmdEncoder::KernelNodeCmdEncoder(size_t arg_count, size_t uniform_size) noexcept
    : ShaderDispatchCmdEncoder{invalid_resource_handle, arg_count, uniform_size} {}

void KernelNodeCmdEncoder::update_uniform(size_t i, const void *data) noexcept {
    auto &arg = arguments()[i];
    LUISA_ASSERT(arg.tag == Argument::Tag::UNIFORM, "Argument type mismatches");
    auto offset = arg.uniform.offset;
    auto size = arg.uniform.size;
    std::memcpy(_argument_buffer.data() + offset, data, size);
}

void KernelNodeCmdEncoder::update_buffer(size_t i, uint64_t handle, size_t offset, size_t size) noexcept {
    auto &arg = arguments()[i];
    LUISA_ASSERT(arg.tag == Argument::Tag::BUFFER, "Argument type mismatches");
    arg.buffer.handle = handle;
    arg.buffer.offset = offset;
    arg.buffer.size = size;
}