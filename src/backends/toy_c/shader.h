#pragma once
#include <luisa/core/dynamic_module.h>
#include <luisa/core/stl/string.h>
#include <luisa/vstl/common.h>
#include <luisa/vstl/md5.h>
struct MemoryManager;
namespace lc::toy_c {
class LCStream;
class LCDevice;
using namespace luisa;
using namespace luisa::compute;
class LCShader : public vstd::IOperatorNewBase {
    vstd::func_ptr_t<void(uint3 thd_id, uint3 blk_id, uint3 dsp_id, uint3 dsp_size, uint3 ker_id, void *args)> kernel;
    vstd::func_ptr_t<uint32_t(uint32_t)> get_usage;
    void _emplace_arg(luisa::span<const Argument> arguments, std::byte const *uniform_data, luisa::vector<std::byte> &arg_buffer);

public:
    uint3 block_size;
    LCShader(DynamicModule &dyn_module, luisa::span<const Type *const> arg_types, luisa::string_view kernel_name);
    ~LCShader();
    void dispatch(LCDevice* device, LCStream* stream, MemoryManager &manager, uint3 size, luisa::span<const Argument> arguments, std::byte const *uniform_data, luisa::vector<std::byte> &arg_buffer);
    void dispatch(LCDevice* device, LCStream* stream, MemoryManager &manager, luisa::span<uint3 const> size, luisa::span<const Argument> arguments, std::byte const *uniform_data, luisa::vector<std::byte> &arg_buffer);
};
}// namespace lc::toy_c