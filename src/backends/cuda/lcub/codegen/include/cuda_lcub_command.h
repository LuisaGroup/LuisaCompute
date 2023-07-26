
//
//  created by MuGdxy on 2023/4/10
//
#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/backends/ext/registry.h>
#include <functional>

namespace luisa::compute::cuda {
#define LUISA_MAKE_COMMAND_COMMON_CREATE(Cmd)                        \
    template<typename... Args>                                       \
        requires(std::is_constructible_v<Cmd, Args && ...>)          \
    [[nodiscard]] static auto create(Args &&...args) noexcept {      \
        return luisa::make_unique<Cmd>(std::forward<Args>(args)...); \
    }

#define LUISA_MAKE_COMMAND_COMMON(Cmd, Type)   \
    /*friend class luisa::compute::CmdDeser;*/ \
    /*friend class luisa::compute::CmdSer;*/   \
    LUISA_MAKE_COMMAND_COMMON_CREATE(Cmd)      \
    luisa::compute::StreamTag stream_tag() const noexcept override { return Type; }

class CudaLCubCommand : public luisa::compute::CustomCommand {
    friend lc::validation::Stream;

public:
    explicit CudaLCubCommand(const std::function<void(cudaStream_t)> &f) noexcept
        : CustomCommand{}, func{f} {}
    std::function<void(cudaStream_t)> func;
    LUISA_MAKE_COMMAND_COMMON(CudaLCubCommand, luisa::compute::StreamTag::COMPUTE);
    [[nodiscard]] virtual uint64_t uuid() const noexcept { return static_cast<uint64_t>(CustomCommandUUID::CUDA_LCUB_COMMAND); }
    void accept(luisa::compute::CommandVisitor &visitor) const noexcept override { LUISA_ERROR_WITH_LOCATION("No Impl"); }
    void accept(luisa::compute::MutableCommandVisitor &visitor) noexcept override;
};

#undef LUISA_MAKE_COMMAND_COMMON_CREATE
#undef LUISA_MAKE_COMMAND_COMMON
}// namespace sphere
