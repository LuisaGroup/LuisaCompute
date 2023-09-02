
//
//  created by MuGdxy on 2023/4/10
//

#pragma once

#include <cuda.h>
#include <luisa/core/stl/functional.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/backends/ext/registry.h>
#include <luisa/runtime/graph/graph_capture_command.h>

namespace luisa::compute::cuda {

class CudaLCubCommand final : public luisa::compute::CustomCommand, public luisa::compute::graph::GraphCaptureCommand {

public:
    friend lc::validation::Stream;
    luisa::function<void(CUstream)> func;
    virtual void capture_on(uint64_t stream_handle) const noexcept override {
		func(reinterpret_cast<CUstream>(stream_handle));
	}
public:
    explicit CudaLCubCommand(luisa::function<void(CUstream)> f) noexcept
        : CustomCommand{}, func{std::move(f)} {}
    [[nodiscard]] StreamTag stream_tag() const noexcept override { return StreamTag::COMPUTE; }
    [[nodiscard]] uint64_t uuid() const noexcept override {
        return static_cast<uint64_t>(CustomCommandUUID::CUDA_LCUB_COMMAND);
    }
};

}// namespace luisa::compute::cuda
