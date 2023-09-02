#pragma once

namespace luisa::compute::graph {
class GraphCaptureCommand {
public:
    virtual void capture_on(uint64_t stream_handle) const noexcept = 0;
};
}// namespace luisa::compute::graph