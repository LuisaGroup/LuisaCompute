#pragma once

#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/buffer.h>
namespace luisa::compute {
class DMLGraph {
public:
    virtual ~DMLGraph() noexcept = default;
    [[nodiscard]] virtual unique_ptr<Command> build(int batchSize, int input, int layer, int hiddenDim, int output) noexcept = 0;

protected:
    [[nodiscard]] virtual unique_ptr<Command> forward(
        Argument::Buffer input_buffer,
        Argument::Buffer output_buffer,
        Argument::Buffer weights_buffer) noexcept = 0;

public:
    template<typename InputBuffer, typename OutputBuffer, typename WeightBuffer>
        requires is_buffer_or_view_v<InputBuffer> &&
                 is_buffer_or_view_v<OutputBuffer> &&
                 is_buffer_or_view_v<WeightBuffer>
    [[nodiscard]] unique_ptr<Command> forward(InputBuffer const &input, OutputBuffer const &output, WeightBuffer const &weights) noexcept {
        auto to_buffer_arg = []<typename T>(const T &t) noexcept {
            if constexpr (is_buffer_view_v<T>) {
                return Argument::Buffer{
                    t.handle(),
                    t.offset_bytes(),
                    t.size_bytes()};
            } else {
                return Argument::Buffer{
                    t.handle(),
                    0ull,
                    t.size_bytes()};
            }
        };
        return forward(to_buffer_arg(input), to_buffer_arg(output), to_buffer_arg(weights));
    }
};

class DirectMLExt : public DeviceExtension {
public:
    ~DirectMLExt() noexcept = default;
    static constexpr luisa::string_view name = "DirectMLExt";
    [[nodiscard]] virtual unique_ptr<DMLGraph> create_graph(bool half_precision) noexcept = 0;
};

}// namespace luisa::compute
