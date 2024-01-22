#pragma once

#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/buffer.h>
namespace luisa::compute {
struct FusedActivation {
    enum class Type : uint32_t {
        NONE,
        ELU,
        HARD_SIGMOID,
        IDENTITY,
        LEAKY_RELU,
        LINEAR,
        PARAMETRIC_SOFTPLUS,
        RELU,
        SCALED_ELU,
        SCALED_TANH,
        SIGMOID,
        SOFTPLUS,
        SOFTSIGN,
        TANH,
        THRESHOLDED_RELU,
        SHRINK,
        CELU,
    };
    Type type = Type::NONE;
    float param1 = 0.0f;
    float param2 = 0.0f;
    FusedActivation() = default;

    explicit FusedActivation(Type type, float param1 = 0.0f, float param2 = 0.0f)
        : type(type), param1(param1), param2(param2) {}

    static FusedActivation none() noexcept {
        return FusedActivation();
    }

    static FusedActivation elu(float alpha = 1.0f) noexcept {
        return FusedActivation(Type::ELU, alpha);
    }

    static FusedActivation hard_sigmoid(float alpha = 0.2f, float beta = 0.5f) noexcept {
        return FusedActivation(Type::HARD_SIGMOID, alpha, beta);
    }

    static FusedActivation identity() noexcept {
        return FusedActivation(Type::IDENTITY);
    }

    static FusedActivation leakly_relu(float alpha = 0.01f) noexcept {
        return FusedActivation(Type::LEAKY_RELU, alpha);
    }

    static FusedActivation linear(float alpha, float beta) noexcept {
        return FusedActivation(Type::LINEAR, alpha, beta);
    }

    static FusedActivation parametric_softplus(float alpha, float beta) noexcept {
        return FusedActivation(Type::PARAMETRIC_SOFTPLUS, alpha, beta);
    }

    static FusedActivation relu() noexcept {
        return FusedActivation(Type::RELU);
    }

    static FusedActivation scaled_elu(float alpha = 1.67326319217681884765625f, float gamma = 1.05070102214813232421875f) noexcept {
        return FusedActivation(Type::SCALED_ELU, alpha, gamma);
    }

    static FusedActivation scaled_tanh(float alpha = 1.0f, float beta = 0.5f) noexcept {
        return FusedActivation(Type::SCALED_TANH, alpha, beta);
    }

    static FusedActivation sigmoid() noexcept {
        return FusedActivation(Type::SIGMOID);
    }

    static FusedActivation softplus(float steepness = 1.0f) noexcept {
        return FusedActivation(Type::SOFTPLUS, steepness);
    }

    static FusedActivation softsign() noexcept {
        return FusedActivation(Type::SOFTSIGN);
    }

    static FusedActivation tanh() noexcept {
        return FusedActivation(Type::TANH);
    }

    static FusedActivation thresholded_relu(float alpha = 1.0f) noexcept {
        return FusedActivation(Type::THRESHOLDED_RELU, alpha);
    }

    static FusedActivation shrink(float bias = 0.0f, float threshold = 0.5f) noexcept {
        return FusedActivation(Type::SHRINK, bias, threshold);
    }

    static FusedActivation celu(float alpha = 1.0f) noexcept {
        return FusedActivation(Type::CELU, alpha);
    }
};

class DMLGraph {
public:
    virtual ~DMLGraph() noexcept = default;
    [[nodiscard]] virtual unique_ptr<Command> build() noexcept = 0;

protected:
    [[nodiscard]] virtual unique_ptr<Command> forward(
        Argument::Buffer input_buffer,
        Argument::Buffer output_buffer,
        Argument::Buffer weights_buffer) noexcept = 0;

public:
    [[nodiscard]] virtual size_t input_buffer_size_bytes() const noexcept = 0;
    [[nodiscard]] virtual size_t output_buffer_size_bytes() const noexcept = 0;
    [[nodiscard]] virtual size_t weight_buffer_size_bytes() const noexcept = 0;
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
    [[nodiscard]] virtual unique_ptr<DMLGraph> create_graph(
        uint32_t batch_size,
        uint32_t input_elements,
        uint32_t out_elements,
        luisa::span<const uint32_t> hidden_layer_elements,
        luisa::span<const FusedActivation> activations,
        bool half_precision = false) noexcept = 0;
};

}// namespace luisa::compute
