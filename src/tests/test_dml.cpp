#include <luisa/luisa-compute.h>
#include <luisa/backends/ext/dml_ext.h>
#include <fstream>
using namespace luisa;
using namespace luisa::compute;

template<int I, int O>
void NNLayerRelu(const float *inputs, float *outputs, const float *w) {
    for (int o = 0; o < O; o++) {
        float res = 0;
        const float *b = w + I * o;
        for (int i = 0; i < I; i += 2) {
            res += inputs[i] * b[i];
            res += inputs[i + 1] * b[i + 1];
        }
        outputs[o] = std::max(0.f, res);
    }
}
template<int I, int O>
void NNLayer(const float *inputs, float *outputs, const float *w) {
    for (int o = 0; o < O; o++) {
        float res = 0;
        const float *b = w + I * o;
        for (int i = 0; i < I; i++) {
            res += inputs[i] * b[i];
        }
        outputs[o] = res;
    }
}
template<int I, int H, int O>
void NNForward(const float *inputs, float *outputs, const float *weights) {
    const float *weightHead = weights;
    float cache0[H];
    float cache1[H];

    NNLayerRelu<I, H>(inputs, cache0, weightHead);
    weightHead += (I)*H;

    NNLayerRelu<H, H>(cache0, cache1, weightHead);
    weightHead += (H)*H;
    NNLayerRelu<H, H>(cache1, cache0, weightHead);
    weightHead += (H)*H;
    NNLayer<H, O>(cache0, outputs, weightHead);
}

int main(int argc, char *argv[]) {
    auto ctx = Context(argv[0]);
    auto device = ctx.create_device("dx");
    auto stream = device.create_stream();
    auto ext = device.extension<DirectMLExt>();
    FusedActivation activation[] = {
        // Try different activation
        FusedActivation::none(),
        FusedActivation::none()
        // FusedActivation::scaled_elu()
        // FusedActivation::sigmoid()
    };
    uint hidden_layer[] = {
        2};
    auto graph = ext->create_graph(1, 2, 2, hidden_layer, activation, false);
    stream << graph->build();

    luisa::vector<float> inputs{};
    inputs.push_back(3);
    inputs.push_back(4);

    luisa::vector<float> outputs{};
    outputs.resize(graph->output_buffer_size_bytes() / sizeof(float));

    luisa::vector<float> weights{};
    auto vv = graph->weight_buffer_size_bytes();
    weights.resize(graph->weight_buffer_size_bytes() / sizeof(float));

    weights[0] = 5;
    weights[1] = 6;
    weights[2] = 7;
    weights[3] = 8;
    weights[4] = 11;
    weights[5] = 12;
    weights[6] = 13;
    weights[7] = 14;
    /*
    matrix sequence:
    [input[0], input[1]] * [weights[0], weights[2]  * activation
                            weights[1], weights[3]]
    which is:
    [input[0] * weights[0] + input[1] * weights[1],
     input[0] * weights[2] + input[0] * weights[3]] * activation
    
    with activation is none, result should be:
    [3 * 5 + 4 * 6,      =          [39,
     3 * 7 + 4 * 8]                  53]

    [39 * 11 + 53 * 12,      =          [1065,
     39 * 13 + 53 * 14]                  1249]

    */

    auto ipt = device.create_buffer<float>(inputs.size());
    auto w = device.create_buffer<float>(weights.size());
    auto opt = device.create_buffer<float>(outputs.size());

    stream << w.copy_from(weights.data()) << ipt.copy_from(inputs.data())
           << graph->forward(ipt, opt, w)
           << opt.copy_to(outputs.data()) << synchronize();
    size_t idx = 0;
    for (auto &&i : outputs) {
        LUISA_INFO("Output: {} is: {}", idx, i);
        idx++;
    }
}