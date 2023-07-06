#include <luisa/luisa-compute.h>
#include <luisa/backends/ext/dml_ext.h>
#include <fstream>
#include <iostream>
using namespace luisa;
using namespace luisa::compute;


template<int I, int O>
void NNLayerRelu(const float* inputs, float* outputs, const float* w)
{
    for (int o = 0; o < O; o++)
    {
        float res = 0;
        const float* b = w + I * o;
        for (int i = 0; i < I; i += 2)
        {
            res += inputs[i] * b[i];
            res += inputs[i + 1] * b[i + 1];
        }
        outputs[o] = std::max(0.f, res);
    }
}
template<int I, int O>
void NNLayer(const float* inputs, float* outputs, const float* w)
{
    for (int o = 0; o < O; o++)
    {
        float res = 0;
        const float* b = w + I * o;
        for (int i = 0; i < I; i++)
        {
            res += inputs[i] * b[i];
        }
        outputs[o] = res;
    }
}
template<int I, int H, int O>
void NNForward(const float* inputs, float* outputs, const float* weights)
{
    const float* weightHead = weights;
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

int main(int argc, char* argv[])
{
	auto ctx = luisa::compute::Context(argv[0]);
	auto device = ctx.create_device("dx");
	auto stream = device.create_stream();
	auto ext = device.extension<luisa::compute::DirectMLExt>();

	auto graph = ext->Build(stream, 1, 64, 3, 128, 50, false);

    std::vector<float> inputs{};
    inputs.resize(64);
    inputs[3] = 1.f;
    std::vector<float> outputs{};
    std::vector<float> weights{};
    outputs.resize(50);
    auto weightSize = 64 * 128 + 128 * 128 + 128 * 128 + 128 * 50;
    weights.resize(weightSize);
    std::ifstream file("E:/OLLF/CPP_params.txt");
    if (file.good())
    {
        std::string str;
        int line = 0;
        while (std::getline(file, str))
        {
            weights[line] = std::atof(str.c_str());
            line++;
        }
        file.close();
    }

	auto ipt = device.create_buffer<float>(64);
	auto w = device.create_buffer<float>(weightSize);
    auto opt = device.create_buffer<float>(50);

	stream << w.copy_from(weights.data())<<ipt.copy_from(inputs.data());
    stream << ext->Forward(graph.get(), ipt, opt, w);
	stream << luisa::compute::synchronize();
	stream<<opt.copy_to(outputs.data())<< luisa::compute::synchronize();

    std::vector<float> outputsCPU{};
    outputsCPU.resize(50);
    NNForward<64, 128, 50>(inputs.data(), outputsCPU.data(), weights.data());

    float error = 0.f;
    for (int i = 0; i < outputsCPU.size(); i++)
    {
        error += abs(outputs[i] - outputsCPU[i]);
    }
    std::cout << error << std::endl;
}