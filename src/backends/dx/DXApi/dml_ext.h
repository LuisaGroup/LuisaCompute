#pragma once
#include <luisa/backends/ext/dml_ext.h>
#include "../d3dx12.h"

struct IDMLOperator;
struct IDMLCompiledOperator;
class IDMLDevice;
namespace lc::dx {
class LCDevice;
using namespace luisa::compute;
class Device;

class DxDirectMLExt final : public DirectMLExt, public vstd::IOperatorNewBase {
    std::mutex dmlDeviceMtx;

public:
    DeviceInterface *device;
    DxDirectMLExt(DeviceInterface *device);
    ~DxDirectMLExt(){};
    luisa::unique_ptr<DMLGraph> create_graph(
        uint32_t batch_size,
        uint32_t input_elements,
        uint32_t out_elements,
        luisa::span<const uint32_t> hidden_layer_elements,
        luisa::span<const FusedActivation> activations,
        bool half_precision) noexcept override;
};
}// namespace lc::dx