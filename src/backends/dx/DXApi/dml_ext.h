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
    luisa::unique_ptr<DMLGraph> create_graph(bool half) noexcept override;
};
}// namespace lc::dx