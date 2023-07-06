#pragma once
#include <luisa/backends/ext/dml_ext.h>
#include "../d3dx12.h"

struct IDMLOperator;
struct IDMLCompiledOperator;
class IDMLDevice;
namespace lc::dx 
{
    class LCDevice;
    using namespace luisa::compute;
    class Device;


    class DxDirectMLExt final : public DirectMLExt, public vstd::IOperatorNewBase 
    {
    public:
        DeviceInterface* device;
        DxDirectMLExt(DeviceInterface* device);
        ~DxDirectMLExt() {};
        virtual luisa::unique_ptr<DMLGraph> Build(Stream& stream, int batchSize, int input, int layer, int hiddenDim, int output, bool half)override;
        virtual luisa::unique_ptr<Command> Forward(DMLGraph* graph, luisa::compute::Resource& input, luisa::compute::Resource& output, luisa::compute::Resource& weights)override;
    protected:
    };
}