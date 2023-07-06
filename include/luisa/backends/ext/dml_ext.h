#pragma once

#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/luisa-compute.h>

namespace luisa::compute 
{
    class DMLGraph
    {
    };
    class DirectMLExt : public DeviceExtension 
    {

    public:
        ~DirectMLExt() noexcept = default;
        static constexpr luisa::string_view name = "DirectMLExt";
        virtual unique_ptr<DMLGraph> Build(Stream& stream, int batchSize, int input, int layer, int hiddenDim, int output, bool half) = 0;
        virtual unique_ptr<Command> Forward(DMLGraph* graph, Resource& input, Resource& output, Resource& weights) = 0;
    };

}// namespace luisa::compute

