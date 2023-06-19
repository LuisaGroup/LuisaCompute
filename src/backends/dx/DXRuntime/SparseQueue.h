#pragma once
#include <DXRuntime/Device.h>
#include <DXRuntime/CommandQueue.h>
#include <DXApi/CmdQueueBase.h>
#include <luisa/runtime/command_list.h>

namespace lc::dx {
using namespace luisa::compute;
class SparseQueue : public CmdQueueBase {
    
public:
    CommandQueue queue;
    SparseQueue(
        Device *device);
    ~SparseQueue();
    void Execute(
        CommandList &&cmdList);
    void Sync();
};
}// namespace lc::dx
