#pragma once
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/command_list.h>
#include <luisa/vstl/lockfree_array_queue.h>
struct MemoryManager;
namespace lc::toy_c {
class LCDevice;
class Event;
using namespace luisa;
using namespace luisa::compute;
class LCStream : public vstd::IOperatorNewBase {
public:
    DeviceInterface::StreamLogCallback print_callback;
    LCStream();
    void dispatch(MemoryManager &manager, LCDevice *device, CommandList &&cmdlist);
    ~LCStream();
};
}// namespace lc::toy_c