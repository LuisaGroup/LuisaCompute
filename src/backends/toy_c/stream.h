#pragma once
#include <luisa/core/fiber.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/command_list.h>
#include <luisa/vstl/lockfree_array_queue.h>
struct MemoryManager;
namespace lc::toy_c {
class Event;
using namespace luisa;
using namespace luisa::compute;
class LCStream : public vstd::IOperatorNewBase {
public:
    struct Signal {
        Event *evt;
        uint64_t fence;
    };
    struct Wait {
        Event *evt;
        uint64_t fence;
    };
private:
    luisa::fiber::event _evt;
public:
    DeviceInterface::StreamLogCallback print_callback;
    LCStream();
    void dispatch(MemoryManager& manager, CommandList &&cmdlist);
    void sync();
    ~LCStream();
};
}// namespace lc::toy_c