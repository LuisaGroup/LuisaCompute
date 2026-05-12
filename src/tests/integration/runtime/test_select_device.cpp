// Device selection test demonstrating how to enumerate and select
// specific hardware devices by name (GeForce, Radeon RX, Arc).

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/core/logging.h>

using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_select_device(Device &device) {
    auto argv = boost::ut::detail::cfg::largv;
    Context context{argv[0]};

    // Get hardware device names for the specified backend
    luisa::vector<luisa::string> device_names = context.backend_device_names(device.backend_name());
    if (device_names.empty()) {
        LUISA_WARNING("No hardware device found.");
        expect(false) << "No hardware device found.";
        return;
    }

    // Print all available devices
    size_t device_index = 0;
    for (auto &&i : device_names) {
        LUISA_INFO("Found hardware device: {}", i);
    }

    // Find device with "GeForce" or "Radeon RX" or "Arc"
    for (size_t i = 0; i < device_names.size(); ++i) {
        luisa::string &device_name = device_names[i];
        if (device_name.find("GeForce") != luisa::string::npos ||
            device_name.find("Radeon RX") != luisa::string::npos ||
            device_name.find("Arc") != luisa::string::npos) {
            LUISA_INFO("Select device: {}", device_name);
            device_index = i;
        }
    }

    // Create device with the selected index
    DeviceConfig device_config{
        .device_index = device_index};
    [[maybe_unused]] auto selected_device = context.create_device(device.backend_name(), &device_config);
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
    auto &device = dc->device;
    test_select_device(device);
}
