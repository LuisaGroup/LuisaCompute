#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/core/logging.h>
using namespace luisa::compute;
int main(int argc, char *argv[]) {
    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    // get hardware-device names
    luisa::vector<luisa::string> device_names = context.backend_device_names(argv[1]);
    if (device_names.empty()) {
        LUISA_WARNING("No haredware device found.");
        exit(1);
    }
    size_t device_index = 0;
    for(auto&& i : device_names){
        LUISA_INFO("Found hardware device: {}", i);
    }
    // find device with "GeForce" or "Radeon RX" or "Arc"
    for (size_t i = 0; i < device_names.size(); ++i) {
        luisa::string& device_name = device_names[i];
        if (device_name.find("GeForce") != luisa::string::npos ||
            device_name.find("Radeon RX") != luisa::string::npos ||
            device_name.find("Arc") != luisa::string::npos) {
            LUISA_INFO("Select device: {}", device_name);
            device_index = i;
        }
    }
    DeviceConfig device_config{
        .device_index = device_index};
    Device device = context.create_device(argv[1], &device_config);
}
