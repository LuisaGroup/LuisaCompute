// Example: Using a custom DirectX Agility SDK via DirectXDeviceConfigExt
//
// This example demonstrates how to programmatically specify a custom Agility SDK
// version and DLL path at device creation time by subclassing DirectXDeviceConfigExt.
//
// Run (ensure your custom D3D12Core.dll is in the path returned by GetSDKPath()):
//   ./bin/example_dx_custom_sdk

#include <iostream>
#include <luisa/luisa-compute.h>
#include <luisa/backends/ext/dx_config_ext.h>

using namespace luisa::compute;

// A custom DirectXDeviceConfigExt that specifies a custom Agility SDK.
// Override GetSDKVersion() and GetSDKPath() to point to your SDK.
//
// NOTE: This example uses hardcoded values. In your application, you might
// read these from environment variables, a configuration file, or command-line args.
class CustomSDKConfig final : public DirectXDeviceConfigExt {
public:
    // Return a custom Agility SDK version.
    // Return 0 (default) to keep the built-in D3D12_PREVIEW_SDK_VERSION (717).
    [[nodiscard]] uint32_t GetSDKVersion() const noexcept override {
        // Example: use Agility SDK version 619 (for SM 6.9 / Work Graphs).
        // Change this to match the SDK you downloaded.
        return 619u;
    }

    // Return a custom Agility SDK DLL path.
    // Return empty (default) to keep the built-in ".\\D3D12\\".
    // The path should point to the directory containing D3D12Core.dll.
    [[nodiscard]] luisa::string_view GetSDKPath() const noexcept override {
        // Example: look for D3D12Core.dll in "./D3D12_619/"
        return "./D3D12_619/";
    }
};

int main(int argc, char *argv[]) {
    // Create context
    Context ctx{argv[0]};

    // Create the custom SDK config extension
    auto ext = luisa::make_unique<CustomSDKConfig>();

    // Fill in device configuration
    DeviceConfig config{};
    config.extension = std::move(ext);

    // Create the DX device with the custom Agility SDK configuration
    auto device = ctx.create_device("dx", &config);
    std::cout << "Device created: " << device.backend_name() << std::endl;
    std::cout << "Custom Agility SDK configured." << std::endl;
    std::cout << "Check the log output for:" << std::endl;
    std::cout << "  \"Using custom Agility SDK version: 619\"" << std::endl;
    std::cout << "  \"Using custom Agility SDK path: ./D3D12_619/\"" << std::endl;

    // Create a stream and run a simple hello-world kernel to verify the device works
    Stream stream = device.create_stream(StreamTag::COMPUTE);

    // Simple buffer write kernel
    constexpr uint N = 16u;
    auto buffer = device.create_buffer<uint>(N);
    auto kernel = device.compile<1>([&]() noexcept {
        auto tid = dispatch_x();
        buffer->write(tid, tid * 2u);
    });

    stream << kernel().dispatch(N) << synchronize();
    std::vector<uint> result(N);
    stream << buffer.copy_to(result.data()) << synchronize();

    bool ok = true;
    for (auto i = 0u; i < N && ok; i++) {
        if (result[i] != i * 2u) {
            std::cerr << "Mismatch at index " << i
                      << ": expected " << (i * 2u)
                      << ", got " << result[i] << std::endl;
            ok = false;
        }
    }
    if (ok) {
        std::cout << "Kernel execution verified successfully." << std::endl;
    }

    return ok ? 0 : 1;
}
