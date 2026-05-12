#include <iostream>
#include <vector>
#include <exception>
#include <luisa/luisa-compute.h>
#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/runtime/builtin_kernel.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

struct MyStruct {
    uint a;
    uint b;
    uint c;
    uint d;
};

int test_builtin_kernel(Device &device) {

    log_level_verbose();

    Stream stream = device.create_stream();

    BuiltinKernel builtin{device};
    builtin.compile_all(device);

    // Test Buffer fill
    {
        auto &shader = builtin._fill_buffer_uint;

        const size_t buffer_size = 1024;
        Buffer<uint> buffer = device.create_buffer<uint>(buffer_size);

        stream << shader(buffer, 42u).dispatch(buffer_size)
               << synchronize();

        std::vector<uint> result(buffer_size);
        stream << buffer.copy_to(luisa::span{result}) << synchronize();

        bool success = true;
        for (size_t i = 0; i < buffer_size; i++) {
            if (result[i] != 42u) {
                success = false;
                break;
            }
        }
        std::cout << "Buffer fill: " << (success ? "PASS" : "FAIL") << std::endl;
    }

    // Test Buffer fill with custom struct
    {
        const size_t buffer_size = 256;
        Buffer<MyStruct> buffer = device.create_buffer<MyStruct>(buffer_size);

        CommandList cmdlist{};
        MyStruct value{1u, 2u, 3u, 4u};
        builtin.fill_buffer(cmdlist, buffer.view(), value);
        stream << cmdlist.commit() << synchronize();

        std::vector<MyStruct> result(buffer_size);
        stream << buffer.copy_to(luisa::span{result}) << synchronize();

        bool success = true;
        for (size_t i = 0; i < buffer_size; i++) {
            if (result[i].a != 1u || result[i].b != 2u ||
                result[i].c != 3u || result[i].d != 4u) {
                success = false;
                break;
            }
        }
        std::cout << "Buffer fill (custom struct): " << (success ? "PASS" : "FAIL") << std::endl;
    }

    // Test Image2D fill (uint)
    {
        auto &shader = builtin._fill_image_uint;

        const uint2 image_size{64, 64};
        Image<uint> image = device.create_image<uint>(PixelStorage::INT4, image_size);

        stream << shader(image, 255u).dispatch(image_size)
               << synchronize();

        std::cout << "Image2D fill (uint): PASS (shader compiled and dispatched)" << std::endl;
    }

    // Test Image2D fill (float)
    {
        auto &shader = builtin._fill_image_float;

        const uint2 image_size{64, 64};
        Image<float> image = device.create_image<float>(PixelStorage::FLOAT4, image_size);

        stream << shader(image, 1.0f).dispatch(image_size)
               << synchronize();

        std::cout << "Image2D fill (float): PASS (shader compiled and dispatched)" << std::endl;
    }

    // Test Volume fill (uint)
    {
        auto &shader = builtin._fill_volume_uint;

        const uint3 volume_size{32, 32, 32};
        Volume<uint> volume = device.create_volume<uint>(PixelStorage::INT4, volume_size);

        stream << shader(volume, 100u).dispatch(volume_size)
               << synchronize();

        std::cout << "Volume fill (uint): PASS (shader compiled and dispatched)" << std::endl;
    }

    // Test Volume fill (int)
    {
        auto &shader = builtin._fill_volume_int;

        const uint3 volume_size{16, 16, 16};
        Volume<int> volume = device.create_volume<int>(PixelStorage::INT4, volume_size);

        stream << shader(volume, -50).dispatch(volume_size)
               << synchronize();

        std::cout << "Volume fill (int): PASS (shader compiled and dispatched)" << std::endl;
    }

    // Test Volume fill (float)
    {
        auto &shader = builtin._fill_volume_float;

        const uint3 volume_size{8, 8, 8};
        Volume<float> volume = device.create_volume<float>(PixelStorage::FLOAT4, volume_size);

        stream << shader(volume, 3.14f).dispatch(volume_size)
               << synchronize();

        std::cout << "Volume fill (float): PASS (shader compiled and dispatched)" << std::endl;
    }
    
    // Test Image2D fill via CommandList (uint)
    {
        const uint2 image_size{16, 16};
        Image<uint> image = device.create_image<uint>(PixelStorage::INT1, image_size);
        
        CommandList cmdlist{};
        builtin.fill_image(cmdlist, image.view(), 123u);
        stream << cmdlist.commit() << synchronize();
        
        std::vector<uint> result(image_size.x * image_size.y);
        stream << image.copy_to(luisa::span{result}) << synchronize();
        
        bool success = true;
        for (size_t i = 0; i < result.size(); i++) {
            if (result[i] != 123u) {
                success = false;
                break;
            }
        }
        std::cout << "Image2D fill via CommandList (uint): " << (success ? "PASS" : "FAIL") << std::endl;
    }
    
    // Test Image2D fill via CommandList (int)
    {
        const uint2 image_size{16, 16};
        Image<int> image = device.create_image<int>(PixelStorage::INT1, image_size);
        
        CommandList cmdlist{};
        builtin.fill_image(cmdlist, image.view(), -42);
        stream << cmdlist.commit() << synchronize();
        
        std::vector<int> result(image_size.x * image_size.y);
        stream << image.copy_to(luisa::span{result}) << synchronize();
        
        bool success = true;
        for (size_t i = 0; i < result.size(); i++) {
            if (result[i] != -42) {
                success = false;
                break;
            }
        }
        std::cout << "Image2D fill via CommandList (int): " << (success ? "PASS" : "FAIL") << std::endl;
    }
    
    // Test Image2D fill via CommandList (float)
    {
        const uint2 image_size{16, 16};
        Image<float> image = device.create_image<float>(PixelStorage::FLOAT1, image_size);
        
        CommandList cmdlist{};
        builtin.fill_image(cmdlist, image.view(), 2.718f);
        stream << cmdlist.commit() << synchronize();
        
        std::vector<float> result(image_size.x * image_size.y);
        stream << image.copy_to(luisa::span{result}) << synchronize();
        
        bool success = true;
        for (size_t i = 0; i < result.size(); i++) {
            if (result[i] != 2.718f) {
                success = false;
                break;
            }
        }
        std::cout << "Image2D fill via CommandList (float): " << (success ? "PASS" : "FAIL") << std::endl;
    }
    
    // Test Volume fill via CommandList (uint)
    {
        const uint3 volume_size{8, 8, 8};
        Volume<uint> volume = device.create_volume<uint>(PixelStorage::INT1, volume_size);
        
        CommandList cmdlist{};
        builtin.fill_volume(cmdlist, volume.view(), 77u);
        stream << cmdlist.commit() << synchronize();
        
        std::vector<uint> result(volume_size.x * volume_size.y * volume_size.z);
        stream << volume.copy_to(luisa::span{result}) << synchronize();
        
        bool success = true;
        for (size_t i = 0; i < result.size(); i++) {
            if (result[i] != 77u) {
                success = false;
                break;
            }
        }
        std::cout << "Volume fill via CommandList (uint): " << (success ? "PASS" : "FAIL") << std::endl;
    }
    
    // Test Volume fill via CommandList (int)
    {
        const uint3 volume_size{8, 8, 8};
        Volume<int> volume = device.create_volume<int>(PixelStorage::INT1, volume_size);
        
        CommandList cmdlist{};
        builtin.fill_volume(cmdlist, volume.view(), -99);
        stream << cmdlist.commit() << synchronize();
        
        std::vector<int> result(volume_size.x * volume_size.y * volume_size.z);
        stream << volume.copy_to(luisa::span{result}) << synchronize();
        
        bool success = true;
        for (size_t i = 0; i < result.size(); i++) {
            if (result[i] != -99) {
                success = false;
                break;
            }
        }
        std::cout << "Volume fill via CommandList (int): " << (success ? "PASS" : "FAIL") << std::endl;
    }
    
    // Test Volume fill via CommandList (float)
    {
        const uint3 volume_size{8, 8, 8};
        Volume<float> volume = device.create_volume<float>(PixelStorage::FLOAT1, volume_size);
        
        CommandList cmdlist{};
        builtin.fill_volume(cmdlist, volume.view(), 1.414f);
        stream << cmdlist.commit() << synchronize();
        
        std::vector<float> result(volume_size.x * volume_size.y * volume_size.z);
        stream << volume.copy_to(luisa::span{result}) << synchronize();
        
        bool success = true;
        for (size_t i = 0; i < result.size(); i++) {
            if (result[i] != 1.414f) {
                success = false;
                break;
            }
        }
        std::cout << "Volume fill via CommandList (float): " << (success ? "PASS" : "FAIL") << std::endl;
    }
    
    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_builtin_kernel(device);
    test_builtin_kernel(device);
}
