
#include <algorithm>
#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <dsl/syntax.h>

using namespace luisa;
using namespace luisa::compute;

using std::max;
namespace {
enum TestType {
    SAME_INSTANCE = 0,
    COPY_TEXTURE = 1,
    COPY_BUFFER = 2,
    RW_TEXTURE = 3
};
}

// test 2D texture of float4  (without offset & region)
int test_texture_upload_download(Device& device, int width, int height, int lod, TestType test_type)
{
    LUISA_INFO("===================================================");
    LUISA_INFO("test_texture_upload_download {}x{}, lod={}, test={}", width, height, lod, test_type);
    auto image0 = device.create_image<float>(PixelStorage::FLOAT4, width, height, lod);
    auto image1 = device.create_image<float>(PixelStorage::FLOAT4, width, height, lod);
    int total_size = 0;
    for (int i=0; i<lod; ++i) {
        // LUISA_INFO("i={} A:{} B:{}", i, max(1, width>>i), max(1, height>>i));
        total_size += max(1, width>>i) * max(1, height>>i);
    }
    LUISA_INFO("total_size = {}", total_size);

    // random generate data
    std::vector<float> input(total_size * 4);
    for (int i=0; i<total_size*4; ++i)
        input[i] = (float)rand()*1e-6;
    // upload texture
    auto stream = device.create_stream();
    for (int i=0, offset=0; i<lod; ++i) {
        stream << image0.view(i).copy_from(input.data() + offset*4);
        offset += max(1, width>>i) * max(1, height>>i);
    }

    if (test_type == COPY_TEXTURE) {
        for (int i=0, offset=0; i<lod; ++i)
            stream << image1.view(i).copy_from(image0.view(i));
    }
    if (test_type == COPY_BUFFER) {
        auto buf = device.create_buffer<float4>(width * height);
        for (int i=0, offset=0; i<lod; ++i) {
            stream << image0.view(i).copy_to(buf.view());
            stream << image1.view(i).copy_from(buf.view());
        }
    }
    if (test_type == RW_TEXTURE) {
        Kernel2D rw_kernel = [](ImageVar<float> image0, ImageVar<float> image1) noexcept {
            Var coord = dispatch_id().xy();
            image1.write(coord, image0.read(coord));
        };
        Kernel2D fill_kernel = [](ImageVar<float> image1) noexcept {
            Var coord = dispatch_id().xy();
            image1.write(coord, make_float4(1.0f));
        };
        auto shader = device.compile(rw_kernel);
        auto shader1 = device.compile(fill_kernel);
        for (int i=0, offset=0; i<lod; ++i)
            stream << shader( image0.view(i), image1.view(i)).dispatch(max(width>>i,1), max(height>>i,1));
            // stream << shader1(image1.view(i)).dispatch(max(width>>i,1), max(height>>i,1));
        // stream << synchronize();
        LUISA_WARNING("=====3");
    }

    // download texture
    std::vector<float> output(total_size * 4);
    for (int i=0, offset=0; i<lod; ++i) {
        stream << (test_type? image1: image0).view(i).copy_to(output.data() + offset*4);
        offset += max(1, width>>i) * max(1, height>>i);
    }
    stream << synchronize();
    // check correctness
    for (int i=0; i<total_size*4; ++i) {
        if (input[i] != output[i]) {
            LUISA_ERROR("data mismatch at i={}", i);
            return -1;
        }
    }
    LUISA_INFO("OK");
    return 0;
}



int main(int argc, char* argv[])
{
    // log_level_info();
    log_level_verbose();
    Context context{argv[0]};
    auto device = context.create_device("ispc");

    for (int i=3; i<4; ++i) {
        TestType test_type = (TestType)i;
        test_texture_upload_download(device, 1024, 1024, 1, test_type) ||
        test_texture_upload_download(device, 346, 987, 1, test_type) ||
        test_texture_upload_download(device, 4567, 4575, 1, test_type) ||
        test_texture_upload_download(device, 1234567, 3, 1, test_type) ||
        test_texture_upload_download(device, 1234567, 1, 1, test_type) ||
        test_texture_upload_download(device, 1024, 1024, 5, test_type) ||
        test_texture_upload_download(device, 1234, 1234, 5, test_type) ||
        test_texture_upload_download(device, 1234567, 1, 20, test_type) ||
        test_texture_upload_download(device, 1234, 1234, 11, test_type);
    }
}
