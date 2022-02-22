
#include <algorithm>
#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <dsl/syntax.h>

using namespace luisa;
using namespace luisa::compute;


// test 2D texture of float4  (without offset & region)
int test_texture_upload_download(Device& device, int width, int height, int lod)
{
    LUISA_INFO("test_texture_upload_download {}x{}, lod={}", width, height, lod);
    auto image0 = device.create_image<float>(PixelStorage::FLOAT4, width, height, lod);
    int total_size = 0;
    for (int i=0; i<lod; ++i)
        total_size += max(1, width>>i) * max(1, height>>i);

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
    // download texture
    std::vector<float> output(total_size * 4);
    for (int i=0, offset=0; i<lod; ++i) {
        stream << image0.view(i).copy_to(output.data() + offset*4);
        offset += max(1, width>>i) * max(1, height>>i);
    }
    stream << synchronize();
    // check correctness
    for (int i=0; i<total_size*4; ++i) {
        if (input[i] != output[i]) {
            LUISA_ERROR("ERROR: data mismatch");
            return -1;
        }
    }
    LUISA_INFO("OK");
    return 0;
}


int main(int argc, char* argv[])
{
    log_level_info();
    // log_level_verbose();
    Context context{argv[0]};
    auto device = context.create_device("ispc");

    test_texture_upload_download(device, 1024, 1024, 1) ||
    test_texture_upload_download(device, 346, 987, 1) ||
    test_texture_upload_download(device, 4567, 4575, 1) ||
    test_texture_upload_download(device, 1234567, 3, 1) ||
    test_texture_upload_download(device, 1234567, 1, 1) ||
    test_texture_upload_download(device, 1234, 1234, 5) ||
    test_texture_upload_download(device, 1234567, 1, 20) ||
    test_texture_upload_download(device, 1234, 1234, 11);
}
