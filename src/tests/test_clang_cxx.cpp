#include <stb/stb_image_write.h>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/rtx/accel.h>

#include <luisa/gui/window.h>
#include <luisa/runtime/swapchain.h>

#include <luisa/clangcxx/compiler.h>

using namespace luisa;
using namespace luisa::compute;
using namespace std::string_view_literals;

static bool kTestRuntime = false;
static std::string kTestName = "lang";
static std::string kBackend = "dx";

int main(int argc, char *argv[]) {
    for (int i = 0; i < argc; i++) {
        auto argV = luisa::string(argv[i]);
        kTestRuntime |= (argV == "--with_runtime");
        auto _ = luisa::string("--test_name=");
        if (argV.starts_with(_)) {
            kTestName = argV.substr(_.size());
        }
        using namespace std::string_view_literals;
        constexpr auto backend_option = "--backend="sv;
        if (argV.starts_with(backend_option)) {
            kBackend = argV.substr(backend_option.size());
        }
    }

    Context context{argv[0]};
    // DeviceConfig config{.headless = true};
    static constexpr uint width = 1920;
    static constexpr uint height = 1080;
    Device device = context.create_device(kBackend, /*&config*/ nullptr);
    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    // compile cxx shader
    {
        auto src_relative = "./../../src/tests/cxx_shaders/test." + kTestName + ".cpp";
        auto inc_relative = "./../../src/tests/cxx_shaders";
        auto shader_path = std::filesystem::canonical(context.runtime_directory() / src_relative);
        auto include_path = std::filesystem::canonical(context.runtime_directory() / inc_relative);
        auto compiler = luisa::clangcxx::Compiler(
            ShaderOption{
                .compile_only = true,
                .name = "test.bin"});
        compiler.create_shader(context, device, {}, shader_path, include_path);
        LUISA_INFO("{}", luisa::clangcxx::Compiler::lsp_compile_commands(
            context, {}, include_path, "test." + kTestName + ".cpp", include_path
        ));
    }
    if (kTestRuntime) {
        Callable linear_to_srgb = [&](Var<float3> x) noexcept {
            return saturate(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                                   12.92f * x,
                                   x <= 0.00031308f));
        };
        if (kTestName == "mandelbrot") {
            auto mandelbrot_shader = device.load_shader<2, Image<float>>("test.bin");
            auto texture = device.create_image<float>(PixelStorage::FLOAT4, width, height);
            Window window{"test func", uint2(width, height)};
            Swapchain swap_chain{device.create_swapchain(
                window.native_handle(),
                stream,
                uint2(width, height),
                false, false, 2)};
            auto ldr_image = device.create_image<float>(swap_chain.backend_storage(), uint2(width, height));

            Kernel2D hdr2ldr_kernel = [&](ImageVar<float> hdr_image, ImageFloat ldr_image, Float scale, Bool is_hdr) noexcept {
                UInt2 coord = dispatch_id().xy();
                Float4 hdr = hdr_image.read(coord);
                Float3 ldr = hdr.xyz() * scale;
                $if (!is_hdr) {
                    ldr = linear_to_srgb(ldr);
                };
                ldr_image.write(coord, make_float4(ldr, 1.0f));
            };
            auto hdr2ldr_shader = device.compile(hdr2ldr_kernel);

            auto blk_size = mandelbrot_shader.block_size();
            LUISA_INFO("{}, {}, {}", blk_size.x, blk_size.y, blk_size.z);
            while (!window.should_close()) {
                window.poll_events();
                stream << mandelbrot_shader(texture).dispatch(width, height)
                       << hdr2ldr_shader(texture, ldr_image, 1.0f, true).dispatch(width, height)
                       << swap_chain.present(ldr_image);
            }
        }
        if (kTestName == "rtx") {
            auto raytracing_shader = device.load_shader<2, Buffer<float4>, Accel, uint32_t>("test.bin");

            Kernel2D colorspace_kernel = [&](BufferFloat4 hdr_image, BufferUInt ldr_image) noexcept {
                UInt i = dispatch_y() * dispatch_size_x() + dispatch_x();
                Float3 hdr = hdr_image.read(i).xyz();
                UInt3 ldr = make_uint3(round(clamp(linear_to_srgb(hdr), 0.f, 1.f) * 255.0f));
                ldr_image.write(i, ldr.x | (ldr.y << 8u) | (ldr.z << 16u) | (255u << 24u));
            };
            auto colorspace_shader = device.compile(colorspace_kernel);

            Kernel1D set_transform_kernel = [&](AccelVar accel, Float4x4 matrix, UInt offset) noexcept {
                accel.set_instance_transform(dispatch_id().x + offset, matrix);
            };
            auto set_transform_shader = device.compile(set_transform_kernel);

            std::array vertices{
                float3(-0.5f, -0.5f, 0.0f),
                float3(0.5f, -0.5f, 0.0f),
                float3(0.0f, 0.5f, 0.0f)};
            std::array indices{0u, 1u, 2u};
            Stream stream = device.create_stream();
            Buffer<float3> vertex_buffer = device.create_buffer<float3>(3u);
            Buffer<Triangle> triangle_buffer = device.create_buffer<Triangle>(1u);
            stream << vertex_buffer.copy_from(vertices.data())
                   << triangle_buffer.copy_from(indices.data());

            Accel accel = device.create_accel();
            Mesh mesh = device.create_mesh(vertex_buffer, triangle_buffer);
            accel.emplace_back(mesh, scaling(1.5f));
            accel.emplace_back(mesh, translation(float3(-0.25f, 0.0f, 0.1f)) *
                                         rotation(float3(0.0f, 0.0f, 1.0f), 0.5f));
            stream << mesh.build() << accel.build();

            static constexpr uint width = 512u;
            static constexpr uint height = 512u;
            Buffer<float4> hdr_image = device.create_buffer<float4>(width * height);
            Buffer<uint> ldr_image = device.create_buffer<uint>(width * height);
            std::vector<uint8_t> pixels(width * height * 4u);

            Clock clock;
            clock.tic();
            static constexpr uint spp = 1024u;
            for (uint i = 0u; i < spp; i++) {
                float t = static_cast<float>(i) * (1.0f / spp);
                vertices[2].y = 0.5f - 0.2f * t;
                float4x4 m = translation(float3(-0.25f + t * 0.15f, 0.0f, 0.1f)) *
                             rotation(float3(0.0f, 0.0f, 1.0f), 0.5f + t * 0.5f);

                stream << vertex_buffer.copy_from(vertices.data())
                       << set_transform_shader(accel, m, 1u).dispatch(1)
                       << mesh.build()
                       << accel.build()
                       << raytracing_shader(hdr_image, accel, i).dispatch(width, height);
                if (i == 511u) {
                    float4x4 mm = translation(make_float3(0.0f, 0.0f, 0.3f)) *
                                  rotation(make_float3(0.0f, 0.0f, 1.0f), radians(180.0f));
                    accel.emplace_back(mesh, mm, true);
                    stream << accel.update_instance_buffer();
                }
            }
            stream << colorspace_shader(hdr_image, ldr_image).dispatch(width, height)
                   << ldr_image.copy_to(pixels.data())
                   << synchronize();
            double time = clock.toc();

            LUISA_INFO("Time: {} ms", time);
            stbi_write_png("test_rtx.png", width, height, 4, pixels.data(), 0);
        }
    }
    return 0;
}