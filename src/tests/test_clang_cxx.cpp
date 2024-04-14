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
#include <luisa/backends/ext/raster_ext.hpp>
#include <luisa/runtime/raster/raster_shader.h>

using namespace luisa;
using namespace luisa::compute;
using namespace std::string_view_literals;

static bool kTestRuntime = false;
static bool kUseExport = false;
static std::string kTestName = "lang";
static std::string kBackend = "dx";

int main(int argc, char *argv[]) {
    for (int i = 0; i < argc; i++) {
        auto argV = luisa::string(argv[i]);
        kTestRuntime |= (argV == "--with_runtime");
        kUseExport |= (argV == "--export");
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
    DeviceConfig config{.headless = !kTestRuntime};
    static constexpr uint width = 1920;
    static constexpr uint height = 1080;
    Device device = context.create_device(kBackend, &config);
    // compile cxx shader
    {
        auto src_relative = "./../../src/tests/cxx_shaders/test." + kTestName + ".cpp";
        auto inc_relative = "./../../src/tests/cxx_shaders";
        auto shader_path = std::filesystem::canonical(context.runtime_directory() / src_relative);
        auto include_path = std::filesystem::canonical(context.runtime_directory() / inc_relative);
        luisa::vector<luisa::string> defines;
        // Enable debug for printer
        if (kTestName == "printer") {
            defines.emplace_back("DEBUG");
        }
        auto iter = vstd::range_linker{
            vstd::make_ite_range(defines),
            vstd::transform_range{[&](auto &&v) { return luisa::string_view{v}; }}}
                        .i_range();
        auto inc_iter = vstd::range_linker{
            vstd::make_ite_range(luisa::span{&include_path, 1}),
            vstd::transform_range{
                [&](auto &&path) { return luisa::to_string(path); }}}
                            .i_range();
        if (kUseExport) {
            auto lib = luisa::clangcxx::Compiler::export_callables(
                device, iter, shader_path, inc_iter);
            luisa::string lib_str;
            for (auto &&i : lib.callable_map()) {
                lib_str += i.first;
                lib_str += ": ";
                lib_str += i.second->return_type()->description();
                lib_str += "(";
                for (auto &&a : i.second->arguments()) {
                    lib_str += a.type()->description();
                    lib_str += ", ";
                }
                if (!i.second->arguments().empty())
                    lib_str.erase(lib_str.end() - 2, lib_str.end());
                lib_str += ")\n";
            }
            LUISA_INFO("Export functions: \n{}", lib_str);
            auto ser_data = lib.serialize();
            LUISA_INFO("Serialized size: {} bytes", ser_data.size());

        } else {
            luisa::clangcxx::Compiler::create_shader(
                ShaderOption{
                    .compile_only = true,
                    .name = "test.bin"},
                device, iter, shader_path, inc_iter);
        }
    }
    if (kTestRuntime) {
        Callable linear_to_srgb = [&](Var<float3> x) noexcept {
            return saturate(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                                   12.92f * x,
                                   x <= 0.00031308f));
        };
        if (kTestName == "mandelbrot") {
            Stream stream = device.create_stream(StreamTag::GRAPHICS);
            auto mandelbrot_shader = device.load_shader<2, Image<float>>("test.bin");
            auto texture = device.create_image<float>(PixelStorage::FLOAT4, width, height);
            Window window{"test func", uint2(width, height)};
            Swapchain swap_chain{device.create_swapchain(
                stream,
                SwapchainOption{
                    .display = window.native_display(),
                    .window = window.native_handle(),
                    .size = uint2(width, height),
                    .wants_hdr = false,
                    .wants_vsync = false,
                    .back_buffer_count = 2})};
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
            stream << synchronize();
        }
        if (kTestName == "printer") {
            auto printer_shader = device.load_shader<1>("test.bin");
            Stream stream = device.create_stream();
            stream.set_log_callback([&](auto &&f) {
                LUISA_INFO("device: {}", f);
            });
            stream << printer_shader().dispatch(64) << synchronize();
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
        if (kTestName == "raster") {
            auto raster_ext = device.extension<RasterExt>();
            auto types = {
                Type::of<float4x4>(),
                Type::of<float3>()};
            auto shader = device.load_raster_shader<float, float>("test.bin");
        }
    }
    return 0;
}