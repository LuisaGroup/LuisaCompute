#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cstdarg> // va_start

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

static void log_msg(const char *fmt, ...) {
    static auto t0 = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();
    printf("[%lld ms] ", (long long)ms);
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    printf("\n");
    fflush(stdout);
}

// A simple compute kernel (no ray tracing) to test pipeline creation after RT destroy
Kernel1D clear_kernel = [](BufferFloat4 buf) noexcept {
    buf.write(dispatch_x(), make_float4(0.f));
};

// Motion blur ray tracing kernel
Callable generate_ray = [](Float2 p) noexcept {
    auto origin = make_float3(0.f, 1.5f, 2.5f);
    auto target = make_float3(0.f, 0.f, 0.f);
    auto up = make_float3(0.f, 1.f, 0.f);
    auto front = normalize(target - origin);
    auto right = normalize(cross(front, up));
    up = cross(right, front);
    auto fov = radians(45.f);
    auto image_plane_height = tan(fov / 2.f);
    auto image_plane_width = image_plane_height;
    up *= image_plane_height;
    right *= image_plane_width;
    auto uv = p / make_float2(512.f, 512.f) * 2.f - 1.f;
    return make_ray(origin, normalize(uv.x * right - uv.y * up + front));
};

Kernel2D rt_kernel = [](BufferFloat4 image, AccelVar accel, UInt frame_index) noexcept {
    auto coord = dispatch_id().xy();
    auto ray = generate_ray(make_float2(coord) + 0.5f);
    auto time = 0.5f;  // fixed time for simplicity
    auto hit = accel.intersect_motion(ray, time, {});
    auto color = make_float3(0.f);
    $if (hit->is_triangle()) {
        color = make_float3(1.f, 0.f, 0.f);
    };
    image.write(coord.y * dispatch_size_x() + coord.x, make_float4(color, 1.0f));
};

struct MotionBlurScene {
    Buffer<float3> vertex_buffer;
    Buffer<Triangle> triangle_buffer;
    Mesh mesh;
    Accel accel;
    Shader2D<Buffer<float4>, Accel, uint> rt_shader;
    Buffer<float4> image;
    Stream stream;

    static MotionBlurScene create(Device &device) {
        log_msg("  [Scene::create] creating stream...");
        auto stream = device.create_stream(StreamTag::GRAPHICS);

        log_msg("  [Scene::create] creating vertex/triangle buffers...");
        static constexpr uint keyframe_count = 2u;
        std::array vertices{
            // keyframe 0
            float3(-0.5f, -0.5f, 0.0f),
            float3(0.5f, -0.5f, 0.0f),
            float3(-0.1f, 0.5f, 0.0f),
            // keyframe 1
            float3(-0.5f, -0.5f, 0.0f),
            float3(0.5f, -0.5f, 0.0f),
            float3(0.1f, 0.5f, 0.0f),
        };
        std::array indices{0u, 1u, 2u};

        auto vertex_buffer = device.create_buffer<float3>(3u * keyframe_count);
        auto triangle_buffer = device.create_buffer<Triangle>(1u);
        stream << vertex_buffer.copy_from(luisa::span{vertices})
               << triangle_buffer.copy_from(luisa::span{indices});

        log_msg("  [Scene::create] creating mesh with motion blur...");
        AccelOption mesh_option;
        mesh_option.motion.keyframe_count = keyframe_count;
        mesh_option.motion.time_start = 0.f;
        mesh_option.motion.time_end = 1.f;
        auto mesh = device.create_mesh(vertex_buffer, triangle_buffer, mesh_option);

        log_msg("  [Scene::create] creating accel...");
        auto accel = device.create_accel();
        accel.emplace_back(mesh, scaling(2.f));

        log_msg("  [Scene::create] building mesh + accel...");
        stream << mesh.build() << accel.build() << synchronize();

        log_msg("  [Scene::create] compiling RT shader (motion blur)...");
        auto rt_shader = device.compile(rt_kernel);
        log_msg("  [Scene::create] RT shader compiled");

        log_msg("  [Scene::create] creating image buffer...");
        auto image = device.create_buffer<float4>(512u * 512u);

        log_msg("  [Scene::create] done");
        return MotionBlurScene{
            std::move(vertex_buffer),
            std::move(triangle_buffer),
            std::move(mesh),
            std::move(accel),
            std::move(rt_shader),
            std::move(image),
            std::move(stream),
        };
    }

    void render(uint frames) {
        log_msg("  [Scene::render] dispatching %u frames...", frames);
        for (uint i = 0; i < frames; i++) {
            stream << rt_shader(image, accel, i).dispatch(512u, 512u);
        }
        stream << synchronize();
        log_msg("  [Scene::render] done");
    }

    void destroy() {
        log_msg("  [Scene::destroy] synchronizing stream...");
        stream << synchronize();
        log_msg("  [Scene::destroy] releasing resources (RAII)...");
        // All resources will be destroyed by RAII when this struct goes out of scope.
        // Explicitly move to empty to trigger destruction in controlled order:
        image = {};
        log_msg("  [Scene::destroy] image released");
        rt_shader = {};
        log_msg("  [Scene::destroy] rt_shader released");
        accel = {};
        log_msg("  [Scene::destroy] accel released");
        mesh = {};
        log_msg("  [Scene::destroy] mesh released");
        triangle_buffer = {};
        log_msg("  [Scene::destroy] triangle_buffer released");
        vertex_buffer = {};
        log_msg("  [Scene::destroy] vertex_buffer released");
        stream = {};
        log_msg("  [Scene::destroy] stream released, all done");
    }
};

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <backend>\n  backend: vk, dx, cuda\n", argv[0]);
        return 1;
    }

    log_msg("=== test_mesh_8: Motion Blur Lifecycle Test ===");
    log_msg("Backend: %s", argv[1]);

    Context context{argv[0]};
    Device device = context.create_device(argv[1]);
    log_msg("Device created");

    // === ITERATION 1: Create, use, destroy motion blur scene ===
    log_msg("");
    log_msg("=== ITERATION 1: Create + Render + Destroy ===");
    {
        auto scene = MotionBlurScene::create(device);
        scene.render(16);  // Dispatch 16 frames of motion blur ray tracing
        scene.destroy();
    }
    log_msg("=== ITERATION 1 COMPLETE ===");

    // === POST-DESTROY TEST: Can we still create compute pipelines? ===
    log_msg("");
    log_msg("=== POST-DESTROY TEST: Creating new compute shader... ===");
    {
        auto stream2 = device.create_stream(StreamTag::GRAPHICS);
        log_msg("  New stream created");

        log_msg("  Compiling new compute shader...");
        auto compute_shader = device.compile(clear_kernel);
        log_msg("  Compute shader compiled successfully!");

        auto buf = device.create_buffer<float4>(1024u);
        stream2 << compute_shader(buf).dispatch(1024u) << synchronize();
        log_msg("  Compute shader dispatched and synchronized OK");
    }
    log_msg("=== POST-DESTROY TEST PASSED ===");

    // === ITERATION 2: Create, use, destroy again ===
    log_msg("");
    log_msg("=== ITERATION 2: Create + Render + Destroy ===");
    {
        auto scene = MotionBlurScene::create(device);
        scene.render(16);
        scene.destroy();
    }
    log_msg("=== ITERATION 2 COMPLETE ===");

    // === FINAL TEST: Compute pipeline after second destroy ===
    log_msg("");
    log_msg("=== FINAL TEST: Creating compute shader after 2nd destroy... ===");
    {
        auto stream3 = device.create_stream(StreamTag::GRAPHICS);
        log_msg("  New stream created");

        // Use a slightly different kernel to avoid cache hits
        Kernel1D fill_kernel = [](BufferFloat4 buf, Float val) noexcept {
            buf.write(dispatch_x(), make_float4(val));
        };

        log_msg("  Compiling new compute shader...");
        auto fill_shader = device.compile(fill_kernel);
        log_msg("  Compute shader compiled successfully!");

        auto buf = device.create_buffer<float4>(1024u);
        stream3 << fill_shader(buf, 1.0f).dispatch(1024u) << synchronize();
        log_msg("  Compute shader dispatched and synchronized OK");
    }
    log_msg("=== FINAL TEST PASSED ===");

    log_msg("");
    log_msg("=== ALL TESTS PASSED — lifecycle is correct ===");
    return 0;
}
