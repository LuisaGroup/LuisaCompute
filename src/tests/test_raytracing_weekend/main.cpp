#include <iostream>

#include <luisa/core/clock.h>

#include "rtweekend.h"

#include "hittable_list.h"
#include "sphere.h"
#include "ray.h"
#include "camera.h"
#include "material.h"

hittable_list random_scene() {
    hittable_list world;

    auto ground_material = make_shared<lambertian>(float3(0.5, 0.5, 0.5));
    materials.push_back(ground_material);
    world.add(make_shared<sphere>(float3(0, -1000, 0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_float();
            float3 center(a + 0.9f * random_float(), 0.2f, b + 0.9f * random_float());

            if (length(center - float3(4, 0.2, 0)) > 0.9f) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8f) {
                    // diffuse
                    auto albedo = make_float3(random_float(), random_float(), random_float()) * make_float3(random_float(), random_float(), random_float());
                    sphere_material = make_shared<lambertian>(albedo);
                    materials.push_back(sphere_material);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95f) {
                    // metal
                    auto albedo = make_float3(random_float(0.5, 1), random_float(0.5, 1), random_float(0.5, 1));
                    auto fuzz = random_float(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    materials.push_back(sphere_material);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    materials.push_back(sphere_material);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    materials.push_back(material1);
    world.add(make_shared<sphere>(float3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(float3(0.4, 0.2, 0.1));
    materials.push_back(material2);
    world.add(make_shared<sphere>(float3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(float3(0.7, 0.6, 0.5), 0.0);
    materials.push_back(material3);
    world.add(make_shared<sphere>(float3(4, 1, 0), 1.0, material3));

    return world;
}

Float3 ray_color(const ray &r_, const hittable &world, UInt depth, UInt &seed) {
    ray r = r_;
    Float3 ret = make_float3(1.0f);
    hit_record rec;
    $loop {
        $if(depth <= 0) {
            ret *= make_float3(0);
            $break;
        };

        $if(world.hit(r, 0.001f, infinity, rec)) {
            ray scattered;
            Float3 attenuation;
            for (uint i = 0; i < materials.size(); i++) {
                $if(rec.mat_index == i) {
                    $if(materials[i]->scatter(r, rec, attenuation, scattered, seed)) {
                        r = scattered;
                        depth -= 1u;
                        ret *= attenuation;
                        $continue;
                    };
                    ret *= make_float3(0);
                    $break;
                };
            }
        };

        Float3 unit_direction = normalize(r.direction());
        Float t = 0.5f * (unit_direction.y + 1.0f);
        ret *= (1.0f - t) * make_float3(1.0f, 1.0f, 1.0f) + t * make_float3(0.5f, 0.7f, 1.0f);
        $break;
    };
    return ret;
};

int main(int argc, char *argv[]) {

    // Image
    constexpr float aspect_ratio = 3.0f / 2.0f;
    constexpr uint image_width = 1200;
    constexpr uint image_height = static_cast<int>(image_width / aspect_ratio);
    constexpr uint samples_per_pixel = 1024;
    constexpr uint max_depth = 50;

    // World
    auto world = random_scene();

    // hittable_list world;

    // auto material_ground = make_shared<lambertian>(float3(0.8, 0.8, 0.0));
    // auto material_center = make_shared<lambertian>(float3(0.1, 0.2, 0.5));
    // auto material_left   = make_shared<dielectric>(1.5);
    // auto material_right  = make_shared<metal>(float3(0.8, 0.6, 0.2), 0.0);

    // materials.push_back(material_ground);
    // materials.push_back(material_center);
    // materials.push_back(material_left);
    // materials.push_back(material_right);

    // world.add(make_shared<sphere>(float3( 0.0, -100.5, -1.0), 100.0, material_ground));
    // world.add(make_shared<sphere>(float3( 0.0,    0.0, -1.0),   0.5, material_center));
    // world.add(make_shared<sphere>(float3(-1.0,    0.0, -1.0),   0.5, material_left));
    // world.add(make_shared<sphere>(float3(-1.0,    0.0, -1.0), -0.45, material_left));
    // world.add(make_shared<sphere>(float3( 1.0,    0.0, -1.0),   0.5, material_right));

    // Camera

    float3 lookfrom(13, 2, 3);
    float3 lookat(0, 0, 0);
    float3 vup(0, 1, 0);
    float dist_to_focus = 10.0;
    float aperture = 0.1f;

    camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

    // Init

    Context context{argv[0]};
    if (argc <= 1) { exit(1); }
    Device device = context.create_device(argv[1]);
    Stream stream = device.create_stream();
    constexpr uint2 resolution = make_uint2(image_width, image_height);
    Image<uint> seed_image = device.create_image<uint>(PixelStorage::INT1, resolution);
    Image<float> accum_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    luisa::vector<std::byte> host_image(accum_image.size_bytes());

    // Render

    Kernel2D render_kernel = [&](ImageUInt seed_image, ImageFloat accum_image, UInt sample_index) {
        UInt2 coord = dispatch_id().xy();
        UInt2 size = dispatch_size().xy();
        $if(sample_index == 0u) {
            seed_image.write(coord, make_uint4(tea(coord.x, coord.y)));
            accum_image.write(coord, make_float4(make_float3(0.0f), 1.0f));
        };

        UInt seed = seed_image.read(coord).x;
        Float2 uv = make_float2((coord.x + frand(seed)) / (size.x - 1.0f), (size.y - 1u - coord.y + frand(seed)) / (size.y - 1.0f));
        ray r = cam.get_ray(uv, seed);
        Float3 pixel_color = ray_color(r, world, max_depth, seed);

        Float3 accum_color = lerp(accum_image.read(coord).xyz(), pixel_color, 1.0f / (sample_index + 1.0f));
        accum_image.write(coord, make_float4(accum_color, 1.0f));
        seed_image.write(coord, make_uint4(seed));
    };

    Shader2D<Image<uint>, Image<float>, uint> render = device.compile(render_kernel);

    Clock clk;
    for (uint sample_index = 0u; sample_index < samples_per_pixel; sample_index++) {
        stream << render(seed_image, accum_image, sample_index)
                      .dispatch(resolution)
               << [sample_index, samples_per_pixel, &clk] {
                      LUISA_INFO("Samples: {} / {} ({:.1f}s)",
                                 sample_index + 1u, samples_per_pixel,
                                 clk.toc() * 1e-3);
                  };
    }

    // Gamma Correct

    Kernel2D gamma_kernel = [&](ImageFloat accum_image, ImageFloat output) {
        UInt2 coord = dispatch_id().xy();
        output.write(coord, make_float4(sqrt(accum_image.read(coord).xyz()), 1.0f));
    };

    Shader2D<Image<float>, Image<float>> gamma_correct = device.compile(gamma_kernel);
    auto output_image = device.create_image<float>(PixelStorage::BYTE4, resolution);

    stream << gamma_correct(accum_image, output_image).dispatch(resolution);
    stream << output_image.copy_to(host_image.data())
           << synchronize();
    stbi_write_png("test_rt_weekend.png", resolution.x, resolution.y, 4, host_image.data(), 0);
}

