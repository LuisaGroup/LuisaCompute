#import <UIKit/UIKit.h>
#import <QuartzCore/CAMetalLayer.h>

#include <Metal/Metal.hpp>

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <luisa/gui/window.h>

#if defined(LUISA_IOS_ON_DEVICE_AIR)
#include <luisa/luisa-compute.h>
#include <luisa/runtime/rhi/resource.h>

#if defined(LUISA_IOS_RUNTIME_DEVICE)
#include "metal4_device_conformance.h"
#endif
#include "metal4_ios_path_tracing_kernel.h"
#include "metal_air_pipeline.h"
#include "metal_static_backend.h"
#include "metal_xir_pipeline.h"
#include "rendering/path_tracing_test.h"
#endif

namespace {

constexpr auto image_width = 512u;
constexpr auto image_height = 512u;
constexpr auto samples_per_pixel = LUISA_IOS_PATH_TRACING_SPP;
constexpr auto bytes_per_pixel = 4u;
constexpr auto row_bytes = image_width * bytes_per_pixel;

struct alignas(16) RootArguments {
    MTL::ResourceID output;
    std::uint64_t padding_0{};
    std::uint32_t sample_count{};
    std::array<std::uint32_t, 3u> padding_1{};
};

struct alignas(16) DispatchSize {
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t z;
    std::uint32_t padding;
};

static_assert(sizeof(RootArguments) == 32u);
static_assert(offsetof(RootArguments, sample_count) == 16u);
static_assert(sizeof(DispatchSize) == 16u);

[[nodiscard]] double elapsed_ms(
    std::chrono::steady_clock::time_point begin,
    std::chrono::steady_clock::time_point end) noexcept {
    return std::chrono::duration<double, std::milli>{end - begin}.count();
}

#if defined(LUISA_IOS_ON_DEVICE_AIR)
[[nodiscard]] luisa::compute::metal::MetalAIRVersion parse_air_version(
    std::string_view text) noexcept {
    using luisa::compute::metal::MetalAIRVersion;
    MetalAIRVersion version{};
    uint32_t *components[]{
        &version.major, &version.minor, &version.patch};
    for (auto component = 0u; component < 3u; component++) {
        auto separator = text.find('.');
        auto token = text.substr(0u, separator);
        if (token.empty()) { return {}; }
        auto [end, error] = std::from_chars(
            token.data(), token.data() + token.size(),
            *components[component]);
        if (error != std::errc{} ||
            end != token.data() + token.size()) {
            return {};
        }
        if (separator == std::string_view::npos) { break; }
        text.remove_prefix(separator + 1u);
    }
    return version;
}
#endif

[[nodiscard]] NSString *string_from_error(NS::Error *error) noexcept {
    if (error == nullptr) { return @"unknown error"; }
    auto description = error->localizedDescription();
    return description == nullptr ? @"unknown error" :
                                    [NSString stringWithUTF8String:description->utf8String()];
}

[[nodiscard]] NSString *sha256_hex(const void *bytes, size_t size) noexcept {
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256(bytes, static_cast<CC_LONG>(size), digest.data());
    auto text = [NSMutableString stringWithCapacity:digest.size() * 2u];
    for (auto byte : digest) { [text appendFormat:@"%02x", byte]; }
    return text;
}

[[nodiscard]] NSURL *documents_url(NSString *filename) noexcept {
    auto urls = [[NSFileManager defaultManager]
        URLsForDirectory:NSDocumentDirectory
               inDomains:NSUserDomainMask];
    return [[urls firstObject] URLByAppendingPathComponent:filename];
}

[[nodiscard]] UIImage *image_from_rgba8(
    const void *bytes, uint32_t width, uint32_t height) noexcept {
    auto image_row_bytes = static_cast<size_t>(width) * bytes_per_pixel;
    auto data = [NSData dataWithBytes:bytes
                               length:image_row_bytes * height];
    auto provider = CGDataProviderCreateWithCFData(
        (__bridge CFDataRef)data);
    auto color_space = CGColorSpaceCreateDeviceRGB();
    auto bitmap_info = static_cast<CGBitmapInfo>(
        static_cast<uint32_t>(kCGBitmapByteOrder32Big) |
        static_cast<uint32_t>(kCGImageAlphaLast));
    auto cg_image = CGImageCreate(
        width, height, 8u, 32u, image_row_bytes,
        color_space, bitmap_info, provider, nullptr, false,
        kCGRenderingIntentDefault);
    auto image = [UIImage imageWithCGImage:cg_image
                                     scale:1.0
                               orientation:UIImageOrientationUp];
    CGImageRelease(cg_image);
    CGColorSpaceRelease(color_space);
    CGDataProviderRelease(provider);
    return image;
}

}// namespace

@interface LuisaRenderOutcome : NSObject
@property(nonatomic, strong) UIImage *image;
@property(nonatomic, copy) NSString *status;
@property(nonatomic, copy) NSDictionary *metadata;
@end

@implementation LuisaRenderOutcome
@end

using LuisaProgressHandler = void (^)(uint64_t completed_spp,
                                      double elapsed_ms);
using LuisaMilestoneHandler = void (^)(LuisaRenderOutcome *outcome);

namespace {

[[nodiscard]] LuisaRenderOutcome *failure(
    NSString *stage, NSString *message,
    NSMutableDictionary *metadata) noexcept {
    auto status = [NSString stringWithFormat:@"%@ failed: %@", stage, message];
    metadata[@"success"] = @NO;
    metadata[@"failed_stage"] = stage;
    metadata[@"error"] = message;
    NSLog(@"LUISA_IOS_METAL4_PATH_TRACING success=0 stage='%@' error='%@'",
          stage, message);
    auto outcome = [LuisaRenderOutcome new];
    outcome.status = status;
    outcome.metadata = metadata;
    return outcome;
}

void persist_metadata(NSDictionary *metadata) noexcept {
    NSError *error = nil;
    auto json = [NSJSONSerialization dataWithJSONObject:metadata
                                                options:NSJSONWritingPrettyPrinted |
                                                        NSJSONWritingSortedKeys
                                                  error:&error];
    if (json == nil) {
        NSLog(@"LUISA_IOS_METAL4_PATH_TRACING metadata_error='%@'",
              error.localizedDescription);
        return;
    }
    auto url = documents_url(@"luisa_metal4_path_tracing.json");
    if (![json writeToURL:url options:NSDataWritingAtomic error:&error]) {
        NSLog(@"LUISA_IOS_METAL4_PATH_TRACING metadata_write_error='%@'",
              error.localizedDescription);
    } else {
        NSLog(@"LUISA_IOS_METAL4_PATH_TRACING metadata='%@'", url.path);
    }
}

#if defined(LUISA_IOS_RUNTIME_DEVICE)
[[nodiscard]] LuisaRenderOutcome *
render_path_tracing_runtime(
    luisa::compute::Window *window,
    LuisaProgressHandler progress_handler,
    LuisaMilestoneHandler milestone_handler) noexcept {
    using namespace luisa;
    using namespace luisa::compute;
    using namespace luisa::compute::metal;

    auto metadata = [NSMutableDictionary dictionary];
    metadata[@"width"] = @(image_width);
    metadata[@"height"] = @(image_height);
    metadata[@"samples_per_pixel"] = @(samples_per_pixel);
    metadata[@"shader_generation"] =
        @"device AST -> XIR -> LLVM -> LLVM 14 downgrade -> AIR";
    metadata[@"runtime_path"] =
        @"Luisa DeviceInterface -> MTL4 queue/command buffer/encoder";
    metadata[@"system"] = [[UIDevice currentDevice] systemVersion];

    if (@available(iOS 26.0, *)) {
        auto native_device = NS::TransferPtr(
            MTL::CreateSystemDefaultDevice());
        if (!native_device) {
            return failure(
                @"device", @"Metal device is unavailable", metadata);
        }
        auto device_name = [NSString stringWithUTF8String:
                                         native_device->name()->utf8String()];
        auto supports_metal4 = native_device->supportsFamily(
            MTL::GPUFamilyMetal4);
        auto supports_apple9 = native_device->supportsFamily(
            MTL::GPUFamilyApple9);
        auto supports_apple10 = native_device->supportsFamily(
            MTL::GPUFamilyApple10);
        metadata[@"device"] = device_name;
        metadata[@"metal4"] = @(supports_metal4);
        metadata[@"apple9"] = @(supports_apple9);
        metadata[@"apple10"] = @(supports_apple10);
        metadata[@"mtl4_acceleration_structure_build"] =
            @(supports_apple9);
        if (!supports_metal4) {
            return failure(
                @"feature_guard",
                @"GPUFamilyMetal4 is not supported", metadata);
        }

        auto file_manager = [NSFileManager defaultManager];
        auto application_support_urls = [file_manager
            URLsForDirectory:NSApplicationSupportDirectory
                   inDomains:NSUserDomainMask];
        auto application_support_url = [application_support_urls firstObject];
        if (application_support_url == nil) {
            return failure(
                @"runtime data directory",
                @"Application Support directory is unavailable", metadata);
        }
        auto runtime_data_url = [application_support_url
            URLByAppendingPathComponent:@"LuisaCompute"
                            isDirectory:YES];
        NSError *directory_error = nil;
        if (![file_manager
                createDirectoryAtURL:runtime_data_url
          withIntermediateDirectories:YES
                           attributes:nil
                                error:&directory_error]) {
            return failure(
                @"runtime data directory",
                directory_error.localizedDescription, metadata);
        }
        auto runtime_data_path = runtime_data_url.path;
        metadata[@"runtime_data_path"] = runtime_data_path;

        auto device_begin = std::chrono::steady_clock::now();
        Context context{
            luisa::string_view{},
            luisa::string_view{runtime_data_path.UTF8String}};
        Device device = context.create_device("metal4");
        auto device_end = std::chrono::steady_clock::now();
        metadata[@"runtime_device_ms"] = @(
            elapsed_ms(device_begin, device_end));
        metadata[@"thread_execution_width"] = @(
            device.compute_warp_size());
        auto query_string = [&device](luisa::string_view property) noexcept {
            auto value = device.query(property);
            return [NSString stringWithUTF8String:value.c_str()];
        };
        auto query_bool = [&query_string](
                              luisa::string_view property) noexcept {
            return [query_string(property) isEqualToString:@"true"];
        };
        metadata[@"luisa_device"] = query_string("device_name");
        metadata[@"luisa_gpu_family"] = query_string("metal4_gpu_family");
        metadata[@"reported_capabilities"] = @{
            @"metal4_runtime" : @(query_bool("metal4_runtime")),
            @"address_driven_acceleration_structures" : @(
                query_bool("metal4_address_driven_acceleration_structures")),
            @"component_motion" : @(
                query_bool("metal4_component_motion")),
            @"primitive_motion_blur" : @(
                query_bool("metal_motion_blur")),
            @"ray_tracing" : @(query_bool("metal_ray_tracing")),
            @"ray_tracing_from_render" : @(
                query_bool("metal_ray_tracing_from_render")),
            @"function_pointers" : @(
                query_bool("metal_function_pointers")),
            @"function_pointers_from_render" : @(
                query_bool("metal_function_pointers_from_render")),
            @"dynamic_libraries" : @(
                query_bool("metal_dynamic_libraries")),
            @"render_dynamic_libraries" : @(
                query_bool("metal_render_dynamic_libraries")),
            @"argument_buffer_tier" : @(
                [query_string("metal_argument_buffer_tier") integerValue]),
            @"read_write_texture_tier" : @(
                [query_string("metal_read_write_texture_tier") integerValue]),
            @"raster_order_groups" : @(
                query_bool("metal_raster_order_groups")),
            @"bc_texture_compression" : @(
                query_bool("metal_bc_texture_compression")),
            @"shader_barycentric_coordinates" : @(
                query_bool("metal_shader_barycentric_coordinates")),
            @"pull_model_interpolation" : @(
                query_bool("metal_pull_model_interpolation"))
        };

        auto conformance = run_metal4_device_conformance(
            device, image_width, image_height, samples_per_pixel);
        metadata[@"matrix_motion_valid"] = @(
            conformance.matrix_motion_valid);
        metadata[@"matrix_motion_hit_count"] = @(
            conformance.matrix_motion_hit_count);
        metadata[@"matrix_motion_centroid_delta"] = @(
            conformance.matrix_motion_centroid_delta);
        metadata[@"component_motion_exercised"] = @(
            conformance.component_motion_exercised);
        metadata[@"component_motion_valid"] = @(
            conformance.component_motion_valid);
        metadata[@"component_motion_hit_count"] = @(
            conformance.component_motion_hit_count);
        metadata[@"component_motion_centroid_delta"] = @(
            conformance.component_motion_centroid_delta);
        if (!conformance.success) {
            auto stage = [NSString stringWithUTF8String:
                                       conformance.failed_stage.c_str()];
            auto error = [NSString stringWithUTF8String:
                                       conformance.error.c_str()];
            return failure(stage, error, metadata);
        }

        metadata[@"renderer"] = @"hardware RTX Cornell path tracing";
        metadata[@"pipeline_compile_ms"] = @(
            conformance.path_trace_compile_ms);
        metadata[@"dispatch_readback_ms"] = @(
            conformance.path_trace_dispatch_readback_ms);
        metadata[@"acceleration_build_ms"] = @(
            conformance.acceleration_build_ms);
        metadata[@"acceleration_structure_path"] =
            [NSString stringWithUTF8String:
                          conformance.acceleration_structure_path.c_str()];
        metadata[@"motion_blur"] = [NSString stringWithUTF8String:
                                                 conformance.motion_blur.c_str()];
        metadata[@"component_motion"] = [NSString stringWithUTF8String:
                                                      conformance.component_motion.c_str()];
        metadata[@"abi_layout_checksum"] = @(conformance.abi_layout_checksum);
        metadata[@"atomic_value"] = @(conformance.atomic_value);
        metadata[@"texture_read_rgba"] = @[
            @(conformance.texture_read[0u]),
            @(conformance.texture_read[1u]),
            @(conformance.texture_read[2u]),
            @(conformance.texture_read[3u])
        ];
        metadata[@"native_include_checksum"] = @(
            conformance.native_include_checksum);
        metadata[@"native_include_ms"] = @(
            conformance.native_include_ms);
        metadata[@"timeline_value"] = @(conformance.timeline_value);
        metadata[@"compute_abi_ms"] = @(conformance.compute_abi_ms);
        metadata[@"timeline_event_ms"] = @(conformance.timeline_event_ms);
        metadata[@"matrix_motion_hit_count"] = @(
            conformance.matrix_motion_hit_count);
        metadata[@"matrix_motion_centroid_delta"] = @(
            conformance.matrix_motion_centroid_delta);
        metadata[@"component_motion_exercised"] = @(
            conformance.component_motion_exercised);
        metadata[@"component_motion_hit_count"] = @(
            conformance.component_motion_hit_count);
        metadata[@"component_motion_centroid_delta"] = @(
            conformance.component_motion_centroid_delta);
        metadata[@"motion_instance_ms"] = @(
            conformance.motion_instance_ms);
        metadata[@"shader_log_message"] = [NSString stringWithUTF8String:
                                                        conformance.printer_message.c_str()];
        metadata[@"shader_log_ms"] = @(conformance.printer_ms);
        metadata[@"bindless_value"] = @(conformance.bindless_value);
        metadata[@"indirect_checksum"] = @(conformance.indirect_checksum);
        metadata[@"bindless_indirect_ms"] = @(
            conformance.bindless_indirect_ms);
        metadata[@"raster_compile_ms"] = @(
            conformance.raster_compile_ms);
        metadata[@"raster_dispatch_readback_ms"] = @(
            conformance.raster_dispatch_readback_ms);
        metadata[@"raster_colored_pixels"] = @(
            conformance.raster_colored_pixels);
        metadata[@"raster_stencil_colored_pixels"] = @(
            conformance.raster_stencil_colored_pixels);
        metadata[@"raster_center_rgba"] = @[
            @(conformance.raster_center[0u]),
            @(conformance.raster_center[1u]),
            @(conformance.raster_center[2u]),
            @(conformance.raster_center[3u])
        ];
        metadata[@"path_trace_nonblack_pixels"] = @(
            conformance.path_trace_nonblack_pixels);
        metadata[@"path_trace_max_channel"] = @(
            conformance.path_trace_max_channel);
        metadata[@"path_trace_mean_luma"] = @(
            conformance.path_trace_mean_luma);
        auto address_driven_as =
            conformance.acceleration_structure_path == "true";
        auto exercised_features =
            [NSMutableDictionary dictionaryWithDictionary:@{
                @"static_device_interface" : @"passed",
                @"xir_llvm_air_codegen" : @"passed",
                @"llvm_14_downgrade" : @"passed",
                @"mtl4_compiler_queue_command_buffer" : @"passed",
                @"compute_encoder" : @"passed",
                @"shader_logging" : @"passed",
                @"bool_byte_abi" : @"passed",
                @"device_atomics" : @"passed",
                @"direct_texture_io" : @"passed",
                @"external_callable_native_include" : @"passed",
                @"unsigned_timeline_event" : @"passed",
                @"bindless_resources" : @"passed",
                @"gpu_indirect_dispatch" : @"passed",
                @"raster_encoder" : @"passed",
                @"raster_base_instance" : @"passed",
                @"raster_d24s8_stencil" : @"passed",
                @"raster_d32s8a24_stencil" : @"passed",
                @"primitive_motion" : @"passed",
                @"matrix_motion" : @"passed",
                @"component_motion" :
                    (conformance.component_motion_exercised ?
                         @"passed" : @"guarded_unsupported"),
                @"address_driven_acceleration_structure" :
                    (address_driven_as ?
                         @"passed" : @"guarded_unsupported"),
                @"compatibility_acceleration_structure_bridge" :
                    (address_driven_as ? @"not_used" : @"passed"),
                @"closest_any_hit_ray_tracing" : @"passed",
                @"shader_execution_reordering" : @"passed",
                @"window_swapchain_present" : @"pending",
                @"repository_path_tracing" : @"pending"
            }];
        metadata[@"exercised_features"] = exercised_features;

        auto pixels = std::move(conformance.pixels);
        auto dispatch_ms = conformance.path_trace_dispatch_readback_ms;
        auto pixel_count = pixels.size() * sizeof(pixels.front());
        auto pixel_sha = sha256_hex(pixels.data(), pixel_count);
        auto image = image_from_rgba8(
            pixels.data(), image_width, image_height);
        auto png = UIImagePNGRepresentation(image);
        auto png_url = documents_url(
            @"luisa_metal4_path_tracing.png");
        NSError *write_error = nil;
        if (![png writeToURL:png_url
                     options:NSDataWritingAtomic
                       error:&write_error]) {
            return failure(
                @"PNG write", write_error.localizedDescription,
                metadata);
        }

        metadata[@"pixel_sha256"] = pixel_sha;
        metadata[@"png_path"] = png_url.path;
        metadata[@"renderer"] =
            @"interactive repository MIS path tracing plus Metal4 conformance";
        metadata[@"interactive"] = @YES;
        metadata[@"presentation_path"] =
            @"UIKit CAMetalLayer -> Luisa Window -> Luisa Swapchain -> MTL4 present";
        metadata[@"interactive_snapshot_spp"] =
            @(LUISA_IOS_INTERACTIVE_SNAPSHOT_SPP);

        if (window == nullptr || window->native_handle() == 0u) {
            return failure(
                @"native window",
                @"UIKit CAMetalLayer was not wrapped by a Luisa Window",
                metadata);
        }

        bool snapshot_valid = false;
        NSString *snapshot_error = nil;
        UIImage *repository_image = nil;
        NSString *repository_status = nil;
        double last_progress_ms = -1000.0;
        auto process_snapshot = [&] (
                                    uint2 repository_resolution,
                                    uint64_t repository_spp,
                                    double repository_elapsed_ms,
                                    const luisa::vector<std::array<uint8_t, 4u>>
                                        &repository_pixels) noexcept {
            uint32_t repository_nonblack_pixels = 0u;
            uint8_t repository_max_channel = 0u;
            uint64_t repository_channel_sum = 0u;
            for (auto pixel : repository_pixels) {
                auto rgb_sum = static_cast<uint32_t>(pixel[0u]) +
                               pixel[1u] + pixel[2u];
                repository_channel_sum += rgb_sum;
                if (rgb_sum > 6u) { repository_nonblack_pixels++; }
                repository_max_channel = std::max(
                    repository_max_channel,
                    std::max(pixel[0u], std::max(pixel[1u], pixel[2u])));
            }
            if (repository_pixels.empty() ||
                repository_nonblack_pixels < repository_pixels.size() / 8u ||
                repository_max_channel < 32u) {
                snapshot_error =
                    @"the interactive repository image is empty or degenerate";
                window->set_should_close();
                return;
            }
            auto repository_pixel_bytes =
                repository_pixels.size() * sizeof(repository_pixels.front());
            auto repository_pixel_sha = sha256_hex(
                repository_pixels.data(), repository_pixel_bytes);
            repository_image = image_from_rgba8(
                repository_pixels.data(),
                repository_resolution.x, repository_resolution.y);
            auto repository_png = UIImagePNGRepresentation(repository_image);
            auto repository_png_url = documents_url(
                @"luisa_test_path_tracing.png");
            NSError *repository_write_error = nil;
            if (![repository_png writeToURL:repository_png_url
                                    options:NSDataWritingAtomic
                                      error:&repository_write_error]) {
                snapshot_error = repository_write_error.localizedDescription;
                window->set_should_close();
                return;
            }
            auto repository_mean_luma =
                static_cast<double>(repository_channel_sum) /
                static_cast<double>(repository_pixels.size() * 3u * 255u);
            metadata[@"repository_path_tracing_source"] =
                @"examples/rendering/path_tracing.cpp";
            metadata[@"repository_path_tracing_width"] =
                @(repository_resolution.x);
            metadata[@"repository_path_tracing_height"] =
                @(repository_resolution.y);
            metadata[@"repository_path_tracing_spp"] = @(repository_spp);
            metadata[@"repository_path_tracing_elapsed_ms"] =
                @(repository_elapsed_ms);
            metadata[@"repository_path_tracing_nonblack_pixels"] =
                @(repository_nonblack_pixels);
            metadata[@"repository_path_tracing_max_channel"] =
                @(repository_max_channel);
            metadata[@"repository_path_tracing_mean_luma"] =
                @(repository_mean_luma);
            metadata[@"repository_path_tracing_pixel_sha256"] =
                repository_pixel_sha;
            metadata[@"repository_path_tracing_png_path"] =
                repository_png_url.path;
            exercised_features[@"window_swapchain_present"] = @"passed";
            exercised_features[@"repository_path_tracing"] = @"passed";
            metadata[@"success"] = @YES;
            repository_status = [NSString stringWithFormat:
                @"Luisa Path Tracing live on %@\n"
                 "%ux%u, %llu spp, %.2f s\n"
                 "Window -> Swapchain -> Metal4 AIR\n"
                 "continuing to accumulate samples...",
                device_name,
                repository_resolution.x, repository_resolution.y,
                static_cast<unsigned long long>(repository_spp),
                repository_elapsed_ms * 1.0e-3];
            NSLog(@"LUISA_IOS_REPOSITORY_PATH_TRACING success=1 interactive=1 source='examples/rendering/path_tracing.cpp' presentation='Window->Swapchain->MTL4' device='%@' size=%ux%u spp=%llu elapsed_ms=%.6f nonblack=%u max_channel=%u mean_luma=%.9f pixel_sha256=%@ png='%@'",
                  device_name,
                  repository_resolution.x, repository_resolution.y,
                  static_cast<unsigned long long>(repository_spp),
                  repository_elapsed_ms,
                  repository_nonblack_pixels, repository_max_channel,
                  repository_mean_luma, repository_pixel_sha,
                  repository_png_url.path);
            NSLog(@"LUISA_IOS_METAL4_PATH_TRACING success=1 runtime=DeviceInterface renderer=RTX interactive=1 device='%@' size=%ux%u spp=%llu as_path='%@' raster_pixels=%u stencil_pixels=%u log='%@' compile_ms=%.6f dispatch_readback_ms=%.6f pixel_sha256=%@ png='%@'",
                  device_name,
                  repository_resolution.x, repository_resolution.y,
                  static_cast<unsigned long long>(repository_spp),
                  metadata[@"acceleration_structure_path"],
                  conformance.raster_colored_pixels,
                  conformance.raster_stencil_colored_pixels,
                  metadata[@"shader_log_message"],
                  [metadata[@"pipeline_compile_ms"] doubleValue],
                  dispatch_ms, repository_pixel_sha,
                  repository_png_url.path);
            snapshot_valid = true;
            persist_metadata(metadata);
            if (milestone_handler != nil) {
                auto milestone = [LuisaRenderOutcome new];
                milestone.image = repository_image;
                milestone.status = repository_status;
                milestone.metadata = metadata;
                milestone_handler(milestone);
            }
        };

        auto repository_test = luisa::ref::run_path_tracing_test(
            device,
            luisa::ref::PathTracingTestOptions{
                .offline = false,
                .spp = 0u,
                .max_spp_per_dispatch = 1u,
                .window = window,
                .snapshot_spp = LUISA_IOS_INTERACTIVE_SNAPSHOT_SPP,
                .progress_callback = [&] (uint64_t completed_spp,
                                           double elapsed_ms) noexcept {
                    if (progress_handler != nil &&
                        (completed_spp == 1u ||
                         elapsed_ms - last_progress_ms >= 100.0)) {
                        last_progress_ms = elapsed_ms;
                        progress_handler(completed_spp, elapsed_ms);
                    }
                },
                .snapshot_callback = process_snapshot});
        if (!repository_test.success) {
            return failure(
                @"repository path tracing",
                [NSString stringWithUTF8String:
                              repository_test.error.c_str()],
                metadata);
        }
        if (snapshot_error != nil) {
            return failure(
                @"interactive repository snapshot",
                snapshot_error, metadata);
        }
        if (!snapshot_valid && repository_test.completed_spp != 0u) {
            process_snapshot(
                repository_test.resolution,
                repository_test.completed_spp,
                repository_test.elapsed_ms,
                repository_test.pixels);
        }
        if (snapshot_error != nil) {
            return failure(
                @"interactive repository snapshot",
                snapshot_error, metadata);
        }
        auto outcome = [LuisaRenderOutcome new];
        outcome.image = repository_image;
        outcome.status = snapshot_valid ?
            [repository_status stringByAppendingString:
                                  @"\nrendering stopped"] :
            @"Interactive rendering stopped before the evidence snapshot.";
        outcome.metadata = metadata;
        return outcome;
    }
    return failure(
        @"availability", @"iOS 26.0 or newer is required", metadata);
}
#endif

[[nodiscard]] LuisaRenderOutcome *render_path_tracing(
    luisa::compute::Window *window,
    LuisaProgressHandler progress_handler,
    LuisaMilestoneHandler milestone_handler) noexcept {
#if defined(LUISA_IOS_RUNTIME_DEVICE)
    return render_path_tracing_runtime(
        window, progress_handler, milestone_handler);
#else
    static_cast<void>(window);
    static_cast<void>(progress_handler);
    static_cast<void>(milestone_handler);
#endif
    auto metadata = [NSMutableDictionary dictionary];
#if !defined(LUISA_IOS_ON_DEVICE_AIR)
    metadata[@"air_sha256"] = @LUISA_IOS_AIR_SHA256;
#endif
    metadata[@"width"] = @(image_width);
    metadata[@"height"] = @(image_height);
    metadata[@"samples_per_pixel"] = @(samples_per_pixel);
    metadata[@"root_argument_size"] = @(sizeof(RootArguments));
    metadata[@"dispatch_size_size"] = @(sizeof(DispatchSize));
#if defined(LUISA_IOS_ON_DEVICE_AIR)
    metadata[@"shader_generation"] =
        @"device AST -> XIR -> LLVM -> LLVM 14 downgrade -> AIR";
#else
    metadata[@"shader_generation"] =
        @"host XIR -> LLVM -> AIR AOT; device MTL4 pipeline creation";
#endif
    metadata[@"runtime_path"] = @"MTL4 compiler + queue + command buffer + compute encoder";

    if (@available(iOS 26.0, *)) {
        auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
        if (!device) {
            return failure(@"device", @"Metal device is unavailable", metadata);
        }
        auto device_name = [NSString stringWithUTF8String:device->name()->utf8String()];
        auto supports_metal4 = device->supportsFamily(MTL::GPUFamilyMetal4);
        auto supports_apple9 = device->supportsFamily(MTL::GPUFamilyApple9);
        auto supports_apple10 = device->supportsFamily(MTL::GPUFamilyApple10);
        metadata[@"device"] = device_name;
        metadata[@"metal4"] = @(supports_metal4);
        metadata[@"apple9"] = @(supports_apple9);
        metadata[@"apple10"] = @(supports_apple10);
        metadata[@"mtl4_acceleration_structure_build"] = @(supports_apple9);
        metadata[@"system"] = [[UIDevice currentDevice] systemVersion];
#if defined(LUISA_IOS_ON_DEVICE_AIR)
        NSLog(@"LUISA_IOS_METAL4_PATH_TRACING device='%@' metal4=%d apple9=%d apple10=%d codegen=device",
              device_name, supports_metal4, supports_apple9, supports_apple10);
#else
        NSLog(@"LUISA_IOS_METAL4_PATH_TRACING device='%@' metal4=%d apple9=%d apple10=%d air_sha256=%s",
              device_name, supports_metal4, supports_apple9, supports_apple10,
              LUISA_IOS_AIR_SHA256);
#endif
        if (!supports_metal4) {
            return failure(@"feature_guard",
                           @"GPUFamilyMetal4 is not supported", metadata);
        }

        NS::Error *metal_error = nullptr;
        NS::SharedPtr<MTL::Library> library;
#if defined(LUISA_IOS_ON_DEVICE_AIR)
        using namespace luisa::compute;
        using namespace luisa::compute::metal;
        auto ast_begin = std::chrono::steady_clock::now();
        auto kernel = make_ios_path_tracing_kernel();
        auto ast_end = std::chrono::steady_clock::now();
        auto option = ShaderOption{
            .enable_cache = false,
            .enable_fast_math = true,
            .enable_debug_info = false,
            .compile_only = true,
            .name = "luisa_ios_path_tracing_device_codegen"};
        auto xir_begin = std::chrono::steady_clock::now();
        auto module = metal_translate_ast_to_xir(
            kernel.function()->function(), option);
        auto xir_end = std::chrono::steady_clock::now();
        auto deployment = parse_air_version(
            LUISA_IOS_AIR_DEPLOYMENT_VERSION);
        auto sdk = parse_air_version(LUISA_IOS_AIR_SDK_VERSION);
        if (deployment.major == 0u || sdk.major == 0u) {
            return failure(@"AIR target", @"invalid deployment or SDK version", metadata);
        }
        auto air_begin = std::chrono::steady_clock::now();
        auto air = metal_codegen_air(
            *module, option,
            metal_air_target_for_ios(deployment, sdk));
        auto air_end = std::chrono::steady_clock::now();
        metadata[@"ast_build_ms"] = @(elapsed_ms(ast_begin, ast_end));
        metadata[@"xir_opt_ms"] = @(elapsed_ms(xir_begin, xir_end));
        metadata[@"llvm_air_ms"] = @(elapsed_ms(air_begin, air_end));
        metadata[@"air_bytes"] = @(air.library.size());
        metadata[@"air_root_argument_size"] = @(air.root_argument_size);
        metadata[@"air_target"] = [NSString stringWithFormat:
                                                @"iOS %u.%u.%u SDK %u.%u.%u",
                                                deployment.major, deployment.minor, deployment.patch,
                                                sdk.major, sdk.minor, sdk.patch];
        if (air.library.empty() || air.root_argument_size != sizeof(RootArguments)) {
            return failure(@"AIR codegen", @"empty library or invalid root ABI", metadata);
        }
        auto generated_air_sha = sha256_hex(
            air.library.data(), air.library.size());
        metadata[@"air_sha256"] = generated_air_sha;
        auto library_data = dispatch_data_create(
            air.library.data(), air.library.size(), nullptr,
            DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        auto library_begin = std::chrono::steady_clock::now();
        library = NS::TransferPtr(device->newLibrary(
            library_data, &metal_error));
        auto library_end = std::chrono::steady_clock::now();
        dispatch_release(library_data);
        metadata[@"library_load_ms"] = @(elapsed_ms(library_begin, library_end));
#else
        auto metallib_path = [[NSBundle mainBundle]
            pathForResource:@"luisa_ios_path_tracing"
                     ofType:@"metallib"];
        if (metallib_path == nil) {
            return failure(@"bundle", @"AIR metallib resource is missing", metadata);
        }
        auto metallib_data = [NSData dataWithContentsOfFile:metallib_path];
        if (metallib_data == nil) {
            return failure(@"bundle", @"AIR metallib resource is unreadable", metadata);
        }
        auto bundled_air_sha = sha256_hex(
            metallib_data.bytes, metallib_data.length);
        metadata[@"bundled_air_sha256"] = bundled_air_sha;
        auto expected_air_sha = @LUISA_IOS_AIR_SHA256;
        auto air_sha_matches = [bundled_air_sha isEqualToString:expected_air_sha];
        metadata[@"air_sha_matches"] = @(air_sha_matches);
        if (!air_sha_matches) {
            return failure(@"bundle", @"AIR metallib SHA-256 mismatch", metadata);
        }
        auto metallib_url = [NSURL fileURLWithPath:metallib_path];
        auto library_begin = std::chrono::steady_clock::now();
        library = NS::TransferPtr(device->newLibrary(
            reinterpret_cast<NS::URL *>(metallib_url), &metal_error));
        auto library_end = std::chrono::steady_clock::now();
        metadata[@"library_load_ms"] = @(elapsed_ms(library_begin, library_end));
#endif
        if (!library) {
            return failure(@"AIR library", string_from_error(metal_error), metadata);
        }

        auto compiler_descriptor = NS::TransferPtr(
            MTL4::CompilerDescriptor::alloc()->init());
        compiler_descriptor->setLabel(NS::String::string(
            "Luisa iOS AIR compiler", NS::UTF8StringEncoding));
        metal_error = nullptr;
        auto compiler = NS::TransferPtr(
            device->newCompiler(compiler_descriptor.get(), &metal_error));
        if (!compiler) {
            return failure(@"MTL4 compiler", string_from_error(metal_error), metadata);
        }

        auto function_descriptor = NS::TransferPtr(
            MTL4::LibraryFunctionDescriptor::alloc()->init());
        function_descriptor->setLibrary(library.get());
        function_descriptor->setName(NS::String::string(
            "kernel_main", NS::UTF8StringEncoding));
        auto pipeline_descriptor = NS::TransferPtr(
            MTL4::ComputePipelineDescriptor::alloc()->init());
        pipeline_descriptor->setComputeFunctionDescriptor(
            function_descriptor.get());
        pipeline_descriptor->setMaxTotalThreadsPerThreadgroup(64u);
        pipeline_descriptor->setThreadGroupSizeIsMultipleOfThreadExecutionWidth(true);
        pipeline_descriptor->setLabel(NS::String::string(
            "Luisa iOS AIR path tracer", NS::UTF8StringEncoding));
        metal_error = nullptr;
        auto pipeline_begin = std::chrono::steady_clock::now();
        auto pipeline = NS::TransferPtr(compiler->newComputePipelineState(
            pipeline_descriptor.get(), nullptr, &metal_error));
        auto pipeline_end = std::chrono::steady_clock::now();
        metadata[@"pipeline_compile_ms"] = @(elapsed_ms(
            pipeline_begin, pipeline_end));
        if (!pipeline) {
            return failure(@"MTL4 pipeline", string_from_error(metal_error), metadata);
        }
        metadata[@"thread_execution_width"] = @(pipeline->threadExecutionWidth());
        metadata[@"max_threads_per_threadgroup"] = @(
            pipeline->maxTotalThreadsPerThreadgroup());

        auto texture_descriptor = NS::TransferPtr(
            MTL::TextureDescriptor::alloc()->init());
        texture_descriptor->setTextureType(MTL::TextureType2D);
        texture_descriptor->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
        texture_descriptor->setWidth(image_width);
        texture_descriptor->setHeight(image_height);
        texture_descriptor->setDepth(1u);
        texture_descriptor->setMipmapLevelCount(1u);
        texture_descriptor->setArrayLength(1u);
        texture_descriptor->setSampleCount(1u);
        texture_descriptor->setStorageMode(MTL::StorageModePrivate);
        texture_descriptor->setUsage(
            MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
        auto output = NS::TransferPtr(
            device->newTexture(texture_descriptor.get()));
        if (!output) {
            return failure(@"texture", @"output allocation failed", metadata);
        }

        auto shared_options = MTL::ResourceStorageModeShared |
                              MTL::ResourceHazardTrackingModeTracked;
        RootArguments root{
            .output = output->gpuResourceID(),
            .sample_count = samples_per_pixel};
        DispatchSize dispatch_size{
            .x = image_width, .y = image_height, .z = 1u, .padding = 0u};
        auto root_buffer = NS::TransferPtr(device->newBuffer(
            &root, sizeof(root), shared_options));
        auto dispatch_buffer = NS::TransferPtr(device->newBuffer(
            &dispatch_size, sizeof(dispatch_size), shared_options));
        auto readback = NS::TransferPtr(device->newBuffer(
            row_bytes * image_height, shared_options));
        if (!root_buffer || !dispatch_buffer || !readback) {
            return failure(@"buffer", @"argument/readback allocation failed", metadata);
        }

        auto table_descriptor = NS::TransferPtr(
            MTL4::ArgumentTableDescriptor::alloc()->init());
        table_descriptor->setInitializeBindings(true);
        table_descriptor->setMaxBufferBindCount(2u);
        table_descriptor->setMaxTextureBindCount(0u);
        table_descriptor->setMaxSamplerStateBindCount(0u);
        metal_error = nullptr;
        auto table = NS::TransferPtr(
            device->newArgumentTable(table_descriptor.get(), &metal_error));
        if (!table) {
            return failure(@"MTL4 argument table",
                           string_from_error(metal_error), metadata);
        }
        table->setAddress(root_buffer->gpuAddress(), 0u);
        table->setAddress(dispatch_buffer->gpuAddress(), 1u);

        auto allocator = NS::TransferPtr(device->newCommandAllocator());
        auto command_buffer = NS::TransferPtr(device->newCommandBuffer());
        if (!allocator || !command_buffer) {
            return failure(@"MTL4 command buffer",
                           @"allocator or command buffer creation failed", metadata);
        }
        command_buffer->setLabel(NS::String::string(
            "Luisa iOS AIR path tracing", NS::UTF8StringEncoding));
        command_buffer->beginCommandBuffer(allocator.get());

        auto dispatch_encoder = command_buffer->computeCommandEncoder();
        if (dispatch_encoder == nullptr) {
            return failure(@"MTL4 dispatch encoder",
                           @"compute encoder creation failed", metadata);
        }
        dispatch_encoder->barrierAfterQueueStages(
            MTL::StageAll, MTL::StageAll, MTL4::VisibilityOptionDevice);
        dispatch_encoder->setComputePipelineState(pipeline.get());
        dispatch_encoder->setArgumentTable(table.get());
        dispatch_encoder->dispatchThreadgroups(
            MTL::Size{image_width / 8u, image_height / 8u, 1u},
            MTL::Size{8u, 8u, 1u});
        dispatch_encoder->endEncoding();

        auto copy_encoder = command_buffer->computeCommandEncoder();
        if (copy_encoder == nullptr) {
            return failure(@"MTL4 copy encoder",
                           @"compute encoder creation failed", metadata);
        }
        copy_encoder->barrierAfterQueueStages(
            MTL::StageAll, MTL::StageAll, MTL4::VisibilityOptionDevice);
        copy_encoder->copyFromTexture(
            output.get(), 0u, 0u, MTL::Origin{0u, 0u, 0u},
            MTL::Size{image_width, image_height, 1u},
            readback.get(), 0u, row_bytes, row_bytes * image_height);
        copy_encoder->endEncoding();

        auto residency_descriptor = NS::TransferPtr(
            MTL::ResidencySetDescriptor::alloc()->init());
        residency_descriptor->setInitialCapacity(5u);
        metal_error = nullptr;
        auto residency = NS::TransferPtr(device->newResidencySet(
            residency_descriptor.get(), &metal_error));
        if (!residency) {
            return failure(@"MTL4 residency set",
                           string_from_error(metal_error), metadata);
        }
        const MTL::Allocation *allocations[]{
            pipeline.get(), output.get(), root_buffer.get(),
            dispatch_buffer.get(), readback.get()};
        residency->addAllocations(allocations, std::size(allocations));
        residency->commit();
        command_buffer->useResidencySet(residency.get());
        command_buffer->endCommandBuffer();

        auto queue_descriptor = NS::TransferPtr(
            MTL4::CommandQueueDescriptor::alloc()->init());
        queue_descriptor->setLabel(NS::String::string(
            "Luisa iOS MTL4 queue", NS::UTF8StringEncoding));
        metal_error = nullptr;
        auto queue = NS::TransferPtr(device->newMTL4CommandQueue(
            queue_descriptor.get(), &metal_error));
        if (!queue) {
            return failure(@"MTL4 queue", string_from_error(metal_error), metadata);
        }

        auto semaphore = dispatch_semaphore_create(0);
        auto commit_options = NS::TransferPtr(
            MTL4::CommitOptions::alloc()->init());
        auto gpu_ms = 0.0;
        std::string feedback_error;
        commit_options->addFeedbackHandler(
            MTL4::CommitFeedbackHandlerFunction{
                [&](MTL4::CommitFeedback *feedback) noexcept {
                    auto begin = feedback->GPUStartTime();
                    auto end = feedback->GPUEndTime();
                    if (end > begin) { gpu_ms = (end - begin) * 1.0e3; }
                    if (auto error = feedback->error()) {
                        feedback_error = error->localizedDescription()->utf8String();
                    }
                    dispatch_semaphore_signal(semaphore);
                }});
        const MTL4::CommandBuffer *command_buffers[]{command_buffer.get()};
        auto submit_begin = std::chrono::steady_clock::now();
        queue->commit(command_buffers, 1u, commit_options.get());
        dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
        auto submit_end = std::chrono::steady_clock::now();
        metadata[@"gpu_ms"] = @(gpu_ms);
        metadata[@"submit_to_feedback_ms"] = @(elapsed_ms(
            submit_begin, submit_end));
        if (!feedback_error.empty()) {
            return failure(
                @"MTL4 execution",
                [NSString stringWithUTF8String:feedback_error.c_str()], metadata);
        }

        auto pixel_bytes = readback->contents();
        auto pixel_count = row_bytes * image_height;
        auto pixel_sha = sha256_hex(pixel_bytes, pixel_count);
        auto image = image_from_rgba8(
            pixel_bytes, image_width, image_height);
        auto png = UIImagePNGRepresentation(image);
        auto png_url = documents_url(@"luisa_metal4_path_tracing.png");
        NSError *write_error = nil;
        if (![png writeToURL:png_url
                     options:NSDataWritingAtomic
                       error:&write_error]) {
            return failure(@"PNG write", write_error.localizedDescription, metadata);
        }

        metadata[@"success"] = @YES;
        metadata[@"pixel_sha256"] = pixel_sha;
        metadata[@"png_path"] = png_url.path;
        auto status = [NSString stringWithFormat:
                                    @"Metal4 AIR path tracing complete\n%@\n%ux%u, %u spp\n"
                                     "pipeline %.2f ms, GPU %.2f ms\npixel SHA %@",
                                    device_name, image_width, image_height, samples_per_pixel,
                                    [metadata[@"pipeline_compile_ms"] doubleValue], gpu_ms,
                                    [pixel_sha substringToIndex:16u]];
        NSLog(@"LUISA_IOS_METAL4_PATH_TRACING success=1 device='%@' size=%ux%u spp=%u pipeline_ms=%.6f gpu_ms=%.6f pixel_sha256=%@ png='%@'",
              device_name, image_width, image_height, samples_per_pixel,
              [metadata[@"pipeline_compile_ms"] doubleValue], gpu_ms,
              pixel_sha, png_url.path);
        auto outcome = [LuisaRenderOutcome new];
        outcome.image = image;
        outcome.status = status;
        outcome.metadata = metadata;
        return outcome;
    }
    return failure(@"availability", @"iOS 26.0 or newer is required", metadata);
}

}// namespace

@interface LuisaMetalView : UIView
@end

@implementation LuisaMetalView

+ (Class)layerClass {
    return [CAMetalLayer class];
}

@end

@interface LuisaPathTracingViewController : UIViewController
@end

@implementation LuisaPathTracingViewController {
    LuisaMetalView *_metal_view;
    UILabel *_status_label;
    UIActivityIndicatorView *_spinner;
    std::unique_ptr<luisa::compute::Window> _luisa_window;
    BOOL _started;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = UIColor.blackColor;

    _metal_view = [LuisaMetalView new];
    _metal_view.translatesAutoresizingMaskIntoConstraints = YES;
    _metal_view.backgroundColor = UIColor.blackColor;
    _metal_view.opaque = YES;
    _metal_view.clipsToBounds = YES;
    [self.view addSubview:_metal_view];

    _status_label = [UILabel new];
    _status_label.translatesAutoresizingMaskIntoConstraints = NO;
    _status_label.textColor = UIColor.whiteColor;
    _status_label.font = [UIFont monospacedSystemFontOfSize:13.0
                                                     weight:UIFontWeightRegular];
    _status_label.numberOfLines = 0;
    _status_label.textAlignment = NSTextAlignmentCenter;
    _status_label.backgroundColor = [UIColor colorWithWhite:0.0 alpha:0.62];
    _status_label.layer.cornerRadius = 10.0;
    _status_label.layer.masksToBounds = YES;
#if defined(LUISA_IOS_ON_DEVICE_AIR)
    _status_label.text =
        @"Generating AST -> XIR -> LLVM -> AIR on this iPhone...";
#else
    _status_label.text =
        @"Loading host-generated XIR -> LLVM -> AIR and running it on Metal4...";
#endif
    [self.view addSubview:_status_label];

    _spinner = [[UIActivityIndicatorView alloc]
        initWithActivityIndicatorStyle:UIActivityIndicatorViewStyleLarge];
    _spinner.translatesAutoresizingMaskIntoConstraints = NO;
    _spinner.color = UIColor.whiteColor;
    [_spinner startAnimating];
    [self.view addSubview:_spinner];

    auto guide = self.view.safeAreaLayoutGuide;
    [NSLayoutConstraint activateConstraints:@[
        [_status_label.leadingAnchor constraintEqualToAnchor:guide.leadingAnchor
                                                    constant:12.0],
        [_status_label.trailingAnchor constraintEqualToAnchor:guide.trailingAnchor
                                                     constant:-12.0],
        [_status_label.bottomAnchor constraintEqualToAnchor:guide.bottomAnchor
                                                 constant:-12.0],
        [_status_label.heightAnchor constraintGreaterThanOrEqualToConstant:78.0],
        [_spinner.centerXAnchor constraintEqualToAnchor:_metal_view.centerXAnchor],
        [_spinner.centerYAnchor constraintEqualToAnchor:_metal_view.centerYAnchor],
    ]];
}

- (void)viewDidLayoutSubviews {
    [super viewDidLayoutSubviews];
    auto safe_frame = self.view.safeAreaLayoutGuide.layoutFrame;
    auto side = std::min(
        CGRectGetWidth(safe_frame), CGRectGetHeight(safe_frame));
    _metal_view.frame = CGRectMake(
        CGRectGetMidX(safe_frame) - side * 0.5,
        CGRectGetMidY(safe_frame) - side * 0.5,
        side, side);
}

- (void)viewDidAppear:(BOOL)animated {
    [super viewDidAppear:animated];
    if (_started) { return; }
    _started = YES;
    [self.view layoutIfNeeded];
    auto metal_layer = static_cast<CAMetalLayer *>(_metal_view.layer);
    metal_layer.contentsScale = self.view.window.windowScene.screen.nativeScale;
    metal_layer.opaque = YES;
    auto native_layer = reinterpret_cast<uint64_t>(
        (__bridge void *)metal_layer);
    _luisa_window = std::make_unique<luisa::compute::Window>(
        "Luisa Path Tracing",
        luisa::make_uint2(1024u),
        luisa::compute::Window::NativeHandle{
            .window = native_layer,
            .display = 0u});
    auto render_window = _luisa_window.get();
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        @autoreleasepool {
            auto outcome = render_path_tracing(
                render_window,
                ^(uint64_t completed_spp, double elapsed_ms) {
                    auto spp_per_second = elapsed_ms > 0.0 ?
                        static_cast<double>(completed_spp) * 1000.0 / elapsed_ms :
                        0.0;
                    auto status = [NSString stringWithFormat:
                        @"Luisa Path Tracing live\n"
                         "1024x1024, %llu spp, %.1f spp/s\n"
                         "Window -> Swapchain -> Metal4 AIR",
                        static_cast<unsigned long long>(completed_spp),
                        spp_per_second];
                    dispatch_async(dispatch_get_main_queue(), ^{
                        [self->_spinner stopAnimating];
                        self->_status_label.text = status;
                    });
                },
                ^(LuisaRenderOutcome *milestone) {
                    dispatch_async(dispatch_get_main_queue(), ^{
                        [self->_spinner stopAnimating];
                        self->_status_label.text = milestone.status;
                    });
                });
            persist_metadata(outcome.metadata);
            dispatch_async(dispatch_get_main_queue(), ^{
                [self->_spinner stopAnimating];
                self->_status_label.text = outcome.status;
                self->_luisa_window.reset();
                self->_started = NO;
            });
        }
    });
}

- (void)viewWillDisappear:(BOOL)animated {
    [super viewWillDisappear:animated];
    if (_luisa_window) { _luisa_window->set_should_close(); }
}

@end

@interface LuisaPathTracingSceneDelegate : UIResponder<UIWindowSceneDelegate>
@property(nonatomic, strong) UIWindow *window;
@end

@implementation LuisaPathTracingSceneDelegate

- (void)scene:(UIScene *)scene
    willConnectToSession:(UISceneSession *)session
                 options:(UISceneConnectionOptions *)connectionOptions {
    (void)session;
    (void)connectionOptions;
    if (![scene isKindOfClass:[UIWindowScene class]]) { return; }
    self.window = [[UIWindow alloc]
        initWithWindowScene:static_cast<UIWindowScene *>(scene)];
    self.window.rootViewController = [LuisaPathTracingViewController new];
    [self.window makeKeyAndVisible];
}

@end

@interface LuisaPathTracingAppDelegate : UIResponder<UIApplicationDelegate>
@end

@implementation LuisaPathTracingAppDelegate

- (BOOL)application:(UIApplication *)application
    didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    (void)application;
    (void)launchOptions;
    luisa_compute_metal4_register_static_backend();
    return YES;
}

@end

int main(int argc, char *argv[]) {
    @autoreleasepool {
        return UIApplicationMain(
            argc, argv, nil,
            NSStringFromClass([LuisaPathTracingAppDelegate class]));
    }
}
