#import <UIKit/UIKit.h>

#include <Metal/Metal.hpp>
#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include <luisa/luisa-compute.h>

#include "metal_static_backend.h"
#include "rendering/path_tracing_test.h"

namespace {

constexpr auto backend_name = LUISA_IOS_BENCHMARK_BACKEND_NAME;
constexpr auto default_spp = 64u;
constexpr auto default_iterations = 3u;
constexpr auto default_max_spp_per_dispatch = 1u;

struct BenchmarkArguments {
    uint32_t spp{default_spp};
    uint32_t iterations{default_iterations};
    uint32_t max_spp_per_dispatch{default_max_spp_per_dispatch};
    bool clear_cache{};
};

struct PixelMetrics {
    uint64_t nonblack_pixels{};
    uint8_t max_channel{};
    double mean_luma{};
    NSString *sha256{};
};

[[nodiscard]] double elapsed_ms(
    std::chrono::steady_clock::time_point begin,
    std::chrono::steady_clock::time_point end) noexcept {
    return std::chrono::duration<double, std::milli>{end - begin}.count();
}

[[nodiscard]] NSURL *documents_url(NSString *filename) noexcept {
    auto urls = [[NSFileManager defaultManager]
        URLsForDirectory:NSDocumentDirectory
               inDomains:NSUserDomainMask];
    return [[urls firstObject] URLByAppendingPathComponent:filename];
}

[[nodiscard]] NSString *sha256_hex(const void *bytes, size_t size) noexcept {
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256(bytes, static_cast<CC_LONG>(size), digest.data());
    auto text = [NSMutableString stringWithCapacity:digest.size() * 2u];
    for (auto byte : digest) { [text appendFormat:@"%02x", byte]; }
    return text;
}

[[nodiscard]] PixelMetrics measure_pixels(
    const luisa::vector<std::array<uint8_t, 4u>> &pixels) noexcept {
    PixelMetrics metrics;
    long double luma_sum = 0.0;
    for (auto pixel : pixels) {
        auto rgb_sum = static_cast<uint32_t>(pixel[0u]) +
                       static_cast<uint32_t>(pixel[1u]) +
                       static_cast<uint32_t>(pixel[2u]);
        if (rgb_sum > 6u) { metrics.nonblack_pixels++; }
        metrics.max_channel = std::max(
            metrics.max_channel,
            std::max(pixel[0u], std::max(pixel[1u], pixel[2u])));
        luma_sum += 0.2126L * pixel[0u] +
                    0.7152L * pixel[1u] +
                    0.0722L * pixel[2u];
    }
    if (!pixels.empty()) {
        metrics.mean_luma = static_cast<double>(
            luma_sum / static_cast<long double>(pixels.size() * 255u));
        metrics.sha256 = sha256_hex(
            pixels.data(), pixels.size() * sizeof(pixels.front()));
    } else {
        metrics.sha256 = @"";
    }
    return metrics;
}

[[nodiscard]] UIImage *image_from_rgba8(
    const void *bytes, uint32_t width, uint32_t height) noexcept {
    constexpr auto bytes_per_pixel = 4u;
    auto row_bytes = static_cast<size_t>(width) * bytes_per_pixel;
    auto data = [NSData dataWithBytes:bytes length:row_bytes * height];
    auto provider = CGDataProviderCreateWithCFData(
        (__bridge CFDataRef)data);
    auto color_space = CGColorSpaceCreateDeviceRGB();
    auto bitmap_info = static_cast<CGBitmapInfo>(
        static_cast<uint32_t>(kCGBitmapByteOrder32Big) |
        static_cast<uint32_t>(kCGImageAlphaLast));
    auto cg_image = CGImageCreate(
        width, height, 8u, 32u, row_bytes,
        color_space, bitmap_info, provider, nullptr, false,
        kCGRenderingIntentDefault);
    auto image = [UIImage imageWithCGImage:cg_image];
    CGImageRelease(cg_image);
    CGColorSpaceRelease(color_space);
    CGDataProviderRelease(provider);
    return image;
}

[[nodiscard]] uint32_t parse_positive_integer(
    NSArray<NSString *> *arguments, NSString *option,
    uint32_t fallback) noexcept {
    auto index = [arguments indexOfObject:option];
    if (index == NSNotFound || index + 1u >= arguments.count) {
        return fallback;
    }
    auto value = [arguments[index + 1u] longLongValue];
    if (value <= 0 || value > std::numeric_limits<uint32_t>::max()) {
        return fallback;
    }
    return static_cast<uint32_t>(value);
}

[[nodiscard]] BenchmarkArguments parse_arguments() noexcept {
    auto process_arguments = [NSProcessInfo processInfo].arguments;
    return {
        .spp = parse_positive_integer(
            process_arguments, @"--spp", default_spp),
        .iterations = parse_positive_integer(
            process_arguments, @"--iterations", default_iterations),
        .max_spp_per_dispatch = parse_positive_integer(
            process_arguments, @"--max-spp-per-dispatch",
            default_max_spp_per_dispatch),
        .clear_cache = [process_arguments containsObject:@"--clear-cache"]};
}

[[nodiscard]] NSURL *create_runtime_data_directory(
    bool clear_cache, NSError **error) noexcept {
    auto urls = [[NSFileManager defaultManager]
        URLsForDirectory:NSApplicationSupportDirectory
               inDomains:NSUserDomainMask];
    auto url = [[urls firstObject]
        URLByAppendingPathComponent:@"LuisaComputeBenchmark"
                        isDirectory:YES];
    auto file_manager = [NSFileManager defaultManager];
    if (clear_cache && [file_manager fileExistsAtPath:url.path] &&
        ![file_manager removeItemAtURL:url error:error]) {
        return nil;
    }
    if (![file_manager createDirectoryAtURL:url
                withIntermediateDirectories:YES
                                 attributes:nil
                                      error:error]) {
        return nil;
    }
    return url;
}

[[nodiscard]] NSDictionary *run_benchmark() noexcept {
    using namespace luisa;
    using namespace luisa::compute;

    auto arguments = parse_arguments();
    auto report = [NSMutableDictionary dictionary];
    report[@"schema_version"] = @1;
    report[@"backend"] = [NSString stringWithUTF8String:backend_name];
    report[@"source"] = @"examples/rendering/path_tracing.cpp";
    report[@"resolution"] = @[@1024, @1024];
    report[@"spp"] = @(arguments.spp);
    report[@"iterations"] = @(arguments.iterations);
    report[@"max_spp_per_dispatch"] = @(
        arguments.max_spp_per_dispatch);
    report[@"clear_cache_requested"] = @(arguments.clear_cache);
    report[@"cache_scope"] =
        @"backend-specific app sandbox; iteration 0 is process-first, later iterations are in-process warm";
#if defined(LUISA_IOS_BENCHMARK_METAL4)
    report[@"shader_path"] =
        @"AST -> XIR opt -> LLVM IR -> LLVM opt/downgrade -> Metal AIR";
#else
    report[@"shader_path"] = @"AST -> MSL -> Metal runtime source compiler";
#endif

    auto native_device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    if (!native_device) {
        report[@"success"] = @NO;
        report[@"error"] = @"Metal device is unavailable";
        return report;
    }
    report[@"device"] = [NSString stringWithUTF8String:
                                      native_device->name()->utf8String()];
    report[@"system"] = [UIDevice currentDevice].systemVersion;
    report[@"metal4_family"] = @(
        native_device->supportsFamily(MTL::GPUFamilyMetal4));
    report[@"apple9_family"] = @(
        native_device->supportsFamily(MTL::GPUFamilyApple9));
    report[@"apple10_family"] = @(
        native_device->supportsFamily(MTL::GPUFamilyApple10));

    NSError *directory_error = nil;
    auto runtime_data_url = create_runtime_data_directory(
        arguments.clear_cache, &directory_error);
    if (runtime_data_url == nil) {
        report[@"success"] = @NO;
        report[@"error"] = directory_error.localizedDescription;
        return report;
    }

    auto device_begin = std::chrono::steady_clock::now();
    Context context{
        luisa::string_view{},
        luisa::string_view{runtime_data_url.path.UTF8String}};
    auto device_config = DeviceConfig{
        // DefaultBinaryIO only creates its persistent .cache/.data roots for
        // non-headless devices. No window is created until explicitly asked,
        // so this remains an offline benchmark while making process-warm
        // shader-cache measurements meaningful on iOS.
        .headless = false,
        .use_lmdb = false};
    Device device = context.create_device(backend_name, &device_config);
    auto device_end = std::chrono::steady_clock::now();
    report[@"device_creation_ms"] = @(
        elapsed_ms(device_begin, device_end));
    report[@"thread_execution_width"] = @(device.compute_warp_size());

    auto runs = [NSMutableArray arrayWithCapacity:arguments.iterations];
    report[@"runs"] = runs;
    luisa::ref::PathTracingTestResult final_result;
    auto all_runs_succeeded = true;
    for (auto iteration = 0u; iteration < arguments.iterations; iteration++) {
        auto result = luisa::ref::run_path_tracing_test(
            device,
            luisa::ref::PathTracingTestOptions{
                .offline = true,
                .spp = arguments.spp,
                .max_spp_per_dispatch = arguments.max_spp_per_dispatch,
                .collect_stage_timings = true});
        auto run = [NSMutableDictionary dictionary];
        run[@"iteration"] = @(iteration);
        run[@"warm_state"] = iteration == 0u ?
                                 @"process-first" :
                                 @"in-process-warm";
        run[@"success"] = @(result.success);
        run[@"completed_spp"] = @(result.completed_spp);
        run[@"scene_setup_cpu_ms"] = @(result.scene_setup_cpu_ms);
        run[@"acceleration_build_ms"] = @(
            result.acceleration_build_ms);
        run[@"kernel_definition_ms"] = @(
            result.kernel_definition_ms);
        run[@"shader_compile_ms"] = @(result.shader_compile_ms);
        run[@"initialization_ms"] = @(result.initialization_ms);
        run[@"render_ms"] = @(result.render_ms);
        run[@"readback_ms"] = @(result.readback_ms);
        run[@"total_ms"] = @(result.total_ms);
        run[@"render_spp_per_second"] = result.render_ms > 0.0 ?
                                            @(static_cast<double>(result.completed_spp) /
                                              result.render_ms * 1000.0) :
                                            @0.0;
        if (result.success) {
            auto metrics = measure_pixels(result.pixels);
            run[@"nonblack_pixels"] = @(metrics.nonblack_pixels);
            run[@"max_channel"] = @(metrics.max_channel);
            run[@"mean_luma"] = @(metrics.mean_luma);
            run[@"pixel_sha256"] = metrics.sha256;
            final_result = std::move(result);
        } else {
            run[@"error"] = [NSString stringWithUTF8String:
                                          result.error.c_str()];
            all_runs_succeeded = false;
        }
        [runs addObject:run];
        if (!all_runs_succeeded) { break; }
    }

    if (all_runs_succeeded && !final_result.pixels.empty()) {
        auto image = image_from_rgba8(
            final_result.pixels.data(),
            final_result.resolution.x,
            final_result.resolution.y);
        auto png = UIImagePNGRepresentation(image);
        auto png_url = documents_url(
            @"luisa_ios_path_tracing_benchmark.png");
        NSError *png_error = nil;
        if (![png writeToURL:png_url
                     options:NSDataWritingAtomic
                       error:&png_error]) {
            report[@"error"] = png_error.localizedDescription;
            all_runs_succeeded = false;
        } else {
            report[@"png_path"] = png_url.path;
        }
    }
    report[@"success"] = @(all_runs_succeeded);
    return report;
}

[[nodiscard]] bool persist_report(NSDictionary *report) noexcept {
    NSError *error = nil;
    auto json = [NSJSONSerialization dataWithJSONObject:report
                                                options:NSJSONWritingPrettyPrinted |
                                                        NSJSONWritingSortedKeys
                                                  error:&error];
    if (json == nil) {
        NSLog(@"LUISA_IOS_BACKEND_BENCHMARK JSON serialization failed: %@",
              error.localizedDescription);
        return false;
    }
    auto url = documents_url(
        @"luisa_ios_path_tracing_benchmark.json");
    if (![json writeToURL:url
                  options:NSDataWritingAtomic
                    error:&error]) {
        NSLog(@"LUISA_IOS_BACKEND_BENCHMARK JSON write failed: %@",
              error.localizedDescription);
        return false;
    }
    NSLog(@"LUISA_IOS_BACKEND_BENCHMARK success=%d backend=%s json='%@'",
          [report[@"success"] boolValue], backend_name, url.path);
    return true;
}

}// namespace

@interface LuisaBenchmarkViewController : UIViewController
@property(nonatomic, strong) UILabel *statusLabel;
@end

@implementation LuisaBenchmarkViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = UIColor.blackColor;
    self.statusLabel = [UILabel new];
    self.statusLabel.translatesAutoresizingMaskIntoConstraints = NO;
    self.statusLabel.textColor = UIColor.whiteColor;
    self.statusLabel.font = [UIFont monospacedSystemFontOfSize:18.0
                                                        weight:UIFontWeightRegular];
    self.statusLabel.numberOfLines = 0;
    self.statusLabel.textAlignment = NSTextAlignmentCenter;
    self.statusLabel.text = [NSString stringWithFormat:
                                          @"Luisa %@ benchmark\nstarting...",
                                          [NSString stringWithUTF8String:backend_name]];
    [self.view addSubview:self.statusLabel];
    [NSLayoutConstraint activateConstraints:@[
        [self.statusLabel.centerXAnchor constraintEqualToAnchor:
                                            self.view.centerXAnchor],
        [self.statusLabel.centerYAnchor constraintEqualToAnchor:
                                            self.view.centerYAnchor],
        [self.statusLabel.leadingAnchor constraintGreaterThanOrEqualToAnchor:
                                            self.view.leadingAnchor
                                                                    constant:24.0],
        [self.statusLabel.trailingAnchor constraintLessThanOrEqualToAnchor:
                                             self.view.trailingAnchor
                                                                  constant:-24.0]
    ]];

    dispatch_async(
        dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
            @autoreleasepool {
                auto report = run_benchmark();
                auto persisted = persist_report(report);
                auto success = [report[@"success"] boolValue] && persisted;
                dispatch_async(dispatch_get_main_queue(), ^{
                    self.statusLabel.text = success ?
                                                @"Benchmark complete" :
                                                @"Benchmark failed";
                    std::fflush(stdout);
                    std::fflush(stderr);
                    std::_Exit(success ? EXIT_SUCCESS : EXIT_FAILURE);
                });
            }
        });
}

@end

@interface LuisaRenderingExampleSceneDelegate : UIResponder<UIWindowSceneDelegate>
@property(nonatomic, strong) UIWindow *window;
@end

@implementation LuisaRenderingExampleSceneDelegate

- (void)scene:(UIScene *)scene
    willConnectToSession:(UISceneSession *)session
                 options:(UISceneConnectionOptions *)connectionOptions {
    static_cast<void>(session);
    static_cast<void>(connectionOptions);
    if (![scene isKindOfClass:[UIWindowScene class]]) { return; }
    self.window = [[UIWindow alloc]
        initWithWindowScene:static_cast<UIWindowScene *>(scene)];
    self.window.rootViewController = [LuisaBenchmarkViewController new];
    [self.window makeKeyAndVisible];
}

@end

@interface LuisaBenchmarkAppDelegate : UIResponder<UIApplicationDelegate>
@end

@implementation LuisaBenchmarkAppDelegate

- (BOOL)application:(UIApplication *)application
    didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    static_cast<void>(application);
    static_cast<void>(launchOptions);
#if defined(LUISA_IOS_BENCHMARK_METAL4)
    luisa_compute_metal4_register_static_backend();
#else
    luisa_compute_metal_register_static_backend();
#endif
    return YES;
}

@end

int main(int argc, char *argv[]) {
    @autoreleasepool {
        return UIApplicationMain(
            argc, argv, nil,
            NSStringFromClass([LuisaBenchmarkAppDelegate class]));
    }
}
