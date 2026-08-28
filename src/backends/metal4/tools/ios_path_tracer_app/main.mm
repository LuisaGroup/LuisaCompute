#import <UIKit/UIKit.h>

#include <Metal/Metal.hpp>

#include <CommonCrypto/CommonDigest.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <string>

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

[[nodiscard]] UIImage *image_from_rgba8(const void *bytes) noexcept {
    auto data = [NSData dataWithBytes:bytes
                               length:row_bytes * image_height];
    auto provider = CGDataProviderCreateWithCFData(
        (__bridge CFDataRef)data);
    auto color_space = CGColorSpaceCreateDeviceRGB();
    auto bitmap_info = static_cast<CGBitmapInfo>(
        static_cast<uint32_t>(kCGBitmapByteOrder32Big) |
        static_cast<uint32_t>(kCGImageAlphaLast));
    auto cg_image = CGImageCreate(
        image_width, image_height, 8u, 32u, row_bytes,
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

[[nodiscard]] LuisaRenderOutcome *render_path_tracing() noexcept {
    auto metadata = [NSMutableDictionary dictionary];
    metadata[@"air_sha256"] = @LUISA_IOS_AIR_SHA256;
    metadata[@"width"] = @(image_width);
    metadata[@"height"] = @(image_height);
    metadata[@"samples_per_pixel"] = @(samples_per_pixel);
    metadata[@"root_argument_size"] = @(sizeof(RootArguments));
    metadata[@"dispatch_size_size"] = @(sizeof(DispatchSize));
    metadata[@"shader_generation"] =
        @"host XIR -> LLVM -> AIR AOT; device MTL4 pipeline creation";
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
        NSLog(@"LUISA_IOS_METAL4_PATH_TRACING device='%@' metal4=%d apple9=%d apple10=%d air_sha256=%s",
              device_name, supports_metal4, supports_apple9, supports_apple10,
              LUISA_IOS_AIR_SHA256);
        if (!supports_metal4) {
            return failure(@"feature_guard",
                           @"GPUFamilyMetal4 is not supported", metadata);
        }

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
        NS::Error *metal_error = nullptr;
        auto library_begin = std::chrono::steady_clock::now();
        auto library = NS::TransferPtr(device->newLibrary(
            reinterpret_cast<NS::URL *>(metallib_url), &metal_error));
        auto library_end = std::chrono::steady_clock::now();
        metadata[@"library_load_ms"] = @(elapsed_ms(library_begin, library_end));
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
        auto image = image_from_rgba8(pixel_bytes);
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

@interface LuisaPathTracingViewController : UIViewController
@end

@implementation LuisaPathTracingViewController {
    UIImageView *_image_view;
    UILabel *_status_label;
    UIActivityIndicatorView *_spinner;
    BOOL _started;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = UIColor.blackColor;

    _image_view = [UIImageView new];
    _image_view.translatesAutoresizingMaskIntoConstraints = NO;
    _image_view.contentMode = UIViewContentModeScaleAspectFit;
    _image_view.backgroundColor = UIColor.blackColor;
    [self.view addSubview:_image_view];

    _status_label = [UILabel new];
    _status_label.translatesAutoresizingMaskIntoConstraints = NO;
    _status_label.textColor = UIColor.whiteColor;
    _status_label.font = [UIFont monospacedSystemFontOfSize:13.0
                                                     weight:UIFontWeightRegular];
    _status_label.numberOfLines = 0;
    _status_label.textAlignment = NSTextAlignmentCenter;
    _status_label.text =
        @"Loading host-generated XIR -> LLVM -> AIR and running it on Metal4...";
    [self.view addSubview:_status_label];

    _spinner = [[UIActivityIndicatorView alloc]
        initWithActivityIndicatorStyle:UIActivityIndicatorViewStyleLarge];
    _spinner.translatesAutoresizingMaskIntoConstraints = NO;
    _spinner.color = UIColor.whiteColor;
    [_spinner startAnimating];
    [self.view addSubview:_spinner];

    auto guide = self.view.safeAreaLayoutGuide;
    [NSLayoutConstraint activateConstraints:@[
        [_image_view.topAnchor constraintEqualToAnchor:guide.topAnchor],
        [_image_view.leadingAnchor constraintEqualToAnchor:guide.leadingAnchor],
        [_image_view.trailingAnchor constraintEqualToAnchor:guide.trailingAnchor],
        [_image_view.heightAnchor constraintEqualToAnchor:guide.heightAnchor
                                               multiplier:0.72],
        [_status_label.topAnchor constraintEqualToAnchor:_image_view.bottomAnchor
                                                constant:8.0],
        [_status_label.leadingAnchor constraintEqualToAnchor:guide.leadingAnchor
                                                    constant:12.0],
        [_status_label.trailingAnchor constraintEqualToAnchor:guide.trailingAnchor
                                                     constant:-12.0],
        [_spinner.centerXAnchor constraintEqualToAnchor:_image_view.centerXAnchor],
        [_spinner.centerYAnchor constraintEqualToAnchor:_image_view.centerYAnchor],
    ]];
}

- (void)viewDidAppear:(BOOL)animated {
    [super viewDidAppear:animated];
    if (_started) { return; }
    _started = YES;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        @autoreleasepool {
            auto outcome = render_path_tracing();
            persist_metadata(outcome.metadata);
            dispatch_async(dispatch_get_main_queue(), ^{
                [self->_spinner stopAnimating];
                self->_image_view.image = outcome.image;
                self->_status_label.text = outcome.status;
            });
        }
    });
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
