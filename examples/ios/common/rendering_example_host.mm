#import <UIKit/UIKit.h>
#import <QuartzCore/CAMetalLayer.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

#include <unistd.h>

#include <luisa/gui/window.h>

#include "metal_static_backend.h"

#ifndef LUISA_IOS_RENDERING_EXAMPLE_NAME
#define LUISA_IOS_RENDERING_EXAMPLE_NAME "Luisa Rendering Example"
#endif

int luisa_ios_rendering_example_main(int argc, char *argv[]);

@interface LuisaRenderingMetalView : UIView
@end

@implementation LuisaRenderingMetalView

+ (Class)layerClass {
    return [CAMetalLayer class];
}

@end

@interface LuisaRenderingExampleViewController : UIViewController<UIGestureRecognizerDelegate>
- (luisa::compute::Window::NativeHandle)
    nativeHandleForName:(luisa::string_view)name
                   size:(luisa::uint2)size;
@end

namespace {

[[nodiscard]] luisa::compute::Window::NativeHandle native_window_provider(
    void *userdata, luisa::string_view name, luisa::uint2 size) noexcept {
    auto controller = (__bridge LuisaRenderingExampleViewController *)userdata;
    return [controller nativeHandleForName:name size:size];
}

[[nodiscard]] luisa::compute::Key translate_ui_key(UIKey *key) noexcept {
    auto characters = key.charactersIgnoringModifiers.uppercaseString;
    if ([characters isEqualToString:UIKeyInputUpArrow]) {
        return luisa::compute::KEY_UP;
    }
    if ([characters isEqualToString:UIKeyInputDownArrow]) {
        return luisa::compute::KEY_DOWN;
    }
    if ([characters isEqualToString:UIKeyInputLeftArrow]) {
        return luisa::compute::KEY_LEFT;
    }
    if ([characters isEqualToString:UIKeyInputRightArrow]) {
        return luisa::compute::KEY_RIGHT;
    }
    if (characters.length != 1u) { return luisa::compute::KEY_UNKNOWN; }
    auto c = [characters characterAtIndex:0u];
    if ((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) {
        return static_cast<luisa::compute::Key>(c);
    }
    switch (c) {
        case ' ': return luisa::compute::KEY_SPACE;
        case '-': return luisa::compute::KEY_MINUS;
        case '=':
        case '+': return luisa::compute::KEY_EQUAL;
        case 0x1b: return luisa::compute::KEY_ESCAPE;
        default: return luisa::compute::KEY_UNKNOWN;
    }
}

[[nodiscard]] luisa::compute::KeyModifiers translate_ui_modifiers(
    UIKeyModifierFlags flags) noexcept {
    auto modifiers = luisa::compute::KeyModifiers{};
    if ((flags & UIKeyModifierShift) != 0u) {
        modifiers |= luisa::compute::KEY_MODIFIER_SHIFT_BIT;
    }
    if ((flags & UIKeyModifierControl) != 0u) {
        modifiers |= luisa::compute::KEY_MODIFIER_CONTROL_BIT;
    }
    if ((flags & UIKeyModifierAlternate) != 0u) {
        modifiers |= luisa::compute::KEY_MODIFIER_ALT_BIT;
    }
    if ((flags & UIKeyModifierCommand) != 0u) {
        modifiers |= luisa::compute::KEY_MODIFIER_SUPER_BIT;
    }
    if ((flags & UIKeyModifierAlphaShift) != 0u) {
        modifiers |= luisa::compute::KEY_MODIFIER_CAPS_LOCK_BIT;
    }
    return modifiers;
}

}// namespace

@implementation LuisaRenderingExampleViewController {
    LuisaRenderingMetalView *_metal_view;
    UILabel *_status_label;
    UIActivityIndicatorView *_spinner;
    luisa::uint2 _requested_resolution;
    BOOL _has_requested_resolution;
    BOOL _started;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = UIColor.blackColor;

    _metal_view = [LuisaRenderingMetalView new];
    _metal_view.translatesAutoresizingMaskIntoConstraints = YES;
    _metal_view.backgroundColor = UIColor.blackColor;
    _metal_view.opaque = YES;
    _metal_view.clipsToBounds = YES;
    _metal_view.multipleTouchEnabled = YES;
    [self.view addSubview:_metal_view];

    auto primary_pan = [[UIPanGestureRecognizer alloc]
        initWithTarget:self action:@selector(handlePrimaryPan:)];
    primary_pan.minimumNumberOfTouches = 1u;
    primary_pan.maximumNumberOfTouches = 1u;
    primary_pan.delegate = self;
    [_metal_view addGestureRecognizer:primary_pan];

    auto secondary_pan = [[UIPanGestureRecognizer alloc]
        initWithTarget:self action:@selector(handleSecondaryPan:)];
    secondary_pan.minimumNumberOfTouches = 2u;
    secondary_pan.maximumNumberOfTouches = 2u;
    secondary_pan.delegate = self;
    [_metal_view addGestureRecognizer:secondary_pan];

    auto pinch = [[UIPinchGestureRecognizer alloc]
        initWithTarget:self action:@selector(handlePinch:)];
    pinch.delegate = self;
    [_metal_view addGestureRecognizer:pinch];

    _status_label = [UILabel new];
    _status_label.translatesAutoresizingMaskIntoConstraints = NO;
    _status_label.textColor = UIColor.whiteColor;
    _status_label.font = [UIFont monospacedSystemFontOfSize:12.0
                                                     weight:UIFontWeightRegular];
    _status_label.numberOfLines = 0;
    _status_label.textAlignment = NSTextAlignmentCenter;
    _status_label.backgroundColor = [UIColor colorWithWhite:0.0 alpha:0.62];
    _status_label.layer.cornerRadius = 10.0;
    _status_label.layer.masksToBounds = YES;
    _status_label.text = [NSString stringWithFormat:
        @"%s\npreparing Window -> Swapchain -> Metal4 AIR",
        LUISA_IOS_RENDERING_EXAMPLE_NAME];
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
        [_status_label.heightAnchor constraintGreaterThanOrEqualToConstant:54.0],
        [_spinner.centerXAnchor constraintEqualToAnchor:_metal_view.centerXAnchor],
        [_spinner.centerYAnchor constraintEqualToAnchor:_metal_view.centerYAnchor],
    ]];
}

- (BOOL)canBecomeFirstResponder {
    return YES;
}

- (uint64_t)nativeMetalLayerHandle {
    return reinterpret_cast<uint64_t>(
        (__bridge void *)static_cast<CAMetalLayer *>(_metal_view.layer));
}

- (luisa::float2)renderCoordinatesForPoint:(CGPoint)point {
    auto bounds = _metal_view.bounds;
    if (!_has_requested_resolution ||
        CGRectGetWidth(bounds) <= 0.0 || CGRectGetHeight(bounds) <= 0.0) {
        return luisa::make_float2();
    }
    auto x = std::clamp(
        static_cast<float>(point.x / CGRectGetWidth(bounds)), 0.0f, 1.0f);
    auto y = std::clamp(
        static_cast<float>(point.y / CGRectGetHeight(bounds)), 0.0f, 1.0f);
    return luisa::make_float2(
        x * static_cast<float>(_requested_resolution.x),
        y * static_cast<float>(_requested_resolution.y));
}

- (void)postPan:(UIPanGestureRecognizer *)recognizer
         button:(luisa::compute::MouseButton)button {
    if (!_has_requested_resolution) { return; }
    auto position = [self renderCoordinatesForPoint:
        [recognizer locationInView:_metal_view]];
    auto handle = [self nativeMetalLayerHandle];
    luisa::compute::Window::post_native_cursor_position_event(
        handle, position);
    switch (recognizer.state) {
        case UIGestureRecognizerStateBegan:
            luisa::compute::Window::post_native_mouse_button_event(
                handle, button, luisa::compute::ACTION_PRESSED, position);
            break;
        case UIGestureRecognizerStateEnded:
        case UIGestureRecognizerStateCancelled:
        case UIGestureRecognizerStateFailed:
            luisa::compute::Window::post_native_mouse_button_event(
                handle, button, luisa::compute::ACTION_RELEASED, position);
            break;
        default: break;
    }
}

- (void)handlePrimaryPan:(UIPanGestureRecognizer *)recognizer {
    [self postPan:recognizer button:luisa::compute::MOUSE_BUTTON_LEFT];
}

- (void)handleSecondaryPan:(UIPanGestureRecognizer *)recognizer {
    [self postPan:recognizer button:luisa::compute::MOUSE_BUTTON_RIGHT];
}

- (void)handlePinch:(UIPinchGestureRecognizer *)recognizer {
    if (!_has_requested_resolution ||
        recognizer.state != UIGestureRecognizerStateChanged) {
        return;
    }
    auto scale = std::max(static_cast<double>(recognizer.scale), 1e-6);
    auto scroll_y = static_cast<float>(std::log2(scale) * 2.0);
    luisa::compute::Window::post_native_scroll_event(
        [self nativeMetalLayerHandle], luisa::make_float2(0.0f, scroll_y));
    recognizer.scale = 1.0;
}

- (BOOL)gestureRecognizer:(UIGestureRecognizer *)gestureRecognizer
    shouldRecognizeSimultaneouslyWithGestureRecognizer:
        (UIGestureRecognizer *)otherGestureRecognizer {
    (void)gestureRecognizer;
    (void)otherGestureRecognizer;
    return YES;
}

- (void)postPresses:(NSSet<UIPress *> *)presses
             action:(luisa::compute::Action)action {
    auto handle = [self nativeMetalLayerHandle];
    for (UIPress *press in presses) {
        auto ui_key = press.key;
        if (ui_key == nil) { continue; }
        auto key = translate_ui_key(ui_key);
        if (key == luisa::compute::KEY_UNKNOWN) { continue; }
        luisa::compute::Window::post_native_key_event(
            handle, key, translate_ui_modifiers(ui_key.modifierFlags), action);
    }
}

- (void)pressesBegan:(NSSet<UIPress *> *)presses
            withEvent:(UIPressesEvent *)event {
    [self postPresses:presses action:luisa::compute::ACTION_PRESSED];
    [super pressesBegan:presses withEvent:event];
}

- (void)pressesEnded:(NSSet<UIPress *> *)presses
            withEvent:(UIPressesEvent *)event {
    [self postPresses:presses action:luisa::compute::ACTION_RELEASED];
    [super pressesEnded:presses withEvent:event];
}

- (void)pressesCancelled:(NSSet<UIPress *> *)presses
                withEvent:(UIPressesEvent *)event {
    [self postPresses:presses action:luisa::compute::ACTION_RELEASED];
    [super pressesCancelled:presses withEvent:event];
}

- (void)viewDidLayoutSubviews {
    [super viewDidLayoutSubviews];
    auto safe_frame = self.view.safeAreaLayoutGuide.layoutFrame;
    auto available_width = CGRectGetWidth(safe_frame);
    auto available_height = CGRectGetHeight(safe_frame);
    auto aspect = _has_requested_resolution && _requested_resolution.y != 0u ?
        static_cast<CGFloat>(_requested_resolution.x) /
            static_cast<CGFloat>(_requested_resolution.y) :
        1.0;
    auto width = available_width;
    auto height = width / aspect;
    if (height > available_height) {
        height = available_height;
        width = height * aspect;
    }
    _metal_view.frame = CGRectMake(
        CGRectGetMidX(safe_frame) - width * 0.5,
        CGRectGetMidY(safe_frame) - height * 0.5,
        width, height);
}

- (luisa::compute::Window::NativeHandle)
    nativeHandleForName:(luisa::string_view)name
                   size:(luisa::uint2)size {
    __block luisa::compute::Window::NativeHandle native{};
    auto configure = ^{
        self->_requested_resolution = size;
        self->_has_requested_resolution = YES;
        [self.view setNeedsLayout];
        [self.view layoutIfNeeded];
        auto layer = static_cast<CAMetalLayer *>(self->_metal_view.layer);
        layer.contentsScale = self.view.window.windowScene.screen.nativeScale;
        layer.opaque = YES;
        native.window = reinterpret_cast<uint64_t>((__bridge void *)layer);
        auto title = [NSString stringWithUTF8String:
            std::string{name}.c_str()];
        self->_status_label.text = [NSString stringWithFormat:
            @"%s\n%@, %ux%u\nWindow -> Swapchain -> Metal4 AIR",
            LUISA_IOS_RENDERING_EXAMPLE_NAME,
            title, size.x, size.y];
        [self->_spinner stopAnimating];
        NSLog(@"LUISA_IOS_RENDERING_EXAMPLE window=1 name='%s' title='%@' "
              "size=%ux%u presentation='Window->Swapchain->MTL4'",
              LUISA_IOS_RENDERING_EXAMPLE_NAME,
              title, size.x, size.y);
    };
    if ([NSThread isMainThread]) {
        configure();
    } else {
        dispatch_sync(dispatch_get_main_queue(), configure);
    }
    return native;
}

- (void)viewDidAppear:(BOOL)animated {
    [super viewDidAppear:animated];
    [self becomeFirstResponder];
    if (_started) { return; }
    _started = YES;
    luisa::compute::Window::set_native_handle_provider(
        native_window_provider, (__bridge void *)self);
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        @autoreleasepool {
            auto executable = std::string{
                NSBundle.mainBundle.executablePath.UTF8String};
            auto backend = std::string{"metal4"};
            std::array<char *, 2u> arguments{
                executable.data(), backend.data()};
            NSLog(@"LUISA_IOS_RENDERING_EXAMPLE start=1 name='%s' "
                  "backend='metal4'",
                  LUISA_IOS_RENDERING_EXAMPLE_NAME);
            auto exit_code = luisa_ios_rendering_example_main(
                static_cast<int>(arguments.size()), arguments.data());
            NSLog(@"LUISA_IOS_RENDERING_EXAMPLE finished=1 name='%s' "
                  "exit_code=%d",
                  LUISA_IOS_RENDERING_EXAMPLE_NAME, exit_code);
            luisa::compute::Window::request_close_all_native_windows();
            luisa::compute::Window::clear_native_handle_provider();
            dispatch_async(dispatch_get_main_queue(), ^{
                [self->_spinner stopAnimating];
                self->_status_label.text = [NSString stringWithFormat:
                    @"%s\nfinished with exit code %d",
                    LUISA_IOS_RENDERING_EXAMPLE_NAME, exit_code];
                self->_started = NO;
            });
        }
    });
}

- (void)viewWillDisappear:(BOOL)animated {
    [super viewWillDisappear:animated];
    luisa::compute::Window::request_close_all_native_windows();
}

@end

@interface LuisaRenderingExampleSceneDelegate : UIResponder<UIWindowSceneDelegate>
@property(nonatomic, strong) UIWindow *window;
@end

@implementation LuisaRenderingExampleSceneDelegate

- (void)scene:(UIScene *)scene
    willConnectToSession:(UISceneSession *)session
                 options:(UISceneConnectionOptions *)connectionOptions {
    (void)session;
    (void)connectionOptions;
    if (![scene isKindOfClass:[UIWindowScene class]]) { return; }
    self.window = [[UIWindow alloc]
        initWithWindowScene:static_cast<UIWindowScene *>(scene)];
    self.window.rootViewController =
        [LuisaRenderingExampleViewController new];
    [self.window makeKeyAndVisible];
}

@end

@interface LuisaRenderingExampleAppDelegate : UIResponder<UIApplicationDelegate>
@end

@implementation LuisaRenderingExampleAppDelegate

- (BOOL)application:(UIApplication *)application
    didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    (void)application;
    (void)launchOptions;
    luisa_compute_metal4_register_static_backend();
    auto urls = [[NSFileManager defaultManager]
        URLsForDirectory:NSApplicationSupportDirectory
               inDomains:NSUserDomainMask];
    auto data_url = [urls firstObject];
    NSError *error = nil;
    if (![[NSFileManager defaultManager]
            createDirectoryAtURL:data_url
     withIntermediateDirectories:YES
                      attributes:nil
                           error:&error]) {
        NSLog(@"Failed to create iOS example data directory: %@", error);
        return NO;
    }
    if (chdir(data_url.path.fileSystemRepresentation) != 0) {
        NSLog(@"Failed to enter iOS example data directory: %@", data_url.path);
        return NO;
    }
    return YES;
}

@end

int main(int argc, char *argv[]) {
    @autoreleasepool {
        return UIApplicationMain(
            argc, argv, nil,
            NSStringFromClass([LuisaRenderingExampleAppDelegate class]));
    }
}
