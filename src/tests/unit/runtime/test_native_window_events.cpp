#include <luisa/gui/window.h>

#include "ut/ut.hpp"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr uint64_t native_handle = 0x1234u;

[[nodiscard]] Window::NativeHandle test_native_window_provider(
    void *, luisa::string_view, uint2) noexcept {
    return {.window = native_handle};
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "provider_backed_window_queues_native_events"_test = [] {
        Window::set_native_handle_provider(test_native_window_provider);
        Window window{"native event test", 640u, 360u};

        auto mouse_events = 0u;
        auto cursor_events = 0u;
        auto key_events = 0u;
        auto scroll_events = 0u;
        auto size_events = 0u;
        auto last_position = make_float2();
        auto last_scroll = make_float2();
        auto last_size = make_uint2();
        auto last_key = KEY_UNKNOWN;
        auto last_modifiers = KeyModifiers{};
        auto last_action = ACTION_UNKNOWN;

        window
            .set_mouse_callback(
                [&](MouseButton button, Action action, float2 position) noexcept {
                    expect(button == MOUSE_BUTTON_LEFT);
                    last_action = action;
                    last_position = position;
                    ++mouse_events;
                })
            .set_cursor_position_callback([&](float2 position) noexcept {
                last_position = position;
                ++cursor_events;
            })
            .set_key_callback(
                [&](Key key, KeyModifiers modifiers, Action action) noexcept {
                    last_key = key;
                    last_modifiers = modifiers;
                    last_action = action;
                    ++key_events;
                })
            .set_scroll_callback([&](float2 offset) noexcept {
                last_scroll = offset;
                ++scroll_events;
            })
            .set_window_size_callback([&](uint2 size) noexcept {
                last_size = size;
                ++size_events;
            });

        Window::post_native_mouse_button_event(
            native_handle, MOUSE_BUTTON_LEFT,
            ACTION_PRESSED, make_float2(10.0f, 20.0f));
        Window::post_native_cursor_position_event(
            native_handle, make_float2(30.0f, 40.0f));
        Window::post_native_key_event(
            native_handle, KEY_W, KEY_MODIFIER_SHIFT_BIT,
            ACTION_PRESSED);
        Window::post_native_scroll_event(
            native_handle, make_float2(1.0f, -2.0f));
        Window::post_native_window_size_event(
            native_handle, make_uint2(1280u, 720u));

        expect(window.is_mouse_button_down(MOUSE_BUTTON_LEFT));
        expect(window.is_key_down(KEY_W));
        expect(eq(mouse_events, 0u));
        window.poll_events();

        expect(eq(mouse_events, 1u));
        expect(eq(cursor_events, 1u));
        expect(eq(key_events, 1u));
        expect(eq(scroll_events, 1u));
        expect(eq(size_events, 1u));
        expect(all(last_position == make_float2(30.0f, 40.0f)));
        expect(all(last_scroll == make_float2(1.0f, -2.0f)));
        expect(all(last_size == make_uint2(1280u, 720u)));
        expect(last_key == KEY_W);
        expect(eq(last_modifiers, KeyModifiers{KEY_MODIFIER_SHIFT_BIT}));
        expect(last_action == ACTION_PRESSED);

        Window::post_native_mouse_button_event(
            native_handle, MOUSE_BUTTON_LEFT,
            ACTION_RELEASED, make_float2(30.0f, 40.0f));
        Window::post_native_key_event(
            native_handle, KEY_W, 0u, ACTION_RELEASED);
        expect(!window.is_mouse_button_down(MOUSE_BUTTON_LEFT));
        expect(!window.is_key_down(KEY_W));
        window.poll_events();
        expect(eq(mouse_events, 2u));
        expect(eq(key_events, 2u));
        expect(last_action == ACTION_RELEASED);

        expect(!window.should_close());
        Window::request_close_all_native_windows();
        expect(window.should_close());
        Window::clear_native_handle_provider();
    };

    return 0;
}
