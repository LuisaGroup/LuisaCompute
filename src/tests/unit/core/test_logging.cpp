#include "ut/ut.hpp"
#include <luisa/core/logging.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <mutex>

using namespace boost::ut;
using namespace boost::ut::literals;

#ifndef LUISA_CUSTOM_LOGGER

struct CapturedMessage {
    luisa::string level;
    luisa::string message;
};

#endif

void _reg_logging_custom_sink() {

    "logging_custom_sink_captures_messages"_test = [] {
        std::mutex mtx;
        luisa::vector<CapturedMessage> captured;

        auto sink = luisa::detail::create_sink_with_callback(
            [&](const char *level, const char *message) {
                std::lock_guard lock{mtx};
                captured.push_back({luisa::string{level}, luisa::string{message}});
            });

        auto &logger = luisa::detail::default_logger();
        auto original_sinks = logger.sinks();
        logger.sinks().clear();
        logger.sinks().push_back(sink);
        luisa::log_level_verbose();

        luisa::log_verbose("verbose_msg_42");
        luisa::log_info("info_msg_100");
        luisa::log_warning("warning_msg_xyz");
        luisa::log_flush();

        logger.sinks() = original_sinks;
        luisa::log_level_verbose();

        expect(captured.size() >= 3u) << "expected at least 3 captured messages, got " << captured.size();

        bool found_verbose = false, found_info = false, found_warning = false;
        for (auto &cm : captured) {
            if (cm.message.find("verbose_msg_42") != luisa::string::npos) found_verbose = true;
            if (cm.message.find("info_msg_100") != luisa::string::npos) found_info = true;
            if (cm.message.find("warning_msg_xyz") != luisa::string::npos) found_warning = true;
        }
        expect(found_verbose) << "custom sink should capture verbose message";
        expect(found_info) << "custom sink should capture info message";
        expect(found_warning) << "custom sink should capture warning message";
    };
}

void _reg_logging_level_filtering() {

    "logging_level_filtering_via_sink"_test = [] {
        std::mutex mtx;
        luisa::vector<CapturedMessage> captured;

        auto sink = luisa::detail::create_sink_with_callback(
            [&](const char *level, const char *message) {
                std::lock_guard lock{mtx};
                captured.push_back({luisa::string{level}, luisa::string{message}});
            });

        auto &logger = luisa::detail::default_logger();
        auto original_sinks = logger.sinks();
        logger.sinks().clear();
        logger.sinks().push_back(sink);

        luisa::log_level_warning();
        captured.clear();

        luisa::log_verbose("should_be_filtered_verbose");
        luisa::log_info("should_be_filtered_info");
        luisa::log_warning("should_appear_warning");
        luisa::log_flush();

        logger.sinks() = original_sinks;
        luisa::log_level_verbose();

        bool found_filtered_verbose = false;
        bool found_filtered_info = false;
        bool found_warning = false;
        for (auto &cm : captured) {
            if (cm.message.find("should_be_filtered_verbose") != luisa::string::npos) found_filtered_verbose = true;
            if (cm.message.find("should_be_filtered_info") != luisa::string::npos) found_filtered_info = true;
            if (cm.message.find("should_appear_warning") != luisa::string::npos) found_warning = true;
        }
        expect(!found_filtered_verbose) << "verbose should be filtered at warning level";
        expect(!found_filtered_info) << "info should be filtered at warning level";
        expect(found_warning) << "warning should pass at warning level";
    };
}

void _reg_logging_add_sink() {

    "logging_add_sink_preserves_existing"_test = [] {
        std::mutex mtx;
        luisa::vector<CapturedMessage> captured;

        auto sink = luisa::detail::create_sink_with_callback(
            [&](const char *level, const char *message) {
                std::lock_guard lock{mtx};
                captured.push_back({luisa::string{level}, luisa::string{message}});
            });

        luisa::detail::default_logger_add_sink(sink);
        luisa::log_level_verbose();

        luisa::log_info("add_sink_test_msg");
        luisa::log_flush();

        auto &sinks = luisa::detail::default_logger().sinks();
        sinks.erase(std::remove(sinks.begin(), sinks.end(), sink), sinks.end());

        bool found = false;
        for (auto &cm : captured) {
            if (cm.message.find("add_sink_test_msg") != luisa::string::npos) found = true;
        }
        expect(found) << "added sink should capture messages alongside default sink";
    };
}

void _reg_logging_formatted_args() {

    "logging_formatted_args_in_sink"_test = [] {
        std::mutex mtx;
        luisa::vector<CapturedMessage> captured;

        auto sink = luisa::detail::create_sink_with_callback(
            [&](const char *level, const char *message) {
                std::lock_guard lock{mtx};
                captured.push_back({luisa::string{level}, luisa::string{message}});
            });

        auto &logger = luisa::detail::default_logger();
        auto original_sinks = logger.sinks();
        logger.sinks().clear();
        logger.sinks().push_back(sink);
        luisa::log_level_verbose();

        luisa::log_info("answer is {}", 42);
        luisa::log_info("pi is {:.2f}", 3.14159);
        luisa::log_info("bool={}, str={}", true, "hello");
        luisa::log_flush();

        logger.sinks() = original_sinks;
        luisa::log_level_verbose();

        bool found_42 = false, found_pi = false, found_bool = false;
        for (auto &cm : captured) {
            if (cm.message.find("answer is 42") != luisa::string::npos) found_42 = true;
            if (cm.message.find("pi is 3.14") != luisa::string::npos) found_pi = true;
            if (cm.message.find("bool=true") != luisa::string::npos) found_bool = true;
        }
        expect(found_42) << "formatted int arg should appear in sink";
        expect(found_pi) << "formatted float arg should appear in sink";
        expect(found_bool) << "formatted bool/string args should appear in sink";
    };
}

void _luisa_reg_logging_basic_functionality() {

    "logging_basic_no_crash"_test = [] {
        luisa::log_level_verbose();
        luisa::log_verbose("Verbose message from test");
        luisa::log_info("Info message from test");
        luisa::log_warning("Warning message from test");
        luisa::log_verbose("Verbose with args: {}, {}", 42, 3.14);
        luisa::log_info("Info with args: {}, {}", "test_string", true);
        luisa::log_warning("Warning with args: {}, {}", 'a', 123u);
        luisa::log_flush();
        expect(true) << "basic logging calls completed without crash";
    };
}

void _luisa_reg_logging_macros() {

    "logging_macros_no_crash"_test = [] {
        luisa::log_level_verbose();
        LUISA_VERBOSE("Macro verbose message");
        LUISA_INFO("Macro info message");
        LUISA_WARNING("Macro warning message");
        LUISA_VERBOSE("Verbose macro: {}, {}", 1, 2);
        LUISA_INFO("Info macro: {}, {}, {}", "a", "b", "c");
        LUISA_WARNING("Warning macro: value = {}", 3.14159);
        expect(true) << "logging macros completed without crash";
    };
}

void _luisa_reg_logging_with_location_macros() {

    "logging_location_macros_no_crash"_test = [] {
        luisa::log_level_verbose();
        LUISA_VERBOSE_WITH_LOCATION("Verbose with location: {}", 100);
        LUISA_INFO_WITH_LOCATION("Info with location: {}", 200);
        LUISA_WARNING_WITH_LOCATION("Warning with location: {}", 300);
        expect(true) << "location macros completed without crash";
    };
}

void _luisa_reg_logging_log_level_transitions() {

    "logging_level_transitions_no_crash"_test = [] {
        luisa::log_level_verbose();
        luisa::log_level_info();
        luisa::log_level_warning();
        luisa::log_level_error();
        luisa::log_level_verbose();
        luisa::log_level_warning();
        luisa::log_level_info();
        luisa::log_level_verbose();
        luisa::log_flush();
        expect(true) << "rapid level transitions completed without crash";
    };
}

void _luisa_reg_logging_complex_format_strings() {

    "logging_complex_format_strings"_test = [] {
        luisa::log_level_verbose();
        LUISA_INFO("Integer: {}", -123456);
        LUISA_INFO("Unsigned: {}", 123456u);
        LUISA_INFO("Float: {}", 3.14159265f);
        LUISA_INFO("Double: {}", 2.718281828459045);
        LUISA_INFO("String: {}", "test_string");
        LUISA_INFO("Boolean: {}", true);
        LUISA_INFO("Pointer: {}", static_cast<void *>(nullptr));
        LUISA_INFO("Hex: {:x}", 255);
        LUISA_INFO("Binary: {:b}", 170);
        LUISA_INFO("Scientific: {:e}", 12345.6789);
        LUISA_INFO("Fixed: {:.2f}", 3.14159);
        expect(true) << "complex format strings completed without crash";
    };
}

void _luisa_reg_logging_empty_and_special_messages() {

    "logging_empty_and_special_messages"_test = [] {
        luisa::log_level_verbose();
        LUISA_INFO("");
        LUISA_INFO("Special chars: \t\n\r!@#$%^&*()_+-=[]{}|;':\",./<>?");
        LUISA_INFO("Braces: {{escaped}}, {{{{nested}}}}");

        luisa::string long_message;
        for (int i = 0; i < 100; ++i) {
            long_message += "This is a long message repeated multiple times. ";
        }
        LUISA_INFO("Long message: {}", long_message);
        expect(long_message.size() > 4000u) << "long message should be substantial";
    };
}

int main(int argc, char *argv[]) {

    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    _reg_logging_custom_sink();
    _reg_logging_level_filtering();
    _reg_logging_add_sink();
    _reg_logging_formatted_args();
    _luisa_reg_logging_basic_functionality();
    _luisa_reg_logging_macros();
    _luisa_reg_logging_with_location_macros();
    _luisa_reg_logging_log_level_transitions();
    _luisa_reg_logging_complex_format_strings();
    _luisa_reg_logging_empty_and_special_messages();
    return 0;
}
