// Test for luisa/core/stl/filesystem.h.
// Covers: to_string(), path_from_narrow(), path_from_utf8().
//
// The assertions stay code-page agnostic: nothing here depends on how a
// particular ANSI code page maps a given non-ASCII character, only on the
// documented contracts (never throw, never return empty for a non-empty path,
// clear the output on failure, round-trip anything that was decoded).
#include <string>
#include <string_view>
#include "ut/ut.hpp"
#include <luisa/core/logging.h>
#include <luisa/core/stl/filesystem.h>

using namespace boost::ut;
using namespace boost::ut::literals;

// ---- to_string ----
void reg_to_string_ascii_round_trip() {
    "to_string of an ASCII path is the identity"_test = [] {
        auto text = luisa::string{"C:/dev/compute/shaders/test.hlsl"};
        luisa::filesystem::path path;
        expect(luisa::path_from_narrow(text, path)) << "ASCII must decode";
        expect(luisa::to_string(path) == text) << "got: " << luisa::to_string(path);
    };
}

void reg_to_string_empty_path() {
    "to_string of an empty path is empty"_test = [] {
        expect(luisa::to_string(luisa::filesystem::path{}).empty());
    };
}

void reg_to_string_unrepresentable_name() {
    // A directory name made of private-use-area code points (the case that got a
    // real process to __fastfail: path::string<char>() converts through CP_ACP
    // with error checking and throws std::system_error, which is fatal in a
    // build without exception support). to_string() must degrade instead.
    "to_string never throws on a name the code page cannot represent"_test = [] {
        const wchar_t name[] = {L'D', 0xF03Au, 0xF05Cu, L'p', L'r', L'o', L'j', 0};
        auto path = luisa::filesystem::path{std::wstring_view{name}};
        auto text = luisa::to_string(path);
        expect(!text.empty()) << "a non-empty path must not stringify to empty";
        LUISA_INFO("Lossy path string: {}", text);
    };
}

void reg_to_string_lone_surrogate() {
#if defined(LUISA_PLATFORM_WINDOWS)
    // The lossy CP_ACP pass also fails on lone surrogates (invalid Unicode
    // scalars); the UTF-8 fallback replaces them so the result stays non-empty.
    "to_string survives a lone surrogate in the file name"_test = [] {
        std::wstring name = L"surrogate";
        name.insert(4u, 1u, static_cast<wchar_t>(0xD800u));// high surrogate, no low pair
        auto path = luisa::filesystem::path{name};
        auto text = luisa::to_string(path);
        expect(!text.empty()) << "a non-empty path must not stringify to empty";
        LUISA_INFO("Lone-surrogate path string: {}", text);
    };
#endif
}

// ---- path_from_narrow ----
void reg_path_from_narrow_empty() {
    "path_from_narrow accepts the empty string"_test = [] {
        auto path = luisa::filesystem::path{"C:/not/empty.txt"};
        expect(luisa::path_from_narrow(luisa::string_view{}, path))
            << "an empty path is valid, not a failure";
        expect(path.empty()) << "the output is cleared first";
    };
}

void reg_path_from_narrow_ascii() {
    "path_from_narrow decodes ASCII and round-trips through to_string"_test = [] {
        auto text = luisa::string_view{"relative/dir name/with space.txt"};
        luisa::filesystem::path path;
        expect(luisa::path_from_narrow(text, path));
        expect(!path.empty());
        expect(luisa::to_string(path) == text) << "got: " << luisa::to_string(path);
    };
}

// ---- path_from_utf8 ----
void reg_path_from_utf8_ascii() {
    "path_from_utf8 agrees with path_from_narrow for ASCII text"_test = [] {
        auto text = luisa::string_view{"C:/dev/compute/shaders/test.hlsl"};
        luisa::filesystem::path narrow_path;
        luisa::filesystem::path utf8_path;
        expect(luisa::path_from_narrow(text, narrow_path));
        expect(luisa::path_from_utf8(text, utf8_path));
        expect(utf8_path == narrow_path) << "ASCII decodes identically";
    };
}

void reg_path_from_utf8_invalid_bytes() {
#if defined(LUISA_PLATFORM_WINDOWS)
    // 0xFF/0xFE are never valid UTF-8 lead bytes: the strict decoder must reject
    // them instead of silently mapping the bytes onto a different name.
    "path_from_utf8 rejects invalid UTF-8 and clears the output"_test = [] {
        luisa::filesystem::path out{"C:/not/empty.txt"};
        expect(!luisa::path_from_utf8("\xff\xfe", out)) << "invalid UTF-8 must fail";
        expect(out.empty()) << "the output is cleared before decoding";
    };
    "path_from_utf8 accepts multi-byte UTF-8 text"_test = [] {
        // "test-\xe6\x96\x87.txt" is UTF-8 for "test-文.txt".
        luisa::filesystem::path out;
        expect(luisa::path_from_utf8("test-\xe6\x96\x87.txt", out))
            << "valid UTF-8 must decode";
        expect(!out.empty());
    };
#else
    // POSIX paths are byte strings: nothing can fail to decode there.
    "path_from_utf8 stores bytes verbatim on POSIX"_test = [] {
        luisa::filesystem::path out;
        expect(luisa::path_from_utf8("\xff\xfe", out)) << "no conversion can fail";
        expect(!out.empty());
    };
#endif
}

void reg_helper_agreement_with_path_api() {
    // The helpers must produce exactly what the ordinary (throwing) path
    // constructor produces for representable text, otherwise every caller that
    // switched to them would resolve a different file.
    "decoded paths behave like constructed ones"_test = [] {
        auto text = luisa::string_view{"C:/Windows/System32"};
        luisa::filesystem::path decoded;
        expect(luisa::path_from_narrow(text, decoded));
        auto constructed = luisa::filesystem::path{text};
        expect(decoded == constructed);
        expect(decoded.is_absolute() == constructed.is_absolute());
        expect(luisa::to_string(decoded) == luisa::to_string(constructed));
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_to_string_ascii_round_trip();
    reg_to_string_empty_path();
    reg_to_string_unrepresentable_name();
    reg_to_string_lone_surrogate();
    reg_path_from_narrow_empty();
    reg_path_from_narrow_ascii();
    reg_path_from_utf8_ascii();
    reg_path_from_utf8_invalid_bytes();
    reg_helper_agreement_with_path_api();
    return 0;
}
