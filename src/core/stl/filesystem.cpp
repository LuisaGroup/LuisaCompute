#include <luisa/core/stl/filesystem.h>
#if defined(LUISA_PLATFORM_WINDOWS) || defined(_WIN32) || defined(_WIN64)
#ifndef UNICODE
#define UNICODE 1
#endif
#ifndef NOMINMAX
#define NOMINMAX 1
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif

#ifndef VC_EXTRALEAN
#define VC_EXTRALEAN 1
#endif

#include <windows.h>
#endif
namespace luisa {
#if defined(LUISA_PLATFORM_WINDOWS) || defined(_WIN32) || defined(_WIN64)
// One WideCharToMultiByte pass. Returns an empty string when the conversion
// fails (only possible in strict mode); a genuinely empty input is handled by
// the caller so an empty result here always means "failed".
luisa::string kfs_convert(const wchar_t *data, int len, UINT code_page,
                                 DWORD flags) {
    const int needed = ::WideCharToMultiByte(code_page, flags, data, len,
                                             nullptr, 0, nullptr, nullptr);
    if (needed <= 0) {
        return {};
    }
    luisa::string out(static_cast<size_t>(needed), '\0');
    if (::WideCharToMultiByte(code_page, flags, data, len, out.data(), needed,
                              nullptr, nullptr) != needed) {
        return {};
    }
    return out;
}
#endif
LUISA_CORE_API luisa::string to_string(const luisa::filesystem::path &path) {
#if defined(LUISA_PLATFORM_WINDOWS) || defined(_WIN32) || defined(_WIN64)
    // Never call path.string<char>() here: it converts through CP_ACP with
    // error checking and throws std::system_error ("No mapping for the
    // Unicode character exists in the target multi-byte code page") on any
    // character that is not representable in the ANSI code page (e.g. a
    // filename containing private-use-area code points). luisa is compiled
    // without C++ exceptions, so that throw terminates the whole process
    // (__fastfail 0xC0000409) - observed in the wild when grep's directory
    // walk hit a directory named 'D<U+F03A><U+F05C>proj'. Convert manually
    // with graceful degradation instead:
    const std::wstring &wide = path.native();
    if (wide.empty()) {
        return {};
    }
    if (wide.size() > static_cast<size_t>(INT_MAX)) {
        return {};
    }
    const int len = static_cast<int>(wide.size());
    // 1. Strict CP_ACP: byte-identical to what path.string<char>() produces
    //    for every representable path (the common case).
    luisa::string out = kfs_convert(wide.data(), len, CP_ACP, WC_ERR_INVALID_CHARS);
    if (!out.empty()) {
        return out;
    }
    // 2. Lossy CP_ACP: unrepresentable characters become the default
    //    replacement char, mirroring how the rest of Windows resolves such
    //    names. The result may not re-resolve to the same file; callers that
    //    re-open the path (grep, glob, read) already handle a failed open.
    out = kfs_convert(wide.data(), len, CP_ACP, 0);
    if (!out.empty()) {
        return out;
    }
    // 3. The lossy pass can still fail on lone surrogates (invalid scalars).
    //    Replace those with '?' and encode as UTF-8, which covers every
    //    remaining valid scalar value.
    std::wstring cleaned(wide);
    for (size_t i = 0; i < cleaned.size(); ++i) {
        const wchar_t wc = cleaned[i];
        const bool lone_high = (wc >= 0xD800 && wc <= 0xDBFF) &&
                               (i + 1 >= cleaned.size() || cleaned[i + 1] < 0xDC00 ||
                                cleaned[i + 1] > 0xDFFF);
        const bool lone_low = (wc >= 0xDC00 && wc <= 0xDFFF) &&
                              (i == 0 || cleaned[i - 1] < 0xD800 ||
                               cleaned[i - 1] > 0xDBFF);
        if (lone_high || lone_low) {
            cleaned[i] = L'?';
        }
    }
    out = kfs_convert(cleaned.data(), len, CP_UTF8, WC_ERR_INVALID_CHARS);
    if (!out.empty()) {
        return out;
    }
    return {};
#else
    // POSIX paths are plain byte strings; no conversion can fail.
    return luisa::string(path.native());
#endif
}

}// namespace luisa

// Unity-build hygiene: the macros defined above before <windows.h>, plus the
// classic windows.h polluters, must not leak into the other translation units
// merged into the same unity-build blob.
#if defined(LUISA_PLATFORM_WINDOWS) || defined(_WIN32) || defined(_WIN64)
#undef UNICODE
#undef NOMINMAX
#undef WIN32_LEAN_AND_MEAN
#undef VC_EXTRALEAN
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif
#ifdef pascal
#undef pascal
#endif
#endif

