#pragma once

#include <filesystem>
#include <luisa/core/stl/string.h>

namespace luisa {

namespace filesystem = std::filesystem;

// Convert a filesystem path to a luisa::string. Never throws (the library is
// built without C++ exceptions): on Windows a name that cannot be represented
// in the ANSI code page is returned lossily (replacement characters, or UTF-8
// for lone-surrogate-only failures) instead of terminating the process, so
// callers re-opening the path must tolerate a failed open.
[[nodiscard]] LUISA_CORE_API luisa::string to_string(const luisa::filesystem::path &path);

// Build a filesystem::path from narrow bytes without ever throwing.
// std::filesystem::path's narrow constructor converts the bytes to the native
// encoding on Windows and throws std::system_error when they cannot be
// represented ("No mapping for the Unicode character exists in the target
// multi-byte code page"). luisa is compiled without C++ exceptions, where that
// failure would terminate the process, so this helper reports the condition
// through its return value:
// * returns true and fills `out` when the bytes are representable;
// * returns false and leaves `out` cleared when they are not - callers treat
//   such a path as "does not exist" instead of as a hard error.
// On POSIX the bytes are stored verbatim, so the conversion cannot fail.
LUISA_CORE_API bool path_from_narrow(luisa::string_view text,
                                     luisa::filesystem::path &out) noexcept;

// Build a filesystem::path from UTF-8 narrow bytes without ever throwing.
// Same contract as path_from_narrow(), but the input bytes are decoded as
// UTF-8 rather than the ANSI code page. Use this for strings the program
// itself produced as UTF-8 - shader/asset paths coming from scripts, JSON or
// XIR interchange text, and file names re-encoded from the wide native
// enumeration - where path_from_narrow()'s code-page decoding would either
// fail (a byte sequence that is valid UTF-8 but not valid in the ACP, e.g. a
// 3-byte CJK name on a GBK machine, decoded with MB_ERR_INVALID_CHARS) or
// silently map the bytes to the wrong name. On failure (input is not valid
// UTF-8) returns false and leaves `out` cleared; callers treat that as
// "does not exist".
// On POSIX the bytes are stored verbatim, so the conversion cannot fail.
LUISA_CORE_API bool path_from_utf8(luisa::string_view text,
                                   luisa::filesystem::path &out) noexcept;

}// namespace luisa
