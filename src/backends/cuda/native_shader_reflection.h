#pragma once

// CUDA native-shader reflection for `NativeShaderExt` on the CUDA backend.
//
// A CUDA kernel carries no resource-class metadata: all it has is a parameter
// list, and the device binary (PTX) only records the *layout* of that list
// (sizes and alignment). This header therefore reflects a native CUDA shader
// from two inputs and cross-checks them:
//
//  * the compiled PTX, which is authoritative for the entry-point name, the
//    parameter layout (`.param` sizes/alignment) and the declared workgroup
//    size (`.maxntid`), and
//  * the `__global__` signature in the source, which is the only place where a
//    parameter's *role* is visible: a pointer parameter is a buffer, and its
//    `const` qualifier makes the binding read-only.
//
// The classification is deliberately small, because the CUDA route of the
// native-shader API supports buffers and scalars only:
//
//  * `T *p` / `T __restrict__ *p`  -> one buffer binding (writable, UAV-class),
//  * `const T *p`                  -> one buffer binding (read-only, SRV-class),
//  * any other parameter           -> a scalar kernel parameter, i.e. one of the
//                                     launcher's `add_uniform` values.
//
// The merge step fails closed whenever the two inputs disagree (a parsing bug,
// an array/reference parameter, a non-64-bit pointer parameter, ...), so a
// mismatch can never silently produce a wrong binding table.
//
// This header is intentionally dependency-free (no CUDA headers, no device),
// so the host-side unit tests can exercise it without a GPU.
#include <cstdint>
#include <cstring>

#include <luisa/core/basic_types.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::cuda::native_shader {

namespace detail {

[[nodiscard]] inline bool is_identifier_start(char c) noexcept {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           c == '_' || c == '$';
}

[[nodiscard]] inline bool is_identifier_char(char c) noexcept {
    return is_identifier_start(c) || (c >= '0' && c <= '9');
}

[[nodiscard]] inline bool is_space(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

[[nodiscard]] inline luisa::string_view trim(luisa::string_view s) noexcept {
    while (!s.empty() && is_space(s.front())) { s.remove_prefix(1u); }
    while (!s.empty() && is_space(s.back())) { s.remove_suffix(1u); }
    return s;
}

// Token-bounded search (`const` must not match `constant`).
[[nodiscard]] inline bool has_token(luisa::string_view text,
                                    luisa::string_view token) noexcept {
    auto pos = text.find(token);
    while (pos != luisa::string_view::npos) {
        auto before_ok = pos == 0u || !is_identifier_char(text[pos - 1u]);
        auto end = pos + token.size();
        auto after_ok = end >= text.size() || !is_identifier_char(text[end]);
        if (before_ok && after_ok) { return true; }
        pos = text.find(token, pos + 1u);
    }
    return false;
}

// Replaces comments and (raw) string/character literals with spaces so that a
// later scan cannot be confused by their contents. Newlines are preserved.
[[nodiscard]] inline luisa::string strip_comments_and_literals(
    luisa::string_view source) noexcept {
    luisa::string result;
    result.resize(source.size(), ' ');
    auto i = size_t{0u};
    auto n = source.size();
    auto copyable = [&](size_t begin, size_t end) noexcept {
        for (auto k = begin; k < end && k < n; k++) {
            if (source[k] == '\n') { result[k] = '\n'; }
        }
    };
    while (i < n) {
        auto c = source[i];
        if (c == '/' && i + 1u < n && source[i + 1u] == '/') {
            auto begin = i;
            while (i < n && source[i] != '\n') { ++i; }
            copyable(begin, i);
        } else if (c == '/' && i + 1u < n && source[i + 1u] == '*') {
            auto begin = i;
            i += 2u;
            while (i + 1u < n && !(source[i] == '*' && source[i + 1u] == '/')) { ++i; }
            i = std::min(n, i + 2u);
            copyable(begin, i);
        } else if (c == 'R' && i + 1u < n && source[i + 1u] == '"') {
            // raw string: R"delim( ... )delim"
            auto paren = source.find('(', i + 2u);
            if (paren == luisa::string_view::npos) { i = n; continue; }
            auto delimiter = source.substr(i + 2u, paren - i - 2u);
            auto closer = luisa::string{")"};
            closer.append(delimiter);
            closer.append("\"");
            auto end = source.find(luisa::string_view{closer}, paren + 1u);
            auto stop = end == luisa::string_view::npos ? n : end + closer.size();
            copyable(i, stop);
            i = stop;
        } else if (c == '"' || c == '\'') {
            auto quote = c;
            auto begin = i;
            ++i;
            while (i < n && source[i] != quote) {
                if (source[i] == '\\' && i + 1u < n) { ++i; }
                ++i;
            }
            i = std::min(n, i + 1u);
            copyable(begin, i);
        } else {
            result[i] = c;
            ++i;
        }
    }
    return result;
}

// Advances past the parenthesised group that starts at `open` and returns the
// index of its matching `)`.
[[nodiscard]] inline size_t skip_matching_paren(
    luisa::string_view text, size_t open) noexcept {
    auto depth = 0;
    for (auto i = open; i < text.size(); i++) {
        if (text[i] == '(') {
            ++depth;
        } else if (text[i] == ')') {
            if (--depth == 0) { return i; }
        }
    }
    return luisa::string_view::npos;
}

// Splits on commas that are not nested in (), [] or {}.
[[nodiscard]] inline bool split_top_level(
    luisa::string_view text,
    luisa::vector<luisa::string_view> &out) noexcept {
    auto depth = 0;
    auto begin = 0u;
    for (auto i = 0u; i < text.size(); i++) {
        switch (text[i]) {
            case '(': case '[': case '{': ++depth; break;
            case ')': case ']': case '}':
                if (--depth < 0) { return false; }
                break;
            case ',':
                if (depth == 0) {
                    out.emplace_back(text.substr(begin, i - begin));
                    begin = i + 1u;
                }
                break;
            default: break;
        }
    }
    if (depth != 0) { return false; }
    if (begin < text.size()) { out.emplace_back(text.substr(begin)); }
    return true;
}

}// namespace detail

// One parameter of the native CUDA kernel, after merging the source signature
// with the compiled parameter list.
struct KernelParameter {
    uint32_t size{0u};      // bytes, as compiled
    uint32_t alignment{0u}; // bytes, as compiled
    bool is_buffer{false};  // pointer-typed in the source signature
    bool read_only{false};  // `const T *` => a read-only binding
};

[[nodiscard]] inline luisa::string_view kind_name(bool read_only) noexcept {
    return read_only ? luisa::string_view{"read-only buffer"} :
                       luisa::string_view{"writable buffer"};
}

// ---------------------------------------------------------------------------
// PTX reflection
// ---------------------------------------------------------------------------

struct ModuleParameter {
    uint32_t size{0u};
    uint32_t alignment{0u};
};

struct ModuleReflection {
    luisa::string entry_point;
    uint3 block_size{0u, 0u, 0u};// PTX `.maxntid` (0 when the kernel declares none)
    luisa::vector<ModuleParameter> parameters;
    luisa::vector<luisa::string> entry_names;// every `.entry` of the module
    luisa::string error;
    [[nodiscard]] bool ok() const noexcept { return error.empty(); }
};

namespace detail {

// Splits the module into per-entry sections: each section starts at the line
// that declares one `.entry` and runs up to the next one. Parameter lists and
// `.maxntid` are read inside a section, so several kernels in one module cannot
// contaminate each other.
struct PtxEntrySection {
    luisa::string name;
    luisa::string_view text;
};

[[nodiscard]] inline luisa::vector<PtxEntrySection> split_ptx_entries(
    luisa::string_view ptx) noexcept {
    luisa::vector<PtxEntrySection> sections;
    luisa::vector<std::pair<luisa::string, size_t>> found;// (name, section start)
    auto i = 0u;
    while (i < ptx.size()) {
        auto line_end = ptx.find('\n', i);
        auto line = ptx.substr(i, line_end == luisa::string_view::npos ?
                                     luisa::string_view::npos :
                                     line_end - i);
        // `.entry` must be a token at the start of a directive position: the
        // modifiers (`.visible`, `.weak`, `.extern`) may precede it.
        auto pos = line.find(".entry");
        while (pos != luisa::string_view::npos) {
            auto before_ok = pos == 0u || is_space(line[pos - 1u]);
            auto after = pos + 6u;
            auto after_ok = after < line.size() && is_space(line[after]);
            if (before_ok && after_ok) { break; }
            pos = line.find(".entry", pos + 1u);
        }
        if (pos != luisa::string_view::npos) {
            auto rest = trim(line.substr(pos + 6u));
            auto name_end = 0u;
            while (name_end < rest.size() && is_identifier_char(rest[name_end])) {
                ++name_end;
            }
            if (name_end > 0u) {
                found.emplace_back(luisa::string{rest.substr(0u, name_end)}, i);
            }
        }
        i = line_end == luisa::string_view::npos ? ptx.size() : line_end + 1u;
    }
    for (auto k = 0u; k < found.size(); k++) {
        auto begin = found[k].second;
        auto end = k + 1u < found.size() ? found[k + 1u].second : ptx.size();
        sections.emplace_back(PtxEntrySection{
            found[k].first, ptx.substr(begin, end - begin)});
    }
    return sections;
}

// Parses one `.param` declaration: ".param [.align N] .<type> name[SIZE]".
[[nodiscard]] inline bool parse_ptx_parameter(
    luisa::string_view declaration, ModuleParameter &parameter,
    luisa::string &error) noexcept {
    declaration = trim(declaration);
    if (declaration.starts_with(".param")) {
        declaration = trim(declaration.substr(6u));
    }
    uint32_t alignment = 0u;
    if (declaration.starts_with(".align")) {
        declaration = trim(declaration.substr(6u));
        auto digits = 0u;
        uint32_t value = 0u;
        while (digits < declaration.size() &&
               declaration[digits] >= '0' && declaration[digits] <= '9') {
            value = value * 10u + static_cast<uint32_t>(declaration[digits] - '0');
            ++digits;
        }
        if (digits == 0u) {
            error = luisa::format("malformed PTX parameter alignment in '{}'.",
                                  declaration);
            return false;
        }
        alignment = value;
        declaration = trim(declaration.substr(digits));
    }
    // the type token
    auto type_end = declaration.find_first_of(" \t");
    auto type = type_end == luisa::string_view::npos ?
                    declaration :
                    declaration.substr(0u, type_end);
    uint32_t size = 0u;
    if (type == ".b8" || type == ".u8" || type == ".s8") {
        size = 1u;
    } else if (type == ".b16" || type == ".u16" || type == ".s16" ||
               type == ".f16" || type == ".b16x2") {
        size = type == ".b16x2" ? 4u : 2u;
    } else if (type == ".b32" || type == ".u32" || type == ".s32" ||
               type == ".f32") {
        size = 4u;
    } else if (type == ".b64" || type == ".u64" || type == ".s64" ||
               type == ".f64") {
        size = 8u;
    } else {
        error = luisa::format(
            "PTX parameter '{}' has an unsupported type; the CUDA "
            "native-shader route reflects only scalar and 64-bit pointer "
            "parameters.",
            declaration);
        return false;
    }
    // the (optional) array suffix of the name gives the size of an aggregate
    if (type_end != luisa::string_view::npos) {
        auto name = trim(declaration.substr(type_end));
        if (auto bracket = name.find('['); bracket != luisa::string_view::npos) {
            auto close = name.find(']', bracket);
            if (close == luisa::string_view::npos) {
                error = luisa::format("malformed PTX parameter '{}'.", declaration);
                return false;
            }
            uint32_t value = 0u;
            for (auto k = bracket + 1u; k < close; k++) {
                if (name[k] < '0' || name[k] > '9') {
                    error = luisa::format(
                        "PTX parameter '{}' has a non-constant array size.",
                        declaration);
                    return false;
                }
                value = value * 10u + static_cast<uint32_t>(name[k] - '0');
            }
            size = value;
        }
    }
    if (size == 0u) {
        error = luisa::format("PTX parameter '{}' has a zero size.", declaration);
        return false;
    }
    parameter.size = size;
    parameter.alignment = alignment == 0u ? std::min(size, 16u) : alignment;
    return true;
}

}// namespace detail

// Reflects the entry `requested` (an empty name means "the only entry") of a
// PTX module: its parameter layout and its declared workgroup size.
[[nodiscard]] inline ModuleReflection reflect_ptx(
    luisa::string_view ptx, luisa::string_view requested) noexcept {
    ModuleReflection reflection;
    auto sections = detail::split_ptx_entries(ptx);
    for (auto &&section : sections) {
        reflection.entry_names.emplace_back(section.name);
    }
    if (sections.empty()) {
        reflection.error =
            "The compiled PTX module declares no `.entry` (kernel); the CUDA "
            "native-shader source must define a `__global__` function.";
        return reflection;
    }
    auto selected = static_cast<size_t>(sections.size());
    if (requested.empty()) {
        if (sections.size() != 1u) {
            reflection.error = luisa::format(
                "The compiled PTX module declares {} kernels ({}); an explicit "
                "`entry_point` is required.",
                sections.size(),
                [&] {
                    luisa::string names;
                    for (auto &&section : sections) {
                        if (!names.empty()) { names.append(", "); }
                        names.append(section.name);
                    }
                    return names;
                }());
            return reflection;
        }
        selected = 0u;
    } else {
        for (auto i = 0u; i < sections.size(); i++) {
            if (sections[i].name == requested) {
                selected = i;
                break;
            }
        }
        if (selected == sections.size()) {
            reflection.error = luisa::format(
                "The compiled PTX module has no kernel named '{}'; declare the "
                "entry point `extern \"C\"` so that its name is not mangled "
                "(the module declares: {}).",
                requested,
                [&] {
                    luisa::string names;
                    for (auto &&section : sections) {
                        if (!names.empty()) { names.append(", "); }
                        names.append(section.name);
                    }
                    return names;
                }());
            return reflection;
        }
    }
    reflection.entry_point = sections[selected].name;
    auto text = sections[selected].text;
    // parameter list
    auto open = text.find('(');
    if (open == luisa::string_view::npos) {
        reflection.error = luisa::format(
            "Malformed PTX: kernel '{}' has no parameter list.",
            reflection.entry_point);
        return reflection;
    }
    auto close = detail::skip_matching_paren(text, open);
    if (close == luisa::string_view::npos) {
        reflection.error = luisa::format(
            "Malformed PTX: the parameter list of kernel '{}' is not "
            "terminated.",
            reflection.entry_point);
        return reflection;
    }
    luisa::vector<luisa::string_view> declarations;
    auto parameters = text.substr(open + 1u, close - open - 1u);
    if (!detail::split_top_level(parameters, declarations)) {
        reflection.error = luisa::format(
            "Malformed PTX: the parameter list of kernel '{}' cannot be "
            "parsed.",
            reflection.entry_point);
        return reflection;
    }
    for (auto &&declaration : declarations) {
        if (detail::trim(declaration).empty()) { continue; }
        ModuleParameter parameter;
        if (!detail::parse_ptx_parameter(declaration, parameter,
                                         reflection.error)) {
            return reflection;
        }
        reflection.parameters.emplace_back(parameter);
    }
    // workgroup size: `.maxntid x, y, z`
    if (auto pos = text.find(".maxntid"); pos != luisa::string_view::npos) {
        auto rest = text.substr(pos + 8u);
        uint component[3] = {0u, 0u, 0u};
        auto count = 0u;
        for (auto i = 0u; i < rest.size() && rest[i] != '\n' && count < 3u;) {
            if (rest[i] < '0' || rest[i] > '9') {
                ++i;
                continue;
            }
            uint32_t value = 0u;
            while (i < rest.size() && rest[i] >= '0' && rest[i] <= '9') {
                value = value * 10u + static_cast<uint32_t>(rest[i] - '0');
                ++i;
            }
            component[count++] = value;
        }
        if (count == 3u) {
            reflection.block_size = uint3{component[0], component[1], component[2]};
        }
    }
    return reflection;
}

// ---------------------------------------------------------------------------
// source reflection
// ---------------------------------------------------------------------------

struct SourceParameter {
    bool is_buffer{false};
    bool read_only{false};
    luisa::string text;// as written (diagnostics)
};

struct SourceReflection {
    luisa::string entry_point;
    luisa::vector<SourceParameter> parameters;
    luisa::vector<luisa::string> kernel_names;// every `__global__` found
    luisa::string error;
    [[nodiscard]] bool ok() const noexcept { return error.empty(); }
    [[nodiscard]] luisa::string joined_kernel_names() const noexcept {
        luisa::string names;
        for (auto &&name : kernel_names) {
            if (!names.empty()) { names.append(", "); }
            names.append(name);
        }
        return names;
    }
};

namespace detail {

// Classifies one parameter of a `__global__` signature: pointer parameters are
// buffers (read-only when `const`), everything else is a scalar kernel
// parameter.
[[nodiscard]] inline bool classify_source_parameter(
    luisa::string_view text, SourceParameter &parameter,
    luisa::string &error) noexcept {
    auto body = trim(text);
    parameter.text = luisa::string{body};
    if (body.empty()) {
        error = "The kernel signature has an empty parameter.";
        return false;
    }
    if (body.find("...") != luisa::string_view::npos) {
        error = luisa::format(
            "Kernel parameter '{}' is variadic, which the CUDA native-shader "
            "route does not support.",
            body);
        return false;
    }
    if (body.find('[') != luisa::string_view::npos ||
        body.find(']') != luisa::string_view::npos) {
        error = luisa::format(
            "Kernel parameter '{}' is an array; pass a pointer to the buffer "
            "instead.",
            body);
        return false;
    }
    if (body.find('&') != luisa::string_view::npos) {
        error = luisa::format(
            "Kernel parameter '{}' is a reference; the CUDA native-shader "
            "route supports pointers to buffers and scalar parameters only.",
            body);
        return false;
    }
    if (body.find('(') != luisa::string_view::npos) {
        error = luisa::format(
            "Kernel parameter '{}' is not a plain type; the CUDA native-shader "
            "route supports pointers to buffers and scalar parameters only.",
            body);
        return false;
    }
    if (body.find('*') != luisa::string_view::npos) {
        parameter.is_buffer = true;
        // `const float *` and `float const *` both describe a read-only buffer;
        // detect the qualifier anywhere in the declaration, which is enough for
        // the supported shapes (`const T *`, `T const *`, `const T *__restrict`).
        parameter.read_only = has_token(body, "const");
    }
    return true;
}

// `__global__` may be preceded or followed by attributes whose own argument
// list must not be mistaken for the parameter list.
[[nodiscard]] inline bool is_function_attribute(luisa::string_view token) noexcept {
    return token == "__launch_bounds__" || token == "__cluster_dims__" ||
           token == "__maxnreg__" || token == "__attribute__";
}

// Parses the `__global__` declarations of a (comment- and literal-stripped)
// translation unit, in source order.
[[nodiscard]] inline bool parse_global_functions(
    luisa::string_view text, luisa::vector<SourceReflection> &kernels,
    luisa::string &error) noexcept {
    constexpr auto attribute = luisa::string_view{"__global__"};
    auto search_from = size_t{0u};
    while (true) {
        auto pos = text.find(attribute, search_from);
        if (pos == luisa::string_view::npos) { break; }
        search_from = pos + attribute.size();
        if (pos > 0u && (is_identifier_char(text[pos - 1u]) ||
                         text[pos - 1u] == '.')) {
            continue;// part of a longer identifier
        }
        // Find the function name: the last identifier before the `(` that is
        // not an attribute's own argument list. A `{`, `}` or `;` before that
        // `(` means the token is not a declaration at all (e.g. a `__global__`
        // mention inside another kernel's body), so this occurrence is skipped.
        auto i = search_from;
        luisa::string name;
        size_t open = luisa::string_view::npos;
        auto malformed = false;
        while (i < text.size()) {
            auto next_open = text.find('(', i);
            auto stopper = text.find_first_of("{};", i);
            if (next_open == luisa::string_view::npos ||
                (stopper != luisa::string_view::npos && stopper < next_open)) {
                break;
            }
            auto end = next_open;
            while (end > i && is_space(text[end - 1u])) { --end; }
            auto begin = end;
            while (begin > i && is_identifier_char(text[begin - 1u])) { --begin; }
            auto token = begin < end ? text.substr(begin, end - begin) :
                                       luisa::string_view{};
            auto qualified = begin > 0u && (text[begin - 1u] == '>' ||
                                            text[begin - 1u] == ':' ||
                                            text[begin - 1u] == '.' ||
                                            text[begin - 1u] == '~');
            if (!token.empty() && !qualified &&
                !is_function_attribute(token)) {
                name = luisa::string{token};
                open = next_open;
                break;
            }
            if (qualified) {
                malformed = true;// e.g. a template specialization: unsupported
                open = next_open;
                break;
            }
            // an attribute's argument list: skip it and keep looking
            auto close = skip_matching_paren(text, next_open);
            if (close == luisa::string_view::npos) {
                error = luisa::format(
                    "Malformed CUDA source: unbalanced parentheses in a "
                    "`__global__` declaration near offset {}.", pos);
                return false;
            }
            i = close + 1u;
        }
        if (malformed) {
            error = luisa::format(
                "Unsupported CUDA source: cannot determine the name of a "
                "`__global__` function near offset {} (templated, qualified or "
                "operator declarations are not supported).", pos);
            return false;
        }
        if (name.empty()) { continue; }// not a declaration
        auto close = skip_matching_paren(text, open);
        if (close == luisa::string_view::npos) {
            error = luisa::format(
                "Malformed CUDA source: the parameter list of kernel '{}' is "
                "not terminated.", name);
            return false;
        }
        SourceReflection kernel;
        kernel.entry_point = std::move(name);
        auto parameters = text.substr(open + 1u, close - open - 1u);
        if (!trim(parameters).empty()) {
            luisa::vector<luisa::string_view> declarations;
            if (!split_top_level(parameters, declarations)) {
                error = luisa::format(
                    "Malformed CUDA source: the parameter list of kernel '{}' "
                    "cannot be parsed.", kernel.entry_point);
                return false;
            }
            for (auto &&declaration : declarations) {
                SourceParameter parameter;
                if (!classify_source_parameter(declaration, parameter, error)) {
                    return false;
                }
                kernel.parameters.emplace_back(std::move(parameter));
            }
        }
        kernels.emplace_back(std::move(kernel));
    }
    return true;
}

}// namespace detail

// Reflects the `__global__` signature that `requested` selects (an empty name
// or the default `main` selects the only `__global__` of the translation unit).
[[nodiscard]] inline SourceReflection reflect_source(
    luisa::string_view source, luisa::string_view requested) noexcept {
    SourceReflection reflection;
    auto text = detail::strip_comments_and_literals(source);
    luisa::vector<SourceReflection> kernels;
    if (!detail::parse_global_functions(text, kernels, reflection.error)) {
        return reflection;
    }
    if (kernels.empty()) {
        reflection.error =
            "The CUDA native-shader source declares no `__global__` function; "
            "the CUDA route compiles kernels, not host code.";
        return reflection;
    }
    for (auto &&kernel : kernels) {
        reflection.kernel_names.emplace_back(kernel.entry_point);
    }
    auto selects_default = requested.empty() || requested == "main";
    auto selected = kernels.size();
    if (selects_default) {
        if (kernels.size() != 1u) {
            reflection.error = luisa::format(
                "The CUDA native-shader source declares {} `__global__` "
                "functions ({}); pass `NativeShaderCompileInfo::entry_point` "
                "explicitly.",
                kernels.size(), reflection.joined_kernel_names());
            return reflection;
        }
        selected = 0u;
    } else {
        for (auto i = 0u; i < kernels.size(); i++) {
            if (kernels[i].entry_point == requested) {
                selected = i;
                break;
            }
        }
        if (selected == kernels.size()) {
            reflection.error = luisa::format(
                "The CUDA native-shader source declares no `__global__` "
                "function named '{}' (it declares: {}).",
                requested, reflection.joined_kernel_names());
            return reflection;
        }
    }
    reflection.entry_point = kernels[selected].entry_point;
    reflection.parameters = std::move(kernels[selected].parameters);
    return reflection;
}

// ---------------------------------------------------------------------------
// merged reflection
// ---------------------------------------------------------------------------

struct Reflection {
    luisa::string entry_point;
    uint3 block_size{0u, 0u, 0u};// 0 when neither the PTX nor the caller says
    luisa::vector<KernelParameter> parameters;// kernel signature order
    uint32_t uniform_bytes{0u};  // total size of the scalar parameters
    luisa::string error;
    [[nodiscard]] bool ok() const noexcept { return error.empty(); }
    [[nodiscard]] size_t buffer_count() const noexcept {
        size_t count = 0u;
        for (auto &&parameter : parameters) {
            if (parameter.is_buffer) { ++count; }
        }
        return count;
    }
    [[nodiscard]] size_t scalar_count() const noexcept {
        return parameters.size() - buffer_count();
    }
    // The launcher's buffer bindings in canonical (parameter) order.
    [[nodiscard]] size_t buffer_parameter_index(size_t buffer_index) const noexcept {
        size_t found = 0u;
        for (auto i = 0u; i < parameters.size(); i++) {
            if (parameters[i].is_buffer && found++ == buffer_index) { return i; }
        }
        return parameters.size();
    }
};

// Merges the two reflections, failing closed on any disagreement: the parameter
// counts must match, and every buffer parameter must be a 64-bit pointer in the
// compiled layout.
[[nodiscard]] inline Reflection merge(ModuleReflection module,
                                      SourceReflection const &source) noexcept {
    Reflection reflection;
    reflection.entry_point = module.entry_point;
    reflection.block_size = module.block_size;
    if (module.entry_point != source.entry_point) {
        reflection.error = luisa::format(
            "The compiled PTX entry point '{}' does not match the reflected "
            "source kernel '{}'.",
            module.entry_point, source.entry_point);
        return reflection;
    }
    if (module.parameters.size() != source.parameters.size()) {
        reflection.error = luisa::format(
            "Kernel '{}' declares {} parameter(s) in the source but {} in the "
            "compiled PTX; the CUDA native-shader reflection cannot classify "
            "the kernel.",
            module.entry_point, source.parameters.size(),
            module.parameters.size());
        return reflection;
    }
    reflection.parameters.reserve(module.parameters.size());
    for (auto i = 0u; i < module.parameters.size(); i++) {
        auto &&compiled = module.parameters[i];
        auto &&declared = source.parameters[i];
        if (declared.is_buffer && compiled.size != sizeof(uint64_t)) {
            reflection.error = luisa::format(
                "Kernel '{}' parameter {} ('{}') is a buffer pointer but "
                "occupies {} bytes in the compiled PTX; the CUDA native-shader "
                "route requires 64-bit device addresses.",
                module.entry_point, i, declared.text, compiled.size);
            return reflection;
        }
        if (!declared.is_buffer && compiled.size == 0u) {
            reflection.error = luisa::format(
                "Kernel '{}' parameter {} ('{}') has a zero size in the "
                "compiled PTX.", module.entry_point, i, declared.text);
            return reflection;
        }
        reflection.parameters.emplace_back(KernelParameter{
            compiled.size, compiled.alignment,
            declared.is_buffer, declared.read_only});
        if (!declared.is_buffer) { reflection.uniform_bytes += compiled.size; }
    }
    return reflection;
}

}// namespace luisa::compute::cuda::native_shader
