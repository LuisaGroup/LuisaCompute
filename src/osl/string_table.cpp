#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include <luisa/core/stl/format.h>
#include <luisa/osl/string_table.h>
#include "translator_impl.h"

namespace luisa::compute::osl {

StringTable::StringTable() noexcept {
    _strings.emplace_back("");
    _map.emplace(luisa::string{""}, empty_string_token);
}

StringToken StringTable::intern(luisa::string_view s) noexcept {
    if (auto iter = _map.find(s); iter != _map.end()) {
        return iter->second;
    }
    auto token = static_cast<StringToken>(_strings.size());
    _strings.emplace_back(s);
    _map.emplace(luisa::string{s}, token);
    return token;
}

luisa::optional<StringToken> StringTable::find(luisa::string_view s) const noexcept {
    if (auto iter = _map.find(s); iter != _map.end()) {
        return iter->second;
    }
    return luisa::nullopt;
}

luisa::string_view StringTable::lookup(StringToken token) const noexcept {
    if (token >= _strings.size()) {
        return {};
    }
    return _strings[token];
}

void StringTable::clear() noexcept {
    _strings.clear();
    _map.clear();
    _strings.emplace_back("");
    _map.emplace(luisa::string{""}, empty_string_token);
}

namespace detail {

namespace {

[[nodiscard]] int32_t format_arg_as_int(const HostFormatArg &arg) noexcept {
    if (auto *p = luisa::get_if<int32_t>(&arg)) { return *p; }
    if (auto *p = luisa::get_if<float>(&arg)) { return static_cast<int32_t>(*p); }
    return 0;
}

[[nodiscard]] double format_arg_as_double(const HostFormatArg &arg) noexcept {
    if (auto *p = luisa::get_if<float>(&arg)) { return static_cast<double>(*p); }
    if (auto *p = luisa::get_if<int32_t>(&arg)) { return static_cast<double>(*p); }
    return 0.0;
}

[[nodiscard]] luisa::string format_single_printf(luisa::string_view spec, double val, bool is_int, bool is_uint) noexcept {
    char fmt_buf[64];
    if (spec.size() >= sizeof(fmt_buf)) {
        return {};
    }
    std::memcpy(fmt_buf, spec.data(), spec.size());
    fmt_buf[spec.size()] = '\0';
    char out_buf[256];
    int n = 0;
    if (is_int) {
        n = std::snprintf(out_buf, sizeof(out_buf), fmt_buf, static_cast<int>(val));
    } else if (is_uint) {
        n = std::snprintf(out_buf, sizeof(out_buf), fmt_buf, static_cast<unsigned int>(static_cast<int64_t>(val)));
    } else {
        n = std::snprintf(out_buf, sizeof(out_buf), fmt_buf, val);
    }
    if (n <= 0) {
        return {};
    }
    if (static_cast<size_t>(n) < sizeof(out_buf)) {
        return luisa::string{out_buf, static_cast<size_t>(n)};
    }
    return luisa::string{out_buf, sizeof(out_buf) - 1u};
}

}// namespace

luisa::optional<luisa::string> eval_osl_format(
    luisa::string_view fmt,
    luisa::span<const HostFormatArg> args) noexcept {
    luisa::string out;
    size_t arg_idx = 0u;
    for (size_t i = 0u; i < fmt.size(); ++i) {
        if (fmt[i] != '%') {
            out.push_back(fmt[i]);
            continue;
        }
        if (i + 1u >= fmt.size()) {
            return luisa::nullopt;
        }
        if (fmt[i + 1u] == '%') {
            out.push_back('%');
            ++i;
            continue;
        }
        size_t spec_start = i;
        size_t j = i + 1u;
        while (j < fmt.size() && (fmt[j] == '-' || fmt[j] == '+' || fmt[j] == ' ' || fmt[j] == '#' || fmt[j] == '0')) {
            ++j;
        }
        while (j < fmt.size() && std::isdigit(static_cast<unsigned char>(fmt[j]))) {
            ++j;
        }
        if (j < fmt.size() && fmt[j] == '.') {
            ++j;
            while (j < fmt.size() && std::isdigit(static_cast<unsigned char>(fmt[j]))) {
                ++j;
            }
        }
        if (j >= fmt.size()) {
            return luisa::nullopt;
        }
        char conv = fmt[j];
        luisa::string_view spec = fmt.substr(spec_start, j - spec_start + 1u);
        if (arg_idx >= args.size()) {
            return luisa::nullopt;
        }
        const auto &arg = args[arg_idx++];
        switch (conv) {
            case 's': {
                if (auto *p = luisa::get_if<luisa::string_view>(&arg)) {
                    out.append(*p);
                } else {
                    return luisa::nullopt;
                }
                break;
            }
            case 'd':
            case 'i': {
                if (!luisa::holds_alternative<int32_t>(arg) && !luisa::holds_alternative<float>(arg)) {
                    return luisa::nullopt;
                }
                out.append(format_single_printf(spec, static_cast<double>(format_arg_as_int(arg)), true, false));
                break;
            }
            case 'u':
            case 'x':
            case 'X':
            case 'o': {
                if (!luisa::holds_alternative<int32_t>(arg) && !luisa::holds_alternative<float>(arg)) {
                    return luisa::nullopt;
                }
                out.append(format_single_printf(spec, static_cast<double>(format_arg_as_int(arg)), false, true));
                break;
            }
            case 'f':
            case 'F':
            case 'e':
            case 'E':
            case 'g':
            case 'G': {
                if (auto *p = luisa::get_if<luisa::float3>(&arg)) {
                    luisa::string sub_spec{spec.substr(0u, spec.size() - 1u)};
                    sub_spec.push_back(conv);
                    out.append(format_single_printf(sub_spec, p->x, false, false));
                    out.push_back(' ');
                    out.append(format_single_printf(sub_spec, p->y, false, false));
                    out.push_back(' ');
                    out.append(format_single_printf(sub_spec, p->z, false, false));
                } else if (luisa::holds_alternative<float>(arg) || luisa::holds_alternative<int32_t>(arg)) {
                    out.append(format_single_printf(spec, format_arg_as_double(arg), false, false));
                } else {
                    return luisa::nullopt;
                }
                break;
            }
            case 'c':
            case 'p':
            case 'v':
            case 'n': {
                if (auto *p = luisa::get_if<luisa::float3>(&arg)) {
                    luisa::string sub_spec{spec.substr(0u, spec.size() - 1u)};
                    sub_spec.push_back('g');
                    out.append(format_single_printf(sub_spec, p->x, false, false));
                    out.push_back(' ');
                    out.append(format_single_printf(sub_spec, p->y, false, false));
                    out.push_back(' ');
                    out.append(format_single_printf(sub_spec, p->z, false, false));
                } else {
                    return luisa::nullopt;
                }
                break;
            }
            case 'm': {
                if (auto *m = luisa::get_if<luisa::float4x4>(&arg)) {
                    luisa::string sub_spec{spec.substr(0u, spec.size() - 1u)};
                    sub_spec.push_back('g');
                    // L = M_osl^T, so OSL row r col c is (*m)[r][c]
                    for (uint32_t r = 0u; r < 4u; ++r) {
                        for (uint32_t c = 0u; c < 4u; ++c) {
                            if (r > 0u || c > 0u) { out.push_back(' '); }
                            out.append(format_single_printf(sub_spec, (*m)[r][c], false, false));
                        }
                    }
                } else {
                    return luisa::nullopt;
                }
                break;
            }
            default:
                return luisa::nullopt;
        }
        i = j;
    }
    return out;
}

luisa::string eval_osl_substr(luisa::string_view s, int32_t start, int32_t len) noexcept {
    int64_t n = static_cast<int64_t>(s.size());
    int64_t st = start;
    if (st < 0) {
        st += n;
    }
    if (st < 0) { st = 0; }
    if (st >= n) {
        return {};
    }
    int64_t end = (len < 0) ? (n + static_cast<int64_t>(len))
                            : (st + static_cast<int64_t>(len));
    end = std::min<int64_t>(end, n);
    if (end <= st) {
        return {};
    }
    return luisa::string{s.substr(static_cast<size_t>(st), static_cast<size_t>(end - st))};
}

int32_t eval_osl_getchar(luisa::string_view s, int32_t index) noexcept {
    if (index < 0 || static_cast<size_t>(index) >= s.size()) {
        return 0;
    }
    return static_cast<int32_t>(static_cast<unsigned char>(s[static_cast<size_t>(index)]));
}

int32_t eval_osl_stoi(luisa::string_view s) noexcept {
    luisa::string tmp{s};
    char *end = nullptr;
    long v = std::strtol(tmp.c_str(), &end, 10);
    if (end == tmp.c_str()) { return 0; }
    return static_cast<int32_t>(v);
}

float eval_osl_stof(luisa::string_view s) noexcept {
    luisa::string tmp{s};
    char *end = nullptr;
    float v = std::strtof(tmp.c_str(), &end);
    if (end == tmp.c_str()) { return 0.0f; }
    return v;
}

int32_t eval_osl_string_hash(luisa::string_view s) noexcept {
    uint64_t h = luisa::hash64(s.data(), s.size(), luisa::hash64_default_seed);
    return static_cast<int32_t>(static_cast<uint32_t>((h >> 32u) ^ h));
}

luisa::vector<luisa::string> eval_osl_split(
    luisa::string_view s, luisa::string_view sep, int32_t maxsplit) noexcept {
    luisa::vector<luisa::string> out;
    if (s.empty()) {
        return out;
    }
    int64_t remaining_splits = (maxsplit < 0)
                                   ? std::numeric_limits<int64_t>::max()
                                   : static_cast<int64_t>(maxsplit);
    auto is_ws = [](char c) noexcept {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
    };
    if (sep.empty()) {
        size_t pos = 0u;
        while (pos < s.size() && is_ws(s[pos])) { ++pos; }
        while (pos < s.size()) {
            if (remaining_splits == 0) {
                out.emplace_back(s.substr(pos));
                break;
            }
            size_t end = pos;
            while (end < s.size() && !is_ws(s[end])) { ++end; }
            out.emplace_back(s.substr(pos, end - pos));
            pos = end;
            while (pos < s.size() && is_ws(s[pos])) { ++pos; }
            if (pos < s.size()) {
                --remaining_splits;
            }
        }
        return out;
    }
    size_t start = 0u;
    while (remaining_splits > 0) {
        size_t found = s.find(sep, start);
        if (found == luisa::string_view::npos) { break; }
        out.emplace_back(s.substr(start, found - start));
        start = found + sep.size();
        --remaining_splits;
    }
    out.emplace_back(s.substr(start));
    return out;
}

}// namespace detail

}// namespace luisa::compute::osl
