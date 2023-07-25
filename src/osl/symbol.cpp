//
// Created by Mike Smith on 2023/7/25.
//

#include <luisa/core/logging.h>

#include <luisa/osl/type.h>
#include <luisa/osl/symbol.h>

namespace luisa::compute::osl {

luisa::string Symbol::dump() const noexcept {
    auto tag_string = [tag = _tag] {
        using namespace std::string_view_literals;
        switch (tag) {
            case Tag::PARAM: return "param"sv;
            case Tag::OUTPUT_PARAM: return "oparam"sv;
            case Tag::LOCAL: return "local"sv;
            case Tag::TEMP: return "temp"sv;
            case Tag::GLOBAL: return "global"sv;
            case Tag::CONST: return "const"sv;
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid symbol tag.");
    }();
    auto s = luisa::format("{}\t{}", tag_string, _type->dump());
    if (is_array()) {
        if (is_unbounded()) {
            s.append("[]");
        } else {
            s.append(luisa::format("[{}]", _array_length));
        }
    }
    s.append("\t").append(_identifier);
    if (!_initial_values.empty()) {
        s.append("\t");
        for (auto &&v : _initial_values) {
            s.append(v.dump()).append(" ");
        }
        s.pop_back();
    }
    if (!_hints.empty()) {
        s.append("\t");
        for (auto &&h : _hints) {
            s.append(h.dump()).append(" ");
        }
        s.pop_back();
    }
    return s;
}

}// namespace luisa::compute::osl