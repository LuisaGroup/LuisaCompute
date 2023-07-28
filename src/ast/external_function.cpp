#include <luisa/core/stl/hash.h>
#include <luisa/core/logging.h>
#include <luisa/ast/external_function.h>

namespace luisa::compute {

ExternalFunction::ExternalFunction(luisa::string name,
                                   const Type *return_type,
                                   luisa::vector<const Type *> argument_types,
                                   luisa::vector<Usage> argument_usages) noexcept
    : _name{std::move(name)}, 
      _return_type{return_type},
      _hash{},
      _argument_types{std::move(argument_types)},
      _argument_usages{std::move(argument_usages)} { _compute_hash(); }

inline void ExternalFunction::_compute_hash() noexcept {
    using namespace std::string_view_literals;
    static thread_local const auto seed = luisa::hash_value("__hash_external_function"sv);
    luisa::string desc;
    desc.reserve(64u);
    desc.append(_return_type ? _return_type->description() : "void")
        .append(" "sv)
        .append(_name)
        .append("(");
    for (auto i = 0u; i < _argument_types.size(); i++) {
        if (i != 0u) { desc.append(", "sv); }
        switch (_argument_usages[i]) {
            case Usage::NONE: desc.append("unused "sv); break;
            case Usage::READ: desc.append("in "sv); break;
            case Usage::WRITE: desc.append("out "sv); break;
            case Usage::READ_WRITE: desc.append("inout "sv); break;
        }
        desc.append(_argument_types[i]->description());
    }
    desc.append(")"sv);
    _hash = luisa::hash_value(desc, seed);
    LUISA_VERBOSE_WITH_LOCATION(
        "Computed hash for external function '{}': {}",
        desc, luisa::hash_to_string(_hash));
}

}// namespace luisa::compute

