//
// Created by Mike Smith on 2021/3/13.
//

#include <core/logging.h>
#include <ast/expression.h>
#include <ast/function_builder.h>

namespace luisa::compute {

void RefExpr::_mark(Usage usage) const noexcept {
    FunctionBuilder::current()->mark_variable_usage(_variable.uid(), usage);
}

void CallExpr::_mark(Usage) const noexcept {
    for (auto arg : _arguments) { arg->mark(Usage::READ); }
    if (is_builtin()) {
        // TODO: builtins
        if (_name == "texture_write") {// texture_write(tex, ...)
            _arguments.front()->mark(Usage::WRITE);
        }
    } else {
        auto f = Function::callable(_uid);
        auto args = f.arguments();
        for (auto i = 0u; i < args.size(); i++) {
            if (auto arg = args[i];
                (static_cast<uint32_t>(f.variable_usage(arg.uid())) & static_cast<uint32_t>(Usage::WRITE)) != 0u
                && (arg.tag() == Variable::Tag::BUFFER
                    || arg.tag() == Variable::Tag::TEXTURE)) {
                _arguments[i]->mark(Usage::WRITE);
            };
        }
    }
}

CallExpr::CallExpr(const Type *type, std::string_view name, CallExpr::ArgumentList args) noexcept
    : Expression{type}, _name{name}, _arguments{args} {
    using namespace std::string_view_literals;
    if (auto prefix = "custom_"sv; _name.starts_with(prefix)) {
        auto uid_str = _name.substr(prefix.size());
        if (auto [p, ec] = std::from_chars(uid_str.data(), uid_str.data() + uid_str.size(), _uid);
            ec != std::errc{}) {
            LUISA_ERROR_WITH_LOCATION("Invalid custom callable function: {}.", _name);
        }
    }
}

}// namespace luisa::compute
