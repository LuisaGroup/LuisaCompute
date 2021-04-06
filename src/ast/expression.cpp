//
// Created by Mike Smith on 2021/3/13.
//

#include "ast/variable.h"
#include <core/logging.h>
#include <ast/expression.h>
#include <ast/function_builder.h>

namespace luisa::compute {

void RefExpr::_mark(Variable::Usage usage) const noexcept {
    FunctionBuilder::current()->mark_variable_usage(_variable.uid(), usage);
}

void CallExpr::_mark(Variable::Usage) const noexcept {
    if (is_builtin()) {
        if (_name == "texture_write") {
            _arguments[0]->mark(Variable::Usage::WRITE);
            for (auto i = 1u; i < _arguments.size(); i++) {
                _arguments[i]->mark(Variable::Usage::READ);
            }
        } else {
            for (auto arg : _arguments) {
                arg->mark(Variable::Usage::READ);
            }
        }
    } else {
        auto f = Function::callable(_uid);
        auto args = f.arguments();
        for (auto i = 0u; i < args.size(); i++) {
            auto arg = args[i];
            _arguments[i]->mark(
                arg.tag() == Variable::Tag::BUFFER || arg.tag() == Variable::Tag::IMAGE
                    ? f.variable_usage(arg.uid())
                    : Variable::Usage::READ);
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
