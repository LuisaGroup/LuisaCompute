#pragma once
#include <luisa/vstl/common.h>
#include <luisa/ast/op.h>
#include <luisa/ast/expression.h>
#include <luisa/ast/type.h>
#include "string_builder.h"
#include <luisa/vstl/md5.h>
#include <luisa/core/stl/hash.h>
namespace lc::hlsl {
class CodegenUtility;
using namespace luisa::compute;
// XXRet T{
//  YY func(A);
// }
struct TemplateFunction {
    vstd::string_view ret_type;
    vstd::string_view body;
    vstd::string_view tmp_type_name;
    char access_place;
    char args_place;
    char temp_type_place;
};
class AccessChain {
public:
    struct AccessNode {
        bool is_struct;
    };
    struct MemberNode {
        uint member_index;
    };
    using Node = vstd::variant<AccessNode, MemberNode>;

private:
    CallOp _op;
    Variable _root_var;
    vstd::vector<Node> _nodes;
    size_t _hash;
    vstd::string _func_name;
    size_t _get_hash() const;
    static vstd::vector<Node> nodes_from_exprs(luisa::span<Expression const *const> exprs);

public:
    AccessChain(
        CallOp op,
        Variable const &root_var,
        luisa::span<Expression const *const> access_expr);
    AccessChain(AccessChain const &) = delete;
    AccessChain(AccessChain &&) = default;
    auto hash() const { return _hash; }
    vstd::string_view func_name() const { return _func_name; }
    void init_name();
    bool operator==(AccessChain const &node) const;
    bool operator!=(AccessChain const &node) const { return !operator==(node); }

    void gen_func_impl(CodegenUtility *util, TemplateFunction const &tmp, luisa::span<Expression const *const> args, vstd::StringBuilder &builder);
    void call_this_func(luisa::span<Expression const *const> args, vstd::StringBuilder &builder, ExprVisitor &visitor) const;
};
struct AccessHash {
    size_t operator()(AccessChain const &c) const {
        return c.hash();
    }
};
}// namespace lc::hlsl
