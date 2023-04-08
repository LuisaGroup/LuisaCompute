#pragma once
#include <vstl/common.h>
#include <ast/op.h>
#include <ast/expression.h>
#include <ast/type.h>
#include <HLSL/string_builder.h>
#include <vstl/md5.h>
#include <core/stl/hash.h>
namespace lc::dx {
class CodegenUtility;
using namespace luisa::compute;
// XXRet T{
//  YY func(A);
// }
struct TemplateFunction {
    vstd::string_view ret_type;
    vstd::string_view body;
    char access_place;
    char args_place;
};
class AccessChain {
public:
    struct AccessNode {};
    struct MemberNode {
        uint member_index;
    };
    using Node = vstd::variant<AccessNode, MemberNode>;

private:
    CallOp _op;
    Type const *_root_type;
    vstd::vector<Node> _nodes;
    size_t _hash;
    vstd::string _func_name;
    size_t _get_hash() const;
    static vstd::vector<Node> nodes_from_exprs(luisa::span<Expression const *const> exprs);

public:
    AccessChain(
        CallOp op,
        Type const *root_type,
        luisa::span<Expression const *const> access_expr);
    AccessChain(AccessChain const &) = delete;
    AccessChain(AccessChain &&) = default;
    auto hash() const { return _hash; }
    vstd::string_view func_name() const { return _func_name; }
    void init_name();
    bool operator==(AccessChain const &node) const;
    bool operator!=(AccessChain const &node) const { return !operator==(node); }

    void gen_func_impl(CodegenUtility *util, TemplateFunction const &tmp, luisa::span<Expression const *const> args, vstd::StringBuilder &builder);
};
struct AccessHash {
    size_t operator()(AccessChain const &c) const {
        return c.hash();
    }
};
}// namespace lc::dx