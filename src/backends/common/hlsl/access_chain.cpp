#include "access_chain.h"
#include "dx_codegen.h"
namespace lc::dx {
AccessChain::AccessChain(
    CallOp op,
    Type const *root_type,
    luisa::span<Expression const *const> exprs) : _op{op}, _nodes{nodes_from_exprs(exprs)}, _root_type{root_type} {
    _hash = _get_hash();
}
void AccessChain::init_name() {
    vstd::vector<uint8_t> bin_vecs;
    auto desc = _root_type->description();
    bin_vecs.push_back_uninitialized(desc.size() + sizeof(CallOp));
    *reinterpret_cast<CallOp*>(bin_vecs.data()) = _op;
    memcpy(bin_vecs.data() + sizeof(CallOp), desc.data(), desc.size());
    for (auto &&i : _nodes) {
        i.multi_visit(
            [&](AccessNode const &) {
                bin_vecs.emplace_back(static_cast<uint8_t>('a'));
            },
            [&](MemberNode const &m) {
                auto last_size = bin_vecs.size();
                bin_vecs.push_back_uninitialized(sizeof(size_t));
                memcpy(bin_vecs.data() + last_size, &m.member_index, sizeof(size_t));
            });
    }
    vstd::MD5 md5{bin_vecs};
    _func_name << 'F' << md5.to_string(false);
}
size_t AccessChain::_get_hash() const {
    auto hash_value = luisa::hash<CallOp>{}(_op);
    hash_value = luisa::hash<size_t>{}(reinterpret_cast<size_t>(_root_type));
    for (auto &&i : _nodes) {
        hash_value = luisa::hash<size_t>{}(i.index(), hash_value);
        i.visit([&]<typename T>(T const &t) {
            if constexpr (std::is_same_v<T, MemberNode>) {
                hash_value = luisa::hash<uint>{}(t.member_index, hash_value);
            }
        });
    }
    return hash_value;
}
bool AccessChain::operator==(AccessChain const &node) const {
    if (_op != node._op || node._nodes.size() != _nodes.size() || node._root_type != _root_type)
        return false;
    for (auto idx : vstd::range(_nodes.size())) {
        auto &l = _nodes[idx];
        auto &r = node._nodes[idx];
        if (!l.visit_or(true, [&]<typename T>(T const &t) {
                if constexpr (std::is_same_v<T, MemberNode>) {
                    return t.member_index == reinterpret_cast<T const *>(r.place_holder())->member_index;
                } else {
                    return true;
                }
            })) {
            return false;
        }
    }
    return true;
}
vstd::vector<AccessChain::Node> AccessChain::nodes_from_exprs(luisa::span<Expression const *const> args) {
    vstd::vector<Node> nodes;
    auto type = args.front()->type();
    nodes.reserve(args.size());
    for (auto index : luisa::span{args}.subspan(1)) {
        switch (type->tag()) {
            case Type::Tag::BUFFER:
            case Type::Tag::VECTOR:
                type = type->element();
                nodes.emplace_back(AccessNode{});
                break;
            case Type::Tag::MATRIX:
                type = Type::vector(type->element(), type->dimension());
                nodes.emplace_back(AccessNode{});
                break;
            case Type::Tag::ARRAY:
                type = type->element();
                nodes.emplace_back(AccessNode{});
                break;
            case Type::Tag::STRUCTURE: {
                auto literal = static_cast<const LiteralExpr *>(index)->value();
                auto member_index =
                    luisa::holds_alternative<int>(literal) ?
                        static_cast<uint>(luisa::get<int>(literal)) :
                        luisa::get<uint>(literal);
                nodes.emplace_back(MemberNode{.member_index = member_index});
                type = type->members()[member_index];
            } break;
        }
    }
    return nodes;
}
void AccessChain::gen_func_impl(CodegenUtility *util, TemplateFunction const &tmp, luisa::span<Expression const *const> args, vstd::StringBuilder &builder) {
    vstd::StringBuilder buffer_name;
    util->GetTypeName(*_root_type, buffer_name, Usage::READ_WRITE, true);
    vstd::StringBuilder chain_str;
    chain_str << "a0"sv;
    size_t arg_idx = 1;
    for (auto &&i : _nodes) {
        i.multi_visit(
            [&](AccessNode const &) {
                chain_str << "[a"sv;
                vstd::to_string(arg_idx, chain_str);
                chain_str << ']';
                ++arg_idx;
            },
            [&](MemberNode const &m) {
                chain_str << ".v"sv;
                vstd::to_string(arg_idx, chain_str);
            });
    }
    arg_idx = 1;
    builder << tmp.ret_type << ' ' << _func_name << '(' << buffer_name << " a0"sv;
    for (auto &&i : _nodes) {
        if (i.is_type_of<AccessNode>()) {
            builder << ",uint a"sv;
            vstd::to_string(arg_idx, builder);
            ++arg_idx;
        }
    }
    size_t arg_start = arg_idx;
    for (auto &&i : args) {
        builder << ',';
        util->GetTypeName(*i->type(), builder, Usage::READ, true);
        builder << " a"sv;
        vstd::to_string(arg_idx, builder);
        ++arg_idx;
    }
    builder << "){\n"sv;
    for (auto &&i : tmp.body) {
        if (i == tmp.access_place) {
            builder << chain_str;
        } else if (i == tmp.args_place) {
            builder << 'a';
            vstd::to_string(arg_start, builder);
            for (auto j : vstd::range(arg_start + 1, arg_idx)) {
                builder << ",a"sv;
                vstd::to_string(j, builder);
            }
        } else if (i == tmp.temp_type_place) {
            builder << tmp.tmp_type_name;
        } else {
            builder << i;
        }
    }
    builder << "}\n"sv;
}
void AccessChain::call_this_func(luisa::span<Expression const *const> args, vstd::StringBuilder &builder, ExprVisitor &visitor) const {
    builder << _func_name << '(';
    assert(!args.empty() && !_nodes.empty() && args.size() > _nodes.size());
    args[0]->accept(visitor);
    for (auto i : vstd::range(1, _nodes.size())) {
        auto &node = _nodes[i];
        if (node.is_type_of<AccessNode>()) {
            builder << ',';
            args[i]->accept(visitor);
        }
    }
    for (auto i : vstd::range(_nodes.size(), args.size())) {
        builder << ',';
        args[i]->accept(visitor);
    }
    builder << ')';
}
}// namespace lc::dx