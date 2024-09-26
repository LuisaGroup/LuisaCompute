#include "access_chain.h"
#include "hlsl_codegen.h"
namespace lc::hlsl {
AccessChain::AccessChain(
    CallOp op,
    Variable const &root_var,
    luisa::span<Expression const *const> exprs) : _op{op}, _root_var{root_var}, _nodes{nodes_from_exprs(exprs)} {
    _hash = _get_hash();
}
void AccessChain::init_name() {
    vstd::vector<uint8_t> bin_vecs;
    auto desc = _root_var.type()->description();
    size_t basic_size = desc.size() + sizeof(CallOp);
    if (_root_var.is_shared()) {
        uint uid = _root_var.uid();
        bin_vecs.push_back_uninitialized(basic_size + sizeof(uint));
        *reinterpret_cast<uint *>(bin_vecs.data() + basic_size) = uid;
    } else {
        bin_vecs.push_back_uninitialized(basic_size);
    }
    auto ptr = bin_vecs.data();
    *reinterpret_cast<CallOp *>(ptr) = _op;
    ptr += sizeof(CallOp);
    std::memcpy(ptr, desc.data(), desc.size());

    for (auto &&i : _nodes) {
        i.multi_visit(
            [&](AccessNode const &) {
                bin_vecs.emplace_back(static_cast<uint8_t>('a'));
            },
            [&](MemberNode const &m) {
                auto last_size = bin_vecs.size();
                bin_vecs.push_back_uninitialized(sizeof(size_t));
                std::memcpy(bin_vecs.data() + last_size, &m.member_index, sizeof(size_t));
            });
    }
    vstd::MD5 md5{bin_vecs};
    _func_name << 'F' << md5.to_string(false);
}
size_t AccessChain::_get_hash() const {
    auto hash_value = luisa::hash<CallOp>{}(_op);
    if (_root_var.is_shared()) {
        hash_value = luisa::hash<size_t>{}(_root_var.uid());
    } else {
        hash_value = luisa::hash<size_t>{}(reinterpret_cast<size_t>(_root_var.type()));
    }
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
    if (_op != node._op || node._nodes.size() != _nodes.size())
        return false;
    if (_root_var.is_shared()) {
        if (node._root_var != _root_var) return false;
    } else {
        if (node._root_var.type() != _root_var.type()) return false;
    }
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
                nodes.emplace_back(AccessNode{false});
                break;
            case Type::Tag::MATRIX:
                type = Type::vector(type->element(), type->dimension());
                nodes.emplace_back(AccessNode{false});
                break;
            case Type::Tag::ARRAY:
                type = type->element();
                nodes.emplace_back(AccessNode{true});
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
            default:
                LUISA_ERROR_WITH_LOCATION("Invalid access chain node type: {}",
                                          type->description());
        }
    }
    return nodes;
}
void AccessChain::gen_func_impl(CodegenUtility *util, TemplateFunction const &tmp, luisa::span<Expression const *const> args, vstd::StringBuilder &builder) {
    size_t arg_start;
    vstd::StringBuilder chain_str;
    size_t arg_idx = 1;
    auto build_access = [&]() {
        for (auto &&i : _nodes) {
            i.multi_visit(
                [&](AccessNode const &n) {
                    if (!_root_var.is_shared() && n.is_struct) {
                        chain_str << ".v";
                    }
                    chain_str << "[a"sv;
                    vstd::to_string(arg_idx, chain_str);
                    chain_str << ']';
                    ++arg_idx;
                },
                [&](MemberNode const &m) {
                    chain_str << ".v"sv;
                    vstd::to_string(m.member_index, chain_str);
                });
        }
    };
    auto build_other_arguments = [&]() {
        arg_start = arg_idx;
        for (auto &&i : args) {
            builder << ',';
            util->GetTypeName(*i->type(), builder, Usage::READ, true);
            builder << " a"sv;
            vstd::to_string(arg_idx, builder);
            ++arg_idx;
        }
    };
    if (_root_var.is_shared()) {
        util->GetVariableName(_root_var, chain_str);
        build_access();
        arg_idx = 1;
        builder << tmp.ret_type << ' ' << _func_name << '(';
        if (_nodes[0].is_type_of<AccessNode>()) {
            builder << "uint a"sv;
            vstd::to_string(arg_idx, builder);
            ++arg_idx;
        }
        for (auto &&i : vstd::ptr_range(_nodes.data() + 1, _nodes.data() + _nodes.size())) {
            if (i.is_type_of<AccessNode>()) {
                builder << ",uint a"sv;
                vstd::to_string(arg_idx, builder);
                ++arg_idx;
            }
        }
        build_other_arguments();
    } else {
        chain_str << "a0"sv;
        build_access();
        arg_idx = 1;
        builder << tmp.ret_type << ' ' << _func_name << '(';
        util->GetTypeName(*_root_var.type(), builder, Usage::READ_WRITE, true);
        builder << " a0"sv;
        for (auto &&i : _nodes) {
            if (i.is_type_of<AccessNode>()) {
                builder << ",uint a"sv;
                vstd::to_string(arg_idx, builder);
                ++arg_idx;
            }
        }
        build_other_arguments();
    }
    builder << "){\n"sv;
    std::bitset<std::numeric_limits<char>::max()> bitsets;
    bitsets[tmp.access_place] = true;
    bitsets[tmp.args_place] = true;
    bitsets[tmp.temp_type_place] = true;
    for (auto &&i : tmp.body) {
        if (bitsets[i]) {
            if (i == tmp.access_place) {
                builder << chain_str;
            } else if (i == tmp.args_place) {
                builder << 'a';
                vstd::to_string(arg_start, builder);
                for (auto j : vstd::range(arg_start + 1, arg_idx)) {
                    builder << ",a"sv;
                    vstd::to_string(j, builder);
                }
            } else {
                builder << tmp.tmp_type_name;
            }
        } else {
            builder << i;
        }
    }
    builder << "}\n"sv;
}
void AccessChain::call_this_func(luisa::span<Expression const *const> args, vstd::StringBuilder &builder, ExprVisitor &visitor) const {
    builder << _func_name << '(';
    LUISA_ASSUME(!args.empty() && !_nodes.empty() && args.size() > _nodes.size());
    if (!_root_var.is_shared()) {
        args[0]->accept(visitor);
        builder << ',';
    }
    for (auto i : vstd::range(0, _nodes.size())) {
        auto &node = _nodes[i];
        if (node.is_type_of<AccessNode>()) {
            args[i + 1]->accept(visitor);
            builder << ',';
        }
    }
    for (auto i : vstd::range(_nodes.size() + 1, args.size())) {
        args[i]->accept(visitor);
        builder << ',';
    }
    builder[builder.size() - 1] = ')';
}
}// namespace lc::hlsl
