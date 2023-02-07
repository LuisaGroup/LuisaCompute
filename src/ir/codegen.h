//
// Created by Mike Smith on 2023/1/13.
//

#pragma once

#include <core/stl.h>
#include <luisa_compute_ir/bindings.hpp>

namespace luisa::compute {

class CppSourceBuilder {

public:
    template<typename T>
    struct BoxedSliceHash {
        using is_avalaunching = void;
        [[nodiscard]] auto operator()(ir::CBoxedSlice<T> slice) const noexcept {
            return luisa::hash64(slice.ptr, slice.len * sizeof(T), Hash64::default_seed);
        }
    };

    template<typename T>
    struct BoxedSliceEqual {
        [[nodiscard]] auto operator()(ir::CBoxedSlice<T> lhs, ir::CBoxedSlice<T> rhs) const noexcept {
            return lhs.len == rhs.len &&
                   (lhs.ptr == rhs.ptr ||
                    std::memcmp(lhs.ptr, rhs.ptr, lhs.len * sizeof(T)) == 0);
        }
    };

    struct PhiAssignment {
        const ir::Node *dst;
        const ir::Node *src;
    };

    struct Context {
        ir::Module module;
        luisa::string signature;
        luisa::string locals;
        luisa::string body;
        luisa::unordered_map<const ir::Node *, luisa::string> node_to_var;
        luisa::unordered_map<const ir::BasicBlock *, luisa::vector<PhiAssignment>> block_to_phis;
        luisa::unordered_set<const ir::Node *> grads;
        bool in_generic_loop{false};
    };

private:
    Context *_ctx;
    luisa::unordered_map<const ir::Type *, luisa::string> _type_names;
    luisa::unordered_map<ir::CBoxedSlice<uint8_t>, luisa::string,
                         BoxedSliceHash<uint8_t>, BoxedSliceEqual<uint8_t>>
        _constant_names;
    // CallableModule's are value types, so we use the entry basic block as the key.
    luisa::unordered_map<const ir::BasicBlock *, luisa::string> _callable_names;
    luisa::string _types;
    luisa::string _symbols;

private:
    template<typename Func>
    static void _iterate(const ir::BasicBlock *bb, const Func &f) noexcept {
        auto node_ref = bb->first;
        while (node_ref != ir::INVALID_REF) {
            auto node = ir::luisa_compute_ir_node_get(node_ref);
            f(node);
            node_ref = node->next;
        }
    }

    void _collect_phis(const ir::BasicBlock *bb) noexcept;
    [[nodiscard]] luisa::string /* name */ _generate_type(const ir::Type *type) noexcept;
    [[nodiscard]] luisa::string /* name */ _generate_constant(const ir::Const &c) noexcept;
    [[nodiscard]] luisa::string /* name */ _generate_callable(const ir::CallableModule &callable) noexcept;
    [[nodiscard]] luisa::string /* name */ _generate_node(const ir::Node *node) noexcept;
    [[nodiscard]] luisa::string /* name */ _generate_node(ir::NodeRef node) noexcept;
    void _generate_argument(const ir::Node *node, bool is_last) noexcept;
    void _generate_kernel(const ir::KernelModule &kernel) noexcept;
    void _generate_indent(uint depth) noexcept;
    // detail instructions
    void _generate_block(const ir::BasicBlock *bb, uint indent = 0u) noexcept;
    void _generate_instr_local(const ir::Node *node, uint indent) noexcept;
    void _generate_instr_user_data(const ir::Node *node, uint indent) noexcept;
    void _generate_instr_invalid(const ir::Node *node, uint indent) noexcept;
    void _generate_instr_const(const ir::Node *node, uint indent) noexcept;
    void _generate_instr_update(const ir::Node *node, uint indent) noexcept;
    void _generate_instr_call(const ir::Node *node, uint indent) noexcept;
    void _generate_instr_phi(const ir::Node *node, uint indent) noexcept;
    void _generate_instr_return(const ir::Node *node, uint indent) noexcept;
    void _generate_instr_loop(const ir::Node *node, uint indent) noexcept;
    void _generate_instr_generic_loop(const ir::Node *node, uint indent) noexcept;
    void _generate_instr_break(const ir::Node *node, uint indent) noexcept;
    void _generate_instr_continue(const ir::Node *node, uint indent) noexcept;
    void _generate_instr_if(const ir::Node *node, uint indent) noexcept;
    void _generate_instr_switch(const ir::Node *node, uint indent) noexcept;
    void _generate_instr_ad_scope(const ir::Node *node, uint indent) noexcept;
    void _generate_instr_ad_detach(const ir::Node *node, uint indent) noexcept;
    void _generate_instr_comment(const ir::Node *node, uint indent) noexcept;
    void _generate_instr_debug(const ir::Node *node, uint indent) noexcept;


private:
    CppSourceBuilder() noexcept = default;

public:
    [[nodiscard]] static luisa::string build(const ir::KernelModule &m) noexcept;
};

}// namespace luisa::compute
