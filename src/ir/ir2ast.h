#pragma once
#include <ast/function_builder.h>
#include <luisa_compute_ir/bindings.hpp>

namespace luisa::compute {

namespace detail {
    class FunctionBuilder;
}

class LC_AST_API IR2AST {
public:
    template<typename T>
    struct BoxedSliceHash {
        using is_avalaunching = void;
        [[nodiscard]] auto operator()(ir::CBoxedSlice<T> slice) const noexcept {
            return luisa::hash64(slice.ptr, slice.len * sizeof(T), hash64_default_seed);
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

    struct IR2ASTContext {
        ir::Module module;
        luisa::shared_ptr<detail::FunctionBuilder> function_builder;
        luisa::unordered_map<const ir::Node *, const Expression *> node_to_exprs;
        luisa::unordered_map<const ir::BasicBlock *, luisa::vector<PhiAssignment>> block_to_phis;
        bool zero_init;
    };

private:
    template<typename Func>
    static void _iterate(const ir::BasicBlock *bb, const Func& f) noexcept {
        auto node_ref = bb->first;
        while (node_ref != ir::INVALID_REF) {
            auto node = ir::luisa_compute_ir_node_get(node_ref);
            f(node);
            node_ref = node->next;
        }
    }

    IR2ASTContext *_ctx;
    
    [[nodiscard]] const Type *_convert_primitive_type(const ir::Primitive &type) noexcept;
    [[nodiscard]] CallOp _decide_make_vector_op(const Type *primitive, size_t length) noexcept;
    [[nodiscard]] CallOp _decide_make_matrix_op(size_t dimension) noexcept;

    [[nodiscard]] const Expression *_convert_constant(const ir::Const &const_) noexcept;
    [[nodiscard]] const Expression *_convert_node(const ir::NodeRef node_ref) noexcept;
    [[nodiscard]] const Expression *_convert_node(const ir::Node *node) noexcept;
    [[nodiscard]] const Type *_convert_type(const ir::Type *type) noexcept;
    [[nodiscard]] const RefExpr * _convert_argument(const ir::Node *node) noexcept;
    [[nodiscard]] const RefExpr *_convert_captured (const ir::Capture &captured) noexcept;
    
    void _convert_block(const ir::BasicBlock *block) noexcept;
    void _convert_instr_local(const ir::Node *node) noexcept;
    void _convert_instr_user_data(const ir::Node *node) noexcept;
    void _convert_instr_invalid(const ir::Node *node) noexcept;
    void _convert_instr_const(const ir::Node *node) noexcept;
    void _convert_instr_update(const ir::Node *node) noexcept;
    const Expression *_convert_instr_call(const ir::Node *node) noexcept;
    void _convert_instr_phi(const ir::Node *node) noexcept;
    void _convert_instr_return(const ir::Node *node) noexcept;
    void _convert_instr_loop(const ir::Node *node) noexcept;
    void _convert_instr_generic_loop(const ir::Node *node) noexcept;
    void _convert_instr_break(const ir::Node *node) noexcept;
    void _convert_instr_continue(const ir::Node *node) noexcept;
    void _convert_instr_if(const ir::Node *node) noexcept;
    void _convert_instr_switch(const ir::Node *node) noexcept;
    void _convert_instr_ad_scope(const ir::Node *node) noexcept;
    void _convert_instr_ad_detach(const ir::Node *node) noexcept;
    void _convert_instr_comment(const ir::Node *node) noexcept;
    void _convert_instr_debug(const ir::Node *node) noexcept;
    void _collect_phis(const ir::BasicBlock *bb) noexcept;
    
    [[nodiscard]] luisa::shared_ptr<detail::FunctionBuilder> convert_kernel(const ir::KernelModule *kernel) noexcept;
    [[nodiscard]] luisa::shared_ptr<detail::FunctionBuilder> convert_callable(const ir::CallableModule *callable) noexcept;
public:
    [[nodiscard]] static luisa::shared_ptr<detail::FunctionBuilder> build(const ir::KernelModule *kernel) noexcept;
};

}