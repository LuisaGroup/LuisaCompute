#include "ut/ut.hpp"
#include <luisa/ast/function.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_reg2mem.h>
#include <luisa/xir/translators/coro_xir2ast.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

// Disambiguate AST Function from XIR Function
using ASTFunction = compute::Function;

namespace {

static constexpr uint32_t kReservedFrameFieldCount = 4u;
static constexpr uint32_t kTargetTokenField = 3u;

struct StatementCounter final : StmtVisitor {
    uint stores = 0u;
    uint returns = 0u;
    uint ifs = 0u;
    uint exprs = 0u;

    void visit(const BreakStmt *) override {}
    void visit(const ContinueStmt *) override {}
    void visit(const ReturnStmt *) override { returns++; }
    void visit(const ScopeStmt *stmt) override {
        for (auto s : stmt->statements()) { s->accept(*this); }
    }
    void visit(const IfStmt *stmt) override {
        ifs++;
        stmt->true_branch()->accept(*this);
        stmt->false_branch()->accept(*this);
    }
    void visit(const LoopStmt *stmt) override { stmt->body()->accept(*this); }
    void visit(const ExprStmt *) override { exprs++; }
    void visit(const SwitchStmt *stmt) override { stmt->body()->accept(*this); }
    void visit(const SwitchCaseStmt *stmt) override { stmt->body()->accept(*this); }
    void visit(const SwitchDefaultStmt *stmt) override { stmt->body()->accept(*this); }
    void visit(const AssignStmt *) override { stores++; }
    void visit(const ForStmt *stmt) override { stmt->body()->accept(*this); }
    void visit(const CommentStmt *) override {}
    void visit(const RayQueryStmt *stmt) override {
        stmt->on_triangle_candidate()->accept(*this);
        stmt->on_procedural_candidate()->accept(*this);
    }
    void visit(const SuspendStmt *) override {}
    void visit(const AutoDiffStmt *stmt) override { stmt->body()->accept(*this); }
    void visit(const PrintStmt *) override {}
    void visit(const DebugBreakStmt *) override {}
};

[[nodiscard]] CallableFunction *make_continuation(Module &m, Value *&frame_arg_out, BasicBlock *&body_out) noexcept {
    auto *cf = m.create_callable(nullptr);
    auto frame_type = Type::structure({
        Type::of<uint32_t>(),
        Type::of<uint32_t>(),
        Type::of<uint32_t>(),
        Type::of<uint32_t>(),
        Type::of<float>(),
    });
    frame_arg_out = cf->create_reference_argument(frame_type);
    body_out = cf->create_body_block();
    return cf;
}

}// namespace

void reg_coro_xir2ast() {

    "simple_continuation_translates_to_callable_ast"_test = [] {
        // given: a continuation with frame load/store patterns and no phi nodes
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_continuation(m, frame_arg, body);

        XIRBuilder b;
        b.set_insertion_point(body);

        uint32_t token_field = kTargetTokenField;
        auto *idx_token = m.create_constant(Type::of<uint32_t>(), &token_field);
        auto *gep_token = b.gep(Type::of<uint32_t>(), frame_arg, {idx_token});
        auto *loaded_token = b.load(Type::of<uint32_t>(), gep_token);

        uint32_t value_field = kReservedFrameFieldCount;
        auto *idx_value = m.create_constant(Type::of<uint32_t>(), &value_field);
        auto *gep_value = b.gep(Type::of<float>(), frame_arg, {idx_value});
        auto *float_val = m.create_constant_one(Type::of<float>());
        b.store(gep_value, float_val);

        b.return_void();

        // when
        auto ast = xir_to_ast_translate_continuation(*cf);

        // then
        expect(ast != nullptr);
        expect(ast->function().tag() == ASTFunction::Tag::CALLABLE);
        expect(ast->arguments().size() == 1u);
        expect(ast->arguments().front().is_reference());

        StatementCounter counter;
        ast->body()->accept(counter);
        expect(counter.stores >= 2u);
        expect(counter.returns == 1u);
    };

    "continuation_with_control_flow_translates_correctly"_test = [] {
        // given: a continuation with a structured if-inst control flow
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_continuation(m, frame_arg, body);

        XIRBuilder b;
        b.set_insertion_point(body);

        uint32_t token_field = kTargetTokenField;
        auto *idx_token = m.create_constant(Type::of<uint32_t>(), &token_field);
        auto *gep_token = b.gep(Type::of<uint32_t>(), frame_arg, {idx_token});
        auto *token_val = b.load(Type::of<uint32_t>(), gep_token);

        auto *zero_c = m.create_constant_zero(Type::of<uint32_t>());
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_NOT_EQUAL, {token_val, zero_c});

        auto *if_inst = b.if_(cond);
        auto *merge_bb = if_inst->create_merge_block();

        b.set_insertion_point(if_inst->create_true_block());
        uint32_t value_field = kReservedFrameFieldCount;
        auto *idx_value = m.create_constant(Type::of<uint32_t>(), &value_field);
        auto *gep1 = b.gep(Type::of<float>(), frame_arg, {idx_value});
        auto *float_val = m.create_constant_one(Type::of<float>());
        b.store(gep1, float_val);
        b.br(merge_bb);

        b.set_insertion_point(if_inst->create_false_block());
        b.br(merge_bb);

        b.set_insertion_point(merge_bb);
        b.return_void();

        // when
        auto ast = xir_to_ast_translate_continuation(*cf);

        // then
        expect(ast != nullptr);
        expect(ast->function().tag() == ASTFunction::Tag::CALLABLE);

        StatementCounter counter;
        ast->body()->accept(counter);
        expect(counter.stores >= 2u);
        expect(counter.returns == 1u);
        expect(counter.ifs == 1u);
    };

    "continuation_translation_preserves_frame_argument_type"_test = [] {
        // given: a continuation with a multi-field frame struct
        Module m;
        auto frame_type = Type::structure({
            Type::of<uint32_t>(),
            Type::of<uint32_t>(),
            Type::of<uint32_t>(),
            Type::of<uint32_t>(),
            Type::of<float>(),
            Type::of<int>(),
        });
        auto *cf = m.create_callable(nullptr);
        auto *frame_arg = cf->create_reference_argument(frame_type);
        auto *body = cf->create_body_block();

        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();

        // when
        auto ast = xir_to_ast_translate_continuation(*cf);

        // then
        expect(ast != nullptr);
        expect(ast->function().tag() == ASTFunction::Tag::CALLABLE);
        expect(ast->arguments().size() == 1u);
        expect(ast->arguments().front().is_reference());
    };

    "non_continuation_callable_does_not_crash_translation"_test = [] {
        // given: a regular callable without frame arg (not a continuation)
        Module m;
        auto *cf = m.create_callable(nullptr);
        cf->create_value_argument(Type::of<float>());
        auto *body = cf->create_body_block();

        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();

        // Using regular xir_to_ast_translate (not continuation-specific)
        auto ast = xir_to_ast_translate(*cf, {});
        expect(ast != nullptr);
        expect(ast->function().tag() == ASTFunction::Tag::CALLABLE);
        expect(ast->arguments().size() == 1u);
        expect(!ast->arguments().front().is_reference());
    };

    "kernel_is_not_a_continuation_callable"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        k->create_body_block();

        expect(k->derived_function_tag() == DerivedFunctionTag::KERNEL);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_coro_xir2ast();
    return 0;
}
