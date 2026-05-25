#include "ut/ut.hpp"
#include <luisa/luisa-compute.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/xir2ast.h>
#include <luisa/xir/translators/xir2text.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

struct StatementCounter final : StmtVisitor {
    uint stores = 0u;
    uint returns = 0u;
    uint ifs = 0u;
    uint loops = 0u;
    uint switches = 0u;
    uint exprs = 0u;
    uint prints = 0u;
    uint suspends = 0u;
    uint coro_binds = 0u;

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
    void visit(const LoopStmt *stmt) override {
        loops++;
        stmt->body()->accept(*this);
    }
    void visit(const ExprStmt *) override { exprs++; }
    void visit(const SwitchStmt *stmt) override {
        switches++;
        stmt->body()->accept(*this);
    }
    void visit(const SwitchCaseStmt *stmt) override { stmt->body()->accept(*this); }
    void visit(const SwitchDefaultStmt *stmt) override { stmt->body()->accept(*this); }
    void visit(const AssignStmt *) override { stores++; }
    void visit(const ForStmt *stmt) override { stmt->body()->accept(*this); }
    void visit(const CommentStmt *) override {}
    void visit(const RayQueryStmt *stmt) override {
        stmt->on_triangle_candidate()->accept(*this);
        stmt->on_procedural_candidate()->accept(*this);
    }
    void visit(const AutoDiffStmt *stmt) override { stmt->body()->accept(*this); }
    void visit(const PrintStmt *) override { prints++; }
    void visit(const DebugBreakStmt *) override {}
    void visit(const SuspendStmt *) override { suspends++; }
    void visit(const CoroBindStmt *) override { coro_binds++; }
};

[[nodiscard]] auto first_definition(Module *module) noexcept {
    for (auto *f : module->function_list()) {
        if (f->is_definition()) { return static_cast<FunctionDefinition *>(f); }
    }
    return static_cast<FunctionDefinition *>(nullptr);
}

[[nodiscard]] auto first_kernel_definition(Module *module) noexcept {
    for (auto *f : module->function_list()) {
        if (f->derived_function_tag() == DerivedFunctionTag::KERNEL) { return static_cast<FunctionDefinition *>(f); }
    }
    return static_cast<FunctionDefinition *>(nullptr);
}

}// namespace

void reg_xir2ast_direct() {

    "xir_to_ast_direct_memory_and_resource_kernel"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        kernel->set_block_size(make_uint3(64u, 1u, 1u));
        auto *buffer = kernel->create_resource_argument(Type::buffer(Type::of<float>()));
        auto *body = kernel->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *local = b.alloca_local(Type::of<float>());
        auto *idx = module.create_dispatch_id();
        auto *zero = module.create_constant_zero(Type::of<uint>());
        auto *x = b.call(Type::of<uint>(), ArithmeticOp::EXTRACT, {idx, zero});
        auto *read = b.call(Type::of<float>(), ResourceReadOp::BUFFER_READ, {buffer, x});
        float two = 2.0f;
        auto *scale = module.create_constant(Type::of<float>(), &two);
        auto *mul = b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {read, scale});
        b.store(local, mul);
        auto *load = b.load(Type::of<float>(), local);
        b.call(ResourceWriteOp::BUFFER_WRITE, {buffer, x, load});
        b.return_void();

        auto ast = xir_to_ast_translate(*kernel, {});
        expect(ast != nullptr);
        auto block_size = ast->block_size();
        expect(block_size.x == 64u);
        expect(block_size.y == 1u);
        expect(block_size.z == 1u);
        expect(ast->arguments().size() == 1u);
        expect(ast->local_variables().size() == 2u);
        StatementCounter counter;
        ast->body()->accept(counter);
        expect(counter.stores >= 1u);
        expect(counter.exprs >= 1u);
        expect(counter.returns == 1u);

        auto roundtrip = ast_to_xir_translate(ast->function(), {});
        expect(roundtrip != nullptr);
        auto text = xir_to_text_translate(roundtrip.get(), false);
        expect(text.find("resource_write buffer_write") != string::npos);
    };

    "xir_to_ast_direct_structured_if"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *buffer = kernel->create_resource_argument(Type::buffer(Type::of<float>()));
        auto *body = kernel->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *idx = module.create_thread_id();
        auto *zero = module.create_constant_zero(Type::of<uint>());
        auto *x = b.call(Type::of<uint>(), ArithmeticOp::EXTRACT, {idx, zero});
        auto *value = b.call(Type::of<float>(), ResourceReadOp::BUFFER_READ, {buffer, x});
        float threshold = 0.0f;
        auto *c = module.create_constant(Type::of<float>(), &threshold);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_GREATER, {value, c});
        auto *if_inst = b.if_(cond);
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(if_inst->create_true_block());
        b.call(ResourceWriteOp::BUFFER_WRITE, {buffer, x, value});
        b.br(merge);
        b.set_insertion_point(if_inst->create_false_block());
        b.call(ResourceWriteOp::BUFFER_WRITE, {buffer, x, c});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();

        auto ast = xir_to_ast_translate(*kernel, {});
        expect(ast != nullptr);
        StatementCounter counter;
        ast->body()->accept(counter);
        expect(counter.ifs == 1u);
        expect(counter.exprs == 2u);
        expect(counter.returns == 1u);
    };

    "xir_to_ast_direct_coroutine_markers_roundtrip"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *local = b.alloca_local(Type::of<float>());
        b.coro_register(local, "local");
        b.coro_suspend(1u);
        b.return_void();

        auto ast = xir_to_ast_translate(*kernel, {});
        expect(ast != nullptr);
        StatementCounter counter;
        ast->body()->accept(counter);
        expect(counter.coro_binds == 1u);
        expect(counter.suspends == 1u);
        expect(counter.returns == 1u);

        auto roundtrip = ast_to_xir_translate(ast->function(), {});
        expect(roundtrip != nullptr);
        auto text = xir_to_text_translate(roundtrip.get(), false);
        expect(text.find("coro_register") != string::npos);
        expect(text.find("coro_suspend 1") != string::npos);
    };
}

void reg_xir2ast_ast_roundtrip() {

    "xir_to_ast_ast_xir_arithmetic_roundtrip"_test = [] {
        Kernel1D kernel = [](BufferFloat buffer) {
            auto idx = dispatch_id().x;
            auto value = buffer->read(idx);
            auto y = value * 2.0f + 1.0f;
            buffer->write(idx, y);
        };
        auto original = ast_to_xir_translate(kernel.function()->function(), {});
        xir_to_ast_normalize_module(original.get());
        auto *def = first_kernel_definition(original.get());
        expect(def != nullptr);
        auto ast = xir_to_ast_translate(*def, {});
        expect(ast != nullptr);
        auto rebuilt = ast_to_xir_translate(ast->function(), {});
        expect(rebuilt != nullptr);
        auto text = xir_to_text_translate(rebuilt.get(), false);
        expect(text.find("arithmetic binary_mul") != string::npos);
        expect(text.find("resource_write buffer_write") != string::npos);
    };

    "xir_to_ast_ast_xir_control_flow_roundtrip"_test = [] {
        Kernel1D kernel = [](BufferFloat buffer) {
            auto idx = dispatch_id().x;
            auto value = buffer->read(idx);
            $if (value > 0.0f) {
                buffer->write(idx, value);
            } $else {
                buffer->write(idx, 0.0f);
            };
        };
        auto original = ast_to_xir_translate(kernel.function()->function(), {});
        xir_to_ast_normalize_module(original.get());
        auto *def = first_kernel_definition(original.get());
        auto ast = xir_to_ast_translate(*def, {});
        expect(ast != nullptr);
        auto rebuilt = ast_to_xir_translate(ast->function(), {});
        expect(rebuilt != nullptr);
        auto text = xir_to_text_translate(rebuilt.get(), false);
        expect(text.find("if") != string::npos);
        expect(text.find("resource_write buffer_write") != string::npos);
    };

    "xir_to_ast_preserves_bound_resource_arguments"_test = [] {
        Kernel1D kernel = [](BufferFloat captured, BufferFloat runtime) {
            auto idx = dispatch_id().x;
            runtime->write(idx, captured->read(idx));
        };
        compute::Function function = kernel.function()->function();
        auto binding = compute::Function::Binding{compute::Function::BufferBinding{0x1234u, 16u, 64u}};
        auto config = XIR2ASTConfig{.bound_arguments = luisa::span{&binding, 1u}};
        auto original = ast_to_xir_translate(function, {});
        auto *def = first_kernel_definition(original.get());
        auto ast = xir_to_ast_translate(*def, config);
        expect(ast != nullptr);
        expect(ast->bound_arguments().size() == 1u);
        expect(ast->unbound_arguments().size() == 1u);
        expect(luisa::holds_alternative<compute::Function::BufferBinding>(ast->bound_arguments().front()));
    };

    "xir_to_ast_ast_xir_coroutine_dsl_roundtrip"_test = [] {
        Kernel1D kernel = [](BufferFloat buffer) {
            auto idx = dispatch_id().x;
            auto value = buffer->read(idx);
            coro_bind(value, "value");
            $suspend("after_read");
            buffer->write(idx, value);
        };
        auto original = ast_to_xir_translate(kernel.function()->function(), {});
        auto *def = first_kernel_definition(original.get());
        expect(def != nullptr);
        auto ast = xir_to_ast_translate(*def, {});
        expect(ast != nullptr);
        StatementCounter counter;
        ast->body()->accept(counter);
        expect(counter.coro_binds == 1u);
        expect(counter.suspends == 1u);
        auto rebuilt = ast_to_xir_translate(ast->function(), {});
        expect(rebuilt != nullptr);
        auto text = xir_to_text_translate(rebuilt.get(), false);
        expect(text.find("coro_register") != string::npos);
        expect(text.find("coro_suspend 1") != string::npos);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_xir2ast_direct();
    reg_xir2ast_ast_roundtrip();
    return 0;
}
