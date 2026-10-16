// Regression for linear-time FunctionBuilder ownership discovery on shared
// expression DAGs.

#include "ut/ut.hpp"

#include <cstdint>

#include <luisa/ast/function_builder.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::detail;
using namespace boost::ut;
using namespace boost::ut::literals;

int main() {
    "kernel_ownership_check_visits_shared_expression_dag_once"_test = [] {
        constexpr auto depth = 40u;
        auto kernel = FunctionBuilder::define_kernel([] {
            auto *builder = FunctionBuilder::current();
            auto *sink = builder->local(Type::of<uint32_t>());
            const Expression *value = builder->literal(
                Type::of<uint32_t>(), uint32_t{1u});
            for (auto i = 0u; i < depth; ++i) {
                value = builder->binary(
                    Type::of<uint32_t>(), BinaryOp::ADD,
                    value, value);
            }
            builder->assign(sink, value);
        });

        expect(kernel != nullptr);
        expect(kernel->body()->statements().size() == 1u);
        auto *statement = kernel->body()->statements().front();
        expect(statement->tag() == Statement::Tag::ASSIGN);
        auto *value = static_cast<const AssignStmt *>(statement)->rhs();
        for (auto i = 0u; i < depth; ++i) {
            expect(value->tag() == Expression::Tag::BINARY);
            auto *binary = static_cast<const BinaryExpr *>(value);
            // This is a 40-node DAG whose occurrence-expanded tree contains
            // 2^40 leaves. Ownership analysis must not expand that tree.
            expect(binary->lhs() == binary->rhs());
            value = binary->lhs();
        }
        expect(value->tag() == Expression::Tag::LITERAL);
    };
}
