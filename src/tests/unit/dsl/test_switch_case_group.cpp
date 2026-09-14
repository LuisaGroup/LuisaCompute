#include "ut/ut.hpp"

#include <luisa/ast/ast2json.h>
#include <luisa/ast/callable_library.h>
#include <luisa/dsl/sugar.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/xir2ast.h>
#include <luisa/xir/verifier.h>

#include <array>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void check_groups(Function function, uint count) {
    auto module = xir::ast_to_xir_translate(function, {});
    expect(xir::xir_verify_module(module.get()).succeeded());
    auto found = 0u;
    for (auto *f : module->function_list()) {
        if (auto *definition = f->definition()) {
            definition->traverse_instructions([&](const xir::Instruction *inst) {
                if (!inst->isa<xir::SwitchInst>()) { return; }
                auto *sw = static_cast<const xir::SwitchInst *>(inst);
                expect(sw->case_count() == count);
                for (auto i = 1u; i < sw->case_count(); i++) {
                    expect(sw->case_block(i) == sw->case_block(0u));
                }
                expect(sw->default_block() != sw->case_block(0u));
                found++;
            });
        }
    }
    expect(found == 1u);
}

template<typename T>
void check_wide_labels(T a, T b, T c) {
    Kernel1D source = [&](Var<Buffer<T>> output, Var<T> selector) {
        $switch (selector) {
            $case (a, b, c) { output.write(0u, selector); };
            $default {};
        };
    };
    check_groups(source.function()->function(), 3u);
    auto xir = xir::ast_to_xir_translate(source.function()->function(), {});
    auto *kernel = xir->function_list().front()->definition();
    auto rebuilt = xir::xir_to_ast_translate(*kernel, {});
    check_groups(rebuilt->function(), 3u);
    CallableLibrary library;
    library.add_callable("wide", source.function());
    CallableLibrary decoded;
    decoded.load(library.serialize());
    check_groups(decoded.get_function("wide"), 3u);
    expect(decoded.get_function("wide").hash() == source.function()->hash());
}

void check_single_label_semantics(Function lhs, Function rhs) {
    expect(lhs.tag() == rhs.tag());
    expect(lhs.block_size().x == rhs.block_size().x);
    expect(lhs.block_size().y == rhs.block_size().y);
    expect(lhs.block_size().z == rhs.block_size().z);
    expect(lhs.arguments().size() == rhs.arguments().size());
    for (auto i = 0u; i < lhs.arguments().size(); i++) {
        auto left = lhs.arguments()[i];
        auto right = rhs.arguments()[i];
        expect(left.uid() == right.uid());
        expect(left.tag() == right.tag());
        expect(*left.type() == *right.type());
    }

    auto left_body = lhs.body();
    auto right_body = rhs.body();
    expect(left_body->statements().size() == right_body->statements().size());
    expect(left_body->statements().size() == 1u);
    if (left_body->statements().size() != 1u || right_body->statements().size() != 1u) { return; }

    auto left_switch = static_cast<const SwitchStmt *>(left_body->statements()[0]);
    auto right_switch = static_cast<const SwitchStmt *>(right_body->statements()[0]);
    expect(left_switch->tag() == Statement::Tag::SWITCH);
    expect(right_switch->tag() == Statement::Tag::SWITCH);
    expect(left_switch->expression()->tag() == right_switch->expression()->tag());
    expect(*left_switch->expression()->type() == *right_switch->expression()->type());
    expect(left_switch->expression()->hash() == right_switch->expression()->hash());

    auto left_cases = left_switch->body()->statements();
    auto right_cases = right_switch->body()->statements();
    expect(left_cases.size() == right_cases.size());
    expect(left_cases.size() == 1u);
    if (left_cases.size() != 1u || right_cases.size() != 1u) { return; }

    auto left_case = static_cast<const SwitchCaseStmt *>(left_cases[0]);
    auto right_case = static_cast<const SwitchCaseStmt *>(right_cases[0]);
    auto left_labels = left_case->expressions();
    auto right_labels = right_case->expressions();
    expect(left_labels.size() == right_labels.size());
    expect(left_labels.size() == 1u);
    if (left_labels.size() != 1u || right_labels.size() != 1u) { return; }
    expect(left_labels[0]->tag() == right_labels[0]->tag());
    expect(*left_labels[0]->type() == *right_labels[0]->type());
    expect(left_labels[0]->hash() == right_labels[0]->hash());
    if (left_labels[0]->tag() == Expression::Tag::LITERAL) {
        auto left_literal = static_cast<const LiteralExpr *>(left_labels[0]);
        auto right_literal = static_cast<const LiteralExpr *>(right_labels[0]);
        expect(left_literal->value().index() == right_literal->value().index());
        expect(luisa::get<int>(left_literal->value().to_variant()) ==
               luisa::get<int>(right_literal->value().to_variant()));
    }

    auto left_case_body = left_case->body()->statements();
    auto right_case_body = right_case->body()->statements();
    expect(left_case_body.size() == right_case_body.size());
    expect(left_case_body.size() == 1u);
    if (left_case_body.size() != 1u || right_case_body.size() != 1u) { return; }
    expect(left_case_body[0]->tag() == Statement::Tag::BREAK);
    expect(right_case_body[0]->tag() == Statement::Tag::BREAK);
}

}// namespace

int main() {
    "shared_case_dsl_and_all_ast_roundtrips"_test = [] {
        Kernel1D source = [](BufferInt output, Int selector) {
            $switch (selector) {
                $case (-7, 0, 25) { output.write(0u, selector); };
                $default {};
            };
        };
        check_groups(source.function()->function(), 3u);
        auto copied = source.function()->duplicate();
        check_groups(copied->function(), 3u);
        expect(copied->function().hash() == source.function()->hash());
        auto encoded = try_to_json(source.function()->function());
        expect(static_cast<bool>(encoded)) << encoded.error;
        expect(encoded.json.find("SWITCH_CASE_GROUP") != string::npos);
        auto decoded = from_json(encoded.json);
        expect(static_cast<bool>(decoded)) << decoded.error;
        if (decoded) {
            check_groups(decoded.function->function(), 3u);
            expect(decoded.function->function().hash() == source.function()->hash());
        }
        CallableLibrary library;
        library.add_callable("switch", source.function());
        auto binary = library.serialize();
        CallableLibrary loaded;
        loaded.load(binary);
        check_groups(loaded.get_function("switch"), 3u);
        // CallableLibrary is not byte-idempotent even for legacy single
        // cases. Rebuild the AST hash instead of trusting its stored hash.
        auto rehashed = loaded.get_function_builder("switch")->duplicate();
        expect(rehashed->hash() == source.function()->hash());
        CallableLibrary second;
        second.load(loaded.serialize());
        check_groups(second.get_function("switch"), 3u);
    };
    "shared_case_host_filtered_span_and_single_label_compatibility"_test = [] {
        for (auto count : {1u, 2u, 3u}) {
            const std::array<int, 3u> labels{-7, 0, 25};
            Kernel1D source = [&](BufferInt output, Int selector) {
                $switch (selector) {
                    $case (luisa::span<const int>{labels.data(), count}) { output.write(0u, selector); };
                    $default {};
                };
            };
            check_groups(source.function()->function(), count);
            auto encoded = try_to_json(source.function()->function());
            expect(static_cast<bool>(encoded));
            expect((encoded.json.find("SWITCH_CASE_GROUP") != string::npos) == (count > 1u));
            if (count == 1u) {
                // Compare at one source location: DSL comments affect hashes.
                auto make_single = [](bool grouped) {
                    return luisa::compute::detail::FunctionBuilder::define_kernel([&] {
                        auto *builder = luisa::compute::detail::FunctionBuilder::current();
                        auto *selector = builder->argument(Type::of<int>());
                        auto *sw = builder->switch_(selector);
                        builder->with(sw->body(), [&] {
                            auto *label = builder->literal(Type::of<int>(), -7);
                            const Expression *labels[]{label};
                            auto *c = grouped ? builder->case_(luisa::span{labels}) : builder->case_(label);
                            builder->with(c->body(), [&] { builder->break_(); });
                        });
                        // Function duplication validates kernel launch metadata.
                        builder->set_block_size(make_uint3(1u, 1u, 1u));
                    });
                };
                auto single = make_single(false);
                auto grouped = make_single(true);
                expect(single->hash() == grouped->hash());
                CallableLibrary a, b;
                a.add_callable("single", single);
                b.add_callable("single", grouped);
                CallableLibrary decoded_a, decoded_b;
                decoded_a.load(a.serialize());
                decoded_b.load(b.serialize());
                auto rehashed_a = decoded_a.get_function_builder("single")->duplicate();
                auto rehashed_b = decoded_b.get_function_builder("single")->duplicate();
                expect(rehashed_a->hash() == single->hash());
                expect(rehashed_b->hash() == grouped->hash());
                expect(rehashed_a->hash() == rehashed_b->hash());
                check_single_label_semantics(rehashed_a->function(), rehashed_b->function());
            }
        }
    };
    "shared_case_signed_and_unsigned_wide_labels"_test = [] {
        check_wide_labels<int64_t>(-1, 0x00000000ffffffffll, 0x1234567800000000ll);
        check_wide_labels<uint64_t>(0xffffffffffffffffull, 0x00000000ffffffffull, 0x1234567800000000ull);
        check_wide_labels<int16_t>(-1, -32768, 32767);
        check_wide_labels<uint16_t>(0u, 32768u, 65535u);
    };
}
