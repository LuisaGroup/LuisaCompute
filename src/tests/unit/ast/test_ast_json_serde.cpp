// Tests for the versioned and bounded AST JSON interchange format.

#include "ut/ut.hpp"

#include <luisa/ast/ast2json.h>
#include <luisa/ast/function_builder.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::detail;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] shared_ptr<const FunctionBuilder> make_kernel() {
    auto twice = FunctionBuilder::define_callable([] {
        auto builder = FunctionBuilder::current();
        builder->set_name("twice");
        auto value = builder->argument(Type::of<float>());
        builder->set_variable_name(value->variable().uid(), "value");
        auto two = builder->literal(Type::of<float>(), 2.0f);
        builder->return_(builder->binary(
            Type::of<float>(), BinaryOp::MUL, value, two));
    });
    return FunctionBuilder::define_kernel([&] {
        auto builder = FunctionBuilder::current();
        builder->set_name("remote_roundtrip");
        builder->set_block_size(uint3{64u, 1u, 1u});
        builder->set_allowed_warp_size(32u);
        auto output = builder->buffer(Type::buffer(Type::of<float>()));
        builder->set_variable_name(output->variable().uid(), "output");
        auto index = builder->swizzle(
            Type::of<uint>(), builder->dispatch_id(), 1u, 0u);
        auto input = builder->literal(Type::of<float>(), 3.0f);
        auto value = builder->call(
            Type::of<float>(), Function{twice.get()}, {input});
        builder->call(CallOp::BUFFER_WRITE, {output, index, value});
    });
}

void test_round_trip() {
    auto source = make_kernel();
    auto encoded = try_to_json(Function{source.get()});
    expect(static_cast<bool>(encoded));
    expect(encoded.error.empty());
    expect(encoded.json.find("luisa.compute.ast") != string::npos);

    auto decoded = from_json(encoded.json);
    expect(static_cast<bool>(decoded)) << decoded.error;
    if (!decoded) { return; }
    auto function = Function{decoded.function.get()};
    expect(function.tag() == Function::Tag::KERNEL);
    expect(all(function.block_size() == uint3{64u, 1u, 1u}));
    expect(function.allowed_warp_size().value_or(0u) == 32u);
    expect(function.custom_callables().size() == 1u);
    expect(function.hash() == Function{source.get()}.hash());

    auto encoded_again = try_to_json(function);
    expect(static_cast<bool>(encoded_again));
    auto decoded_again = from_json(encoded_again.json);
    expect(static_cast<bool>(decoded_again)) << decoded_again.error;
    if (!decoded_again) { return; }
    expect(Function{decoded_again.function.get()}.hash() == function.hash());
}

void test_control_flow_round_trip() {
    auto source = FunctionBuilder::define_kernel([] {
        auto builder = FunctionBuilder::current();
        builder->set_name("control_flow");
        builder->set_block_size(uint3{8u, 2u, 1u});
        auto output = builder->buffer(Type::buffer(Type::of<uint>()));
        auto index = builder->local(Type::of<uint>());
        builder->set_variable_name(index->variable().uid(), "index");
        auto zero = builder->literal(Type::of<uint>(), 0u);
        auto one = builder->literal(Type::of<uint>(), 1u);
        auto four = builder->literal(Type::of<uint>(), 4u);
        builder->assign(index, zero);
        auto condition = builder->binary(
            Type::of<bool>(), BinaryOp::LESS, index, four);
        auto step = builder->binary(
            Type::of<uint>(), BinaryOp::ADD, index, one);
        auto loop = builder->for_(index, condition, step);
        builder->with(loop->body(), [&] {
            auto low_bit = builder->binary(
                Type::of<uint>(), BinaryOp::BIT_AND, index, one);
            auto even = builder->binary(
                Type::of<bool>(), BinaryOp::EQUAL, low_bit, zero);
            auto branch = builder->if_(even);
            builder->with(branch->true_branch(), [&] {
                builder->call(CallOp::BUFFER_WRITE, {output, index, index});
            });
            builder->with(branch->false_branch(), [&] {
                builder->comment_("odd lane");
            });
        });
        auto tail = builder->loop_();
        builder->with(tail->body(), [&] {
            builder->comment_("single iteration");
            builder->break_();
        });
        const Expression *arguments[]{index};
        builder->print_("index = {}", arguments);
    });
    auto encoded = try_to_json(Function{source.get()});
    expect(static_cast<bool>(encoded)) << encoded.error;
    auto decoded = from_json(encoded.json);
    expect(static_cast<bool>(decoded)) << decoded.error;
    if (!decoded) { return; }
    expect(Function{decoded.function.get()}.hash() == Function{source.get()}.hash());
}

void test_scalar_assignment_conversion_round_trip() {
    auto source = FunctionBuilder::define_kernel([] {
        auto builder = FunctionBuilder::current();
        builder->set_block_size(uint3{1u, 1u, 1u});
        auto value = builder->local(Type::of<uint>());
        builder->assign(
            value, builder->literal(Type::of<int>(), 7));
    });
    auto encoded = try_to_json(Function{source.get()});
    expect(static_cast<bool>(encoded)) << encoded.error;
    auto decoded = from_json(encoded.json);
    expect(static_cast<bool>(decoded)) << decoded.error;
    if (decoded) {
        expect(Function{decoded.function.get()}.hash() ==
               Function{source.get()}.hash());
    }

    auto invalid = FunctionBuilder::define_kernel([] {
        auto builder = FunctionBuilder::current();
        builder->set_block_size(uint3{1u, 1u, 1u});
        auto value = builder->local(Type::of<float2>());
        builder->assign(
            value, builder->literal(Type::of<float>(), 1.0f));
    });
    auto invalid_encoded = try_to_json(Function{invalid.get()});
    expect(!static_cast<bool>(invalid_encoded));
    expect(invalid_encoded.error.find("assignment") != string::npos);
}

void test_indirect_dispatch_buffer_round_trip() {
    auto source = FunctionBuilder::define_kernel([] {
        auto builder = FunctionBuilder::current();
        builder->set_name("indirect_dispatch_writer");
        builder->set_block_size(uint3{1u, 1u, 1u});
        auto dispatch_buffer = builder->buffer(
            Type::custom(ast_json_indirect_dispatch_buffer_type_name));
        auto one = builder->literal(Type::of<uint>(), 1u);
        builder->call(
            CallOp::INDIRECT_SET_DISPATCH_COUNT,
            {dispatch_buffer, one});
    });
    auto encoded = try_to_json(Function{source.get()});
    expect(static_cast<bool>(encoded)) << encoded.error;
    auto decoded = from_json(encoded.json);
    expect(static_cast<bool>(decoded)) << decoded.error;
    if (!decoded) { return; }
    auto function = Function{decoded.function.get()};
    expect(function.arguments().size() == 1u);
    expect(function.arguments().front().tag() == Variable::Tag::BUFFER);
    expect(function.arguments().front().type()->description() ==
           ast_json_indirect_dispatch_buffer_type_name);
    expect(function.hash() == Function{source.get()}.hash());
}

void test_ray_query_custom_types_round_trip() {
    for (auto type_name : {ast_json_ray_query_all_type_name,
                           ast_json_ray_query_any_type_name}) {
        auto source = FunctionBuilder::define_kernel([type_name] {
            auto builder = FunctionBuilder::current();
            builder->set_block_size(uint3{1u, 1u, 1u});
            auto query = builder->local(Type::custom(type_name));
            static_cast<void>(builder->ray_query_(query));
        });
        auto encoded = try_to_json(Function{source.get()});
        expect(static_cast<bool>(encoded)) << encoded.error;
        auto decoded = from_json(encoded.json);
        expect(static_cast<bool>(decoded)) << decoded.error;
        if (decoded) {
            expect(Function{decoded.function.get()}.hash() ==
                   Function{source.get()}.hash());
        }
    }
}

void test_unknown_custom_type_rejected() {
    auto source = FunctionBuilder::define_kernel([] {
        auto builder = FunctionBuilder::current();
        builder->set_block_size(uint3{1u, 1u, 1u});
        auto value = builder->local(Type::custom("UntrustedRemoteCustomType"));
        static_cast<void>(builder->ray_query_(value));
    });
    auto encoded = try_to_json(Function{source.get()});
    expect(!static_cast<bool>(encoded));
    expect(encoded.error.find("Custom AST types") != string::npos);
}

void test_nested_custom_type_rejected() {
    auto source = FunctionBuilder::define_kernel([] {
        auto builder = FunctionBuilder::current();
        builder->set_block_size(uint3{1u, 1u, 1u});
        static_cast<void>(builder->buffer(
            Type::buffer(Type::custom(ast_json_ray_query_all_type_name))));
    });
    auto encoded = try_to_json(Function{source.get()});
    expect(!static_cast<bool>(encoded));
    expect(encoded.error.find("nested in buffers") != string::npos);

    // The legacy encoder remains available for source compatibility, so the
    // decoder must independently reject the same unsafe type construction.
    auto decoded = from_json(to_json(Function{source.get()}));
    expect(!static_cast<bool>(decoded));
    expect(decoded.error.find("portable data type") != string::npos);
}

class RemappingResolver final : public ASTJsonBindingResolver {

public:
    [[nodiscard]] bool resolve_buffer(
        const Type *serialized_type, uint64_t serialized_handle,
        size_t serialized_offset,
        size_t serialized_size, Function::BufferBinding &binding,
        string &error) const noexcept override {
        if (serialized_type != Type::buffer(Type::of<float>()) ||
            serialized_handle != 17u) {
            error = "unexpected serialized buffer handle";
            return false;
        }
        binding = Function::BufferBinding{
            99u, serialized_offset + 4u, serialized_size};
        return true;
    }
};

void test_binding_resolver() {
    auto source = FunctionBuilder::define_kernel([] {
        auto builder = FunctionBuilder::current();
        builder->set_block_size(uint3{1u, 1u, 1u});
        auto buffer = builder->buffer_binding(
            Type::buffer(Type::of<float>()), 17u, 8u, 64u);
        auto zero = builder->literal(Type::of<uint>(), 0u);
        auto value = builder->call(
            Type::of<float>(), CallOp::BUFFER_READ, {buffer, zero});
        builder->expression_statement(value);
    });
    auto encoded = try_to_json(Function{source.get()});
    expect(static_cast<bool>(encoded));
    RemappingResolver resolver;
    auto decoded = from_json(encoded.json, {}, &resolver);
    expect(static_cast<bool>(decoded)) << decoded.error;
    if (!decoded) { return; }
    auto bindings = Function{decoded.function.get()}.bound_arguments();
    expect(bindings.size() == 1u);
    auto binding = get<Function::BufferBinding>(bindings.front());
    expect(binding.handle == 99u);
    expect(binding.offset == 12u);
    expect(binding.size == 64u);
}

void cpu_custom(void *, void *) {}
void cpu_custom_destroy(void *) {}

void test_unsafe_node_rejected() {
    auto source = FunctionBuilder::define_kernel([] {
        auto builder = FunctionBuilder::current();
        builder->set_block_size(uint3{1u, 1u, 1u});
        auto argument = builder->literal(Type::of<int>(), 1);
        auto expression = builder->call(
            Type::of<int>(), cpu_custom, cpu_custom_destroy,
            nullptr, argument);
        builder->expression_statement(expression);
    });
    auto encoded = try_to_json(Function{source.get()});
    expect(!static_cast<bool>(encoded));
    expect(encoded.error.find("CPU custom") != string::npos);
}

void test_malformed_documents_rejected() {
    auto empty = from_json("{}");
    expect(!static_cast<bool>(empty));
    expect(!empty.error.empty());

    auto source = make_kernel();
    auto encoded = try_to_json(Function{source.get()});
    expect(static_cast<bool>(encoded));

    auto wrong_version = encoded.json;
    auto version = wrong_version.find("\"version\": 1");
    expect(version != string::npos);
    wrong_version[version + string_view{"\"version\": "}.size()] = '2';
    auto version_result = from_json(wrong_version);
    expect(!static_cast<bool>(version_result));
    expect(version_result.error.find("version") != string::npos);

    auto bad_base64 = encoded.json;
    auto value = bad_base64.find("\"value\": \"");
    expect(value != string::npos);
    bad_base64[value + string_view{"\"value\": \""}.size()] = '!';
    auto base64_result = from_json(bad_base64);
    expect(!static_cast<bool>(base64_result));
    expect(base64_result.error.find("Base64") != string::npos);

    auto too_small = from_json(
        encoded.json, ASTJsonLimits{.max_document_bytes = 8u});
    expect(!static_cast<bool>(too_small));
    expect(too_small.error.find("byte limit") != string::npos);

    auto duplicate_schema = encoded.json;
    duplicate_schema.insert(1u, "\n  \"schema\": \"luisa.compute.ast\",");
    auto duplicate_result = from_json(duplicate_schema);
    expect(!static_cast<bool>(duplicate_result));
    expect(duplicate_result.error.find("duplicate member") != string::npos);

    auto bad_arity = encoded.json;
    auto write = bad_arity.find("\"op\": \"BUFFER_WRITE\"");
    expect(write != string::npos);
    bad_arity.replace(
        write, string_view{"\"op\": \"BUFFER_WRITE\""}.size(),
        "\"op\": \"ASYNC_COPY\"");
    auto arity_result = from_json(bad_arity);
    expect(!static_cast<bool>(arity_result));
    expect(arity_result.error.find("requires") != string::npos);

    auto indirect_source = FunctionBuilder::define_kernel([] {
        auto builder = FunctionBuilder::current();
        builder->set_block_size(uint3{1u, 1u, 1u});
        auto dispatch_buffer = builder->buffer(
            Type::custom(ast_json_indirect_dispatch_buffer_type_name));
        auto one = builder->literal(Type::of<uint>(), 1u);
        builder->call(
            CallOp::INDIRECT_SET_DISPATCH_COUNT,
            {dispatch_buffer, one});
    });
    auto indirect_encoded = try_to_json(Function{indirect_source.get()});
    expect(static_cast<bool>(indirect_encoded));
    auto bad_indirect_arity = indirect_encoded.json;
    auto indirect_count = bad_indirect_arity.find(
        "\"op\": \"INDIRECT_SET_DISPATCH_COUNT\"");
    expect(indirect_count != string::npos);
    bad_indirect_arity.replace(
        indirect_count,
        string_view{"\"op\": \"INDIRECT_SET_DISPATCH_COUNT\""}.size(),
        "\"op\": \"INDIRECT_SET_DISPATCH_KERNEL\"");
    auto indirect_arity_result = from_json(bad_indirect_arity);
    expect(!static_cast<bool>(indirect_arity_result));
    expect(indirect_arity_result.error.find("exactly 5") != string::npos);

    auto tiny_parse_budget = from_json(
        encoded.json,
        ASTJsonLimits{.max_parse_memory_bytes = 1u});
    expect(!static_cast<bool>(tiny_parse_budget));
    expect(tiny_parse_budget.error.find("parse") != string::npos);

    auto tiny_node_budget = from_json(
        encoded.json,
        ASTJsonLimits{.max_nodes = 2u});
    expect(!static_cast<bool>(tiny_node_budget));
    expect(tiny_node_budget.error.find("node count") != string::npos);

    auto structure_source = FunctionBuilder::define_kernel([] {
        auto builder = FunctionBuilder::current();
        builder->set_block_size(uint3{1u, 1u, 1u});
        auto value = builder->local(
            Type::structure(16u, {Type::of<float4>()}));
        builder->assign(value, value);
    });
    auto structure_encoded = try_to_json(
        Function{structure_source.get()});
    expect(static_cast<bool>(structure_encoded));
    auto bad_structure_alignment = structure_encoded.json;
    auto structure_tag = bad_structure_alignment.find(
        "\"tag\": \"STRUCTURE\"");
    expect(structure_tag != string::npos);
    if (structure_tag == string::npos) { return; }
    auto structure_alignment = bad_structure_alignment.find(
        "\"alignment\": 16");
    expect(structure_alignment != string::npos);
    if (structure_alignment == string::npos) { return; }
    bad_structure_alignment.replace(
        structure_alignment,
        string_view{"\"alignment\": 16"}.size(),
        "\"alignment\": 1");
    auto structure_result = from_json(bad_structure_alignment);
    expect(!static_cast<bool>(structure_result));
    expect(structure_result.error.find("member alignment") != string::npos);

    auto byte_buffer_source = FunctionBuilder::define_kernel([] {
        auto builder = FunctionBuilder::current();
        builder->set_block_size(uint3{1u, 1u, 1u});
        auto buffer = builder->buffer(Type::of<ByteBuffer>());
        auto zero = builder->literal(Type::of<uint>(), 0u);
        auto value = builder->call(
            Type::of<uint>(), CallOp::BYTE_BUFFER_READ,
            {buffer, zero});
        builder->expression_statement(value);
    });
    auto byte_buffer_encoded = try_to_json(
        Function{byte_buffer_source.get()});
    expect(static_cast<bool>(byte_buffer_encoded));
    auto bad_atomic = byte_buffer_encoded.json;
    auto byte_buffer_read = bad_atomic.find(
        "\"op\": \"BYTE_BUFFER_READ\"");
    expect(byte_buffer_read != string::npos);
    if (byte_buffer_read == string::npos) { return; }
    bad_atomic.replace(
        byte_buffer_read,
        string_view{"\"op\": \"BYTE_BUFFER_READ\""}.size(),
        "\"op\": \"ATOMIC_FETCH_ADD\"");
    auto atomic_result = from_json(bad_atomic);
    expect(!static_cast<bool>(atomic_result));
    expect(atomic_result.error.find("typed buffer") != string::npos);

    auto invalid_lvalue = FunctionBuilder::define_kernel([] {
        auto builder = FunctionBuilder::current();
        builder->set_block_size(uint3{1u, 1u, 1u});
        auto one = builder->literal(Type::of<uint>(), 1u);
        builder->assign(one, one);
    });
    auto invalid_lvalue_encoded = try_to_json(
        Function{invalid_lvalue.get()});
    expect(!static_cast<bool>(invalid_lvalue_encoded));
    expect(invalid_lvalue_encoded.error.find("not assignable") != string::npos);

    auto invalid_ray_query = FunctionBuilder::define_kernel([] {
        auto builder = FunctionBuilder::current();
        builder->set_block_size(uint3{1u, 1u, 1u});
        auto value = builder->local(Type::of<uint>());
        static_cast<void>(builder->ray_query_(value));
    });
    auto invalid_ray_query_encoded = try_to_json(
        Function{invalid_ray_query.get()});
    expect(!static_cast<bool>(invalid_ray_query_encoded));
    expect(invalid_ray_query_encoded.error.find("non-ray-query") != string::npos);
}

}// namespace

static auto test_ast_json_registration = [] {
    "ast_json_round_trip"_test = test_round_trip;
    "ast_json_control_flow_round_trip"_test =
        test_control_flow_round_trip;
    "ast_json_scalar_assignment_conversion_round_trip"_test =
        test_scalar_assignment_conversion_round_trip;
    "ast_json_indirect_dispatch_buffer_round_trip"_test =
        test_indirect_dispatch_buffer_round_trip;
    "ast_json_ray_query_custom_types_round_trip"_test =
        test_ray_query_custom_types_round_trip;
    "ast_json_unknown_custom_type_rejected"_test =
        test_unknown_custom_type_rejected;
    "ast_json_nested_custom_type_rejected"_test =
        test_nested_custom_type_rejected;
    "ast_json_binding_resolver"_test = test_binding_resolver;
    "ast_json_unsafe_node_rejected"_test = test_unsafe_node_rejected;
    "ast_json_malformed_documents_rejected"_test =
        test_malformed_documents_rejected;
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
}
