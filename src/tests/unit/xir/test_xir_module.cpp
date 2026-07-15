#include "ut/ut.hpp"
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>
#include <luisa/ast/type_registry.h>
#include <luisa/runtime/buffer.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

// ---- KernelFunction ----

// ---- CallableFunction ----

// ---- ExternalFunction ----

// ---- Multiple functions in module ----

// ---- Arguments ----

// ---- BasicBlock ----

// ---- Constants ----

// ---- Undefined ----

// ---- Special Registers ----

// ---- Metadata ----

// ---- Value hierarchy ----

// ---- Traversal ----

// ---- to_string for tags ----

void reg_module_basics() {

    "xir_module_construction"_test = [] {
        Module module;
        auto count = 0u;
        for ([[maybe_unused]] auto *f : module.function_list()) { count++; }
        expect(count == 0u) << "new module should have no functions";
        auto ccount = 0u;
        for ([[maybe_unused]] auto *c : module.constant_list()) { ccount++; }
        expect(ccount == 0u) << "new module should have no constants";
        auto ucount = 0u;
        for ([[maybe_unused]] auto *u : module.undefined_list()) { ucount++; }
        expect(ucount == 0u) << "new module should have no undefined values";
        auto scount = 0u;
        for ([[maybe_unused]] auto *s : module.special_register_list()) { scount++; }
        expect(scount == 0u) << "new module should have no special registers";
    };
}

void reg_kernel() {

    "xir_kernel_function"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        expect(kernel != nullptr) << "create_kernel should return non-null";
        expect(kernel->derived_function_tag() == DerivedFunctionTag::KERNEL);
        expect(kernel->is_definition() == true);
        auto bs = kernel->block_size();
        expect(bs.x == 64u && bs.y == 1u && bs.z == 1u) << "default block size should be (64,1,1)";
        kernel->set_block_size(make_uint3(16u, 8u, 1u));
        bs = kernel->block_size();
        expect(bs.x == 16u && bs.y == 8u && bs.z == 1u);
        auto count = 0u;
        for ([[maybe_unused]] auto *f : module.function_list()) { count++; }
        expect(count == 1u);
    };

    "xir_kernel_body_block"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        expect(kernel->body_block() == nullptr);
        auto *body = kernel->create_body_block();
        expect(body != nullptr);
        expect(kernel->body_block() == body);
        // create_body_block(false) asserts when body already exists, so skip that
        // create_body_block(true) overwrites the existing body block
        auto *body3 = kernel->create_body_block(true);
        expect(body3 != nullptr);
        expect(kernel->body_block() == body3);
    };
}

void reg_callable() {

    "xir_callable_function"_test = [] {
        Module module;
        auto *float_type = Type::of<float>();
        auto *callable = module.create_callable(float_type);
        expect(callable != nullptr);
        expect(callable->derived_function_tag() == DerivedFunctionTag::CALLABLE);
        expect(callable->is_definition() == true);
        expect(callable->type() == float_type);
    };

    "xir_callable_void_return"_test = [] {
        Module module;
        auto *callable = module.create_callable(nullptr);
        expect(callable != nullptr);
        expect(callable->type() == nullptr) << "void callable should have null type";
    };
}

void reg_external() {

    "xir_external_function"_test = [] {
        Module module;
        auto *ext = module.create_external_function(Type::of<int>());
        expect(ext != nullptr);
        expect(ext->derived_function_tag() == DerivedFunctionTag::EXTERNAL);
        expect(ext->is_definition() == false) << "external functions are not definitions";
        expect(ext->definition() == nullptr) << "external functions have no definition";
    };
}

void reg_multi_func() {

    "xir_multiple_functions"_test = [] {
        Module module;
        (void)module.create_kernel();
        (void)module.create_callable(Type::of<float>());
        (void)module.create_external_function(Type::of<int>());
        auto count = 0u;
        for ([[maybe_unused]] auto *f : module.function_list()) { count++; }
        expect(count == 3u) << "module should have 3 functions";
    };
}

void reg_arguments() {

    "xir_value_argument"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *arg = kernel->create_value_argument(Type::of<float>());
        expect(arg != nullptr);
        expect(arg->type() == Type::of<float>());
        expect(arg->derived_argument_tag() == DerivedArgumentTag::VALUE);
        expect(arg->is_value() == true);
        expect(arg->is_reference() == false);
        expect(arg->is_resource() == false);
        expect(arg->is_lvalue() == false);
    };

    "xir_reference_argument"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *arg = kernel->create_reference_argument(Type::of<int>());
        expect(arg != nullptr);
        expect(arg->derived_argument_tag() == DerivedArgumentTag::REFERENCE);
        expect(arg->is_reference() == true);
        expect(arg->is_lvalue() == true) << "reference arguments should be lvalues";
    };

    "xir_resource_argument"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *arg = kernel->create_resource_argument(Type::of<Buffer<float>>());
        expect(arg != nullptr);
        expect(arg->derived_argument_tag() == DerivedArgumentTag::RESOURCE);
        expect(arg->is_resource() == true);
        expect(arg->is_lvalue() == false);
    };

    "xir_multiple_arguments"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        kernel->create_value_argument(Type::of<float>());
        kernel->create_reference_argument(Type::of<int>());
        kernel->create_resource_argument(Type::of<Buffer<float>>());
        auto count = 0u;
        for ([[maybe_unused]] auto *a : kernel->arguments()) { count++; }
        expect(count == 3u);
    };

    "xir_create_argument_by_ref_flag"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *val_arg = kernel->create_argument(Type::of<float>(), false);
        auto *ref_arg = kernel->create_argument(Type::of<int>(), true);
        expect(val_arg->is_value() == true);
        expect(ref_arg->is_reference() == true);
    };
}

void reg_basic_block() {

    "xir_basic_block_creation"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *bb = kernel->create_basic_block();
        expect(bb != nullptr);
        expect(bb->is_terminated() == false);
    };

    "xir_multiple_basic_blocks"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *bb1 = kernel->create_basic_block();
        auto *bb2 = kernel->create_basic_block();
        expect(bb1 != bb2) << "each create_basic_block should return a new block";
        auto count = 0u;
        for ([[maybe_unused]] auto *b : kernel->basic_blocks()) { count++; }
        expect(count == 2u);
    };
}

void reg_constants() {

    "xir_constant_create"_test = [] {
        Module module;
        auto *float_type = Type::of<float>();
        float value = 3.14f;
        auto *c = module.create_constant(float_type, &value);
        expect(c != nullptr);
        expect(c->type() == float_type);
        expect(c->as<float>() == 3.14f) << "constant data should match";
        expect(c->derived_value_tag() == DerivedValueTag::CONSTANT);
    };

    "xir_constant_zero"_test = [] {
        Module module;
        auto *int_type = Type::of<int>();
        auto *zero = module.create_constant_zero(int_type);
        expect(zero != nullptr);
        expect(zero->as<int>() == 0);
    };

    "xir_constant_one"_test = [] {
        Module module;
        auto *int_type = Type::of<int>();
        auto *one = module.create_constant_one(int_type);
        expect(one != nullptr);
        expect(one->as<int>() == 1);
    };

    "xir_constant_deduplication"_test = [] {
        Module module;
        auto *int_type = Type::of<int>();
        int v1 = 42;
        int v2 = 42;
        auto *c1 = module.create_constant(int_type, &v1);
        auto *c2 = module.create_constant(int_type, &v2);
        expect(c1 == c2) << "constants with same type and data should be deduplicated";
    };

    "xir_constant_different_values"_test = [] {
        Module module;
        auto *int_type = Type::of<int>();
        int v1 = 42;
        int v2 = 43;
        auto *c1 = module.create_constant(int_type, &v1);
        auto *c2 = module.create_constant(int_type, &v2);
        expect(c1 != c2) << "constants with different data should not be deduplicated";
    };

    "xir_constant_hash"_test = [] {
        Module module;
        auto *int_type = Type::of<int>();
        int v1 = 42;
        int v2 = 42;
        auto *c1 = module.create_constant(int_type, &v1);
        auto *c2 = module.create_constant(int_type, &v2);
        expect(c1->hash() == c2->hash()) << "same constants should have same hash";
    };

    "xir_constant_list"_test = [] {
        Module module;
        auto *float_type = Type::of<float>();
        float v1 = 1.0f;
        float v2 = 2.0f;
        (void)module.create_constant(float_type, &v1);
        (void)module.create_constant(float_type, &v2);
        auto count = 0u;
        for ([[maybe_unused]] auto *c : module.constant_list()) { count++; }
        expect(count == 2u);
    };

    "xir_constant_zero_one_different"_test = [] {
        Module module;
        auto *int_type = Type::of<int>();
        auto *zero = module.create_constant_zero(int_type);
        auto *one = module.create_constant_one(int_type);
        expect(zero != one) << "zero and one constants should differ";
        expect(zero->as<int>() != one->as<int>());
    };
}

void reg_undefined() {

    "xir_undefined_creation"_test = [] {
        Module module;
        auto *float_type = Type::of<float>();
        auto *undef = module.create_undefined(float_type);
        expect(undef != nullptr);
        expect(undef->type() == float_type);
        expect(undef->derived_value_tag() == DerivedValueTag::UNDEFINED);
    };

    "xir_undefined_deduplication"_test = [] {
        Module module;
        auto *float_type = Type::of<float>();
        auto *u1 = module.create_undefined(float_type);
        auto *u2 = module.create_undefined(float_type);
        expect(u1 == u2) << "undefined values of same type should be deduplicated";
    };

    "xir_undefined_different_types"_test = [] {
        Module module;
        auto *u1 = module.create_undefined(Type::of<float>());
        auto *u2 = module.create_undefined(Type::of<int>());
        expect(u1 != u2) << "undefined values of different types should differ";
    };

    "xir_undefined_list"_test = [] {
        Module module;
        (void)module.create_undefined(Type::of<float>());
        (void)module.create_undefined(Type::of<int>());
        auto count = 0u;
        for ([[maybe_unused]] auto *u : module.undefined_list()) { count++; }
        expect(count == 2u);
    };
}

void reg_special_regs() {

    "xir_special_register_thread_id"_test = [] {
        Module module;
        auto *tid = module.create_thread_id();
        expect(tid != nullptr);
        expect(tid->derived_special_register_tag() == DerivedSpecialRegisterTag::THREAD_ID);
        expect(tid->type() == Type::of<uint3>());
        expect(tid->derived_value_tag() == DerivedValueTag::SPECIAL_REGISTER);
    };

    "xir_special_register_block_id"_test = [] {
        Module module;
        auto *bid = module.create_block_id();
        expect(bid != nullptr);
        expect(bid->derived_special_register_tag() == DerivedSpecialRegisterTag::BLOCK_ID);
        expect(bid->type() == Type::of<uint3>());
    };

    "xir_special_register_dispatch_id"_test = [] {
        Module module;
        auto *did = module.create_dispatch_id();
        expect(did != nullptr);
        expect(did->derived_special_register_tag() == DerivedSpecialRegisterTag::DISPATCH_ID);
        expect(did->type() == Type::of<uint3>());
    };

    "xir_special_register_warp_lane_id"_test = [] {
        Module module;
        auto *wlid = module.create_warp_lane_id();
        expect(wlid != nullptr);
        expect(wlid->derived_special_register_tag() == DerivedSpecialRegisterTag::WARP_LANE_ID);
        expect(wlid->type() == Type::of<uint>());
    };

    "xir_special_register_kernel_id"_test = [] {
        Module module;
        auto *kid = module.create_kernel_id();
        expect(kid != nullptr);
        expect(kid->derived_special_register_tag() == DerivedSpecialRegisterTag::KERNEL_ID);
        expect(kid->type() == Type::of<uint>());
    };

    "xir_special_register_block_size"_test = [] {
        Module module;
        auto *bsz = module.create_block_size();
        expect(bsz != nullptr);
        expect(bsz->derived_special_register_tag() == DerivedSpecialRegisterTag::BLOCK_SIZE);
        expect(bsz->type() == Type::of<uint3>());
    };

    "xir_special_register_dispatch_size"_test = [] {
        Module module;
        auto *dsz = module.create_dispatch_size();
        expect(dsz != nullptr);
        expect(dsz->derived_special_register_tag() == DerivedSpecialRegisterTag::DISPATCH_SIZE);
        expect(dsz->type() == Type::of<uint3>());
    };

    "xir_special_register_warp_size"_test = [] {
        Module module;
        auto *wsz = module.create_warp_size();
        expect(wsz != nullptr);
        expect(wsz->derived_special_register_tag() == DerivedSpecialRegisterTag::WARP_SIZE);
        expect(wsz->type() == Type::of<uint>());
    };

    "xir_special_register_deduplication"_test = [] {
        Module module;
        auto *tid1 = module.create_thread_id();
        auto *tid2 = module.create_thread_id();
        expect(tid1 == tid2) << "same special register tag should be deduplicated";
    };

    "xir_special_register_generic_api"_test = [] {
        Module module;
        auto *sr = module.create_special_register(DerivedSpecialRegisterTag::DISPATCH_ID);
        auto *did = module.create_dispatch_id();
        expect(sr == did) << "generic API and convenience API should return same register";
    };

    "xir_special_register_list"_test = [] {
        Module module;
        (void)module.create_thread_id();
        (void)module.create_block_id();
        (void)module.create_dispatch_id();
        auto count = 0u;
        for ([[maybe_unused]] auto *s : module.special_register_list()) { count++; }
        expect(count == 3u);
    };
}

void reg_metadata() {

    "xir_metadata_name_on_module"_test = [] {
        Module module;
        expect(!module.name().has_value());
        module.set_name("test_module");
        expect(module.name().has_value());
        expect(module.name().value() == "test_module");
    };

    "xir_metadata_name_on_function"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        expect(!kernel->name().has_value());
        kernel->set_name("my_kernel");
        expect(kernel->name().has_value());
        expect(kernel->name().value() == "my_kernel");
    };

    "xir_metadata_name_on_constant"_test = [] {
        Module module;
        int val = 42;
        auto *c = module.create_constant(Type::of<int>(), &val);
        c->set_name("my_constant");
        expect(c->name().has_value());
        expect(c->name().value() == "my_constant");
    };

    "xir_metadata_comment"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        kernel->add_comment("this is a test kernel");
        // Verify comment metadata exists
        auto *meta = kernel->find_metadata(DerivedMetadataTag::COMMENT);
        expect(meta != nullptr) << "comment metadata should be findable";
    };

    "xir_metadata_location"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        kernel->set_location("test_file.cpp", 42);
        auto *meta = kernel->find_metadata(DerivedMetadataTag::LOCATION);
        expect(meta != nullptr) << "location metadata should be findable";
    };

    "xir_metadata_on_argument"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *arg = kernel->create_value_argument(Type::of<float>());
        arg->set_name("input_x");
        expect(arg->name().has_value());
        expect(arg->name().value() == "input_x");
    };
}

void reg_value_hierarchy() {

    "xir_value_tag_function"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        expect(kernel->derived_value_tag() == DerivedValueTag::FUNCTION);
    };

    "xir_value_tag_basic_block"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *bb = kernel->create_basic_block();
        expect(bb->derived_value_tag() == DerivedValueTag::BASIC_BLOCK);
    };

    "xir_value_tag_argument"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *arg = kernel->create_value_argument(Type::of<float>());
        expect(arg->derived_value_tag() == DerivedValueTag::ARGUMENT);
    };

    "xir_value_is_global"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *c = module.create_constant_zero(Type::of<int>());
        auto *sr = module.create_thread_id();
        auto *undef = module.create_undefined(Type::of<float>());
        // All global values
        expect(kernel->is_global() == true);
        expect(c->is_global() == true);
        expect(sr->is_global() == true);
        expect(undef->is_global() == true);
        // Arguments are function-scoped, not global
        auto *arg = kernel->create_value_argument(Type::of<float>());
        expect(arg->is_global() == false);
    };
}

void reg_traversal() {

    "xir_traverse_basic_blocks_default_order"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        // Must terminate the body block before traversal
        // (traverse_basic_blocks calls terminator() which asserts is_terminated())
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.return_void();
        auto count = 0u;
        kernel->traverse_basic_blocks([&](BasicBlock *) { count++; });
        expect(count >= 1u) << "traversal should visit at least the body block";
    };
}

void reg_tag_strings() {

    "xir_function_tag_to_string"_test = [] {
        expect(to_string(DerivedFunctionTag::KERNEL) == "kernel");
        expect(to_string(DerivedFunctionTag::CALLABLE) == "callable");
        expect(to_string(DerivedFunctionTag::EXTERNAL) == "external");
    };

    "xir_value_tag_to_string"_test = [] {
        expect(to_string(DerivedValueTag::UNDEFINED) == "undef");
        expect(to_string(DerivedValueTag::FUNCTION) == "func");
        expect(to_string(DerivedValueTag::BASIC_BLOCK) == "block");
        expect(to_string(DerivedValueTag::INSTRUCTION) == "inst");
        expect(to_string(DerivedValueTag::CONSTANT) == "const");
        expect(to_string(DerivedValueTag::ARGUMENT) == "arg");
        expect(to_string(DerivedValueTag::SPECIAL_REGISTER) == "sreg");
    };

    "xir_special_register_tag_to_string"_test = [] {
        expect(to_string(DerivedSpecialRegisterTag::THREAD_ID) == "thread_id");
        expect(to_string(DerivedSpecialRegisterTag::BLOCK_ID) == "block_id");
        expect(to_string(DerivedSpecialRegisterTag::DISPATCH_ID) == "dispatch_id");
        expect(to_string(DerivedSpecialRegisterTag::DISPATCH_SIZE) == "dispatch_size");
    };

    "xir_argument_tag_strings"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *val = kernel->create_value_argument(Type::of<float>());
        auto *ref = kernel->create_reference_argument(Type::of<int>());
        auto *res = kernel->create_resource_argument(Type::of<Buffer<float>>());
        expect(val->derived_argument_tag() == DerivedArgumentTag::VALUE);
        expect(ref->derived_argument_tag() == DerivedArgumentTag::REFERENCE);
        expect(res->derived_argument_tag() == DerivedArgumentTag::RESOURCE);
    };
}

int main(int argc, char *argv[]) {

    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_module_basics();
    reg_kernel();
    reg_callable();
    reg_external();
    reg_multi_func();
    reg_arguments();
    reg_basic_block();
    reg_constants();
    reg_undefined();
    reg_special_regs();
    reg_metadata();
    reg_value_hierarchy();
    reg_traversal();
    reg_tag_strings();
    return 0;
}
