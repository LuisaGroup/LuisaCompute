// A reference can be read-only through nested calls even when the leaf's
// unknown reference actuals prevent signature promotion at that call site.
#include "ut/ut.hpp"

#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/promote_ref_arg.h>
#include <luisa/xir/verifier.h>

using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {
struct Options {
    unsigned depth{1u};
    bool alias_at_root{};
    bool alias_in_wrapper{};
    bool write_read_argument{};
    bool shared{};
    bool additional_alias_call{};
    bool external{};
    bool recursive{};
};

void check_forwarding(Options options, bool should_promote) {
    Module module;
    auto *type = Type::of<int>();
    auto *one = module.create_constant_one(type);
    auto *zero = module.create_constant_zero(type);
    Function *leaf = options.external ? static_cast<Function *>(module.create_external_function(type)) : static_cast<Function *>(module.create_callable(type));
    leaf->set_name("forwarding_leaf");
    auto *read = leaf->create_reference_argument(type);
    auto *write = leaf->create_reference_argument(type);
    XIRBuilder b;
    if (!options.external) {
        b.set_insertion_point(leaf->definition()->create_body_block());
        b.store(write, one);
        if (options.write_read_argument) { b.store(read, zero); }
        if (options.recursive) { b.call(type, leaf, {read, write}); }
        // The load follows the write, so an aliased snapshot is unsound.
        b.return_(b.load(type, read));
    }
    auto *outer = leaf;
    for (auto level = 0u; level < options.depth; ++level) {
        auto *wrapper = module.create_callable(type);
        auto *input = wrapper->create_reference_argument(type);
        auto *output = wrapper->create_reference_argument(type);
        b.set_insertion_point(wrapper->create_body_block());
        auto *value = b.call(type, outer,
                             {input, options.alias_in_wrapper ? input : output});
        b.return_(value);
        outer = wrapper;
    }
    auto *kernel = module.create_kernel();
    b.set_insertion_point(kernel->create_body_block());
    auto *input = options.shared ? b.alloca_shared(type) : b.alloca_local(type);
    auto *output = b.alloca_local(type);
    b.store(input, zero);
    b.store(output, zero);
    auto *call = b.call(type, outer,
                        {input, options.alias_at_root ? input : output});
    auto *sink = b.alloca_local(type);
    b.store(sink, call);
    if (options.additional_alias_call) {
        b.store(sink, b.call(type, outer, {input, input}));
    }
    b.return_void();

    expect(xir_verify_module(&module).succeeded());
    auto result = promote_ref_arg_pass_run_on_module(&module);
    expect(outer->arguments().front()->is_reference() != should_promote)
        << "transitive read-only effects must be independent of whether "
           "a nested call site can prove its actual references disjoint";
    if (should_promote) {
        expect(result.promoted_ref_arg_count == 1u);
        expect(call->argument(0u)->isa<LoadInst>());
        // This leaf still cannot prove disjoint roots for its reference
        // actuals. Only the closed-world outer call is snapshotted.
        expect(leaf->arguments().front()->is_reference());
        expect(outer->arguments().back()->is_reference());
    } else {
        expect(call->argument(0u) == input);
    }
    expect(xir_verify_module(&module).succeeded());
    expect(!promote_ref_arg_pass_run_on_module(&module).changed())
        << "the signature rewrite must reach a stable result";
}
}// namespace

int main(int argc, char **argv) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    "readonly_reference_forwarded_through_one_wrapper"_test = [] {
        check_forwarding({}, true);
    };
    "readonly_reference_forwarded_through_three_wrappers"_test = [] {
        check_forwarding({.depth = 3u}, true);
    };
    "forwarded_readonly_reference_root_alias_is_not_snapshotted"_test = [] {
        check_forwarding({.alias_at_root = true}, false);
    };
    "forwarding_same_pointer_to_read_and_write_is_not_readonly"_test = [] {
        check_forwarding({.alias_in_wrapper = true}, false);
    };
    "transitive_write_is_not_readonly"_test = [] {
        check_forwarding({.write_read_argument = true}, false);
    };
    "forwarded_shared_memory_is_not_snapshotted"_test = [] {
        check_forwarding({.shared = true}, false);
    };
    "every_outer_call_site_must_prove_disjoint_storage"_test = [] {
        check_forwarding({.additional_alias_call = true}, false);
    };
    "unknown_callee_effects_are_conservative"_test = [] {
        check_forwarding({.external = true}, false);
    };
    "recursive_reference_forwarding_is_conservative"_test = [] {
        check_forwarding({.recursive = true}, false);
    };
    return 0;
}
