// Resource descriptor identity across ordinary calls and ray-query callbacks.
#include "ut/ut.hpp"

#include <luisa/core/logging.h>
#include <luisa/dsl/rtx/ray_query.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/resource_origin.h>
#include <luisa/xir/verifier.h>

#include <array>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "resource_origins_follow_interleaved_call_and_both_callback_edges"_test = [] {
        Module module;
        auto *buffer_type = Type::buffer(Type::of<uint>());
        auto *query_type = Type::of<RayQueryAll>();
        auto *handler = module.create_callable(nullptr);
        auto *query_arg = handler->create_reference_argument(query_type);
        auto *first_scalar = handler->create_value_argument(Type::of<uint>());
        auto *handler_b = handler->create_resource_argument(buffer_type);
        handler->create_value_argument(Type::of<uint>());
        auto *handler_a = handler->create_resource_argument(buffer_type);
        XIRBuilder builder;
        builder.set_insertion_point(handler->create_body_block());
        auto *read = builder.call(Type::of<uint>(), ResourceReadOp::BUFFER_READ, {handler_b, first_scalar});
        builder.call(ResourceWriteOp::BUFFER_WRITE, {handler_a, first_scalar, read});
        builder.return_void();
        auto *relay = module.create_callable(nullptr);
        auto *relay_a = relay->create_resource_argument(buffer_type);
        auto *scalar = relay->create_value_argument(Type::of<uint>());
        auto *relay_b = relay->create_resource_argument(buffer_type);
        builder.set_insertion_point(relay->create_body_block());
        auto *query = builder.alloca_(query_type, AllocaOp::LOCAL);
        std::array<Value *, 4u> captures{scalar, relay_b, scalar, relay_a};
        builder.ray_query_pipeline(query, handler, handler, luisa::span{captures});
        builder.return_void();
        auto *kernel = module.create_kernel();
        auto *kernel_scalar = kernel->create_value_argument(Type::of<uint>());
        auto *root_a = kernel->create_resource_argument(buffer_type);
        kernel->create_value_argument(Type::of<uint>());
        auto *root_b = kernel->create_resource_argument(buffer_type);
        builder.set_insertion_point(kernel->create_body_block());
        builder.call(nullptr, relay, {root_a, kernel_scalar, root_b});
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto origins = analyze_unique_resource_origins(&module);
        expect(origins.contains(root_a) && origins.at(root_a) == root_a);
        expect(origins.contains(root_b) && origins.at(root_b) == root_b);
        expect(origins.contains(relay_a) && origins.at(relay_a) == root_a);
        expect(origins.contains(relay_b) && origins.at(relay_b) == root_b);
        expect(origins.contains(handler_a) && origins.at(handler_a) == root_a);
        expect(origins.contains(handler_b) && origins.at(handler_b) == root_b)
            << "callback formal i + 1 must map capture i, including interleaved scalars";
        expect(!origins.contains(query_arg) && !origins.contains(first_scalar) && !origins.contains(kernel_scalar));
        expect(origins.size() == 6u);
        expect(xir_verify_module(&module).succeeded());
    };

    "resource_origins_reject_conflicting_roots_and_owned_unreachable_edges"_test = [] {
        for (auto unreachable : {false, true}) {
            Module module;
            auto *type = Type::buffer(Type::of<uint>());
            auto *callee = module.create_callable(nullptr);
            auto *formal = callee->create_resource_argument(type);
            XIRBuilder builder;
            builder.set_insertion_point(callee->create_body_block());
            builder.return_void();
            auto *kernel = module.create_kernel();
            auto *a = kernel->create_resource_argument(type);
            auto *b = kernel->create_resource_argument(type);
            builder.set_insertion_point(kernel->create_body_block());
            builder.call(nullptr, callee, {a});
            if (unreachable) {
                builder.return_void();
                builder.set_insertion_point(kernel->create_basic_block());
            }
            builder.call(nullptr, callee, {b});
            builder.return_void();
            auto origins = analyze_unique_resource_origins(&module);
            expect(!origins.contains(formal)) << "all owned incoming edges must agree on one descriptor";
            expect(origins.contains(a) && origins.contains(b));
        }
    };

    "resource_origins_merge_ordinary_and_pipeline_calls_conservatively"_test = [] {
        for (auto ordinary_call : {false, true}) {
            Module module;
            auto *type = Type::buffer(Type::of<uint>());
            auto *query_type = Type::of<RayQueryAll>();
            auto *handler = module.create_callable(nullptr);
            handler->create_reference_argument(query_type);
            auto *formal = handler->create_resource_argument(type);
            XIRBuilder builder;
            builder.set_insertion_point(handler->create_body_block());
            builder.return_void();
            auto *kernel = module.create_kernel();
            auto *a = kernel->create_resource_argument(type);
            auto *b = kernel->create_resource_argument(type);
            builder.set_insertion_point(kernel->create_body_block());
            auto *query = builder.alloca_(query_type, AllocaOp::LOCAL);
            if (ordinary_call) {
                builder.call(nullptr, handler, {query, a});
            } else {
                std::array<Value *, 1u> captures{a};
                builder.ray_query_pipeline(query, handler, handler, luisa::span{captures});
            }
            std::array<Value *, 1u> captures{b};
            builder.ray_query_pipeline(query, handler, handler, luisa::span{captures});
            builder.return_void();
            auto verification = xir_verify_module(&module);
            for (const auto &error : verification.errors) {
                LUISA_WARNING("Resource origin fixture verification failed (ordinary_call={}): {}.", ordinary_call, error.message);
            }
            expect(verification.succeeded());
            expect(!analyze_unique_resource_origins(&module).contains(formal))
                << "a pipeline edge cannot override another conflicting pipeline or ordinary call";
        }
    };

    "resource_origins_do_not_seed_recursive_or_unrooted_descriptors"_test = [] {
        for (auto mutual : {false, true}) {
            Module module;
            auto *type = Type::buffer(Type::of<uint>());
            auto *recursive = module.create_callable(nullptr);
            auto *cycle_arg = recursive->create_resource_argument(type);
            XIRBuilder builder;
            auto *peer = mutual ? module.create_callable(nullptr) : recursive;
            auto *peer_arg = mutual ? peer->create_resource_argument(type) : cycle_arg;
            if (mutual) {
                builder.set_insertion_point(peer->create_body_block());
                builder.call(nullptr, recursive, {peer_arg});
                builder.return_void();
            }
            builder.set_insertion_point(recursive->create_body_block());
            builder.call(nullptr, peer, {cycle_arg});
            builder.return_void();
            auto *leaf = module.create_callable(nullptr);
            auto *leaf_arg = leaf->create_resource_argument(type);
            builder.set_insertion_point(leaf->create_body_block());
            builder.return_void();
            auto *orphan = module.create_callable(nullptr);
            auto *orphan_arg = orphan->create_resource_argument(type);
            builder.set_insertion_point(orphan->create_body_block());
            builder.call(nullptr, leaf, {orphan_arg});
            builder.return_void();
            auto *kernel = module.create_kernel();
            auto *root = kernel->create_resource_argument(type);
            builder.set_insertion_point(kernel->create_body_block());
            builder.call(nullptr, recursive, {root});
            builder.call(nullptr, leaf, {root});
            builder.return_void();
            auto origins = analyze_unique_resource_origins(&module);
            expect(!origins.contains(cycle_arg)) << "a kernel seed does not resolve the recursive descriptor dependency";
            expect(!origins.contains(peer_arg));
            expect(!origins.contains(orphan_arg));
            expect(!origins.contains(leaf_arg)) << "the rooted call does not discard the unrooted incoming edge";
            expect(origins.size() == 1u && origins.contains(root));
        }
    };

    "resource_origins_reject_malformed_unknown_or_escaped_inputs"_test = [] {
        for (auto mode : {0u, 1u, 2u}) {
            Module module;
            auto *type = Type::buffer(Type::of<uint>());
            auto *callee = module.create_callable(nullptr);
            auto *formal = callee->create_resource_argument(type);
            XIRBuilder builder;
            builder.set_insertion_point(callee->create_body_block());
            builder.return_void();
            auto *kernel = module.create_kernel();
            auto *root = kernel->create_resource_argument(type);
            auto *wrong_type = kernel->create_resource_argument(Type::buffer(Type::of<float>()));
            auto *consumer = module.create_callable(nullptr);
            consumer->create_value_argument(Type::of<uint>());
            builder.set_insertion_point(consumer->create_body_block());
            builder.return_void();
            builder.set_insertion_point(kernel->create_body_block());
            builder.call(nullptr, callee, {root});
            // These are deliberately unsupported/malformed IR, not verifier input.
            if (mode == 0u) {
                builder.call(nullptr, callee, {wrong_type});
            } else if (mode == 1u) {
                builder.call(nullptr, callee, {module.create_undefined(type)});
            } else {
                builder.call(nullptr, consumer, {callee});
            }
            builder.return_void();
            expect(!analyze_unique_resource_origins(&module).contains(formal));
            expect(analyze_unique_resource_origins(nullptr).empty());
        }
    };
}
