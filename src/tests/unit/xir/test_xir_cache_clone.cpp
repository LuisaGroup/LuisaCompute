#include "ut/ut.hpp"
#include "ast_xir_cache_clone.h"

#include <luisa/ast/type_registry.h>
#include <luisa/dsl/rtx/ray_query.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/metadata/comment.h>
#include <luisa/xir/module.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] const xir::Function *find_function(const Module &module, luisa::string_view name) noexcept {
    for (auto *function : module.function_list()) {
        if (function->name() == name) { return function; }
    }
    return nullptr;
}

}// namespace

int main() {
    "cache_clone_imports_only_the_owned_function_closure"_test = [] {
        Module target;
        Module independent;
        FunctionDefinition *imported = nullptr;
        FunctionDefinition *other = nullptr;
        {
            Module source;
            auto *uint_type = Type::of<uint>();
            auto *leaf = source.create_callable(uint_type);
            leaf->set_name("leaf");
            auto *parameter = leaf->create_value_argument(uint_type);
            parameter->set_name("leaf_parameter");
            auto *entry = leaf->create_body_block();
            auto *consumer = leaf->create_basic_block();
            auto *producer = leaf->create_basic_block();
            consumer->set_name("consumer");
            producer->set_name("producer");

            // Physical block order is entry, consumer, producer; control flow
            // is entry, producer, consumer. The consumer uses a later value.
            XIRBuilder b;
            b.set_insertion_point(entry);
            b.br(producer);
            b.set_insertion_point(producer);
            const uint literal = 17u;
            auto *constant = source.create_constant(uint_type, &literal);
            constant->set_name("cached_constant");
            auto *sum = b.call(uint_type, ArithmeticOp::BINARY_ADD, {parameter, constant});
            sum->add_comment("first");
            sum->add_comment("second");
            b.br(consumer);
            b.set_insertion_point(consumer);
            b.return_(sum);

            auto *callback = source.create_callable(nullptr);
            callback->set_name("callback");
            callback->create_reference_argument(Type::of<RayQueryAll>());
            b.set_insertion_point(callback->create_body_block());
            auto *undef = source.create_undefined(uint_type);
            b.call(uint_type, leaf, {undef});
            b.return_void();

            auto *root = source.create_kernel();
            root->set_name("root");
            root->set_block_size(make_uint3(8u, 4u, 1u));
            b.set_insertion_point(root->create_body_block());
            auto *query = b.alloca_local(Type::of<RayQueryAll>());
            auto *thread_id = source.create_thread_id();
            auto *extract = b.call(uint_type, ArithmeticOp::EXTRACT, {thread_id, source.create_constant_zero(uint_type)});
            b.call(uint_type, leaf, {extract});
            b.ray_query_pipeline(query, callback, callback);
            b.return_void();

            auto *unreachable = source.create_callable(nullptr);
            unreachable->set_name("unreachable");
            b.set_insertion_point(unreachable->create_body_block());
            b.return_void();
            expect(xir_verify_module(&source).succeeded());

            AstXirCacheCloner cloner{target};
            imported = cloner.import(root);
            expect(cloner.import(root) == imported);
            expect(cloner.import(leaf) == find_function(target, "leaf"));
            expect(cloner.import(callback) == find_function(target, "callback"));
            AstXirCacheCloner independent_cloner{independent};
            other = independent_cloner.import(root);
            expect(other != imported);

            auto *copied_leaf = find_function(target, "leaf");
            expect(copied_leaf != leaf);
            expect(constant->use_list().count_size() == 1_u);
            expect(sum->use_list().count_size() == 1_u);
            auto *copied_constant = target.constant_list().front();
            if (copied_constant->name() != "cached_constant") {
                copied_constant = copied_constant->next();
            }
            expect(copied_constant != constant);
            expect(copied_constant->as<uint>() == 17_u);
            expect(copied_constant->name() == "cached_constant");
            imported->set_name("changed_root");
            expect(root->name() == "root");
            expect(other->name() == "root");
        }

        // The clone remains valid after its entire source module is gone.
        expect(xir_verify_module(&target).succeeded());
        expect(xir_verify_module(&independent).succeeded());
        expect(target.function_list().count_size() == 3_u);
        expect(find_function(target, "unreachable") == nullptr);
        expect(target.constant_list().count_size() == 2_u);
        expect(target.undefined_list().count_size() == 1_u);
        expect(target.special_register_list().count_size() == 1_u);
        expect(static_cast<KernelFunction *>(imported)->block_size().x == 8_u);
        expect(static_cast<KernelFunction *>(imported)->block_size().y == 4_u);

        auto *leaf = find_function(target, "leaf")->definition();
        auto *consumer = leaf->body_block()->next();
        auto *producer = consumer->next();
        auto *sum = producer->instructions().front();
        auto *entry_branch = static_cast<const BranchInst *>(leaf->body_block()->instructions().front());
        auto *ret = static_cast<const ReturnInst *>(consumer->instructions().front());
        expect(entry_branch->target_block() == producer);
        expect(ret->return_value() == sum);
        expect(leaf->arguments().front()->name() == "leaf_parameter");
        auto *comment = static_cast<const CommentMD *>(sum->metadata_list().front());
        expect(comment->comment() == "second");
        expect(static_cast<const CommentMD *>(comment->next())->comment() == "first");

        auto *callback = find_function(target, "callback");
        auto *callback_call = static_cast<const CallInst *>(callback->definition()->body_block()->instructions().front());
        expect(callback_call->callee() == leaf);
        expect(callback_call->argument(0u) == target.undefined_list().front());
        auto *pipeline = static_cast<const RayQueryPipelineInst *>(imported->body_block()->instructions().back()->prev());
        expect(pipeline->on_surface_function() == callback);
        expect(pipeline->on_procedural_function() == callback);
    };
}
