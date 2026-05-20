#include <luisa/xir/passes/inline.h>
#include <luisa/xir/passes/call_graph.h>
#include <luisa/xir/builder.h>
#include <luisa/core/logging.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/undefined.h>
#include <luisa/xir/function.h>

namespace luisa::compute::xir {

namespace detail {

class InlineValueResolver final : public InstructionCloneValueResolver {
    luisa::unordered_map<const Value *, Value *> _map;
public:
    void emplace(const Value *from, Value *to) noexcept { _map.emplace(from, to); }
    [[nodiscard]] Value *resolve(const Value *value) noexcept override {
        if (value == nullptr) return nullptr;
        switch (value->derived_value_tag()) {
            case DerivedValueTag::UNDEFINED:
            case DerivedValueTag::FUNCTION:
            case DerivedValueTag::CONSTANT:
            case DerivedValueTag::SPECIAL_REGISTER:
                return const_cast<Value *>(value);
            default: break;
        }
        auto it = _map.find(value);
        LUISA_ASSERT(it != _map.end(), "Inline: unresolved value.");
        return it->second;
    }
};

[[nodiscard]] static size_t count_instructions(FunctionDefinition *def) noexcept {
    size_t n = 0;
    if (def) def->traverse_instructions([&](const Instruction *) noexcept { ++n; });
    return n;
}

[[nodiscard]] static bool inline_call(CallInst *call, Function *callee) noexcept {
    auto callee_def = callee->definition();
    if (!callee_def) return false;
    auto caller_func = call->parent_function();
    if (!caller_func) return false;
    auto caller_def = caller_func->definition();
    if (!caller_def) return false;

    auto module = caller_func->parent_module();
    XIRBuilder builder;
    InlineValueResolver resolver;

    // Map callee args -> call args
    {
        size_t i = 0;
        for (auto arg : callee->arguments()) {
            resolver.emplace(arg, i < call->argument_count()
                                      ? call->argument(i)
                                      : static_cast<Value *>(module->create_undefined(arg->type())));
            ++i;
        }
    }

    // Collect callee blocks and create mapped blocks in caller
    luisa::vector<BasicBlock *> callee_blocks;
    callee_def->traverse_basic_blocks([&](BasicBlock *bb) noexcept { callee_blocks.push_back(bb); });

    luisa::unordered_map<BasicBlock *, BasicBlock *> block_map;
    luisa::vector<BasicBlock *> new_blocks;
    for (auto bb : callee_blocks) {
        auto nb = caller_func->create_basic_block();
        block_map[bb] = nb;
        new_blocks.push_back(nb);
        resolver.emplace(bb, nb);
    }

    // Create single-exit merge block and return value alloca
    auto merge_bb = caller_func->create_basic_block();
    Instruction *ret_alloca = nullptr;
    if (call->type()) {
        builder.set_insertion_point(call);
        ret_alloca = builder.alloca_local(call->type());
    }

    // Clone instructions from callee into new blocks
    for (size_t i = 0; i < callee_blocks.size(); ++i) {
        builder.set_insertion_point(new_blocks[i]);
        for (auto inst : callee_blocks[i]->instructions()) {
            if (inst->isa<ReturnInst>()) {
                auto r = static_cast<ReturnInst *>(inst);
                if (ret_alloca && r->operand_count() > 0) {
                    auto val = resolver.resolve(r->operand(0));
                    builder.store(ret_alloca, val);
                }
                builder.br(merge_bb);
            } else {
                auto c = inst->clone_with_metadata(builder, resolver);
                LUISA_ASSERT(c, "Inline: clone failed.");
                resolver.emplace(inst, c);
            }
        }
    }

    // Wire caller: split the call block
    auto call_block = call->parent_block();
    auto entry_block = block_map[callee_def->body_block()];

    // Collect instructions after the call
    luisa::vector<Instruction *> to_move;
    bool past = false;
    for (auto inst : call_block->instructions()) {
        if (inst == call) { past = true; continue; }
        if (past) to_move.push_back(inst);
    }

    // Load return value in merge block
    if (ret_alloca) {
        builder.set_insertion_point(merge_bb);
        auto loaded = builder.load(call->type(), ret_alloca);
        call->replace_all_uses_with(loaded);
    }

    // Remove the call
    call->remove_self();

    // Move post-call instructions to merge_bb
    builder.set_insertion_point(merge_bb);
    for (auto inst : to_move) {
        if (!inst->is_terminator()) {
            auto m = inst->remove_self();
            builder.append(std::move(m));
        }
    }

    // Move terminator from call_block to merge_bb
    if (call_block->is_terminated()) {
        auto m = call_block->terminator()->remove_self();
        builder.set_insertion_point(merge_bb);
        if (merge_bb->is_terminated()) merge_bb->terminator()->remove_self();
        builder.append(std::move(m));
    }

    // Branch from call_block to inlined entry
    builder.set_insertion_point(call_block);
    builder.br(entry_block);

    return true;
}

static void run(Module *module, InlineInfo &info) noexcept {
    // Early exit if no callables
    bool has_callables = false;
    for (auto f : module->function_list()) {
        if (f->derived_function_tag() == DerivedFunctionTag::CALLABLE) {
            has_callables = true;
            break;
        }
    }
    if (!has_callables) return;

    auto cg = compute_call_graph(module);

    // Collect callables (safe iteration before modification)
    luisa::vector<Function *> callables;
    for (auto f : module->function_list())
        if (f->derived_function_tag() == DerivedFunctionTag::CALLABLE)
            callables.push_back(f);

    // Defer removal to after iteration to avoid corrupting the list
    luisa::vector<Function *> to_remove;
    for (auto callee : callables) {
        auto def = callee->definition();
        if (!def) continue;
        auto edges = cg.call_edges(def);
        if (edges.empty()) continue;

        size_t n = edges.size();
        bool doit = (n == 1) || (n <= 3 && count_instructions(def) <= 50);
        if (!doit) continue;

        for (auto call : edges)
            if (inline_call(call, callee))
                info.inlined_call_count++;

        to_remove.push_back(callee);
    }
    for (auto callee : to_remove) {
        callee->remove_self();
        info.removed_callable_count++;
    }
}

}// namespace detail

InlineInfo inline_pass_run_on_module(Module *module) noexcept {
    InlineInfo info;
    detail::run(module, info);
    return info;
}

InlineInfo inline_all_pass_run_on_module(Module *module) noexcept {
    InlineInfo info;
    if (!module) return info;
    for (;;) {
        bool has_callables = false;
        for (auto f : module->function_list()) {
            if (f->derived_function_tag() == DerivedFunctionTag::CALLABLE) {
                has_callables = true;
                break;
            }
        }
        if (!has_callables) break;
        auto cg = compute_call_graph(module);
        luisa::vector<Function *> callables;
        for (auto f : module->function_list())
            if (f->derived_function_tag() == DerivedFunctionTag::CALLABLE)
                callables.push_back(f);
        bool progress = false;
        for (auto callee : callables) {
            auto def = callee->definition();
            if (!def) continue;
            auto n = detail::count_instructions(def);
            auto edges = cg.call_edges(def);
            if (n * edges.size() > 100000u) continue;
            for (auto call : edges) {
                if (detail::inline_call(call, callee)) {
                    info.inlined_call_count++;
                    progress = true;
                }
            }
        }
        if (!progress) break;
    }
    return info;
}

}// namespace luisa::compute::xir
