#include <luisa/core/stl/deque.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_cfg_distill.h>

namespace luisa::compute::xir {

namespace detail {

[[nodiscard]] static CoroCfgDistillResult distill_function(FunctionDefinition *def) noexcept {

    CoroCfgDistillResult result;

    luisa::unordered_set<BasicBlock *> reachable;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        reachable.insert(bb);
    });

    luisa::unordered_map<uint32_t, BasicBlock *> token_to_resume;
    for (auto *bb : def->basic_blocks()) {
        for (auto *inst : bb->instructions()) {
            if (inst->derived_instruction_tag() == DerivedInstructionTag::CORO_RESUME) {
                auto *r = static_cast<CoroResumeInst *>(inst);
                token_to_resume[r->token()] = bb;
            }
        }
    }

    // assign blocks to scopes via BFS
    luisa::unordered_map<BasicBlock *, int> block_to_scope;
    luisa::deque<std::pair<BasicBlock *, int>> worklist;

    auto *body = def->body_block();
    worklist.emplace_back(body, 0);
    block_to_scope[body] = 0;

    while (!worklist.empty()) {
        auto [bb, sid] = worklist.front();
        worklist.pop_front();

        // ensure scope vector has enough entries
        while (result.scopes.size() <= static_cast<size_t>(sid)) {
            result.scopes.emplace_back();
            result.scopes.back().scope_id = static_cast<int>(result.scopes.size()) - 1;
        }

        result.scopes[sid].blocks.emplace_back(bb);

        auto *term = bb->terminator();

        // check for CoroSuspendInst
        if (term != nullptr && term->derived_instruction_tag() == DerivedInstructionTag::CORO_SUSPEND) {
            auto *s = static_cast<CoroSuspendInst *>(term);
            result.scopes[sid].suspend_token = s->token();
            result.scopes[sid].suspend_name = s->name();

            // start new scope from matching resume block
            if (auto it = token_to_resume.find(s->token()); it != token_to_resume.end()) {
                auto *resume_bb = it->second;
                if (!block_to_scope.contains(resume_bb)) {
                    auto new_sid = static_cast<int>(result.scopes.size());
                    block_to_scope[resume_bb] = new_sid;
                    worklist.emplace_back(resume_bb, new_sid);
                }
            }
            continue;
        }

        // check for CoroTerminateInst
        if (term != nullptr && term->derived_instruction_tag() == DerivedInstructionTag::CORO_TERMINATE) {
            result.scopes[sid].is_terminal = true;
            continue;
        }

        // follow regular CFG successors
        bb->traverse_successors(true, [&](BasicBlock *succ) noexcept {
            if (block_to_scope.contains(succ)) { return; }
            // check if successor starts a new scope (has CoroResumeInst at beginning)
            auto *first = succ->instructions().front();
            if (first->derived_instruction_tag() == DerivedInstructionTag::CORO_RESUME) {
                auto new_sid = static_cast<int>(result.scopes.size());
                block_to_scope[succ] = new_sid;
                worklist.emplace_back(succ, new_sid);
            } else {
                block_to_scope[succ] = sid;
                worklist.emplace_back(succ, sid);
            }
        });
    }

    // compute edges
    result.edges.resize(result.scopes.size());
    for (size_t i = 0; i < result.scopes.size(); ++i) {
        luisa::unordered_set<size_t> edge_set;
        for (auto *bb : result.scopes[i].blocks) {
            bb->traverse_successors(true, [&](BasicBlock *succ) noexcept {
                if (auto it = block_to_scope.find(succ); it != block_to_scope.end()) {
                    auto j = static_cast<size_t>(it->second);
                    if (j != i) { edge_set.insert(j); }
                }
            });
        }
        // also add implicit edge from suspend to resume
        if (result.scopes[i].suspend_token.has_value()) {
            auto token = *result.scopes[i].suspend_token;
            if (auto it = token_to_resume.find(token); it != token_to_resume.end()) {
                if (auto sit = block_to_scope.find(it->second); sit != block_to_scope.end()) {
                    edge_set.insert(static_cast<size_t>(sit->second));
                }
            }
        }
        result.edges[i].assign(edge_set.begin(), edge_set.end());
    }

    return result;
}

}// namespace detail

CoroCfgDistillResult coro_cfg_distill_pass_run_on_function(Function *f) noexcept {
    auto *def = f->definition();
    if (def == nullptr) { return {}; }
    return detail::distill_function(def);
}

size_t coro_cfg_distill_pass_run_on_module(Module *m) noexcept {
    size_t count = 0u;
    for (auto *f : m->function_list()) {
        if (f->is_definition()) {
            static_cast<void>(detail::distill_function(static_cast<FunctionDefinition *>(f)));
            ++count;
        }
    }
    return count;
}

}// namespace luisa::compute::xir
