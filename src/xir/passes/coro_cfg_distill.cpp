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

    luisa::vector<luisa::unordered_set<BasicBlock *>> scope_visited;
    luisa::deque<std::pair<BasicBlock *, int>> worklist;
    luisa::unordered_set<uint32_t> started_tokens;

    auto *body = def->body_block();
    worklist.emplace_back(body, 0);

    while (!worklist.empty()) {
        auto [bb, sid] = worklist.front();
        worklist.pop_front();

        while (result.scopes.size() <= static_cast<size_t>(sid)) {
            result.scopes.emplace_back();
            result.scopes.back().scope_id = static_cast<int>(result.scopes.size()) - 1;
        }
        while (scope_visited.size() <= static_cast<size_t>(sid)) {
            scope_visited.emplace_back();
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
                bool is_self_loop = !result.scopes[sid].blocks.empty() &&
                                   result.scopes[sid].blocks.front() == resume_bb;
                if (!scope_visited[sid].contains(resume_bb) && !is_self_loop && !started_tokens.contains(s->token())) {
                scope_visited[sid].insert(resume_bb);
                started_tokens.insert(s->token());
                auto new_sid = static_cast<int>(result.scopes.size());
                result.scopes.emplace_back();
                result.scopes.back().scope_id = new_sid;
                result.scopes.back().trigger_token = s->token();
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
            if (scope_visited[sid].contains(succ)) { return; }
            scope_visited[sid].insert(succ);
            auto *first = succ->instructions().front();
            if (first->derived_instruction_tag() == DerivedInstructionTag::CORO_RESUME) {
                auto *r = static_cast<CoroResumeInst *>(first);
                if (started_tokens.contains(r->token())) {
                    worklist.emplace_back(succ, sid);
                } else {
                    started_tokens.insert(r->token());
                    auto new_sid = static_cast<int>(result.scopes.size());
                    result.scopes.emplace_back();
                    result.scopes.back().scope_id = new_sid;
                    result.scopes.back().trigger_token = r->token();
                    worklist.emplace_back(succ, new_sid);
                }
            } else {
                worklist.emplace_back(succ, sid);
            }
        });
    }

    // compute edges — rebuild block_to_scope mapping (last scope wins)
    luisa::unordered_map<BasicBlock *, int> block_to_scope;
    for (size_t i = 0; i < result.scopes.size(); ++i) {
        for (auto *bb : result.scopes[i].blocks) {
            block_to_scope[bb] = static_cast<int>(i);
        }
    }
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
