// Legacy experimental implementation intentionally disabled.
// The active phase-1 lowering lives in canonicalize_control_flow_pass.cpp.
#if 0
#include <luisa/xir/passes/Canonicalize_Control_Flow.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
namespace luisa::compute::xir{

    struct LowerLoops{

          
        luisa::unordered_map<Value*, Value*> value_old_new_map;

        void lowerloops(LoopInst* loop);

        void transform_block(BasicBlock* block);




    };


    /*void  LowerLoops::lowerloops(LoopInst* loop){

        auto prepare = loop->prepare_block();
        auto body = loop->body_block();
        auto update = loop->update_block();
        auto merge = loop->merge_block();

        XIRBuilder builder;
        builder.set_insertion_point(loop->parent_block());
        auto simple_loop_after_lowering= builder.simple_loop();

        auto simple_loop_after_lowering_body = simple_loop_after_lowering->create_body_block();
        auto simple_loop_after_lowering_merge = simple_loop_after_lowering->create_merge_block();
        builder.set_insertion_point(simple_loop_after_lowering_body);
        //prepare block
        {
        auto move_instr = [&](Instruction* inst){

            auto block = inst->parent_block();

            if(inst->derived_instruction_tag() == DerivedInstructionTag::IF && inst == block->instructions().back()){
                auto if_inst_prepare = static_cast<IfInst*>(inst);
                auto cond = if_inst_prepare->condition();
                auto if_inst_simple_loop = builder.if_(cond);
                auto true_block = if_inst_simple_loop->create_true_block();
                auto false_block = if_inst_simple_loop->create_false_block();
                auto merge_block = if_inst_simple_loop->create_merge_block();

                builder.set_insertion_point(false_block);
                builder.break_(simple_loop_after_lowering_merge);
                builder.set_insertion_point(simple_loop_after_lowering_merge);
                return ;
            }

            builder.append(inst->remove_self());
        };
    
        prepare->traverse_instructions(move_instr);
        }    
        auto loop_break = builder.alloca_local(Type::of<bool>());
        auto false_value = builder.insertion_point()->parent_block()->parent_module()->create_constant_zero(Type::of<bool>());
        builder.store(loop_break, false_value);

        
        //body block
        {
            

            builder.set_insertion_point(simple_loop_after_lowering_body);

            auto move_instr = [&](Instruction* inst){

                auto block = inst->parent_block();

                if(inst->is_terminator()){
                    switch (inst->derived_instruction_tag())
                    {
                        case DerivedInstructionTag::BREAK:{
                            auto true_value = inst->parent_block()->parent_module()->create_constant_one(Type::of<bool>());
                            builder.store(loop_break, true_value);
                            builder.break_(simple_loop_after_lowering_merge);
                            builder.set_insertion_point(simple_loop_after_lowering_merge);
                            return;
                        }

                        case DerivedInstructionTag::CONTINUE:{
                            builder.break_(simple_loop_after_lowering_body);
                            builder.set_insertion_point(simple_loop_after_lowering_merge);
                            return;
                        }
                    }
                }
                builder.append(inst->remove_self());
            };  
            
            body->traverse_instructions(move_instr);
            builder.break_(simple_loop_after_lowering_merge);
            builder.set_insertion_point(simple_loop_after_lowering_merge);
        }
            auto if_loop_break =  builder.if_(loop_break);
            //if (loop_break) break;
            {
                auto true_block =if_loop_break->create_true_block();
                builder.set_insertion_point(true_block);
                builder.break_(simple_loop_after_lowering_merge);
                if_loop_break->set_merge_block(if_loop_break->parent_block());

                builder.set_insertion_point(if_loop_break->parent_block());
            }

            // update block
            {
                auto move_instr = [&](Instruction* inst) {
                    builder.append(inst->remove_self());
                };

                update->traverse_instructions(move_instr);
            };

            //merge block

            simple_loop_after_lowering->set_merge_block(merge);

}
#endif

#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/Canonicalize_Control_Flow.h>

namespace luisa::compute::xir {

namespace detail {

struct LoopCollector {
    luisa::unordered_set<BasicBlock *> visited_blocks;
    luisa::vector<LoopInst *> loops;

    void collect_block(BasicBlock *block) noexcept {
        if (block == nullptr || !visited_blocks.emplace(block).second) {
            return;
        }
        for (auto inst : block->instructions()) {
            switch (inst->derived_instruction_tag()) {
                case DerivedInstructionTag::IF: {
                    auto if_inst = static_cast<IfInst *>(inst);
                    collect_block(if_inst->true_block());
                    collect_block(if_inst->false_block());
                    collect_block(if_inst->merge_block());
                    break;
                }
                case DerivedInstructionTag::SWITCH: {
                    auto switch_inst = static_cast<SwitchInst *>(inst);
                    for (auto i = 0u; i < switch_inst->case_count(); i++) {
                        collect_block(switch_inst->case_block(i));
                    }
                    collect_block(switch_inst->default_block());
                    collect_block(switch_inst->merge_block());
                    break;
                }
                case DerivedInstructionTag::LOOP: {
                    auto loop_inst = static_cast<LoopInst *>(inst);
                    collect_block(loop_inst->prepare_block());
                    collect_block(loop_inst->body_block());
                    collect_block(loop_inst->update_block());
                    collect_block(loop_inst->merge_block());
                    loops.emplace_back(loop_inst);
                    break;
                }
                case DerivedInstructionTag::SIMPLE_LOOP: {
                    auto simple_loop = static_cast<SimpleLoopInst *>(inst);
                    collect_block(simple_loop->body_block());
                    collect_block(simple_loop->merge_block());
                    break;
                }
                case DerivedInstructionTag::CONDITIONAL_BRANCH: {
                    auto branch = static_cast<ConditionalBranchInst *>(inst);
                    collect_block(branch->true_block());
                    collect_block(branch->false_block());
                    break;
                }
                case DerivedInstructionTag::BRANCH:
                case DerivedInstructionTag::BREAK:
                case DerivedInstructionTag::CONTINUE: {
                    auto branch = static_cast<BranchTerminatorInstruction *>(inst);
                    collect_block(branch->target_block());
                    break;
                }
                default: break;
            }
        }
    }
};

[[nodiscard]] static bool is_supported_ast2xir_for_loop_shape(const LoopInst *loop) noexcept {
    if (loop == nullptr) { return false; }
    auto prepare = loop->prepare_block();
    auto body = loop->body_block();
    auto update = loop->update_block();
    auto merge = loop->merge_block();
    if (prepare == nullptr || body == nullptr || update == nullptr || merge == nullptr) {
        return false;
    }
    if (!prepare->is_terminated() || !update->is_terminated()) {
        return false;
    }
    auto prepare_terminator = prepare->terminator();
    if (!prepare_terminator->isa<ConditionalBranchInst>()) {
        return false;
    }
    auto cond_branch = static_cast<const ConditionalBranchInst *>(prepare_terminator);
    if (cond_branch->condition() == nullptr ||
        cond_branch->true_block() != body ||
        cond_branch->false_block() != merge) {
        return false;
    }
    auto update_terminator = update->terminator();
    if (!update_terminator->isa<BranchInst>()) {
        return false;
    }
    return static_cast<const BranchInst *>(update_terminator)->target_block() == prepare;
}
static void lower_loop_to_simple_loop(LoopInst *loop, Canonicalize_Control_Flow_Info &info) noexcept {
    if (!is_supported_ast2xir_for_loop_shape(loop)) {
        info.skipped_loop_count++;
        return;
    }

    auto old_prepare = loop->prepare_block();
    auto old_body = loop->body_block();
    auto old_update = loop->update_block();
    auto old_merge = loop->merge_block();
    auto cond_branch = static_cast<ConditionalBranchInst *>(old_prepare->terminator());
    auto cond = cond_branch->condition();

    LUISA_DEBUG_ASSERT(cond != nullptr, "Loop prepare condition must not be null.");

    XIRBuilder builder;
    builder.set_insertion_point(loop->prev());
    auto simple_loop = builder.simple_loop();
    auto simple_body = simple_loop->create_body_block();
    simple_loop->set_merge_block(old_merge);

    builder.set_insertion_point(simple_body);
    while (!old_prepare->instructions().empty()) {
        auto inst = old_prepare->instructions().front();
        if (inst->is_terminator()) {
            break;
        }
        builder.append(inst->remove_self());
    }

    auto guard = builder.if_(cond);
    guard->set_true_target(old_body);
    auto false_block = guard->create_false_block();
    guard->set_merge_block(old_update);

    builder.set_insertion_point(false_block);
    builder.break_(old_merge);

    auto update_branch = static_cast<BranchInst *>(old_update->terminator());
    update_branch->set_target_block(simple_body);

    loop->remove_self();
    info.lowered_loop_count++;
}

static Canonicalize_Control_Flow_Info run_on_function(Function *function) noexcept {
    Canonicalize_Control_Flow_Info info;
    if (auto definition = function->definition()) {
        LoopCollector collector;
        collector.collect_block(definition->body_block());
        for (auto loop : collector.loops) {
            lower_loop_to_simple_loop(loop, info);
        }
    }
    return info;
}

}// namespace detail

LUISA_XIR_API Canonicalize_Control_Flow_Info Canoinicalize_Control_Flow_pass_run_on_Function(Function *func) {
    return detail::run_on_function(func);
}

LUISA_XIR_API Canonicalize_Control_Flow_Info Canoinicalize_Control_Flow_pass_run_on_Module(Module *module) {
    Canonicalize_Control_Flow_Info info;
    for (auto func : module->function_list()) {
        auto function_info = detail::run_on_function(func);
        info.lowered_loop_count += function_info.lowered_loop_count;
        info.skipped_loop_count += function_info.skipped_loop_count;
    }
    return info;
}

}// namespace luisa::compute::xir
#if 0
           */


           /*
           不进行transform区域
           Loop{
           prepare{}
           body{}
           update{}
           merge{}
           }

              ->       不进行transform区域
                       simple_loop{
                        body{
                        prepare
                        if(!cond) break;
                        break_flag;
                        do{

                            body{}
                            // continue => {break;}
                            // break => {break_flag = true ; break ;} 

                        }while(false)

                        update{}
                        }
                        merge{}
                       }



            


           
           */


    struct value_old_to_new_resolver : public InstructionCloneValueResolver{
        luisa::unordered_map<Value*, Value*>* map;
        explicit value_old_to_new_resolver(luisa::unordered_map<Value*, Value*> *map) noexcept : map(map){};
        [[nodiscard]] virtual Value *resolve(const Value *value) noexcept override{
            if(map->contains(static_cast<const Value*>(value))){
                return (*map)[static_cast<const Value*>(value)];
            }
            return const_cast<Value*>(value);
        };
    };

    void LowerLoops::lowerloops(LoopInst* loop)  {

        auto old_prepare = loop->prepare_block();
        auto old_body = loop->body_block();
        auto old_update = loop->update_block();
        auto old_merge = loop->merge_block();

        XIRBuilder builder;
        
        

        auto simple_loop_after_lowering_body_block = loop->prepare_block()->parent_function()->create_basic_block();
        auto simple_loop_after_lowering_merge_block = loop->prepare_block()->parent_function()->create_basic_block();
        
        
        //prepare body
        {
            auto move_instr = [&](Instruction* inst){
                auto block = inst->parent_block();
            };
        };
        }

    void LowerLoops::transform_block(BasicBlock* block){

        XIRBuilder builder;


        block->traverse_instructions([&](Instruction* inst){
            
            auto resolver = value_old_to_new_resolver(&value_old_new_map);
            switch (inst->derived_instruction_tag()) {
                
                case DerivedInstructionTag::LOOP:{
                lowerloops(static_cast<LoopInst*>(inst));
                return;
                }
                case DerivedInstructionTag::ALLOCA:{
                
                builder.set_insertion_point(inst->parent_block());
                static_cast<AllocaInst*>(inst)->clone(builder, resolver);
                return;
                }

                case DerivedInstructionTag::STORE:{
                    builder.set_insertion_point(inst->parent_block());
                    static_cast<StoreInst*>(inst)->clone(builder, resolver);
                    return;
                }

                case DerivedInstructionTag::LOAD:{
                    builder.set_insertion_point(inst->parent_block());
                    static_cast<LoadInst*>(inst)->clone(builder, resolver);
                    return;
                }

                case DerivedInstructionTag::PRINT:{
                    builder.set_insertion_point(inst->parent_block());
                    static_cast<PrintInst*>(inst)->clone(builder, resolver);
                    return;
                }

                
            }



        });
    }

    
    [[nodiscard]] Canonicalize_Control_Flow_Info Canoinicalize_Control_Flow_pass_run_on_Function(Function* func){
        Canonicalize_Control_Flow_Info info;
        LowerLoops lower;
        if(auto def = func->definition()){
            def->traverse_basic_blocks([&](BasicBlock* block){
                block->traverse_instructions([&](Instruction* inst){
                    if(inst->derived_instruction_tag() == DerivedInstructionTag::LOOP){
                        lower.lowerloops(static_cast<LoopInst*>(inst));
                    }
                });
            });
        }
        return info;
    }
     [[nodiscard]] Canonicalize_Control_Flow_Info Canoinicalize_Control_Flow_pass_run_on_Module(Module* module){
        Canonicalize_Control_Flow_Info info;
        for(auto func : module->function_list()){
            Canoinicalize_Control_Flow_pass_run_on_Function(func);
        };
        
        return info;
     };

}
#endif

