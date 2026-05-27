---
name: spv-opt
description: SPIRV-Tools optimizer pass development, IR manipulation, and PassTest registration.
---

# SPIRV-Tools Optimizer

Writing optimizer passes for SPIRV-Tools (`src/ext/SPIRV-Tools`).

## Pass Skeleton

Derive from `spvtools::opt::Pass` (or `MemPass` for mem2reg-style passes), implement `name()` and `Process()`.

```cpp
// source/opt/my_pass.h
#ifndef SOURCE_OPT_MY_PASS_H_
#define SOURCE_OPT_MY_PASS_H_
#include "source/opt/pass.h"
namespace spvtools::opt {
class MyPass : public Pass {
 public:
  const char* name() const override { return "my-pass"; }
  Status Process() override;
  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse | IRContext::kAnalysisCFG;
  }
};
}  // namespace spvtools::opt
#endif
```

```cpp
// source/opt/my_pass.cpp
#include "source/opt/my_pass.h"
namespace spvtools::opt {
Pass::Status MyPass::Process() {
  bool modified = false;
  // Iterate functions
  for (auto& func : *get_module()) {
    for (auto& block : *func) {
      for (auto& inst : *block) {
        // transform inst...
        modified = true;
      }
    }
  }
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}
}  // namespace spvtools::opt
```

**Rules**:
- `name()` must match the `--my-pass` CLI flag used in `RegisterPassFromFlag` (no leading hyphens).
- `Process()` must return `Failure` only on real errors.
- If you modify the module, return `SuccessWithChange`; the pass manager invalidates analyses not listed in `GetPreservedAnalyses()`.
- A single pass instance may only run once; internal state does not reset.

### MemPass base class

Many load/store elimination passes derive from `MemPass` instead of `Pass`:

```cpp
#include "source/opt/mem_pass.h"
class MyMemPass : public MemPass {
  // Inherits helpers: GetPtr(), IsTargetVar(), HasLoads(), DCEInst(),
  // CFGCleanup(), Type2Undef(), KillAllInsts(), etc.
};
```

## Key APIs

### Module / IRContext

```cpp
Module* m = get_module();           // or context()->module()
IRContext* ctx = context();

m->ForEachInst([](Instruction* inst){ /* all insts */ }, true);
ctx->get_def_use_mgr();             // analysis::DefUseManager
ctx->get_type_mgr();                // analysis::TypeManager
ctx->get_constant_mgr();            // analysis::ConstantManager
ctx->get_decoration_mgr();          // analysis::DecorationManager
ctx->cfg();                         // CFG
ctx->InvalidateAnalyses(IRContext::kAnalysisDefUse | IRContext::kAnalysisCFG);
ctx->IsConsistent();                // debug invariant check
```

### Instruction

```cpp
spv::Op opcode = inst->opcode();
uint32_t rid = inst->result_id();   // 0 if none
uint32_t tid = inst->type_id();     // 0 if none

// Operands: inst->begin() .. inst->end()
for (auto& op : *inst) {
  if (spvIsIdType(op.type)) { uint32_t id = op.words[0]; }
}

uint32_t val = inst->GetSingleWordInOperand(idx);
inst->NumInOperands();
inst->SetResultId(new_id);
inst->SetResultType(new_type_id);
inst->SetInOperand(idx, {new_val});
inst->ToBinary(&words);

// Predicates
inst->IsBranch();
inst->IsBlockTerminator();
inst->IsDecoration();
inst->IsConstant();
```

### BasicBlock

```cpp
for (auto& block : *func) {
  uint32_t label = block.id();
  Instruction* label_inst = block.GetLabelInst();
  Instruction* merge = block.GetMergeInst();       // OpSelectionMerge / OpLoopMerge
  Instruction* loop_merge = block.GetLoopMergeInst();
  bool has_phi = block.HasPhiInstructions();

  // Iterate instructions (label not included)
  for (auto& inst : block) { }
  block.ForEachInst([](Instruction* i){ }, true);  // true = include debug lines

  // Terminator helpers
  block.ForEachSuccessorLabel([](uint32_t id){ });
  bool is_loop_header = block.IsLoopHeader();
  uint32_t merge_id = block.MergeBlockIdIfAny();
}
```

### Function

```cpp
for (auto& func : *get_module()) {
  uint32_t func_id = func->DefInst().result_id();
  bool is_entry = func->IsEntryPoint();
  func->ForEachParam([](Instruction* param){ });
  // iterate blocks
  for (auto& block : *func) { }
}
```

### Building Instructions

```cpp
#include "source/opt/ir_builder.h"

// Insert before an instruction
InstructionBuilder b(context(), insertion_point,
  IRContext::kAnalysisInstrToBlockMapping | IRContext::kAnalysisDefUse);

// Or append to end of a block
InstructionBuilder b(context(), parent_block,
  IRContext::kAnalysisInstrToBlockMapping | IRContext::kAnalysisDefUse);

Instruction* add = b.AddBinaryOp(spv::Op::OpIAdd, type_id, lhs, rhs);
Instruction* extract = b.AddCompositeExtract(elem_type_id, composite_id, {idx0, idx1});
Instruction* construct = b.AddCompositeConstruct(type_id, {id0, id1});
Instruction* load = b.AddLoad(type_id, ptr_id);
Instruction* store = b.AddStore(ptr_id, value_id);
Instruction* branch = b.AddBranch(target_id);
Instruction* cbranch = b.AddBranchConditional(cond_id, true_id, false_id);
Instruction* phi = b.AddPhi(type_id, {val0, block0, val1, block1});
Instruction* unary = b.AddUnaryOp(type_id, spv::Op::OpConvertFToS, operand);
Instruction* nullary = b.AddNullaryOp(type_id, spv::Op::OpGroupAll);
```

### Replacing / Killing

```cpp
// Replace all uses of old_id with new_id
ctx->ReplaceAllUsesWith(old_id, new_id);

// Kill an instruction (removes from block, updates def-use)
ctx->KillInst(inst);

// Kill an id and all its uses
ctx->KillDef(id);

// Get the defining instruction for an id
Instruction* def = ctx->get_def_use_mgr()->GetDef(id);

// Get users of an id
ctx->get_def_use_mgr()->ForEachUser(id, [](Instruction* user){ });
ctx->get_def_use_mgr()->NumUsers(id);
```

## Testing

Tests live in `test/opt/`. Use `PassTest<::testing::Test>` fixture.

```cpp
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"
#include "source/opt/my_pass.h"

using MyPassTest = PassTest<::testing::Test>;

TEST_F(MyPassTest, Basic) {
  const std::string before = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%main = OpFunction %void None %void
%entry = OpLabel
OpReturn
OpFunctionEnd
)";
  const std::string after = before;
  SinglePassRunAndCheck<MyPass>(before, after, /*skip_nop=*/false, /*do_validation=*/false);
}
```

**Fixture helpers**:
- `SinglePassRunAndCheck<PassT>(before, after, skip_nop, do_validation, args...)` — exact match.
- `SinglePassRunAndCheck<PassT>(before, after, skip_nop, args...)` — overload without validation.
- `SinglePassRunAndMatch<PassT>(original, do_validation, args...)` — runs pass, disassembles, then checks with Effcee `CHECK:` patterns embedded in `original`. Always skips OpNop. Returns `(disassembly, status)`.
- `SinglePassRunAndFail<PassT>(original, args...)` — expects `Status::Failure`, checks error messages with Effcee.
- `SinglePassRunToBinary<PassT>(assembly, skip_nop, args...)` — returns `(binary, status)`.
- `SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS)` — keep ids from assembly.
- `SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER)` — omit SPIR-V header in output.
- `SetTargetEnv(spv_target_env)` — change target environment (default `SPV_ENV_UNIVERSAL_1_3`).

For match tests, embed `CHECK:` lines as comments in the assembly string:

```cpp
const std::string assembly = R"(
; CHECK: OpReturn
OpCapability Shader
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%main = OpFunction %void None %void
%entry = OpLabel
OpReturn
OpFunctionEnd
)";
SinglePassRunAndMatch<MyPass>(assembly, false);
```

### Multi-pass tests

```cpp
TEST_F(MyPassTest, Pipeline) {
  const std::string before = R"(... )";
  const std::string after = R"(... )";
  AddPass<MyFirstPass>();
  AddPass<MySecondPass>();
  RunAndCheck(before, after);
}
```

### Manual context tests

```cpp
#include "source/opt/build_module.h"

TEST(MyPass, Manual) {
  std::unique_ptr<IRContext> ctx =
    BuildModule(SPV_ENV_UNIVERSAL_1_3, nullptr, assembly,
                SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ASSERT_NE(ctx, nullptr);
  MyPass pass;
  auto status = pass.Run(ctx.get());
  EXPECT_EQ(status, Pass::Status::SuccessWithChange);
  EXPECT_TRUE(ctx->IsConsistent());
}
```

## Registering a Pass

1. Add header to `source/opt/passes.h` (or include directly).
2. Add `CreateMyPassPass()` factory declaration to `include/spirv-tools/optimizer.hpp`:

```cpp
Optimizer::PassToken CreateMyPassPass();
```
3. Implement factory in `source/opt/optimizer.cpp`:

```cpp
Optimizer::PassToken CreateMyPassPass() {
  return MakeUnique<opt::PassToken::Impl>(MakeUnique<opt::MyPass>());
}
```
4. Add CLI flag mapping in `source/opt/optimizer.cpp` inside `Optimizer::RegisterPassFromFlag`:

```cpp
} else if (pass_name == "my-pass") {
  RegisterPass(CreateMyPassPass());
```

Look at nearby passes in `RegisterPassFromFlag` for the exact pattern. Passes with arguments parse `pass_args` before calling `RegisterPass(...)`.

## Analyses & Invalidation

Available `IRContext::Analysis` bits:
- `kAnalysisNone`, `kAnalysisDefUse`, `kAnalysisInstrToBlockMapping`, `kAnalysisDecorations`
- `kAnalysisCFG`, `kAnalysisDominatorAnalysis`, `kAnalysisLoopAnalysis`
- `kAnalysisNameMap`, `kAnalysisScalarEvolution`, `kAnalysisRegisterPressure`
- `kAnalysisValueNumberTable`, `kAnalysisStructuredCFG`, `kAnalysisBuiltinVarId`
- `kAnalysisIdToFuncMapping`, `kAnalysisConstants`, `kAnalysisTypes`
- `kAnalysisDebugInfo`, `kAnalysisLiveness`, `kAnalysisIdToGraphMapping`

After `Process()` returns `SuccessWithChange`, the pass manager automatically calls:
```cpp
ctx->InvalidateAnalysesExceptFor(GetPreservedAnalyses());
```

If you mutate IDs outside normal helpers (e.g. `CompactIdsPass`), manually invalidate `kAnalysisDebugInfo` and any others that become stale mid-pass.

## Pass Manager / Recipes

```cpp
spvtools::Optimizer opt(SPV_ENV_UNIVERSAL_1_3);
opt.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&, const char* msg) {
  std::cerr << msg << std::endl;
});
opt.RegisterPass(spvtools::CreateCompactIdsPass())
    .RegisterPass(spvtools::CreateAggressiveDCEPass());
opt.Run(binary.data(), binary.size(), &optimized);
```

Built-in recipes:
- `RegisterPerformancePasses()` / `RegisterSizePasses()` / `RegisterLegalizationPasses()`

## File Map

| File | Purpose |
|---|---|
| `source/opt/pass.h` / `pass.cpp` | Base `Pass` class |
| `source/opt/mem_pass.h` / `mem_pass.cpp` | `MemPass` base for mem2reg-style passes |
| `source/opt/empty_pass.h` / `null_pass.h` | No-op passes for testing |
| `source/opt/ir_context.h` / `ir_context.cpp` | `IRContext`, analysis management |
| `source/opt/module.h` | `Module`, header, section lists |
| `source/opt/function.h` | `Function` |
| `source/opt/basic_block.h` | `BasicBlock` |
| `source/opt/instruction.h` | `Instruction`, `Operand`, `DebugScope` |
| `source/opt/ir_builder.h` | `InstructionBuilder` |
| `source/opt/def_use_manager.h` | `DefUseManager` |
| `source/opt/type_manager.h` | `TypeManager` |
| `source/opt/constants.h` | `ConstantManager` |
| `source/opt/cfg.h` | `CFG` |
| `source/opt/fold.h` | `Folder` (constant folding) |
| `source/opt/passes.h` | Unified include for all pass headers |
| `test/opt/pass_fixture.h` | `PassTest` fixture |
| `include/spirv-tools/optimizer.hpp` | Public C++ API |
| `source/opt/optimizer.cpp` | Pass factories & CLI flag dispatch |
| `examples/cpp-interface/main.cpp` | Example of assemble / validate / optimize / disassemble |
