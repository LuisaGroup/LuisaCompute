# Fallback queue predicate publication

The portable fallback worker pool had two instances of the same lost-wakeup
bug: task completion and shutdown changed an atomic wait predicate without
holding the mutex used by the condition variable. Both transitions now use
`_task_mutex`. No renderer, shader, frame layout, task scheduling policy or
device arithmetic is changed. This is not a timeout increase or polling fix.

## Formal cause and minimal counterexamples

The condition-variable predicate overload is `while (!pred()) wait(lock)`;
the atomic unlock-and-block is inside `wait`, not inside the preceding
predicate evaluation. See the [C++ condition-variable specification](https://eel.is/c++draft/thread.condition.condvar).
Sequentially consistent atomics do not join those separate operations.

Let M be `_task_mutex`, P the wait predicate, W a waiter and N a notifier.
The original implementation permitted this sequence:

1. W owns M and observes P=false.
2. N publishes P=true and notifies, without acquiring M.
3. W atomically unlocks M and begins waiting, after the only notification.

For shutdown, W is one idle worker and N is the dispatcher destroying its
pool. For completion, W is the dispatcher after a permitted spurious
wakeup, while the final worker completes outside M. That worker may publish
`_thread_working == 0` between the recheck and the next wait.

The fix serializes both publications with the predicate observation and
unlock-and-block. Either W observes the new predicate under M, or N cannot
publish it until W has entered the atomic wait operation. Per-item work is
still outside M; only each worker's completion acknowledgement acquires M.
Shutdown joins workers after releasing M. Task generation, work claiming,
submission serialization and the command queue's FIFO contract are unchanged.

## Permanent regression

`test_fallback_command_queue.cpp` compiles the production queue source,
without loading LLVM/Embree or compiling any shader. It has two Linux
schedule-controlled cases:

- One worker pauses while holding M, after its idle predicate was false.
  A concurrent queue destructor attempts shutdown.
- Two workers begin a job. One remains unfinished while a spurious wakeup
  lets the dispatcher recheck completion and pause in that same gap. The
  last worker is then allowed to finish.

Test-executable-only `pthread_cond_wait` / `pthread_cond_broadcast`
interposition exposes that legal scheduling window. A notification of the
paused condition fails immediately, before a process can hang forever. The
backend contains no test hooks. Scheduling control does not change the wait
predicate, unlock the mutex prematurely, or require a rendering workload.
Other platforms run the portable repeated queue test without interposition.

Both cases fail against the original production source and pass with the
same generic fix. Their original debugger captures show the worker wait,
dispatcher notification and destructor/completion stacks. Ordinary stress
also checks 768 queue lifetimes with 1/2/4 workers, 49,152 empty/nonempty
dispatches and exact item counts per test invocation.

## Reproduction and evidence scope

```bash
cmake --build build-codex-xir --parallel 32
ctest --test-dir build-codex-xir --output-on-failure -j32 -L unit
ctest --test-dir build-codex-xir --output-on-failure -j2 \
  --repeat until-fail:100 -R '^test_fallback_command_queue(_completion)?$'
```

This was discovered during Psycles' full parallel fallback suite: 183/184
completed, while `psycles_luisa_camera_sampling_tests fallback` remained
asleep with three threads. Attaching to that existing process was prohibited
by the host's ptrace policy; no system setting was changed. The separate
minimal processes were launched under a debugger instead. Thus the original
camera stall is consistent with the shutdown race, not claimed to have an
attached debugger proof. The two production-queue counterexamples establish
the bugs independently. The hung test was explicitly terminated and its
suite recorded as failed, not silently replaced by an earlier green run.

Raw evidence lives in `/var/tmp/psycles-hidden-socket-U0KI5L`:
`fallback-queue-gated-red.log`, `fallback-queue-completion-red.log`, their
`*-backtrace.log` files, and `queue-fix-*` build/test logs. Final integration
results are indexed by the parent Psycles validation report.

The full child build uses all 32 threads. Its registered suite passes
155/155; both queue regressions additionally pass 100 repetitions each
(66.47 seconds total). The pre-existing CFG/restructure regressions remain
in the full suite. The parent HIP suite passes 182/182 after this change;
its full 32-way fallback suite passes 184/184 in 89.67 seconds, including
the original camera sampling workload.
