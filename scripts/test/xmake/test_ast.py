import sys
from common import run_cmd, get_targets

AST_TARGETS = [
    "test_ast",
    "test_ast_basic",
    "test_builtin_kernel",
    "test_manual_ast",
]

BACKENDS = ["dx", "vk", "cuda", "metal"]


def main():
    ret = run_cmd(["xmake", "f", "-c"])
    if ret != 0:
        print("ERROR: xmake config failed")
        sys.exit(ret)

    available = set(get_targets())
    targets = [t for t in AST_TARGETS if t in available]

    if not targets:
        print("No ast targets found.")
        sys.exit(1)

    failures = []
    for target in targets:
        print(f"\n=== Building {target} ===")
        if run_cmd(["xmake", "build", target]) != 0:
            failures.append(f"build:{target}")
            continue
        for backend in BACKENDS:
            print(f"--- Running {target} with backend {backend} ---")
            if run_cmd(["xmake", "run", target, backend]) != 0:
                failures.append(f"run:{target}:{backend}")

    if failures:
        print("\n!!! FAILURES !!!")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("\nAll ast tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
