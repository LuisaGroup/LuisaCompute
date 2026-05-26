import sys
from common import run_cmd, get_targets, get_backends

AST_TARGETS = [
    "test_ast",
    "test_ast_basic",
    "test_builtin_kernel",
    "test_manual_ast",
]

def main():
    backends = get_backends(["dx", "vk", "cuda", "metal"])
    available = set(get_targets())
    targets = [t for t in AST_TARGETS if t in available]

    if not targets:
        print("No ast targets found.")
        sys.exit(1)

    failures = []
    for target in targets:
        for backend in backends:
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
