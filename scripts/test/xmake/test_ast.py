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
    args = sys.argv[1:]
    mode = None
    backend = None

    if args and args[0] in ("debug", "release"):
        mode = args.pop(0)
    if args:
        backend = args.pop(0)
    if args:
        print(f"Usage: python {sys.argv[0]} [mode] [backend]")
        sys.exit(1)

    backends = [backend] if backend else BACKENDS

    ret = run_cmd(["xmake", "f", "-m", mode if mode else "release", "-c"])
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
