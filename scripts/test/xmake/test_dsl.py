import sys
from common import run_cmd, get_targets

DSL_TARGETS = [
    "test_binding_group",
    "test_binding_group_template",
    "test_callable",
    "test_constant",
    "test_dsl",
    "test_dsl_sugar",
    "test_dsl_multithread",
    "test_soa",
    "test_soa_subview",
    "test_soa_simple",
    "test_device_math",
    "test_nested_callable",
    "test_calc",
    "test_dsl_matrix",
    "test_var",
    "test_8bit",
    "test_dsl_mathematic",
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
    targets = [t for t in DSL_TARGETS if t in available]

    if not targets:
        print("No dsl targets found.")
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
        print("\nAll dsl tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
