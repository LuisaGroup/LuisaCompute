import sys
from common import run_cmd, get_targets

XIR_TARGETS = [
    "test_ast_to_xir",
    "test_xir_passes",
    "test_xir2ast_translators",
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

    ret = run_cmd(["xmake", "f", "-m", mode if mode else "release", "-c", "--lc_enable_xir=true"])
    if ret != 0:
        print("ERROR: xmake config failed")
        sys.exit(ret)

    available = set(get_targets())
    targets = [t for t in XIR_TARGETS if t in available]

    if not targets:
        print("No xir targets found (lc_enable_xir may be disabled).")
        sys.exit(0)

    failures = []
    for target in targets:
        print(f"\n=== Building {target} ===")
        if run_cmd(["xmake", "build", target]) != 0:
            failures.append(f"build:{target}")
            continue
        print(f"--- Running {target} ---")
        if run_cmd(["xmake", "run", target]) != 0:
            failures.append(f"run:{target}")

    if failures:
        print("\n!!! FAILURES !!!")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("\nAll xir tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
