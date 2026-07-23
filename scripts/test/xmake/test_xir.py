import sys
from common import run_cmd, get_targets, get_backends

XIR_TARGETS = [
    "test_ast_to_xir",
    "test_xir_passes",
    "test_xir2ast_translators",
]


def main():
    get_backends()  # parse for CLI consistency; XIR tests are backend-agnostic
    available = set(get_targets())
    targets = [t for t in XIR_TARGETS if t in available]

    if not targets:
        print("No xir targets found (lc_enable_xir may be disabled).")
        sys.exit(0)

    failures = []
    for target in targets:
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
