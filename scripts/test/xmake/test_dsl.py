import sys
from common import run_cmd, get_targets, get_backends, format_backends

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

def main():
    backends = get_backends(["dx", "vk", "cuda", "hip", "metal"])
    available = set(get_targets())
    targets = [t for t in DSL_TARGETS if t in available]

    if not targets:
        print("No dsl targets found.")
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
        print(f"\nAll dsl tests passed for backends: {format_backends(backends)}.")
        sys.exit(0)


if __name__ == "__main__":
    main()
