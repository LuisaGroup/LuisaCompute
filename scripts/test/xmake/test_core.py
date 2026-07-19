import sys
from common import run_cmd, get_targets, get_backends, format_backends

CORE_TARGETS = [
    "test_basic_traits",
    "test_basic_types",
    "test_binary_file_stream",
    "test_binary_io",
    "test_clock",
    "test_dynamic_module",
    "test_first_fit",
    "test_logging",
    "test_mathematics",
    "test_matrix",
    "test_pool",
    "test_type",
    "test_glslang_spirv",
    "test_normal_encoding",
]

def main():
    backends = get_backends(["dx", "vk", "cuda", "hip", "metal"])
    available = set(get_targets())
    targets = [t for t in CORE_TARGETS if t in available]

    if not targets:
        print("No core targets found.")
        sys.exit(1)

    failures = []
    for target in targets:
        if target == "test_normal_encoding":
            for backend in backends:
                print(f"--- Running {target} with backend {backend} ---")
                if run_cmd(["xmake", "run", target, backend]) != 0:
                    failures.append(f"run:{target}:{backend}")
        else:
            print(f"--- Running {target} ---")
            if run_cmd(["xmake", "run", target]) != 0:
                failures.append(f"run:{target}")

    if failures:
        print("\n!!! FAILURES !!!")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print(f"\nAll core tests passed; device-backed coverage used: {format_backends(backends)}.")
        sys.exit(0)


if __name__ == "__main__":
    main()
