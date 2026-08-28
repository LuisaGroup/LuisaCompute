#!/usr/bin/env bash

set -euo pipefail

usage() {
    sed -n '/^# Configure and build/,/^script_dir=/ {
        /^script_dir=/d
        s/^# \{0,1\}//
        p
    }' "$0"
}

# Configure and build Luisa Metal4 AIR applications for a physical iPhone.
#
# Signed examples and tests:
#   scripts/build_ios_metal4.sh --llvm-dir <ios-llvm>/lib/cmake/llvm \
#       --team <apple-development-team> --mode all
#
# Unsigned cross-link audit:
#   scripts/build_ios_metal4.sh --llvm-dir <ios-llvm>/lib/cmake/llvm \
#       --unsigned --mode all
#
# Options:
#   --llvm-dir PATH           arm64 iOS LLVM 21 lib/cmake/llvm directory.
#   --build-dir PATH          Output directory (default shown below).
#   --team ID                 Apple Development team for automatic signing.
#   --mode MODE               all, examples, or tests (default: all).
#   --configuration NAME      Xcode configuration (default: Release).
#   --deployment-target VER   iOS deployment target (default: 26.0).
#   --jobs N                  Parallel build jobs (default: logical CPU count).
#   --unsigned                Disable code signing for link/audit builds.
#   --configure-only          Configure without building.
#   -h, --help                Show this help.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_dir=$(cd -- "$script_dir/.." && pwd -P)

llvm_dir=${LUISA_IOS_LLVM_DIR:-}
ios_build_dir=${LUISA_IOS_BUILD_DIR:-"$repo_dir/cmake-build-ios-metal4-device-air-xcode"}
development_team=${LUISA_IOS_DEVELOPMENT_TEAM:-}
mode=all
configuration=Release
deployment_target=26.0
jobs=$(sysctl -n hw.logicalcpu 2>/dev/null || sysctl -n hw.ncpu)
unsigned=0
configure_only=0

while (($#)); do
    case "$1" in
        --llvm-dir)
            llvm_dir=${2:?Missing value for --llvm-dir}
            shift 2
            ;;
        --build-dir)
            ios_build_dir=${2:?Missing value for --build-dir}
            shift 2
            ;;
        --team)
            development_team=${2:?Missing value for --team}
            shift 2
            ;;
        --mode)
            mode=${2:?Missing value for --mode}
            shift 2
            ;;
        --configuration)
            configuration=${2:?Missing value for --configuration}
            shift 2
            ;;
        --deployment-target)
            deployment_target=${2:?Missing value for --deployment-target}
            shift 2
            ;;
        --jobs)
            jobs=${2:?Missing value for --jobs}
            shift 2
            ;;
        --unsigned)
            unsigned=1
            shift
            ;;
        --configure-only)
            configure_only=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$mode" in
    all)
        build_examples=ON
        build_tests=ON
        build_targets=(luisa-ios-rendering-examples luisa-ios-device-tests)
        ;;
    examples)
        build_examples=ON
        build_tests=OFF
        build_targets=(luisa-ios-rendering-examples)
        ;;
    tests)
        build_examples=OFF
        build_tests=ON
        build_targets=(luisa-ios-device-tests)
        ;;
    *)
        printf 'Invalid --mode value: %s\n' "$mode" >&2
        exit 2
        ;;
esac

for required_tool in cmake xcodebuild xcrun; do
    if ! command -v "$required_tool" >/dev/null 2>&1; then
        printf 'Required tool not found: %s\n' "$required_tool" >&2
        exit 1
    fi
done

if [[ -z "$llvm_dir" ]]; then
    default_llvm_dir="$repo_dir/cmake-build-llvm21-ios/lib/cmake/llvm"
    if [[ -f "$default_llvm_dir/LLVMConfig.cmake" ]]; then
        llvm_dir=$default_llvm_dir
    else
        printf '%s\n' \
            'Pass --llvm-dir or set LUISA_IOS_LLVM_DIR.' >&2
        exit 1
    fi
fi
if [[ ! -f "$llvm_dir/LLVMConfig.cmake" ]]; then
    printf 'LLVM_DIR does not contain LLVMConfig.cmake: %s\n' "$llvm_dir" >&2
    exit 1
fi
llvm_dir=$(cd -- "$llvm_dir" && pwd -P)
mkdir -p "$ios_build_dir"
ios_build_dir=$(cd -- "$ios_build_dir" && pwd -P)
if ((unsigned == 0)) && [[ -z "$development_team" ]]; then
    printf '%s\n' \
        'Signed builds require --team or LUISA_IOS_DEVELOPMENT_TEAM.' >&2
    exit 1
fi

cmake_args=(
    -S "$repo_dir"
    -B "$ios_build_dir"
    -G Xcode
    -DCMAKE_SYSTEM_NAME=iOS
    -DCMAKE_OSX_SYSROOT=iphoneos
    -DCMAKE_OSX_ARCHITECTURES=arm64
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$deployment_target"
    -DCMAKE_BUILD_TYPE="$configuration"
    -DBUILD_SHARED_LIBS=OFF
    -DLLVM_DIR="$llvm_dir"
    -DLUISA_COMPUTE_BUILD_TESTS=OFF
    -DLUISA_COMPUTE_BUILD_IOS_EXAMPLES="$build_examples"
    -DLUISA_COMPUTE_BUILD_IOS_TESTS="$build_tests"
    -DLUISA_COMPUTE_ENABLE_CLANG_CXX=OFF
    -DLUISA_COMPUTE_ENABLE_CUDA=OFF
    -DLUISA_COMPUTE_ENABLE_DX=OFF
    -DLUISA_COMPUTE_ENABLE_FALLBACK=OFF
    -DLUISA_COMPUTE_ENABLE_GUI=ON
    -DLUISA_COMPUTE_ENABLE_HIP=OFF
    -DLUISA_COMPUTE_ENABLE_METAL=OFF
    -DLUISA_COMPUTE_ENABLE_METAL4=ON
    -DLUISA_COMPUTE_ENABLE_SIMD=OFF
    -DLUISA_COMPUTE_ENABLE_TENSOR=OFF
    -DLUISA_COMPUTE_ENABLE_VULKAN=OFF
)
if [[ -n "$development_team" ]]; then
    cmake_args+=("-DLUISA_IOS_DEVELOPMENT_TEAM=$development_team")
fi
cmake "${cmake_args[@]}"

if ((configure_only)); then
    printf 'Configured iOS Metal4 build at %s\n' "$ios_build_dir"
    exit 0
fi

build_args=(
    --build "$ios_build_dir"
    --config "$configuration"
    --parallel "$jobs"
    --target "${build_targets[@]}"
    -- -parallelizeTargets
)
if ((unsigned)); then
    build_args+=(CODE_SIGNING_ALLOWED=NO)
fi
cmake "${build_args[@]}"

printf 'Built iOS Metal4 mode=%s at %s/bin/%s\n' \
    "$mode" "$ios_build_dir" "$configuration"
if ((unsigned)); then
    printf '%s\n' 'Code signing was disabled; these bundles cannot be installed.'
else
    printf 'Bundles were configured for automatic signing by team %s.\n' \
        "$development_team"
fi
