#!/usr/bin/env bash
# Cross-build and link the library without the applications in LuisaComputeGallery.
set -euo pipefail

usage() {
    echo 'Usage: build_ios_library.sh --backend metal|metal4 [--llvm-dir PATH]'
    echo '       [--build-dir PATH] [--deployment-target VERSION] [--jobs N]'
    echo 'Builds unsigned iPhoneOS arm64 backend link tests; does not run them.'
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_dir=$(cd -- "$script_dir/.." && pwd -P)
backend=metal4
llvm_dir=${LUISA_IOS_LLVM_DIR:-}
build_dir=
deployment_target=26.0
jobs=4
while (($#)); do
    case "$1" in
        --backend) backend=${2:?}; shift 2 ;;
        --llvm-dir) llvm_dir=${2:?}; shift 2 ;;
        --build-dir) build_dir=${2:?}; shift 2 ;;
        --deployment-target) deployment_target=${2:?}; shift 2 ;;
        --jobs) jobs=${2:?}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
case "$backend" in
    metal) enable_metal=ON; enable_metal4=OFF ;;
    metal4) enable_metal=OFF; enable_metal4=ON ;;
    *) printf 'Invalid backend: %s\n' "$backend" >&2; exit 2 ;;
esac
for tool in cmake xcodebuild xcrun; do
    command -v "$tool" >/dev/null || { echo "Required tool not found: $tool" >&2; exit 1; }
done
build_dir=${build_dir:-"$repo_dir/build-ios-$backend"}
cmake_args=(
    -S "$repo_dir" -B "$build_dir" -G Xcode
    -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos
    -DCMAKE_OSX_ARCHITECTURES=arm64
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$deployment_target"
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO
    -DBUILD_SHARED_LIBS=OFF
    -DLUISA_COMPUTE_BUILD_TESTS=OFF
    -DLUISA_COMPUTE_BUILD_IOS_LINK_TESTS=ON
    -DLUISA_COMPUTE_ENABLE_CLANG_CXX=OFF
    -DLUISA_COMPUTE_ENABLE_CUDA=OFF
    -DLUISA_COMPUTE_ENABLE_DX=OFF
    -DLUISA_COMPUTE_ENABLE_FALLBACK=OFF
    -DLUISA_COMPUTE_ENABLE_GUI=ON
    -DLUISA_COMPUTE_ENABLE_HIP=OFF
    -DLUISA_COMPUTE_ENABLE_METAL="$enable_metal"
    -DLUISA_COMPUTE_ENABLE_METAL4="$enable_metal4"
    -DLUISA_COMPUTE_ENABLE_SIMD=OFF
    -DLUISA_COMPUTE_ENABLE_TENSOR=OFF
    -DLUISA_COMPUTE_ENABLE_VULKAN=OFF
)
if [[ "$backend" == metal4 ]]; then
    if [[ ! -f "$llvm_dir/LLVMConfig.cmake" ]]; then
        echo 'Metal4 requires --llvm-dir pointing to arm64 iPhoneOS LLVM 22.' >&2
        exit 1
    fi
    llvm_dir=$(cd -- "$llvm_dir" && pwd -P)
    cmake_args+=("-DLLVM_DIR=$llvm_dir")
fi
cmake "${cmake_args[@]}"
cmake --build "$build_dir" --config Release --parallel "$jobs" \
    --target luisa-ios-link-tests -- CODE_SIGNING_ALLOWED=NO
"$script_dir/audit_ios_link_test.sh" \
    "$build_dir/bin/Release/test_ios_${backend}_link.app/test_ios_${backend}_link"
