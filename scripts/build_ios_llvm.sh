#!/usr/bin/env bash

set -euo pipefail

usage() {
    sed -n '/^# Build a static/,/^script_dir=/ {
        /^script_dir=/d
        s/^# \{0,1\}//
        p
    }' "$0"
}

# Build a static arm64 iPhoneOS LLVM 21 tree for Luisa's Metal4 AIR backend.
#
# Usage:
#   scripts/build_ios_llvm.sh --source /path/to/llvm-project-21.1.8.src
#
# Options:
#   --source PATH             llvm-project root or its llvm/ directory.
#   --llvm-version VERSION    Official source release to download when
#                             --source is omitted (default: 21.1.8).
#   --source-cache PATH       Download/extraction directory.
#   --verify-attestation      Verify the downloaded release with GitHub CLI.
#   --build-dir PATH          Output directory (default: cmake-build-llvm21-ios).
#   --host-llvm-prefix PATH   Host LLVM 21 prefix containing llvm-tblgen.
#   --deployment-target VER   iOS deployment target (default: 26.0).
#   --configuration NAME      CMake configuration (default: Release).
#   --jobs N                  Parallel build jobs (default: logical CPU count).
#   --configure-only          Configure without building.
#   -h, --help                Show this help.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_dir=$(cd -- "$script_dir/.." && pwd -P)

llvm_source=${LUISA_IOS_LLVM_SOURCE:-}
llvm_version=${LUISA_IOS_LLVM_VERSION:-21.1.8}
llvm_source_cache=${LUISA_IOS_LLVM_SOURCE_CACHE:-"$repo_dir/cmake-build-llvm21-ios-source"}
llvm_build_dir=${LUISA_IOS_LLVM_BUILD_DIR:-"$repo_dir/cmake-build-llvm21-ios"}
host_llvm_prefix=${LUISA_HOST_LLVM_PREFIX:-}
deployment_target=26.0
configuration=Release
jobs=$(sysctl -n hw.logicalcpu 2>/dev/null || sysctl -n hw.ncpu)
configure_only=0
verify_attestation=0

while (($#)); do
    case "$1" in
        --source)
            llvm_source=${2:?Missing value for --source}
            shift 2
            ;;
        --llvm-version)
            llvm_version=${2:?Missing value for --llvm-version}
            shift 2
            ;;
        --source-cache)
            llvm_source_cache=${2:?Missing value for --source-cache}
            shift 2
            ;;
        --verify-attestation)
            verify_attestation=1
            shift
            ;;
        --build-dir)
            llvm_build_dir=${2:?Missing value for --build-dir}
            shift 2
            ;;
        --host-llvm-prefix)
            host_llvm_prefix=${2:?Missing value for --host-llvm-prefix}
            shift 2
            ;;
        --deployment-target)
            deployment_target=${2:?Missing value for --deployment-target}
            shift 2
            ;;
        --configuration)
            configuration=${2:?Missing value for --configuration}
            shift 2
            ;;
        --jobs)
            jobs=${2:?Missing value for --jobs}
            shift 2
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

for required_tool in cmake ninja xcrun; do
    if ! command -v "$required_tool" >/dev/null 2>&1; then
        printf 'Required tool not found: %s\n' "$required_tool" >&2
        exit 1
    fi
done

if [[ -z "$llvm_source" ]]; then
    mkdir -p "$llvm_source_cache"
    llvm_source_cache=$(cd -- "$llvm_source_cache" && pwd -P)
    release_dir="$llvm_source_cache/llvm-project-$llvm_version.src"
    release_archive="$llvm_source_cache/llvm-project-$llvm_version.src.tar.xz"
    if [[ ! -d "$release_dir/llvm" ]]; then
        if ! command -v curl >/dev/null 2>&1; then
            printf '%s\n' \
                'curl is required to download the official LLVM source.' >&2
            exit 1
        fi
        if [[ ! -f "$release_archive" ]]; then
            release_url="https://github.com/llvm/llvm-project/releases/download/llvmorg-$llvm_version/llvm-project-$llvm_version.src.tar.xz"
            printf 'Downloading official LLVM %s source...\n' "$llvm_version"
            curl --fail --location --proto '=https' --tlsv1.2 \
                --output "$release_archive" "$release_url"
        fi
        printf 'Extracting %s...\n' "$release_archive"
        tar -xf "$release_archive" -C "$llvm_source_cache"
    fi
    if ((verify_attestation)); then
        if [[ ! -f "$release_archive" ]]; then
            printf 'Cannot verify missing release archive: %s\n' \
                "$release_archive" >&2
            exit 1
        fi
        if ! command -v gh >/dev/null 2>&1; then
            printf '%s\n' \
                'GitHub CLI is required by --verify-attestation.' >&2
            exit 1
        fi
        gh attestation verify --repo llvm/llvm-project "$release_archive"
    fi
    llvm_source=$release_dir
fi
if [[ -d "$llvm_source/llvm" ]]; then
    llvm_source="$llvm_source/llvm"
fi
if [[ ! -f "$llvm_source/CMakeLists.txt" ]]; then
    printf 'LLVM source directory is invalid: %s\n' "$llvm_source" >&2
    exit 1
fi
llvm_source=$(cd -- "$llvm_source" && pwd -P)
llvm_version_file="$llvm_source/../cmake/Modules/LLVMVersion.cmake"
llvm_source_major=$(sed -n \
    's/^[[:space:]]*set(LLVM_VERSION_MAJOR \([0-9][0-9]*\)).*/\1/p' \
    "$llvm_version_file" 2>/dev/null | head -n 1)
if [[ "$llvm_source_major" != 21 ]]; then
    printf 'Luisa Metal4 AIR requires LLVM 21 source, found major %s.\n' \
        "${llvm_source_major:-unknown}" >&2
    exit 1
fi
mkdir -p "$llvm_build_dir"
llvm_build_dir=$(cd -- "$llvm_build_dir" && pwd -P)

if [[ -z "$host_llvm_prefix" ]] && command -v brew >/dev/null 2>&1; then
    host_llvm_prefix=$(brew --prefix llvm@21 2>/dev/null || true)
fi
if [[ -z "$host_llvm_prefix" ]]; then
    printf '%s\n' \
        'Pass --host-llvm-prefix or install host LLVM 21 with Homebrew.' >&2
    exit 1
fi
host_tblgen="$host_llvm_prefix/bin/llvm-tblgen"
host_llvm_config="$host_llvm_prefix/bin/llvm-config"
if [[ ! -x "$host_tblgen" || ! -x "$host_llvm_config" ]]; then
    printf 'Host LLVM 21 tools not found under: %s\n' "$host_llvm_prefix" >&2
    exit 1
fi
host_llvm_version=$($host_llvm_config --version)
case "$host_llvm_version" in
    21.*) ;;
    *)
        printf 'Host llvm-tblgen must be LLVM 21, found %s.\n' \
            "$host_llvm_version" >&2
        exit 1
        ;;
esac

cmake -S "$llvm_source" -B "$llvm_build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE="$configuration" \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_SYSROOT=iphoneos \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$deployment_target" \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLVM_TARGETS_TO_BUILD=AArch64 \
    -DLLVM_HOST_TRIPLE="arm64-apple-ios$deployment_target" \
    -DLLVM_DEFAULT_TARGET_TRIPLE="arm64-apple-ios$deployment_target" \
    -DLLVM_TABLEGEN="$host_tblgen" \
    -DLLVM_BUILD_LLVM_DYLIB=OFF \
    -DLLVM_LINK_LLVM_DYLIB=OFF \
    -DLLVM_BUILD_TOOLS=OFF \
    -DLLVM_BUILD_UTILS=OFF \
    -DLLVM_BUILD_TESTS=OFF \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_BUILD_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_UTILS=OFF \
    -DLLVM_ENABLE_PROJECTS= \
    -DLLVM_ENABLE_RUNTIMES= \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_ENABLE_EH=OFF \
    -DLLVM_ENABLE_RTTI=OFF \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_ENABLE_LIBEDIT=OFF \
    -DLLVM_ENABLE_LIBXML2=OFF \
    -DLLVM_ENABLE_ZLIB=OFF \
    -DLLVM_ENABLE_ZSTD=OFF \
    -DLLVM_ENABLE_CURL=OFF \
    -DLLVM_ENABLE_HTTPLIB=OFF \
    -DLLVM_ENABLE_LIBPFM=OFF \
    -DLLVM_ENABLE_TELEMETRY=OFF

llvm_dir="$llvm_build_dir/lib/cmake/llvm"
if ((configure_only)); then
    printf 'Configured iOS LLVM. LLVM_DIR=%s\n' "$llvm_dir"
    exit 0
fi

cmake --build "$llvm_build_dir" --parallel "$jobs" --target \
    LLVMCore \
    LLVMSupport \
    LLVMBitReader \
    LLVMBitWriter \
    LLVMIRReader \
    LLVMLinker \
    LLVMPasses \
    LLVMAnalysis \
    LLVMTransformUtils \
    LLVMipo \
    LLVMScalarOpts \
    LLVMInstCombine \
    LLVMVectorize

if [[ ! -f "$llvm_dir/LLVMConfig.cmake" ]]; then
    printf 'LLVM build completed without LLVMConfig.cmake at %s\n' \
        "$llvm_dir" >&2
    exit 1
fi
printf 'Built static arm64 iOS LLVM %s.\n' "$host_llvm_version"
printf 'Use LLVM_DIR=%s\n' "$llvm_dir"
