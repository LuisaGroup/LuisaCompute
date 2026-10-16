#!/usr/bin/env bash

set -euo pipefail

usage() {
    sed -n '/^# Build the matched/,/^script_dir=/ {
        /^script_dir=/d
        s/^# \{0,1\}//
        p
    }' "$0"
}

# Build the matched iOS path-tracing benchmark with exactly one Metal backend.
#
# Metal4 AIR:
#   scripts/build_ios_backend_benchmark.sh --backend metal4 \
#       --llvm-dir <ios-llvm>/lib/cmake/llvm --team <team-id>
#
# Legacy Metal MSL:
#   scripts/build_ios_backend_benchmark.sh --backend metal \
#       --team <team-id>
#
# Options:
#   --backend NAME            metal or metal4 (default: metal4).
#   --llvm-dir PATH           arm64 iOS LLVM 21 CMake directory; Metal4 only.
#   --build-dir PATH          Output directory (backend-specific default).
#   --bundle-id ID            Override the backend-specific bundle identifier.
#   --team ID                 Apple Development team for automatic signing.
#   --identity ID             Manual codesign identity; requires --profile.
#   --profile PATH            Installed/downloaded mobileprovision for manual signing.
#   --configuration NAME      Xcode configuration (default: Release).
#   --deployment-target VER   iOS deployment target (default: 26.0).
#   --jobs N                  Parallel build jobs (default: logical CPU count).
#   --unsigned                Disable code signing for link/audit builds.
#   --configure-only          Configure without building.
#   -h, --help                Show this help.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_dir=$(cd -- "$script_dir/.." && pwd -P)

backend=metal4
llvm_dir=${LUISA_IOS_LLVM_DIR:-}
ios_build_dir=
bundle_id=
development_team=${LUISA_IOS_DEVELOPMENT_TEAM:-}
signing_identity=
provisioning_profile=
configuration=Release
deployment_target=26.0
jobs=$(sysctl -n hw.logicalcpu 2>/dev/null || sysctl -n hw.ncpu)
unsigned=0
configure_only=0

while (($#)); do
    case "$1" in
        --backend)
            backend=${2:?Missing value for --backend}
            shift 2
            ;;
        --llvm-dir)
            llvm_dir=${2:?Missing value for --llvm-dir}
            shift 2
            ;;
        --build-dir)
            ios_build_dir=${2:?Missing value for --build-dir}
            shift 2
            ;;
        --bundle-id)
            bundle_id=${2:?Missing value for --bundle-id}
            shift 2
            ;;
        --team)
            development_team=${2:?Missing value for --team}
            shift 2
            ;;
        --identity)
            signing_identity=${2:?Missing value for --identity}
            shift 2
            ;;
        --profile)
            provisioning_profile=${2:?Missing value for --profile}
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

case "$backend" in
    metal)
        enable_metal=ON
        enable_metal4=OFF
        ;;
    metal4)
        enable_metal=OFF
        enable_metal4=ON
        ;;
    *)
        printf 'Invalid --backend value: %s\n' "$backend" >&2
        exit 2
        ;;
esac

for required_tool in cmake xcodebuild xcrun; do
    if ! command -v "$required_tool" >/dev/null 2>&1; then
        printf 'Required tool not found: %s\n' "$required_tool" >&2
        exit 1
    fi
done

manual_signing=0
signing_temp_dir=
if [[ -n "$signing_identity" || -n "$provisioning_profile" ]]; then
    if [[ -z "$signing_identity" || -z "$provisioning_profile" ]]; then
        printf '%s\n' \
            '--identity and --profile must be provided together.' >&2
        exit 2
    fi
    if ((unsigned)); then
        printf '%s\n' \
            '--unsigned cannot be combined with manual signing.' >&2
        exit 2
    fi
    if [[ ! -f "$provisioning_profile" ]]; then
        printf 'Provisioning profile does not exist: %s\n' \
            "$provisioning_profile" >&2
        exit 1
    fi
    for required_tool in codesign plutil security; do
        if ! command -v "$required_tool" >/dev/null 2>&1; then
            printf 'Required signing tool not found: %s\n' \
                "$required_tool" >&2
            exit 1
        fi
    done
    manual_signing=1
    signing_temp_dir=$(mktemp -d)
    trap 'if [[ -n "${signing_temp_dir:-}" && -d "$signing_temp_dir" ]]; then
              find "$signing_temp_dir" -type f -delete
              rmdir "$signing_temp_dir"
          fi' EXIT
    profile_plist="$signing_temp_dir/profile.plist"
    entitlements_plist="$signing_temp_dir/entitlements.plist"
    security cms -D -i "$provisioning_profile" > "$profile_plist"
    plutil -extract Entitlements xml1 \
        -o "$entitlements_plist" "$profile_plist"
    application_identifier=$(plutil -extract \
        Entitlements.application-identifier raw "$profile_plist")
    team_identifier=${application_identifier%%.*}
    profile_bundle_pattern=${application_identifier#"$team_identifier."}
    if [[ -z "$bundle_id" ]]; then
        if [[ "$profile_bundle_pattern" == *'*'* ]]; then
            printf '%s\n' \
                'A wildcard profile requires an explicit --bundle-id.' >&2
            exit 2
        fi
        bundle_id=$profile_bundle_pattern
    fi
    if [[ "$profile_bundle_pattern" == *'*'* ]]; then
        profile_bundle_prefix=${profile_bundle_pattern%'*'}
        if [[ "$bundle_id" != "$profile_bundle_prefix"* ]]; then
            printf "Bundle identifier '%s' is not covered by profile '%s'.\n" \
                "$bundle_id" "$profile_bundle_pattern" >&2
            exit 1
        fi
    elif [[ "$bundle_id" != "$profile_bundle_pattern" ]]; then
        printf "Bundle identifier '%s' does not match profile '%s'.\n" \
            "$bundle_id" "$profile_bundle_pattern" >&2
        exit 1
    fi
fi

if [[ -z "$ios_build_dir" ]]; then
    ios_build_dir="$repo_dir/cmake-build-ios-$backend-benchmark-xcode"
fi
mkdir -p "$ios_build_dir"
ios_build_dir=$(cd -- "$ios_build_dir" && pwd -P)

if ((unsigned == 0 && manual_signing == 0)) && \
        [[ -z "$development_team" ]]; then
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
    -DLUISA_COMPUTE_BUILD_TESTS=OFF
    -DLUISA_COMPUTE_BUILD_IOS_EXAMPLES=OFF
    -DLUISA_COMPUTE_BUILD_IOS_TESTS=OFF
    -DLUISA_COMPUTE_BUILD_IOS_BENCHMARKS=ON
    -DLUISA_COMPUTE_IOS_BENCHMARK_BACKEND="$backend"
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
    if [[ -z "$llvm_dir" ]]; then
        default_llvm_dir="$repo_dir/cmake-build-llvm21-ios/lib/cmake/llvm"
        if [[ -f "$default_llvm_dir/LLVMConfig.cmake" ]]; then
            llvm_dir=$default_llvm_dir
        else
            printf '%s\n' \
                'Metal4 requires --llvm-dir or LUISA_IOS_LLVM_DIR.' >&2
            exit 1
        fi
    fi
    if [[ ! -f "$llvm_dir/LLVMConfig.cmake" ]]; then
        printf 'LLVM_DIR does not contain LLVMConfig.cmake: %s\n' \
            "$llvm_dir" >&2
        exit 1
    fi
    llvm_dir=$(cd -- "$llvm_dir" && pwd -P)
    cmake_args+=("-DLLVM_DIR=$llvm_dir")
fi
if [[ -n "$bundle_id" ]]; then
    cmake_args+=("-DLUISA_COMPUTE_IOS_BENCHMARK_BUNDLE_ID=$bundle_id")
fi
cmake_args+=("-DLUISA_IOS_DEVELOPMENT_TEAM=$development_team")

cmake "${cmake_args[@]}"

if ((configure_only)); then
    printf 'Configured iOS %s benchmark at %s\n' \
        "$backend" "$ios_build_dir"
    exit 0
fi

build_args=(
    --build "$ios_build_dir"
    --config "$configuration"
    --parallel "$jobs"
    --target example_ios_path_tracing_benchmark
)
if ((unsigned || manual_signing)); then
    build_args+=(-- CODE_SIGNING_ALLOWED=NO)
else
    build_args+=(-- -allowProvisioningUpdates)
fi
cmake "${build_args[@]}"

if ((manual_signing)); then
    app="$ios_build_dir/bin/$configuration/example_ios_path_tracing_benchmark.app"
    if [[ ! -d "$app" ]]; then
        printf 'Built application bundle is missing: %s\n' "$app" >&2
        exit 1
    fi
    cp "$provisioning_profile" "$app/embedded.mobileprovision"
    codesign --force --sign "$signing_identity" \
        --entitlements "$entitlements_plist" \
        --timestamp=none "$app"
    codesign --verify --deep --strict --verbose=2 "$app"
fi

printf 'Built iOS %s benchmark at %s/bin/%s\n' \
    "$backend" "$ios_build_dir" "$configuration"
if ((unsigned)); then
    printf '%s\n' 'Code signing was disabled; this bundle cannot be installed.'
elif ((manual_signing)); then
    printf "The bundle was manually signed as '%s' with '%s'.\n" \
        "$bundle_id" "$provisioning_profile"
else
    printf 'The bundle was configured for automatic signing by team %s.\n' \
        "$development_team"
fi
