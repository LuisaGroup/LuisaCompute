#!/usr/bin/env bash

set -euo pipefail

usage() {
    printf '%s\n' \
        'Audit Luisa iPhoneOS bundles after an unsigned or signed build.' \
        '' \
        'Usage:' \
        '  scripts/audit_ios_bundles.sh --bin-dir <xcode-build>/bin/Release' \
        '' \
        'Options:' \
        '  --bin-dir PATH       Directory containing the .app bundles.' \
        '  --expected-count N   Required bundle count (default: 21 for --mode all).' \
        '  -h, --help           Show this help.'
}

bin_dir=
expected_count=21
while (($#)); do
    case "$1" in
        --bin-dir)
            bin_dir=${2:?Missing value for --bin-dir}
            shift 2
            ;;
        --expected-count)
            expected_count=${2:?Missing value for --expected-count}
            shift 2
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

if [[ -z "$bin_dir" || ! -d "$bin_dir" ]]; then
    printf 'Bundle directory does not exist: %s\n' "${bin_dir:-<unset>}" >&2
    exit 1
fi
for required_tool in lipo otool plutil xcrun; do
    if ! command -v "$required_tool" >/dev/null 2>&1; then
        printf 'Required tool not found: %s\n' "$required_tool" >&2
        exit 1
    fi
done

shopt -s nullglob
apps=(
    "$bin_dir"/example_ios_*.app
)
device_test_app="$bin_dir/luisa-metal4-ios-device-air-path-tracer.app"
if [[ -d "$device_test_app" ]]; then
    apps+=("$device_test_app")
fi
if ((${#apps[@]} != expected_count)); then
    printf 'Expected %s iOS bundles, found %s in %s.\n' \
        "$expected_count" "${#apps[@]}" "$bin_dir" >&2
    exit 1
fi

audit_failed=0
for app in "${apps[@]}"; do
    executable_name=$(plutil -extract CFBundleExecutable raw "$app/Info.plist")
    executable="$app/$executable_name"
    if [[ ! -f "$executable" ]]; then
        printf 'Missing bundle executable: %s\n' "$executable" >&2
        audit_failed=1
        continue
    fi
    architectures=$(lipo -archs "$executable")
    if [[ "$architectures" != arm64 ]]; then
        printf 'Expected arm64-only executable: %s (%s)\n' \
            "$executable" "$architectures" >&2
        audit_failed=1
    fi
    if ! xcrun vtool -show-build "$executable" | awk '
        $1 == "platform" && $2 == "IOS" { found = 1 }
        END { exit found ? 0 : 1 }
    '; then
        printf 'Executable is not tagged for the iOS platform: %s\n' \
            "$executable" >&2
        audit_failed=1
    fi
    while IFS= read -r dependency; do
        case "$dependency" in
            /System/*|/usr/lib/*) ;;
            *)
                printf 'Unexpected dynamic dependency: %s -> %s\n' \
                    "$executable_name" "$dependency" >&2
                audit_failed=1
                ;;
        esac
    done < <(otool -L "$executable" | awk 'NR > 1 {print $1}')
done

if ((audit_failed)); then
    exit 1
fi
printf 'Audited %s iOS bundles: arm64, iPhoneOS, Apple system dylibs only.\n' \
    "${#apps[@]}"
printf '%s\n' \
    'Luisa and backend libraries (including LLVM where used) are statically linked.'
