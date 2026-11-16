#!/usr/bin/env bash
# Validate the static-link smoke executable, not a device runtime result.
set -euo pipefail
executable=${1:?Pass the iOS link-test executable}
[[ -f "$executable" ]] || { echo "Missing executable: $executable" >&2; exit 1; }
[[ "$(lipo -archs "$executable")" == arm64 ]] || {
    echo "Expected an arm64-only executable: $executable" >&2; exit 1;
}
xcrun vtool -show-build "$executable" | awk '
    $1 == "platform" && $2 == "IOS" { found = 1 }
    END { exit found ? 0 : 1 }
'
# Capture otool output first so a failed inspection cannot appear as an empty,
# successful dependency list through process substitution.
dependencies=$(otool -L "$executable")
while IFS= read -r dependency; do
    case "$dependency" in
        /System/*|/usr/lib/*) ;;
        *) echo "Unexpected dynamic dependency: $dependency" >&2; exit 1 ;;
    esac
done < <(printf '%s\n' "$dependencies" | awk 'NR > 1 { print $1 }')
echo "Audited $executable: arm64, iPhoneOS, system dylibs only (not device-tested)."
