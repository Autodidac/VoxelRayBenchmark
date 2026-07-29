#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"

no_run=0
clean=0
benchmark=0
benchmark_only=0
triangles=100000
seed=3909885990
backend="vulkan"
epochgui_source=""
use_epochgui=1
while (($#)); do
    case "$1" in
        --no-run) no_run=1; shift ;;
        --clean) clean=1; shift ;;
        --benchmark) benchmark=1; shift ;;
        --benchmark-only) benchmark_only=1; shift ;;
        --triangles)
            [[ $# -ge 2 ]] || { echo '--triangles requires a value' >&2; exit 2; }
            triangles="$2"; shift 2 ;;
        --seed)
            [[ $# -ge 2 ]] || { echo '--seed requires a value' >&2; exit 2; }
            seed="$2"; shift 2 ;;
        --backend)
            [[ $# -ge 2 ]] || { echo '--backend requires vulkan or opengl' >&2; exit 2; }
            backend="$2"; shift 2 ;;
        --epochgui-source)
            [[ $# -ge 2 ]] || { echo '--epochgui-source requires a path' >&2; exit 2; }
            epochgui_source="$2"; shift 2 ;;
        --no-epochgui) use_epochgui=0; shift ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            printf 'Usage: ./build.sh [--clean] [--no-run] [--benchmark|--benchmark-only] [--triangles N] [--seed N] [--backend vulkan|opengl] [--epochgui-source PATH|--no-epochgui]\n' >&2
            exit 2 ;;
    esac
done

case "$backend" in
    vulkan) executable="epoch_voxel_vulkan_demo" ;;
    opengl) executable="epoch_voxel_opengl_demo" ;;
    *) echo '--backend must be vulkan or opengl' >&2; exit 2 ;;
esac

resolve_vcpkg_root() {
    local candidate
    if [[ -n "${VCPKG_ROOT:-}" ]] && [[ -f "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" ]]; then
        cd -- "$VCPKG_ROOT" && pwd
        return
    fi
    for candidate in "$root/vcpkg" "$(dirname "$root")/vcpkg" "$HOME/vcpkg" "/opt/vcpkg"; do
        if [[ -f "$candidate/scripts/buildsystems/vcpkg.cmake" ]]; then
            cd -- "$candidate" && pwd
            return
        fi
    done
    cat >&2 <<'MESSAGE'
Unable to locate vcpkg.
Set VCPKG_ROOT to an existing vcpkg checkout, for example:
  export VCPKG_ROOT="$HOME/vcpkg"
CMake will run manifest installation automatically during configure.
MESSAGE
    exit 1
}

export VCPKG_ROOT="$(resolve_vcpkg_root)"
if (( clean )); then rm -rf "$root/build/linux-ninja-release"; fi
printf 'VCPKG_ROOT=%s\nBackend=%s\n' "$VCPKG_ROOT" "$backend"
cmake --preset linux-ninja-release
cmake --build --preset linux-ninja-release --target "$executable"
if (( ! no_run )); then
    args=(--triangles "$triangles" --seed "$seed")
    if (( benchmark_only )); then args+=(--benchmark-only)
    elif (( benchmark )); then args+=(--benchmark)
    fi
    exec "$root/build/linux-ninja-release/$executable" "${args[@]}"
fi
