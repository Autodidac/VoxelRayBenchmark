#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
triangles="${1:-100000}"
"$root/build.sh" --backend vulkan --benchmark-only --triangles "$triangles"
"$root/build.sh" --backend opengl --benchmark-only --triangles "$triangles"
printf 'Benchmark complete. CSV files are in %s\n' "$root"
