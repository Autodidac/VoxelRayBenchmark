# Epoch unified voxel / triangle traversal benchmark

Standalone C++23 + SDL3 benchmark for testing a shared voxel-backed spatial system across:

- Vulkan 1.3 compute
- OpenGL 4.3+ compute
- exact triangle surfaces
- sparse smoke/fog volume bricks
- coherent camera rays and incoherent secondary-like rays

The benchmark is intentionally software-ray based on both APIs. It measures the fallback tier that remains available without hardware ray-tracing extensions.

The OpenGL backend uses only OpenGL 4.3 core buffer, image, compute, and timer-query APIs. It deliberately avoids OpenGL 4.5 direct-state-access calls so a real 4.3 context remains a valid fallback target.

## Owned floating live traversal inspector

Interactive Vulkan and OpenGL runs keep the real GPU benchmark image in the main window and open the EpochGui traversal inspector as a movable, resizable **owned utility child window**. The parent relationship is declared when the child is created; the tool never reparents or docks itself afterward. The inspector shows the active brick field, visited bricks, global and local BVH stages, exact triangle candidates and hits, volume samples, per-step data, and readable explanations. The compute workload remains fixed at 1280x720. See `LIVE_INSPECTOR.md`.


## What changed

The original fine cell-DDA path has been replaced with:

```text
coarse brick DDA
    -> occupied brick
        -> compact brick-local micro-BVH
            -> exact ray/triangle intersection
```

The default benchmark scene now targets **100,000 triangles** in a physically large, highly clustered sparse world. It automatically sweeps brick sizes:

```text
4^3, 8^3, 16^3, 32^3
```

It also benchmarks two workloads:

1. **Surface only**
   - global triangle BVH
   - brick DDA -> local micro-BVH

2. **Surface + sparse smoke/fog**
   - global BVH for surfaces **plus a separate brick DDA** for volume
   - one unified brick DDA handling both volume integration and local triangle micro-BVHs

The second workload directly tests the reason EpochEngine needs voxels regardless of a small surface-only timing difference.

## Why this represents the engine direction

The same sparse brick hierarchy can become the shared spatial backbone for:

- world streaming and residency
- extremely sparse or enormous coordinates
- editable voxel terrain
- voxel particles, smoke, fog, fire, and other volumetrics
- procedural geometry and analytic primitives
- temporal branches and locally rebuilt historical states
- mixed representation selection
- aggressive empty-space skipping
- compact local triangle packets
- software ray traversal on non-RT hardware
- Vulkan and OpenGL fallback paths

See [BENCHMARK_PLAN.md](BENCHMARK_PLAN.md) for the planned engine tiers and later benchmark phases.

## Controls

- `1`: global triangle BVH
- `2`: brick DDA -> brick-local micro-BVH
- `[` / `]`: change brick size
- `R`: coherent / incoherent ray set
- `V`: surface-only / surface + sparse smoke-fog
- `F1`: shaded result
- `F2`: exact triangle-test heatmap
- `F3`: hierarchy/DDA traversal heatmap
- `F4`: composite diagnostic
  - red: triangle tests
  - green: BVH/DDA traversal
  - blue: volume samples
- left click: select a debug ray
- `D`: dump the selected ray traversal to the console
- `I`: toggle the owned floating traversal inspector
- `B`: run the complete matrix and write CSV
- `Space`: pause camera orbit
- `Escape`: quit

The window title reports:

- current backend and traversal mode
- workload and ray coherence
- instrumented GPU time
- triangle tests per ray
- traversal operations per ray
- volume samples per ray

## Floating traversal inspector

The inspector belongs to the main benchmark window but floats as its own resizable tool window. It is created with the parent set at creation time and uses a separate SDL software renderer. It never changes the measured Vulkan/OpenGL dispatch or presentation size.

It mirrors one selected ray on the CPU from the same scene structures and displays:

- the ray origin, direction, and distance parameter
- the active chunk/brick field
- occupied surface and volume brick columns
- global BVH node bounds
- brick DDA visits
- brick-local micro-BVH nodes
- leaf triangle-reference indices
- exact triangle hits and nearest hit distance
- volume samples
- per-step brick, node, reference, density, and position data
- explicit global and unified traversal pipelines
- a distance-ordered timeline
- readable text and a plain-language explanation

A ray is only `origin + direction * t`; it does not point to a triangle. BVH leaves and brick-local leaves hold triangle indices that become candidates for exact intersection tests.

See [LIVE_INSPECTOR.md](LIVE_INSPECTOR.md) for the full guide.

For a ground-up explanation of rays, BVHs, brick DDA, micro-BVHs, triangle candidates, and volume sampling, see [RAY_TRAVERSAL_EXPLAINED.md](RAY_TRAVERSAL_EXPLAINED.md).


## Automated benchmark

### Windows — both backends

```powershell
.\benchmark.bat
```

The Windows scripts automatically use:

```text
C:\Users\iammi\source\repos\vcpkg
```

unless `VCPKG_ROOT` points to another valid checkout.

Run a larger scene:

```powershell
.\benchmark.bat -Triangles 1000000
```

Run one backend:

```powershell
.\benchmark.bat -Backend Vulkan -Triangles 100000
.\benchmark.bat -Backend OpenGL -Triangles 100000
```

Force the generator:

```powershell
.\benchmark.bat -Generator VS2022
```


### Rerun only OpenGL after a completed Vulkan run

```powershell
.\benchmark.bat -Backend OpenGL -Generator VS2022 -Triangles 100000
```

This preserves the existing Vulkan CSV and writes only `voxel_ray_benchmark_opengl.csv`.

### Linux — both backends

```bash
export VCPKG_ROOT="$HOME/vcpkg"
./benchmark.sh 100000
```

### Output

- `voxel_ray_benchmark_vulkan.csv`
- `voxel_ray_benchmark_opengl.csv`

Each CSV records:

- backend, workload, ray set, traversal mode, and brick size
- median, average, p10, and p90 GPU time
- nanoseconds per ray
- global BVH node tests
- brick DDA steps
- micro-BVH node tests
- exact triangle tests
- volume samples and volume-brick visits
- hits and maximum traversal work
- CPU structure-build time
- structure memory footprint

A speed ratio above `1.0x` means the unified brick path was faster than the paired global-BVH path for that row.

## Interactive build and run

### Windows Vulkan

```powershell
.\build.bat -Backend Vulkan
```

### Windows OpenGL

```powershell
.\build.bat -Backend OpenGL
```

### Linux Vulkan

```bash
./build.sh --backend vulkan
```

### Linux OpenGL

```bash
./build.sh --backend opengl
```

Useful Windows options:

```powershell
.\build.bat -Clean -Backend Vulkan
.\build.bat -NoRun -Backend OpenGL
.\build.bat -Backend Vulkan -Triangles 1000000
.\build.bat -Backend Vulkan -Benchmark
.\build.bat -Backend Vulkan -BenchmarkOnly
.\build.bat -Backend Vulkan -EpochGuiSource C:\path\to\EpochGui
.\build.bat -Backend Vulkan -NoEpochGui
```

Useful Linux options:

```bash
./build.sh --clean --backend vulkan
./build.sh --no-run --backend opengl
./build.sh --backend vulkan --triangles 1000000
./build.sh --backend vulkan --benchmark-only
./build.sh --backend vulkan --epochgui-source ../EpochGui
./build.sh --backend vulkan --no-epochgui
```

## Interactive `B` behavior

Pressing `B` runs the complete matrix, writes the backend CSV, waits for GPU completion, and exits the executable cleanly. This deliberately avoids returning to the live swapchain/render loop after the benchmark has reused its command and timing resources. Re-run the build command to reopen the interactive view.

## EpochGui integration

The inspector uses [EpochGui](https://github.com/Autodidac/EpochGui) for renderer-neutral control layout, hit testing, segmented controls, and its embedded bitmap font. CMake fetches EpochGui by default. A local checkout can be supplied instead:

```powershell
cmake --preset windows-vs2022-release -DEPOCHGUI_SOURCE_DIR=C:/path/to/EpochGui
```

For an offline emergency build, the inspector has a minimal SDL fallback:

```powershell
cmake --preset windows-vs2022-release -DEPOCH_VISUALIZER_USE_EPOCHGUI=OFF
```

The fallback keeps the live traversal tool functional but does not use EpochGui layout/font services.

## vcpkg manifest through CMake

Dependencies are declared in `vcpkg.json`. The scripts only select the vcpkg CMake toolchain and a CMake preset. During CMake configuration, vcpkg manifest mode installs the required packages automatically.

The scripts never call `vcpkg install` directly.

Manifest dependencies include:

- SDL3
- Vulkan headers and loader
- `shaderc` host tools for `glslc`
- OpenGL
- GLAD generated for the OpenGL 4.3 core API

## Benchmark fairness

- Both traversal modes use the same generated scene, camera, rays, triangles, shading, and output resolution.
- Both surface modes finish with the same exact Möller-Trumbore ray/triangle test.
- Timed shaders compile out debug counters and selected-ray logging.
- Instrumented work counters are gathered in a separate dispatch.
- Vulkan uses timestamp queries.
- OpenGL uses `GL_TIME_ELAPSED` queries.
- The hybrid accepts triangle hits only inside the active brick interval.
- In the volume workload, both modes use the same sparse brick volume data.
- The global BVH path performs a second brick traversal for volume; the unified path reuses its existing brick traversal.

## What the result means

Surface-only timing answers whether the voxel broad phase beats a software global BVH for exact triangles.

The volume workload answers the more important engine question:

> Once EpochEngine already needs sparse bricks for volumetrics, editing, streaming, procedural content, and fallback rendering, does one unified traversal remain competitive with—or beat—separate specialized structures?

A result that is equal or slightly faster is meaningful because the unified path also supplies services not represented by surface timing alone.

## Current limits

- The CPU BVH builder uses median splitting, not SAH, HLBVH, or GPU construction.
- Volume integration is one homogeneous sample per crossed occupied brick; it represents traversal cost, not final production-quality scattering.
- The sparse volume sources are static procedural smoke/fog proxies, not a particle simulation yet.
- No hardware RT comparison is included yet.
- No dirty-brick incremental rebuild benchmark is included yet.
- No streaming, temporal-branch switching, SDF, or analytic-primitive payload is included yet.
- CPU SIMD fallback remains planned rather than implemented in this package.

## Build fix 0.5.1

This package fixes the MSVC `C2668` ambiguity in `src/ray_inspector.cpp` caused by
the local hit-test helper sharing the name `contains` with EpochGui. The local helper
is now named `ui_contains`. No benchmark shader or timing code was changed.


## Build fix 0.5.2

The live inspector adds a second C++ translation unit. The original benchmark's
`benchmark_common.hpp` defined several non-template free functions directly in the header
without `inline`, which caused MSVC `LNK2005`/`LNK1169` duplicate-symbol failures between
`main_vulkan.obj` or `main_opengl.obj` and `ray_inspector.obj`. Version 0.5.2 marks those
shared-header implementations `inline`. This is an ODR/linkage correction only; scene
generation, traversal, shaders, benchmark timing, and CSV output are unchanged.
