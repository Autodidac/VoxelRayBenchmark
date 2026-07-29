# EpochEngine unified spatial benchmark plan

## Decision being tested

EpochEngine needs a sparse voxel hierarchy even when exact visible surfaces remain triangles. The useful comparison is not only:

```text
voxel traversal versus triangle BVH
```

It is:

```text
one shared sparse spatial backbone
versus
multiple independent acceleration structures and traversal passes
```

The shared backbone should be retained when it is equal, slightly faster, or acceptably close in raw surface timing while materially reducing duplicated systems, duplicated world queries, and duplicated streaming state.

## Implemented tier

### Tier 1 — portable GPU software rays

Implemented in this package:

```text
SDL3
├── Vulkan 1.3 compute
└── OpenGL 4.3+ compute

shared scene data
├── global triangle BVH
└── sparse brick table
    ├── volume payload
    └── brick-local triangle micro-BVH
```

This tier works without Vulkan hardware ray-tracing extensions and gives EpochEngine a modern GPU fallback path.

## Planned runtime tiers

### Tier 0 — CPU fallback

Planned:

- C++23 job-system traversal
- SIMD packet traversal where profitable
- headless/server compatibility
- deterministic verification path
- low-end and unsupported-GPU fallback

### Tier 1 — OpenGL/Vulkan compute fallback

Implemented baseline:

- global BVH software traversal
- sparse brick DDA
- brick-local micro-BVHs
- exact triangles
- sparse volumetric integration

### Tier 2 — Vulkan ray query

Planned:

- retain brick hierarchy for streaming, volume, terrain, temporal state, and representation selection
- use hardware ray queries for local resident triangle packets where faster
- compare direct local software micro-BVH against chunk-local hardware BLAS

### Tier 3 — Vulkan RT pipeline

Planned:

- TLAS of resident chunks/instances
- triangle BLAS for conventional meshes
- procedural AABBs for voxel/SDF/analytic payloads
- shared brick hierarchy remains the world-residency and representation-routing layer

## Representation routing target

```text
sparse world brick
├── empty: jump to brick exit
├── triangle packet: local micro-BVH or hardware BLAS
├── exact voxel payload: voxel hit
├── SDF payload: sphere tracing / specialized intersection
├── volume payload: smoke, fog, fire, cloud, particles
├── analytic primitive: direct intersection
└── procedural payload: generate/evaluate locally
```

The ray should select the cheapest exact representation available in the entered region.

## Current benchmark matrix

For each backend:

- coherent primary rays
- incoherent secondary-like rays
- brick sizes `4^3`, `8^3`, `16^3`, `32^3`
- surface-only workload
- surface + sparse smoke/fog workload

Surface comparison:

```text
global BVH
versus
brick DDA -> local micro-BVH
```

Unified comparison:

```text
global BVH -> exact surfaces
+ separate brick DDA -> volume

versus

one brick DDA
├── volume integration
└── local micro-BVH -> exact surfaces
```

## Scale sweep

Run at minimum:

```text
100,000 triangles
1,000,000 triangles
10,000,000 triangles
```

The largest case may require reducing resolution or benchmark sample count later. Do not infer scaling from the 100k case alone.

## Metrics to retain

### GPU

- median, average, p10, p90 dispatch time
- nanoseconds per ray
- BVH node tests
- brick steps
- micro-BVH node tests
- exact triangle tests
- volume samples
- occupied brick visits
- maximum traversal work

### CPU and memory

- global BVH build time
- each brick variant build time
- dirty-brick rebuild time — planned
- triangle buffer size
- global BVH size
- brick headers
- local node/reference size
- duplication ratio
- streamed resident bytes — planned

## Engine-specific follow-up workloads

### Editable terrain

- modify a localized region
- rebuild only dirty bricks
- compare rebuild latency against global BVH rebuild/refit
- measure frame-time spikes and temporary memory

### World streaming and enormous coordinates

- move through a multi-zone sparse world
- load/unload brick pages
- use local floating origins or integer world-page coordinates
- measure residency changes without rebuilding unrelated geometry

### Temporal branches

- fork a local timeline region
- share unchanged brick pages between branches
- rebuild only changed branch-local bricks
- measure branch-switch latency and memory sharing

### Voxel particles and volumetrics

- animated sparse particle occupancy
- density/emission updates without triangle rebuilds
- temporal smoke trails
- multiple scattering-quality tiers
- compact volume clipmaps for distance

### Mixed representation

- triangles
- exact voxels
- SDF bricks
- analytic spheres/capsules/planes
- procedural terrain patches
- representation selection cost and correctness

### Hardware RT comparison

- software global BVH
- software brick + micro-BVH
- global hardware BLAS/TLAS
- brick/chunk-local hardware BLAS selected through the same sparse backbone

## Decision rules

Prefer the unified brick backbone when it:

1. wins GPU time; or
2. remains within a small measured margin while replacing a separate volume/world-query pass; or
3. materially improves update, streaming, temporal, or fallback behavior without unacceptable memory growth.

Reject or redesign a configuration when:

- triangle reference duplication grows uncontrollably
- brick headers dominate memory
- incoherent-ray divergence erases traversal savings
- dirty updates require broad rebuilds
- a brick size wins one workload but catastrophically loses another

The expected final design may use different brick sizes or hierarchy levels per payload type rather than one global fixed size.

## Live traversal diagnostics

The interactive package now includes a separate live inspector that mirrors one selected ray on the CPU through both the global and unified structures. It visualizes node visits, brick steps, local micro-BVH visits, volume samples, and exact hits in real time.

This diagnostic path is intentionally not part of benchmark timing. Automated GPU dispatches, timestamp queries, shaders, and CSV output remain unchanged. The inspector exists to explain aggregate timing differences and to help identify rays or regions that deserve new benchmark workloads.
