# What the ray benchmark is actually doing

## A ray is not a triangle pointer

A ray is only three pieces of information:

```text
origin       where the ray begins
direction    which way it travels
t range      how far it is allowed to travel
```

A point on the ray is:

```text
P(t) = origin + direction * t
```

For a camera ray, the origin is the camera and the direction passes through one pixel. The ray does not know which object or triangle it will hit.

Triangle indices are stored inside acceleration structures. The traversal algorithm discovers candidate triangle indices and tests them. The nearest successful intersection becomes the result.

## Why not test every triangle?

If a scene has 100,000 triangles, the slow direct method is:

```text
for every ray:
    test triangle 0
    test triangle 1
    ...
    test triangle 99,999
```

At 1280x720, that would mean 921,600 rays. Testing every ray against every triangle would approach 92 billion triangle tests for one image.

Acceleration structures avoid most of those tests.

## Global BVH path

A BVH is a tree of bounding boxes.

```text
root box: entire scene
├── box: left half of scene
│   ├── smaller box
│   └── smaller box
└── box: right half of scene
    ├── smaller box
    └── smaller box
```

Each leaf contains triangle indices, not triangles pointed to by the ray.

Traversal:

```text
1. Test the ray against the root box.
2. If it misses, reject the entire scene.
3. If it hits, test the child boxes.
4. Reject every box the ray misses.
5. Continue until reaching a leaf.
6. Read the triangle indices stored in that leaf.
7. Perform exact ray/triangle tests.
8. Keep the hit with the smallest valid t.
```

The orange boxes in the inspector show the hierarchy regions currently being considered.

## Brick DDA plus micro-BVH path

The voxel-style path divides the world into coarse 3-D bricks. It does not mean the whole map is stored like Minecraft. A brick is a spatial address and a container for optional data.

A brick header can say:

```text
empty
surface triangle references exist
volume density exists
both surface and volume exist
local micro-BVH starts at index N
```

DDA means the ray advances from one brick boundary to the next instead of taking tiny fixed steps.

Traversal:

```text
1. Intersect the ray with the world grid.
2. Determine the first brick coordinate.
3. Compute which X, Y, or Z brick boundary comes next.
4. Advance directly to that boundary.
5. Read the new brick header.
6. Skip the brick immediately if it is empty.
7. If surface data exists, traverse that brick's micro-BVH.
8. Read triangle indices from local leaves and test them exactly.
9. If volume data exists, integrate density over the ray segment in that brick.
10. Continue until a surface blocks the ray or the ray leaves the world.
```

The green grid cells in the inspector are bricks visited by this method. Purple boxes are local micro-BVH regions.

## What `t` means

`t` is distance along the ray when the direction is normalized.

```text
t = 0       ray origin
t = 5       point five world units along the ray
t = 100     point one hundred units along the ray
```

The nearest visible triangle is the valid hit with the lowest `t`.

## What the counters mean

### Triangle tests

Exact mathematical ray/triangle intersection tests. These are the final expensive candidates after broad rejection.

### Global BVH nodes

Bounding boxes visited in the scene-wide tree.

### Brick steps

Coarse grid cells crossed by DDA. Empty bricks can be skipped without touching triangles.

### Micro-BVH nodes

Bounding boxes visited inside occupied bricks.

### Volume samples

Ray segments integrated through smoke, fog, gas, or another density field.

### GPU time

Aggregate time for the complete 1280x720 dispatch. One-ray visualization explains behavior; GPU time proves whether that behavior helps across all rays.

## Why the two methods can have close timings

The voxel route may perform far fewer exact triangle tests but spend more time on:

- brick boundary calculations
- brick-header reads
- local hierarchy traversal
- divergent control flow
- scattered memory access

The global BVH may test more triangles but walk a simpler, more coherent hierarchy.

That is why the correct engine design is hybrid:

```text
coherent surface + volume rays
    often benefit from unified brick traversal

surface-only or incoherent secondary rays
    often benefit from the global BVH or hardware ray tracing

navigation, occupancy, destruction, streaming, and temporal reconstruction
    benefit from the shared voxel/brick spatial representation
```

The benchmark is measuring where each method should be selected, not trying to prove that one structure replaces every other structure.
