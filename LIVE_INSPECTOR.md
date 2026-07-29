# Live traversal inspector — owned floating child window

The interactive Vulkan and OpenGL executables use two windows that belong to one application:

```text
main benchmark window
└── owned utility child: traversal inspector
```

The inspector is created with `SDL_CreateWindowWithProperties` and the parent relationship is supplied through `SDL_PROP_WINDOW_CREATE_PARENT_POINTER` **before the window exists**. It is a movable, resizable floating tool window. It does not appear as a second independent application, and hiding or destroying the parent applies to the child.

The inspector no longer calls `SDL_SetWindowParent`, no longer embeds itself into the main client area, and no longer resizes or repositions itself every frame. This removes the unstable reparent-and-dock path that caused Win32 crashes.

## Benchmark isolation

The visualization does not change the measured compute dispatch:

- benchmark shaders are unchanged
- Vulkan timestamps still bracket only the compute dispatch
- OpenGL `GL_TIME_ELAPSED` still brackets only the compute dispatch
- the traced GPU image remains 1280x720
- the child inspector uses an SDL software renderer after the benchmark image is presented
- `--benchmark-only` does not create the inspector

## What a ray actually is

A ray contains only:

```text
origin       where it starts
 direction   which way it travels
 t range     how far along it may travel
```

Mathematically, a point along it is:

```text
P(t) = origin + direction * t
```

A ray is **not** a pointer to a triangle. Triangle indices appear later, inside BVH or micro-BVH leaf nodes. Traversal finds candidate leaves, then exact ray/triangle tests determine whether a triangle is hit. The smallest valid `t` is the first visible surface.

## Global BVH traversal

```text
ray
→ root bounding box
→ nearer child boxes
→ reject whole groups of triangles
→ leaf containing triangle indices
→ exact ray/triangle tests
→ nearest hit
```

For surface-plus-volume work, the global path then performs a separate brick walk for smoke or fog.

## Brick DDA plus micro-BVH traversal

```text
ray
→ world grid bounds
→ cross brick boundaries with DDA
→ skip empty bricks
→ inspect occupied brick header
→ traverse that brick's micro-BVH
→ leaf containing local triangle indices
→ exact ray/triangle tests
→ sample volume in the same brick
```

The performance tradeoff is between:

- global hierarchy node tests
- empty-space brick steps
- local micro-BVH node tests
- exact triangle tests
- memory locality
- volume traversal reuse
- coherence between neighboring rays

## What the inspector displays

- selected ray origin and direction
- top-down brick field
- empty, surface, volume, and mixed brick columns
- brick coordinates and flattened brick index
- global BVH and local micro-BVH bounds
- leaf reference counts
- triangle candidate and hit indices
- exact hit distance `t`
- volume segments and density
- per-method metrics and GPU sample time
- event timeline and current traversal stage

The status panel explicitly states that leaf indices refer to triangles while the ray itself does not.

## Controls

Main window:

- left click: select a ray
- `I`: open or close the owned floating inspector
- `Space`: pause the camera orbit

Inspector window:

- `Play/Pause`: animate the captured traversal
- `Prev/Next`: move one event
- `Reset`: return to the first event
- `Capture`: rebuild both diagnostic traces from the current camera and selected pixel
- `-/+`: playback rate
- `Both/BVH/Voxel`: comparison visibility
- timeline click: seek
- keyboard: Space, Left, Right, Home, Enter, 1, 2, 3, minus, equals

Playback remains snapshot-based. Pausing freezes event progression, and a capture changes only when the selected pixel/configuration changes or `Capture` is pressed.
