# Epoch voxel live inspector 0.6.0

## Combined workspace

- Replaced the independent inspector window on supported platforms with a borderless SDL child window docked to the main Vulkan/OpenGL host.
- Expanded the interactive host to reserve a right-side panel while preserving a fixed 1280x720 compute workload.
- Added correct aspect-fit presentation and click-to-ray mapping through the visible left-side scene viewport.
- Added automatic child-panel resizing with a 440-680 pixel width range.
- Added a standalone fallback when native child-window parenting is unavailable.

## Visualization depth

- Added a top-down world/brick map with Y collapsed.
- Displays empty, surface, volume, and mixed brick columns.
- Displays bricks visited by both traversal methods.
- Displays current global BVH and micro-BVH bounds.
- Displays the exact hit triangle.
- Added active pipeline stages for the global and unified methods.
- Added current-step details for node indices, bounds, brick coordinates, flattened indices, local node counts, triangle references, density, position, and hit distance.
- Increased text and control sizes.
- Added wrapped explanations and a larger event timeline.

## Playback behavior

- Captures are stable snapshots instead of being regenerated every frame.
- Pause now freezes both event progression and the diagnostic ray data.
- Playback stops at the final event rather than immediately wrapping.
- Added an explicit Capture button and Enter-key shortcut.
- Selecting another pixel, brick variant, ray set, or workload automatically creates a new capture.
- Opening the workspace pauses the main camera so the left viewport initially matches the captured ray.

## Benchmark isolation

- `shaders/trace.comp` is byte-for-byte unchanged from 0.5.2.
- Vulkan and OpenGL compute dispatch dimensions remain 1280x720.
- The wider host window affects presentation only.
- Vulkan timestamps and OpenGL elapsed-time queries still bracket only compute work.
- `--benchmark-only` does not create the panel.
