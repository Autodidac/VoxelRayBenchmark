# Epoch voxel live inspector 0.6.1

## Window-lifetime fix

- Removed the docked/reparented child-window implementation.
- Removed every runtime call to `SDL_SetWindowParent`.
- Creates the inspector with `SDL_CreateWindowWithProperties`.
- Supplies `SDL_PROP_WINDOW_CREATE_PARENT_POINTER` before creation.
- Marks the inspector as an SDL utility window so it behaves as an owned tool window rather than a second application.
- Keeps the inspector movable and resizable.
- Verifies the parent relationship with `SDL_GetWindowParent`.
- Creates the SDL software renderer while the child is hidden, then shows it.
- Main Vulkan/OpenGL windows are back to a normal 1280x720 presentation area and no longer reserve dock width.

## Explanation improvement

- Added an always-visible statement that a ray is `origin + direction * t`.
- Clarified that BVH and micro-BVH leaves contain triangle indices; the ray itself contains no triangle pointer.
- Updated the guide with explicit global-BVH and brick-DDA traversal sequences.

## Benchmark isolation

- `shaders/trace.comp` is unchanged.
- Benchmark dispatch dimensions remain 1280x720.
- Vulkan/OpenGL timing scopes are unchanged.
- The inspector remains outside timestamped GPU work.
