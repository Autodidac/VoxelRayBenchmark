# Sample result

`voxel_ray_benchmark_vulkan_rtx5080.csv` is the latest supplied Vulkan run used while designing the live inspector.

Treat its absolute milliseconds as machine- and run-specific. The useful stable signal was the relative pattern:

- global BVH favored surface-only and incoherent rays
- unified `4^3`/`8^3` brick traversal was competitive or faster for coherent surface-plus-volume work
- larger bricks increased incoherent traversal cost
- `8^3` was the strongest general compromise between performance, memory, and duplicated triangle references

Use the live inspector to explain individual rays, then use a fresh CSV from the target machine for final policy decisions.
