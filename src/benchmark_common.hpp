#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace epoch::voxel_demo {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr u32 kGridX = 256;
constexpr u32 kGridY = 64;
constexpr u32 kGridZ = 256;
constexpr std::array<u32, 4> kBrickSizes{4, 8, 16, 32};
constexpr u32 kMetricCount = 10;
constexpr u32 kMaxDebugEvents = 256;
constexpr u32 kBenchmarkWarmups = 8;
constexpr u32 kBenchmarkSamples = 64;
constexpr u32 kGlobalLeafSize = 8;
constexpr u32 kMicroLeafSize = 4;

[[noreturn]] inline void fail(std::string_view message) {
    throw std::runtime_error(std::string{message});
}

struct Vec3 {
    float x{};
    float y{};
    float z{};
};

struct alignas(16) Vec4 {
    float x{};
    float y{};
    float z{};
    float w{};
};

[[nodiscard]] constexpr Vec3 operator+(Vec3 lhs, Vec3 rhs) noexcept {
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

[[nodiscard]] constexpr Vec3 operator-(Vec3 lhs, Vec3 rhs) noexcept {
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

[[nodiscard]] constexpr Vec3 operator*(Vec3 value, float scalar) noexcept {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

[[nodiscard]] constexpr Vec3 operator/(Vec3 value, float scalar) noexcept {
    return {value.x / scalar, value.y / scalar, value.z / scalar};
}

[[nodiscard]] constexpr float dot(Vec3 lhs, Vec3 rhs) noexcept {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

[[nodiscard]] constexpr Vec3 cross(Vec3 lhs, Vec3 rhs) noexcept {
    return {
        lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.x * rhs.y - lhs.y * rhs.x,
    };
}

[[nodiscard]] inline float length(Vec3 value) noexcept {
    return std::sqrt(dot(value, value));
}

[[nodiscard]] inline Vec3 normalize(Vec3 value) noexcept {
    const float magnitude = length(value);
    return magnitude > 1.0e-8F ? value / magnitude : Vec3{};
}

[[nodiscard]] constexpr Vec3 minimum(Vec3 lhs, Vec3 rhs) noexcept {
    return {std::min(lhs.x, rhs.x), std::min(lhs.y, rhs.y), std::min(lhs.z, rhs.z)};
}

[[nodiscard]] constexpr Vec3 maximum(Vec3 lhs, Vec3 rhs) noexcept {
    return {std::max(lhs.x, rhs.x), std::max(lhs.y, rhs.y), std::max(lhs.z, rhs.z)};
}

[[nodiscard]] constexpr Vec4 to_vec4(Vec3 value, float w = 0.0F) noexcept {
    return {value.x, value.y, value.z, w};
}

struct Aabb {
    Vec3 minimumPoint{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()};
    Vec3 maximumPoint{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()};

    void expand(Vec3 point) noexcept {
        minimumPoint = minimum(minimumPoint, point);
        maximumPoint = maximum(maximumPoint, point);
    }

    void expand(const Aabb& bounds) noexcept {
        expand(bounds.minimumPoint);
        expand(bounds.maximumPoint);
    }

    [[nodiscard]] Vec3 extent() const noexcept {
        return maximumPoint - minimumPoint;
    }

    [[nodiscard]] Vec3 centroid() const noexcept {
        return (minimumPoint + maximumPoint) * 0.5F;
    }
};

[[nodiscard]] inline Aabb clipped_bounds(const Aabb& bounds, const Aabb& clip) noexcept {
    Aabb result;
    result.minimumPoint = maximum(bounds.minimumPoint, clip.minimumPoint);
    result.maximumPoint = minimum(bounds.maximumPoint, clip.maximumPoint);
    return result;
}

struct alignas(16) TriangleGpu {
    Vec4 v0;
    Vec4 v1;
    Vec4 v2;
    Vec4 color;
};
static_assert(sizeof(TriangleGpu) == 64);

struct alignas(16) BvhNodeGpu {
    Vec4 minimumPoint;
    Vec4 maximumPoint;
    std::array<u32, 4> meta{};
};
static_assert(sizeof(BvhNodeGpu) == 48);

struct alignas(16) BrickHeaderGpu {
    u32 nodeOffset{};
    u32 nodeCount{};
    u32 referenceOffset{};
    u32 referenceCount{};
    Vec4 volume{}; // rgb scattering/emission tint, alpha extinction density
};
static_assert(sizeof(BrickHeaderGpu) == 32);

struct alignas(16) TraversalConfigGpu {
    u32 headerOffset{};
    u32 brickSize{};
    u32 brickDimensionX{};
    u32 brickDimensionY{};
    u32 brickDimensionZ{};
    u32 rayMode{};
    u32 variantIndex{};
    u32 workloadMode{}; // 0 surface-only, 1 surface + sparse volume
};
static_assert(sizeof(TraversalConfigGpu) == 32);

struct alignas(16) DebugEventGpu {
    Vec4 positionT;
    std::array<u32, 4> data{};
};
static_assert(sizeof(DebugEventGpu) == 32);

struct alignas(16) DebugBufferGpu {
    u32 count{};
    u32 overflow{};
    std::array<u32, 2> padding{};
    std::array<DebugEventGpu, kMaxDebugEvents> events{};
};

struct MetricsSnapshot {
    u32 rays{};
    u32 bvhNodeTests{};
    u32 brickSteps{};
    u32 microBvhNodeTests{};
    u32 occupiedBricks{};
    u32 triangleTests{};
    u32 volumeSamples{};
    u32 volumeBricks{};
    u32 hits{};
    u32 maximumTraversal{};
};
static_assert(sizeof(MetricsSnapshot) == kMetricCount * sizeof(u32));

struct alignas(16) PushConstants {
    Vec4 cameraPosition;
    Vec4 cameraForward;
    Vec4 cameraRight;
    Vec4 cameraUp;
    Vec4 worldMinimum;
    Vec4 cellSize;
    std::array<u32, 4> imageMode{};
    std::array<u32, 4> gridSelected{};
};
static_assert(sizeof(PushConstants) == 128);

struct BrickVariant {
    u32 brickSize{};
    u32 dimensionX{};
    u32 dimensionY{};
    u32 dimensionZ{};
    u32 headerOffset{};
    u32 headerCount{};
    u32 nodeOffset{};
    u32 nodeCount{};
    u32 referenceOffset{};
    u32 referenceCount{};
    u32 occupiedBricks{};
    u32 volumeBricks{};
    double buildMs{};

    [[nodiscard]] std::size_t byte_size() const noexcept {
        return static_cast<std::size_t>(headerCount) * sizeof(BrickHeaderGpu) +
               static_cast<std::size_t>(nodeCount) * sizeof(BvhNodeGpu) +
               static_cast<std::size_t>(referenceCount) * sizeof(u32);
    }
};

struct DemoOptions {
    u32 targetTriangles{100'000};
    u32 seed{0xE90C'2026u};
    bool benchmarkOnStart{};
    bool benchmarkOnly{};
};

[[nodiscard]] inline u32 parse_u32(std::string_view text, std::string_view option) {
    u32 value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        fail(std::format("Invalid value for {}: {}", option, text));
    }
    return value;
}

[[nodiscard]] inline DemoOptions parse_options(int argc, char** argv) {
    DemoOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument{argv[index]};
        if (argument == "--benchmark") {
            options.benchmarkOnStart = true;
        } else if (argument == "--benchmark-only") {
            options.benchmarkOnStart = true;
            options.benchmarkOnly = true;
        } else if (argument == "--triangles") {
            if (++index >= argc) {
                fail("--triangles requires a value.");
            }
            options.targetTriangles = std::max(parse_u32(argv[index], "--triangles"), 12u);
        } else if (argument == "--seed") {
            if (++index >= argc) {
                fail("--seed requires a value.");
            }
            options.seed = parse_u32(argv[index], "--seed");
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: epoch_voxel_*_demo [--triangles N] [--seed N] [--benchmark|--benchmark-only]\n"
                << "  --triangles N    Target scene triangle count; rounded to whole boxes.\n"
                << "  --seed N         Deterministic scene seed.\n"
                << "  --benchmark      Run the full matrix immediately, then remain interactive.\n"
                << "  --benchmark-only Run the full matrix, write CSV, then exit.\n";
            std::exit(0);
        } else {
            fail(std::format("Unknown argument: {}", argument));
        }
    }
    return options;
}

struct SceneData {
    std::vector<TriangleGpu> triangles;
    std::vector<BvhNodeGpu> bvhNodes;
    std::vector<u32> bvhTriangleReferences;
    std::vector<BrickHeaderGpu> brickHeaders;
    std::vector<BvhNodeGpu> microBvhNodes;
    std::vector<u32> microTriangleReferences;
    std::vector<BrickVariant> brickVariants;
    Vec3 worldMinimum{-128.0F, 0.0F, -128.0F};
    Vec3 cellSize{1.0F, 1.0F, 1.0F};
    u32 gridX{kGridX};
    u32 gridY{kGridY};
    u32 gridZ{kGridZ};
    double bvhBuildMs{};
    double voxelBuildMs{};
};

[[nodiscard]] inline Aabb triangle_bounds(const TriangleGpu& triangle) noexcept {
    Aabb bounds;
    bounds.expand(Vec3{triangle.v0.x, triangle.v0.y, triangle.v0.z});
    bounds.expand(Vec3{triangle.v1.x, triangle.v1.y, triangle.v1.z});
    bounds.expand(Vec3{triangle.v2.x, triangle.v2.y, triangle.v2.z});
    return bounds;
}

inline void add_triangle(SceneData& scene, Vec3 a, Vec3 b, Vec3 c, Vec3 color) {
    scene.triangles.push_back({to_vec4(a, 1.0F), to_vec4(b, 1.0F), to_vec4(c, 1.0F), to_vec4(color, 1.0F)});
}

inline void add_box(SceneData& scene, Vec3 center, Vec3 halfExtent, Vec3 color) {
    const Vec3 p000{center.x - halfExtent.x, center.y - halfExtent.y, center.z - halfExtent.z};
    const Vec3 p001{center.x - halfExtent.x, center.y - halfExtent.y, center.z + halfExtent.z};
    const Vec3 p010{center.x - halfExtent.x, center.y + halfExtent.y, center.z - halfExtent.z};
    const Vec3 p011{center.x - halfExtent.x, center.y + halfExtent.y, center.z + halfExtent.z};
    const Vec3 p100{center.x + halfExtent.x, center.y - halfExtent.y, center.z - halfExtent.z};
    const Vec3 p101{center.x + halfExtent.x, center.y - halfExtent.y, center.z + halfExtent.z};
    const Vec3 p110{center.x + halfExtent.x, center.y + halfExtent.y, center.z - halfExtent.z};
    const Vec3 p111{center.x + halfExtent.x, center.y + halfExtent.y, center.z + halfExtent.z};

    add_triangle(scene, p001, p101, p111, color);
    add_triangle(scene, p001, p111, p011, color);
    add_triangle(scene, p100, p000, p010, color);
    add_triangle(scene, p100, p010, p110, color);
    add_triangle(scene, p000, p001, p011, color);
    add_triangle(scene, p000, p011, p010, color);
    add_triangle(scene, p101, p100, p110, color);
    add_triangle(scene, p101, p110, p111, color);
    add_triangle(scene, p010, p011, p111, color);
    add_triangle(scene, p010, p111, p110, color);
    add_triangle(scene, p000, p100, p101, color);
    add_triangle(scene, p000, p101, p001, color);
}

[[nodiscard]] inline Aabb bounds_for_triangle(const SceneData& scene, u32 triangleIndex, const Aabb* clip) noexcept {
    const Aabb bounds = triangle_bounds(scene.triangles[triangleIndex]);
    return clip != nullptr ? clipped_bounds(bounds, *clip) : bounds;
}

inline u32 build_bvh_node(
    const SceneData& scene,
    std::vector<BvhNodeGpu>& nodes,
    std::vector<u32>& references,
    std::vector<u32>& indices,
    u32 begin,
    u32 end,
    u32 leafSize,
    const Aabb* clip) {
    const u32 nodeIndex = static_cast<u32>(nodes.size());
    nodes.emplace_back();

    Aabb bounds;
    Aabb centroidBounds;
    for (u32 index = begin; index < end; ++index) {
        const Aabb triangleBounds = bounds_for_triangle(scene, indices[index], clip);
        bounds.expand(triangleBounds);
        centroidBounds.expand(triangleBounds.centroid());
    }

    BvhNodeGpu node{};
    node.minimumPoint = to_vec4(bounds.minimumPoint);
    node.maximumPoint = to_vec4(bounds.maximumPoint);

    const u32 count = end - begin;
    if (count <= leafSize) {
        node.meta[2] = static_cast<u32>(references.size());
        node.meta[3] = count;
        references.insert(
            references.end(),
            indices.begin() + static_cast<std::ptrdiff_t>(begin),
            indices.begin() + static_cast<std::ptrdiff_t>(end));
        nodes[nodeIndex] = node;
        return nodeIndex;
    }

    const Vec3 centroidExtent = centroidBounds.extent();
    const int axis = centroidExtent.x >= centroidExtent.y && centroidExtent.x >= centroidExtent.z
        ? 0
        : (centroidExtent.y >= centroidExtent.z ? 1 : 2);
    const u32 middle = begin + count / 2;

    auto centroid_axis = [&](u32 triangleIndex) {
        const Vec3 centroid = bounds_for_triangle(scene, triangleIndex, clip).centroid();
        return axis == 0 ? centroid.x : (axis == 1 ? centroid.y : centroid.z);
    };

    std::nth_element(
        indices.begin() + static_cast<std::ptrdiff_t>(begin),
        indices.begin() + static_cast<std::ptrdiff_t>(middle),
        indices.begin() + static_cast<std::ptrdiff_t>(end),
        [&](u32 lhs, u32 rhs) { return centroid_axis(lhs) < centroid_axis(rhs); });

    node.meta[0] = build_bvh_node(scene, nodes, references, indices, begin, middle, leafSize, clip);
    node.meta[1] = build_bvh_node(scene, nodes, references, indices, middle, end, leafSize, clip);
    nodes[nodeIndex] = node;
    return nodeIndex;
}

[[nodiscard]] inline u32 flatten_3d(u32 x, u32 y, u32 z, u32 width, u32 height) noexcept {
    return x + width * (y + height * z);
}

struct SmokeSource {
    Vec3 center;
    float radius{};
    Vec3 color;
    float density{};
};

[[nodiscard]] inline Vec4 procedural_volume(Vec3 brickCenter, Vec3 brickHalfExtent) noexcept {
    static constexpr std::array<SmokeSource, 6> sources{{
        {{-58.0F, 15.0F, -54.0F}, 28.0F, {0.48F, 0.62F, 0.88F}, 0.16F},
        {{ 44.0F, 11.0F, -38.0F}, 22.0F, {0.90F, 0.48F, 0.32F}, 0.20F},
        {{ 12.0F, 23.0F,  52.0F}, 34.0F, {0.54F, 0.82F, 0.58F}, 0.13F},
        {{-82.0F,  8.0F,  64.0F}, 18.0F, {0.74F, 0.54F, 0.86F}, 0.22F},
        {{ 82.0F, 18.0F,  76.0F}, 26.0F, {0.72F, 0.78F, 0.92F}, 0.15F},
        {{  0.0F, 34.0F,   0.0F}, 20.0F, {0.82F, 0.84F, 0.90F}, 0.12F},
    }};

    float density = 0.0F;
    Vec3 weightedColor{};
    const float supportPadding = length(brickHalfExtent);
    for (const SmokeSource& source : sources) {
        const float distance = length(brickCenter - source.center);
        const float support = source.radius + supportPadding;
        if (distance >= support) {
            continue;
        }
        const float normalized = std::clamp(1.0F - distance / support, 0.0F, 1.0F);
        const float contribution = source.density * normalized * normalized;
        density += contribution;
        weightedColor = weightedColor + source.color * contribution;
    }

    if (density <= 1.0e-5F) {
        return {};
    }
    const Vec3 color = weightedColor / density;
    return {color.x, color.y, color.z, std::min(density, 0.65F)};
}

[[nodiscard]] inline std::array<u32, 6> triangle_brick_range(
    const SceneData& scene,
    const TriangleGpu& triangle,
    u32 brickSize,
    u32 brickX,
    u32 brickY,
    u32 brickZ) {
    const Aabb bounds = triangle_bounds(triangle);
    const Vec3 brickWorldSize = scene.cellSize * static_cast<float>(brickSize);
    auto convert = [](float value, float minimumValue, float step, u32 dimension) {
        const int raw = static_cast<int>(std::floor((value - minimumValue) / step));
        return static_cast<u32>(std::clamp(raw, 0, static_cast<int>(dimension) - 1));
    };

    return {
        convert(bounds.minimumPoint.x, scene.worldMinimum.x, brickWorldSize.x, brickX),
        convert(bounds.maximumPoint.x, scene.worldMinimum.x, brickWorldSize.x, brickX),
        convert(bounds.minimumPoint.y, scene.worldMinimum.y, brickWorldSize.y, brickY),
        convert(bounds.maximumPoint.y, scene.worldMinimum.y, brickWorldSize.y, brickY),
        convert(bounds.minimumPoint.z, scene.worldMinimum.z, brickWorldSize.z, brickZ),
        convert(bounds.maximumPoint.z, scene.worldMinimum.z, brickWorldSize.z, brickZ),
    };
}

inline void build_brick_variant(SceneData& scene, u32 brickSize) {
    const auto start = std::chrono::steady_clock::now();
    BrickVariant variant{};
    variant.brickSize = brickSize;
    variant.dimensionX = (scene.gridX + brickSize - 1) / brickSize;
    variant.dimensionY = (scene.gridY + brickSize - 1) / brickSize;
    variant.dimensionZ = (scene.gridZ + brickSize - 1) / brickSize;
    variant.headerOffset = static_cast<u32>(scene.brickHeaders.size());
    variant.nodeOffset = static_cast<u32>(scene.microBvhNodes.size());
    variant.referenceOffset = static_cast<u32>(scene.microTriangleReferences.size());

    const u64 brickCount64 = static_cast<u64>(variant.dimensionX) * variant.dimensionY * variant.dimensionZ;
    if (brickCount64 > std::numeric_limits<u32>::max()) {
        fail("Brick header table exceeded 32-bit indexing.");
    }
    const u32 brickCount = static_cast<u32>(brickCount64);
    variant.headerCount = brickCount;
    std::vector<u32> counts(brickCount, 0u);

    for (const TriangleGpu& triangle : scene.triangles) {
        const auto range = triangle_brick_range(
            scene, triangle, brickSize, variant.dimensionX, variant.dimensionY, variant.dimensionZ);
        for (u32 z = range[4]; z <= range[5]; ++z) {
            for (u32 y = range[2]; y <= range[3]; ++y) {
                for (u32 x = range[0]; x <= range[1]; ++x) {
                    ++counts[flatten_3d(x, y, z, variant.dimensionX, variant.dimensionY)];
                }
            }
        }
    }

    std::vector<u32> offsets(brickCount + 1, 0u);
    u64 totalAssignments = 0;
    for (u32 index = 0; index < brickCount; ++index) {
        offsets[index] = static_cast<u32>(totalAssignments);
        totalAssignments += counts[index];
        if (totalAssignments > std::numeric_limits<u32>::max()) {
            fail(std::format("{}^3 brick assignment table exceeded 32-bit indexing.", brickSize));
        }
    }
    offsets[brickCount] = static_cast<u32>(totalAssignments);

    std::vector<u32> assignments(static_cast<std::size_t>(totalAssignments));
    std::vector<u32> cursors(offsets.begin(), offsets.end() - 1);
    for (u32 triangleIndex = 0; triangleIndex < scene.triangles.size(); ++triangleIndex) {
        const auto range = triangle_brick_range(
            scene, scene.triangles[triangleIndex], brickSize,
            variant.dimensionX, variant.dimensionY, variant.dimensionZ);
        for (u32 z = range[4]; z <= range[5]; ++z) {
            for (u32 y = range[2]; y <= range[3]; ++y) {
                for (u32 x = range[0]; x <= range[1]; ++x) {
                    const u32 brickIndex = flatten_3d(x, y, z, variant.dimensionX, variant.dimensionY);
                    assignments[cursors[brickIndex]++] = triangleIndex;
                }
            }
        }
    }

    scene.brickHeaders.resize(scene.brickHeaders.size() + brickCount);
    const Vec3 brickWorldSize = scene.cellSize * static_cast<float>(brickSize);

    for (u32 z = 0; z < variant.dimensionZ; ++z) {
        for (u32 y = 0; y < variant.dimensionY; ++y) {
            for (u32 x = 0; x < variant.dimensionX; ++x) {
                const u32 brickIndex = flatten_3d(x, y, z, variant.dimensionX, variant.dimensionY);
                const u32 begin = offsets[brickIndex];
                const u32 end = offsets[brickIndex + 1];
                BrickHeaderGpu& header = scene.brickHeaders[variant.headerOffset + brickIndex];

                Aabb brickBounds;
                brickBounds.minimumPoint = scene.worldMinimum + Vec3{
                    static_cast<float>(x) * brickWorldSize.x,
                    static_cast<float>(y) * brickWorldSize.y,
                    static_cast<float>(z) * brickWorldSize.z};
                brickBounds.maximumPoint = minimum(
                    brickBounds.minimumPoint + brickWorldSize,
                    scene.worldMinimum + Vec3{
                        static_cast<float>(scene.gridX) * scene.cellSize.x,
                        static_cast<float>(scene.gridY) * scene.cellSize.y,
                        static_cast<float>(scene.gridZ) * scene.cellSize.z});
                const Vec3 brickCenter = (brickBounds.minimumPoint + brickBounds.maximumPoint) * 0.5F;
                const Vec3 brickHalfExtent = (brickBounds.maximumPoint - brickBounds.minimumPoint) * 0.5F;
                header.volume = procedural_volume(brickCenter, brickHalfExtent);
                if (header.volume.w > 0.0F) {
                    ++variant.volumeBricks;
                }

                if (begin == end && header.volume.w <= 0.0F) {
                    continue;
                }
                ++variant.occupiedBricks;

                if (begin != end) {
                    header.nodeOffset = static_cast<u32>(scene.microBvhNodes.size());
                    header.referenceOffset = static_cast<u32>(scene.microTriangleReferences.size());
                    std::vector<u32> indices(
                        assignments.begin() + static_cast<std::ptrdiff_t>(begin),
                        assignments.begin() + static_cast<std::ptrdiff_t>(end));

                    build_bvh_node(
                        scene,
                        scene.microBvhNodes,
                        scene.microTriangleReferences,
                        indices,
                        0u,
                        static_cast<u32>(indices.size()),
                        kMicroLeafSize,
                        &brickBounds);

                    header.nodeCount = static_cast<u32>(scene.microBvhNodes.size()) - header.nodeOffset;
                    header.referenceCount = static_cast<u32>(scene.microTriangleReferences.size()) - header.referenceOffset;
                }
            }
        }
    }

    variant.nodeCount = static_cast<u32>(scene.microBvhNodes.size()) - variant.nodeOffset;
    variant.referenceCount = static_cast<u32>(scene.microTriangleReferences.size()) - variant.referenceOffset;
    const auto end = std::chrono::steady_clock::now();
    variant.buildMs = std::chrono::duration<double, std::milli>(end - start).count();
    scene.voxelBuildMs += variant.buildMs;
    scene.brickVariants.push_back(variant);
}

inline SceneData build_scene(const DemoOptions& options) {
    SceneData scene;
    scene.triangles.reserve(options.targetTriangles + 12u);

    std::mt19937 random{options.seed};
    std::uniform_real_distribution<float> clusterPosition{-108.0F, 108.0F};
    std::normal_distribution<float> clusterOffset{0.0F, 7.5F};
    std::uniform_real_distribution<float> widthDistribution{0.2F, 1.45F};
    std::uniform_real_distribution<float> heightDistribution{0.35F, 9.0F};
    std::uniform_real_distribution<float> colorDistribution{0.22F, 0.95F};
    std::uniform_int_distribution<u32> clusterSelection{0u, 31u};

    std::array<Vec3, 32> clusters{};
    for (Vec3& cluster : clusters) {
        cluster = {clusterPosition(random), 0.0F, clusterPosition(random)};
    }
    clusters[0] = {0.0F, 0.0F, 0.0F};

    const u32 targetBoxes = std::max((options.targetTriangles + 11u) / 12u, 1u);
    for (u32 boxIndex = 0; boxIndex < targetBoxes; ++boxIndex) {
        const Vec3 cluster = clusters[clusterSelection(random)];
        const float halfX = widthDistribution(random);
        const float halfZ = widthDistribution(random);
        const float halfY = heightDistribution(random) * 0.5F;
        const float x = std::clamp(cluster.x + clusterOffset(random), -126.0F + halfX, 126.0F - halfX);
        const float z = std::clamp(cluster.z + clusterOffset(random), -126.0F + halfZ, 126.0F - halfZ);
        const Vec3 color{
            colorDistribution(random),
            colorDistribution(random),
            colorDistribution(random)};
        add_box(scene, {x, 0.15F + halfY, z}, {halfX, halfY, halfZ}, color);
    }

    std::vector<u32> indices(scene.triangles.size());
    std::iota(indices.begin(), indices.end(), 0u);
    scene.bvhNodes.reserve(scene.triangles.size() / 2u + 1u);
    scene.bvhTriangleReferences.reserve(scene.triangles.size());

    const auto bvhStart = std::chrono::steady_clock::now();
    build_bvh_node(
        scene,
        scene.bvhNodes,
        scene.bvhTriangleReferences,
        indices,
        0u,
        static_cast<u32>(indices.size()),
        kGlobalLeafSize,
        nullptr);
    const auto bvhEnd = std::chrono::steady_clock::now();
    scene.bvhBuildMs = std::chrono::duration<double, std::milli>(bvhEnd - bvhStart).count();

    for (const u32 brickSize : kBrickSizes) {
        build_brick_variant(scene, brickSize);
    }
    return scene;
}

} // namespace epoch::voxel_demo
