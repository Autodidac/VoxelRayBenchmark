#include "ray_inspector.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <format>
#include <limits>
#include <string_view>

#if EPOCH_VISUALIZER_USE_EPOCHGUI
import epoch.gui;
import epoch.gui.font;
#endif

namespace epoch::voxel_demo {
namespace {

constexpr float kInfinity = std::numeric_limits<float>::max();
constexpr float kEpsilon = 1.0e-4F;
constexpr std::size_t kInspectorEventLimit = 8192;

#if EPOCH_VISUALIZER_USE_EPOCHGUI
namespace gui = epochengine::gui_lib;
using UiVec2 = gui::Vec2;
using UiRect = gui::Rect;
[[nodiscard]] bool ui_contains(UiRect rect, UiVec2 point) noexcept { return gui::contains(rect, point); }
#else
struct UiVec2 { float x{}; float y{}; };
struct UiRect { UiVec2 position{}; UiVec2 size{}; };
[[nodiscard]] bool ui_contains(UiRect rect, UiVec2 point) noexcept {
    return point.x >= rect.position.x && point.x <= rect.position.x + rect.size.x &&
           point.y >= rect.position.y && point.y <= rect.position.y + rect.size.y;
}
#endif

using CpuRay = InspectorRay;

struct CpuHit {
    float t{kInfinity};
    u32 triangleIndex{};
    bool found{};
};

struct DdaState {
    std::array<int, 3> coordinate{};
    std::array<int, 3> step{};
    Vec3 nextT{};
    Vec3 deltaT{};
    float currentT{};
};

[[nodiscard]] Vec3 vec3(Vec4 value) noexcept { return {value.x, value.y, value.z}; }
[[nodiscard]] Vec3 component_multiply(Vec3 a, Vec3 b) noexcept { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
[[nodiscard]] Vec3 component_divide(Vec3 a, Vec3 b) noexcept { return {a.x / b.x, a.y / b.y, a.z / b.z}; }
[[nodiscard]] float component(Vec3 v, int axis) noexcept { return axis == 0 ? v.x : (axis == 1 ? v.y : v.z); }
void set_component(Vec3& v, int axis, float value) noexcept {
    if (axis == 0) v.x = value;
    else if (axis == 1) v.y = value;
    else v.z = value;
}

[[nodiscard]] u32 hash32(u32 value) noexcept {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

[[nodiscard]] float random01(u32& state) noexcept {
    state = hash32(state + 0x9e3779b9u);
    return static_cast<float>(state & 0x00ffffffu) / 16777216.0F;
}

[[nodiscard]] CpuRay generate_ray(const InspectorFrameInput& input) noexcept {
    const u32 width = std::max(input.push.imageMode[0], 1u);
    const u32 height = std::max(input.push.imageMode[1], 1u);
    CpuRay ray{};
    if (input.rayMode == 0u) {
        const float normalizedX = (static_cast<float>(input.selectedX) + 0.5F) / static_cast<float>(width);
        const float normalizedY = (static_cast<float>(input.selectedY) + 0.5F) / static_cast<float>(height);
        const float ndcX = normalizedX * 2.0F - 1.0F;
        const float ndcY = 1.0F - normalizedY * 2.0F;
        ray.origin = vec3(input.push.cameraPosition);
        ray.direction = normalize(
            vec3(input.push.cameraForward) + vec3(input.push.cameraRight) * ndcX + vec3(input.push.cameraUp) * ndcY);
        return ray;
    }

    u32 state = hash32(input.selectedX + input.selectedY * width + 0x51f15e5du);
    const Vec3 extent{
        static_cast<float>(input.push.gridSelected[0]) * input.push.cellSize.x,
        static_cast<float>(input.push.gridSelected[1]) * input.push.cellSize.y,
        static_cast<float>(input.push.gridSelected[2]) * input.push.cellSize.z};
    const Vec3 originRandom{random01(state), random01(state), random01(state)};
    const Vec3 targetRandom{random01(state), random01(state), random01(state)};
    const Vec3 minimumPoint = vec3(input.push.worldMinimum);
    ray.origin = minimumPoint + component_multiply(originRandom, extent);
    ray.origin.y = minimumPoint.y + 0.5F + originRandom.y * extent.y * 0.9F;
    const Vec3 target = minimumPoint + component_multiply(targetRandom, extent);
    ray.direction = normalize(target - ray.origin);
    if (dot(ray.direction, ray.direction) < 0.5F) ray.direction = {0.0F, -1.0F, 0.0F};
    return ray;
}

[[nodiscard]] Vec3 safe_inverse(Vec3 value) noexcept {
    auto inverse = [](float v) {
        const float magnitude = std::max(std::abs(v), 1.0e-20F);
        return v >= 0.0F ? 1.0F / magnitude : -1.0F / magnitude;
    };
    return {inverse(value.x), inverse(value.y), inverse(value.z)};
}

[[nodiscard]] bool intersect_aabb(
    const CpuRay& ray,
    Vec3 minimumPoint,
    Vec3 maximumPoint,
    float maximumT,
    float& nearT,
    float& farT) noexcept {
    const Vec3 inverseDirection = safe_inverse(ray.direction);
    const Vec3 first = component_multiply(minimumPoint - ray.origin, inverseDirection);
    const Vec3 second = component_multiply(maximumPoint - ray.origin, inverseDirection);
    const Vec3 lower = minimum(first, second);
    const Vec3 upper = maximum(first, second);
    nearT = std::max({lower.x, lower.y, lower.z, 0.0F});
    farT = std::min({upper.x, upper.y, upper.z, maximumT});
    return nearT <= farT;
}

[[nodiscard]] bool intersect_triangle(
    const CpuRay& ray,
    const TriangleGpu& triangle,
    float maximumT,
    float& hitT) noexcept {
    const Vec3 v0 = vec3(triangle.v0);
    const Vec3 edge1 = vec3(triangle.v1) - v0;
    const Vec3 edge2 = vec3(triangle.v2) - v0;
    const Vec3 p = cross(ray.direction, edge2);
    const float determinant = dot(edge1, p);
    if (std::abs(determinant) < 1.0e-8F) return false;
    const float inverseDeterminant = 1.0F / determinant;
    const Vec3 tVector = ray.origin - v0;
    const float u = dot(tVector, p) * inverseDeterminant;
    if (u < 0.0F || u > 1.0F) return false;
    const Vec3 q = cross(tVector, edge1);
    const float v = dot(ray.direction, q) * inverseDeterminant;
    if (v < 0.0F || u + v > 1.0F) return false;
    hitT = dot(edge2, q) * inverseDeterminant;
    return hitT > kEpsilon && hitT < maximumT;
}

void record_event(
    InspectorCapture& capture,
    InspectorEventKind kind,
    Vec3 position,
    float t,
    u32 primary = 0,
    u32 secondary = 0,
    u32 tertiary = 0) {
    if (capture.events.size() >= kInspectorEventLimit) {
        capture.truncated = true;
        return;
    }
    capture.events.push_back({kind, position, t, primary, secondary, tertiary});
}

[[nodiscard]] bool inside_grid(const DdaState& state, u32 x, u32 y, u32 z) noexcept {
    return state.coordinate[0] >= 0 && state.coordinate[1] >= 0 && state.coordinate[2] >= 0 &&
           state.coordinate[0] < static_cast<int>(x) &&
           state.coordinate[1] < static_cast<int>(y) &&
           state.coordinate[2] < static_cast<int>(z);
}

[[nodiscard]] DdaState initialize_dda(
    const CpuRay& ray,
    float startT,
    Vec3 gridMinimum,
    Vec3 stepSize,
    u32 dimensionX,
    u32 dimensionY,
    u32 dimensionZ) noexcept {
    DdaState state{};
    state.currentT = startT;
    const Vec3 position = ray.origin + ray.direction * (startT + kEpsilon);
    const Vec3 local = component_divide(position - gridMinimum, stepSize);
    state.coordinate = {
        std::clamp(static_cast<int>(std::floor(local.x)), 0, static_cast<int>(dimensionX) - 1),
        std::clamp(static_cast<int>(std::floor(local.y)), 0, static_cast<int>(dimensionY) - 1),
        std::clamp(static_cast<int>(std::floor(local.z)), 0, static_cast<int>(dimensionZ) - 1)};

    for (std::size_t axis = 0; axis < 3; ++axis) {
        const int componentAxis = static_cast<int>(axis);
        const float direction = component(ray.direction, componentAxis);
        const float minimumAxis = component(gridMinimum, componentAxis);
        const float stepAxis = component(stepSize, componentAxis);
        if (direction > 0.0F) {
            state.step[axis] = 1;
            const float boundary = minimumAxis + static_cast<float>(state.coordinate[axis] + 1) * stepAxis;
            set_component(state.nextT, componentAxis, (boundary - component(ray.origin, componentAxis)) / direction);
            set_component(state.deltaT, componentAxis, stepAxis / direction);
        } else if (direction < 0.0F) {
            state.step[axis] = -1;
            const float boundary = minimumAxis + static_cast<float>(state.coordinate[axis]) * stepAxis;
            set_component(state.nextT, componentAxis, (boundary - component(ray.origin, componentAxis)) / direction);
            set_component(state.deltaT, componentAxis, -stepAxis / direction);
        } else {
            state.step[axis] = 0;
            set_component(state.nextT, componentAxis, kInfinity);
            set_component(state.deltaT, componentAxis, kInfinity);
        }
    }
    return state;
}

void advance_dda(DdaState& state) noexcept {
    const float nextBoundaryT = std::min({state.nextT.x, state.nextT.y, state.nextT.z});
    constexpr float tieTolerance = 1.0e-5F;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const int componentAxis = static_cast<int>(axis);
        if (component(state.nextT, componentAxis) <= nextBoundaryT + tieTolerance) {
            state.coordinate[axis] += state.step[axis];
            set_component(
                state.nextT,
                componentAxis,
                component(state.nextT, componentAxis) + component(state.deltaT, componentAxis));
        }
    }
    state.currentT = nextBoundaryT;
}

void trace_global_bvh(const SceneData& scene, InspectorCapture& capture, CpuHit& hit) {
    if (scene.bvhNodes.empty()) return;
    std::array<u32, 128> stackNodes{};
    std::array<float, 128> stackNear{};
    u32 stackSize = 0;
    float rootNear = 0.0F;
    float rootFar = 0.0F;
    ++capture.metrics.bvhNodeTests;
    if (intersect_aabb(capture.ray, vec3(scene.bvhNodes[0].minimumPoint), vec3(scene.bvhNodes[0].maximumPoint), hit.t, rootNear, rootFar)) {
        stackNodes[stackSize] = 0;
        stackNear[stackSize] = rootNear;
        ++stackSize;
    }

    while (stackSize > 0) {
        --stackSize;
        const u32 nodeIndex = stackNodes[stackSize];
        const float nodeNear = stackNear[stackSize];
        if (nodeNear >= hit.t) continue;
        const BvhNodeGpu& node = scene.bvhNodes[nodeIndex];
        record_event(capture, InspectorEventKind::GlobalBvhNode,
                     capture.ray.origin + capture.ray.direction * nodeNear,
                     nodeNear, nodeIndex, node.meta[3], stackSize);
        if (node.meta[3] > 0) {
            for (u32 index = 0; index < node.meta[3]; ++index) {
                const u32 triangleIndex = scene.bvhTriangleReferences[node.meta[2] + index];
                float triangleT = 0.0F;
                ++capture.metrics.triangleTests;
                if (intersect_triangle(capture.ray, scene.triangles[triangleIndex], hit.t, triangleT)) {
                    hit = {triangleT, triangleIndex, true};
                    record_event(capture, InspectorEventKind::TriangleHit,
                                 capture.ray.origin + capture.ray.direction * triangleT,
                                 triangleT, triangleIndex, nodeIndex, index);
                }
            }
            continue;
        }

        const u32 leftIndex = node.meta[0];
        const u32 rightIndex = node.meta[1];
        float leftNear = 0.0F, leftFar = 0.0F, rightNear = 0.0F, rightFar = 0.0F;
        capture.metrics.bvhNodeTests += 2;
        const bool hitLeft = intersect_aabb(capture.ray, vec3(scene.bvhNodes[leftIndex].minimumPoint),
                                            vec3(scene.bvhNodes[leftIndex].maximumPoint), hit.t, leftNear, leftFar);
        const bool hitRight = intersect_aabb(capture.ray, vec3(scene.bvhNodes[rightIndex].minimumPoint),
                                             vec3(scene.bvhNodes[rightIndex].maximumPoint), hit.t, rightNear, rightFar);
        auto push = [&](u32 index, float nearValue) {
            if (stackSize < stackNodes.size()) {
                stackNodes[stackSize] = index;
                stackNear[stackSize] = nearValue;
                ++stackSize;
            }
        };
        if (hitLeft && hitRight) {
            if (leftNear <= rightNear) { push(rightIndex, rightNear); push(leftIndex, leftNear); }
            else { push(leftIndex, leftNear); push(rightIndex, rightNear); }
        } else if (hitLeft) push(leftIndex, leftNear);
        else if (hitRight) push(rightIndex, rightNear);
    }
}

[[nodiscard]] bool trace_micro_bvh(
    const SceneData& scene,
    const BrickHeaderGpu& header,
    float brickEntry,
    float brickExit,
    InspectorCapture& capture,
    CpuHit& hit) {
    std::array<u32, 128> stackNodes{};
    std::array<float, 128> stackNear{};
    u32 stackSize = 0;
    const float maximumT = std::min(hit.t, brickExit + kEpsilon);
    float rootNear = 0.0F, rootFar = 0.0F;
    ++capture.metrics.microBvhNodeTests;
    if (!intersect_aabb(capture.ray,
                        vec3(scene.microBvhNodes[header.nodeOffset].minimumPoint),
                        vec3(scene.microBvhNodes[header.nodeOffset].maximumPoint),
                        maximumT, rootNear, rootFar)) return false;
    stackNodes[0] = header.nodeOffset;
    stackNear[0] = rootNear;
    stackSize = 1;
    bool foundInBrick = false;

    while (stackSize > 0) {
        --stackSize;
        const u32 nodeIndex = stackNodes[stackSize];
        const float nodeNear = stackNear[stackSize];
        if (nodeNear >= hit.t || nodeNear > brickExit + kEpsilon) continue;
        const BvhNodeGpu& node = scene.microBvhNodes[nodeIndex];
        record_event(capture, InspectorEventKind::MicroBvhNode,
                     capture.ray.origin + capture.ray.direction * nodeNear,
                     nodeNear, nodeIndex, node.meta[3], stackSize);
        if (node.meta[3] > 0) {
            for (u32 index = 0; index < node.meta[3]; ++index) {
                const u32 triangleIndex = scene.microTriangleReferences[node.meta[2] + index];
                float triangleT = 0.0F;
                ++capture.metrics.triangleTests;
                if (intersect_triangle(capture.ray, scene.triangles[triangleIndex],
                                       std::min(hit.t, brickExit + kEpsilon), triangleT) &&
                    triangleT >= brickEntry - kEpsilon && triangleT <= brickExit + kEpsilon) {
                    hit = {triangleT, triangleIndex, true};
                    foundInBrick = true;
                    record_event(capture, InspectorEventKind::TriangleHit,
                                 capture.ray.origin + capture.ray.direction * triangleT,
                                 triangleT, triangleIndex, nodeIndex, index);
                }
            }
            continue;
        }

        const u32 leftIndex = node.meta[0];
        const u32 rightIndex = node.meta[1];
        float leftNear = 0.0F, leftFar = 0.0F, rightNear = 0.0F, rightFar = 0.0F;
        capture.metrics.microBvhNodeTests += 2;
        const float currentMaximum = std::min(hit.t, brickExit + kEpsilon);
        const bool hitLeft = intersect_aabb(capture.ray, vec3(scene.microBvhNodes[leftIndex].minimumPoint),
                                            vec3(scene.microBvhNodes[leftIndex].maximumPoint),
                                            currentMaximum, leftNear, leftFar);
        const bool hitRight = intersect_aabb(capture.ray, vec3(scene.microBvhNodes[rightIndex].minimumPoint),
                                             vec3(scene.microBvhNodes[rightIndex].maximumPoint),
                                             currentMaximum, rightNear, rightFar);
        auto push = [&](u32 index, float nearValue) {
            if (stackSize < stackNodes.size()) {
                stackNodes[stackSize] = index;
                stackNear[stackSize] = nearValue;
                ++stackSize;
            }
        };
        if (hitLeft && hitRight) {
            if (leftNear <= rightNear) { push(rightIndex, rightNear); push(leftIndex, leftNear); }
            else { push(leftIndex, leftNear); push(rightIndex, rightNear); }
        } else if (hitLeft) push(leftIndex, leftNear);
        else if (hitRight) push(rightIndex, rightNear);
    }
    return foundInBrick;
}

void integrate_volume_event(
    const BrickHeaderGpu& header,
    float segmentEntry,
    float segmentExit,
    InspectorCapture& capture) {
    if (header.volume.w <= 0.0F || segmentExit <= segmentEntry) return;
    ++capture.metrics.volumeSamples;
    ++capture.metrics.volumeBricks;
    const float middleT = 0.5F * (segmentEntry + segmentExit);
    record_event(capture, InspectorEventKind::VolumeSegment,
                 capture.ray.origin + capture.ray.direction * middleT,
                 middleT, std::bit_cast<u32>(header.volume.w), capture.metrics.volumeSamples, 0);
}

void trace_volume_only(
    const SceneData& scene,
    const BrickVariant& variant,
    float maximumT,
    InspectorCapture& capture) {
    const Vec3 worldMaximum = scene.worldMinimum + Vec3{
        static_cast<float>(scene.gridX) * scene.cellSize.x,
        static_cast<float>(scene.gridY) * scene.cellSize.y,
        static_cast<float>(scene.gridZ) * scene.cellSize.z};
    float worldEntry = 0.0F, worldExit = 0.0F;
    if (!intersect_aabb(capture.ray, scene.worldMinimum, worldMaximum, maximumT, worldEntry, worldExit)) return;
    worldExit = std::min(worldExit, maximumT);
    const Vec3 brickWorldSize = scene.cellSize * static_cast<float>(variant.brickSize);
    DdaState dda = initialize_dda(capture.ray, worldEntry, scene.worldMinimum, brickWorldSize,
                                  variant.dimensionX, variant.dimensionY, variant.dimensionZ);
    while (dda.currentT <= worldExit && inside_grid(dda, variant.dimensionX, variant.dimensionY, variant.dimensionZ)) {
        ++capture.metrics.brickSteps;
        const u32 x = static_cast<u32>(dda.coordinate[0]);
        const u32 y = static_cast<u32>(dda.coordinate[1]);
        const u32 z = static_cast<u32>(dda.coordinate[2]);
        const u32 brickIndex = flatten_3d(x, y, z, variant.dimensionX, variant.dimensionY);
        const float brickExit = std::min({dda.nextT.x, dda.nextT.y, dda.nextT.z, worldExit});
        const BrickHeaderGpu& header = scene.brickHeaders[variant.headerOffset + brickIndex];
        if (header.volume.w > 0.0F) {
            ++capture.metrics.occupiedBricks;
            record_event(capture, InspectorEventKind::BrickVisit,
                         capture.ray.origin + capture.ray.direction * dda.currentT,
                         dda.currentT, x, y, z);
            integrate_volume_event(header, dda.currentT, brickExit, capture);
        }
        if (brickExit >= worldExit - kEpsilon) break;
        advance_dda(dda);
    }
}

[[nodiscard]] InspectorCapture capture_bvh(
    const SceneData& scene,
    const BrickVariant& variant,
    const InspectorFrameInput& input) {
    InspectorCapture capture{};
    const CpuRay ray = generate_ray(input);
    capture.ray = {ray.origin, ray.direction};
    capture.metrics.rays = 1;
    CpuHit hit{};
    trace_global_bvh(scene, capture, hit);
    if (input.workloadMode != 0u) trace_volume_only(scene, variant, hit.found ? hit.t : kInfinity, capture);
    capture.hit = hit.found;
    capture.triangleIndex = hit.triangleIndex;
    capture.hitT = hit.t;
    capture.metrics.hits = hit.found ? 1u : 0u;
    capture.metrics.maximumTraversal = capture.metrics.bvhNodeTests + capture.metrics.brickSteps;
    return capture;
}

[[nodiscard]] InspectorCapture capture_voxel(
    const SceneData& scene,
    const BrickVariant& variant,
    const InspectorFrameInput& input) {
    InspectorCapture capture{};
    const CpuRay ray = generate_ray(input);
    capture.ray = {ray.origin, ray.direction};
    capture.metrics.rays = 1;
    CpuHit hit{};
    const Vec3 worldMaximum = scene.worldMinimum + Vec3{
        static_cast<float>(scene.gridX) * scene.cellSize.x,
        static_cast<float>(scene.gridY) * scene.cellSize.y,
        static_cast<float>(scene.gridZ) * scene.cellSize.z};
    float worldEntry = 0.0F, worldExit = 0.0F;
    if (intersect_aabb(ray, scene.worldMinimum, worldMaximum, kInfinity, worldEntry, worldExit)) {
        const Vec3 brickWorldSize = scene.cellSize * static_cast<float>(variant.brickSize);
        DdaState dda = initialize_dda(ray, worldEntry, scene.worldMinimum, brickWorldSize,
                                      variant.dimensionX, variant.dimensionY, variant.dimensionZ);
        while (dda.currentT <= worldExit && inside_grid(dda, variant.dimensionX, variant.dimensionY, variant.dimensionZ)) {
            ++capture.metrics.brickSteps;
            const u32 x = static_cast<u32>(dda.coordinate[0]);
            const u32 y = static_cast<u32>(dda.coordinate[1]);
            const u32 z = static_cast<u32>(dda.coordinate[2]);
            const u32 brickIndex = flatten_3d(x, y, z, variant.dimensionX, variant.dimensionY);
            const float brickExit = std::min({dda.nextT.x, dda.nextT.y, dda.nextT.z, worldExit});
            const BrickHeaderGpu& header = scene.brickHeaders[variant.headerOffset + brickIndex];
            record_event(capture, InspectorEventKind::BrickVisit,
                         ray.origin + ray.direction * dda.currentT,
                         dda.currentT, x, y, z);
            const bool hasVolume = input.workloadMode != 0u && header.volume.w > 0.0F;
            const bool hasSurface = header.nodeCount > 0u;
            if (hasVolume || hasSurface) ++capture.metrics.occupiedBricks;
            bool foundInBrick = false;
            if (hasSurface) foundInBrick = trace_micro_bvh(scene, header, dda.currentT, brickExit, capture, hit);
            if (hasVolume) integrate_volume_event(header, dda.currentT, std::min(brickExit, hit.t), capture);
            if (foundInBrick) break;
            if (brickExit >= worldExit - kEpsilon) break;
            advance_dda(dda);
        }
    }
    capture.hit = hit.found;
    capture.triangleIndex = hit.triangleIndex;
    capture.hitT = hit.t;
    capture.metrics.hits = hit.found ? 1u : 0u;
    capture.metrics.maximumTraversal = capture.metrics.brickSteps + capture.metrics.microBvhNodeTests;
    return capture;
}

struct Color { Uint8 r{}, g{}, b{}, a{255}; };

constexpr Color kBackground{12, 15, 22, 255};
constexpr Color kPanel{24, 29, 40, 255};
constexpr Color kPanel2{17, 21, 30, 255};
constexpr Color kBorder{72, 82, 100, 255};
constexpr Color kText{232, 237, 244, 255};
constexpr Color kMuted{160, 172, 190, 255};
constexpr Color kAccent{88, 166, 255, 255};
constexpr Color kBvh{255, 166, 74, 255};
constexpr Color kBrick{84, 212, 130, 255};
constexpr Color kMicro{180, 116, 255, 255};
constexpr Color kVolume{74, 205, 230, 255};
constexpr Color kHit{255, 94, 94, 255};
constexpr Color kSurfaceCell{47, 92, 70, 255};
constexpr Color kVolumeCell{41, 87, 101, 255};
constexpr Color kMixedCell{61, 103, 91, 255};
constexpr float kSmallTextScale = 1.5F;
constexpr float kHeadingTextScale = 2.0F;

void set_color(SDL_Renderer* renderer, Color color) {
    SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
}

[[nodiscard]] SDL_FRect sdl_rect(UiRect rect) noexcept {
    return {rect.position.x, rect.position.y, rect.size.x, rect.size.y};
}

void fill_rect(SDL_Renderer* renderer, UiRect rect, Color color) {
    set_color(renderer, color);
    const SDL_FRect output = sdl_rect(rect);
    SDL_RenderFillRect(renderer, &output);
}

void outline_rect(SDL_Renderer* renderer, UiRect rect, Color color) {
    set_color(renderer, color);
    const SDL_FRect output = sdl_rect(rect);
    SDL_RenderRect(renderer, &output);
}

void draw_line(SDL_Renderer* renderer, float x1, float y1, float x2, float y2, Color color) {
    set_color(renderer, color);
    SDL_RenderLine(renderer, x1, y1, x2, y2);
}

#if EPOCH_VISUALIZER_USE_EPOCHGUI
void draw_text(SDL_Renderer* renderer, float x, float y, std::string_view text, Color color, float scale = 1.0F) {
    set_color(renderer, color);
    float cursorX = x;
    float cursorY = y;
    for (char character : text) {
        if (character == '\n') {
            cursorX = x;
            cursorY += static_cast<float>(gui::font::line_advance) * scale;
            continue;
        }
        const gui::font::BitmapGlyph glyph = gui::font::default_glyph(character);
        for (u32 row = 0; row < gui::font::glyph_height; ++row) {
            for (u32 column = 0; column < gui::font::glyph_width; ++column) {
                if (!gui::font::pixel_on(glyph, column, row)) continue;
                SDL_FRect pixel{
                    cursorX + static_cast<float>(column) * scale,
                    cursorY + static_cast<float>(row) * scale,
                    std::max(scale, 1.0F),
                    std::max(scale, 1.0F)};
                SDL_RenderFillRect(renderer, &pixel);
            }
        }
        cursorX += static_cast<float>(gui::font::glyph_advance) * scale;
    }
}
#else
void draw_text(SDL_Renderer* renderer, float x, float y, std::string_view text, Color color, float scale = 1.0F) {
    set_color(renderer, color);
    const std::string owned{text};
    SDL_SetRenderScale(renderer, std::max(scale, 1.0F), std::max(scale, 1.0F));
    SDL_RenderDebugText(renderer, x / std::max(scale, 1.0F), y / std::max(scale, 1.0F), owned.c_str());
    SDL_SetRenderScale(renderer, 1.0F, 1.0F);
}
#endif

[[nodiscard]] std::vector<std::string> wrap_text(std::string_view text, std::size_t maximumCharacters) {
    std::vector<std::string> lines;
    std::string current;
    std::size_t position = 0;
    while (position < text.size()) {
        while (position < text.size() && text[position] == ' ') ++position;
        if (position >= text.size()) break;
        const std::size_t end = text.find(' ', position);
        const std::string_view word = text.substr(position, end == std::string_view::npos ? text.size() - position : end - position);
        if (!current.empty() && current.size() + 1 + word.size() > maximumCharacters) {
            lines.push_back(std::move(current));
            current.clear();
        }
        if (!current.empty()) current.push_back(' ');
        current.append(word);
        position = end == std::string_view::npos ? text.size() : end + 1;
    }
    if (!current.empty()) lines.push_back(std::move(current));
    return lines;
}

void draw_wrapped_text(
    SDL_Renderer* renderer,
    float x,
    float y,
    std::string_view text,
    Color color,
    float scale,
    std::size_t maximumCharacters,
    std::size_t maximumLines = 8) {
    const auto lines = wrap_text(text, maximumCharacters);
    const float advance = static_cast<float>(
#if EPOCH_VISUALIZER_USE_EPOCHGUI
        gui::font::line_advance
#else
        9
#endif
    ) * scale;
    for (std::size_t index = 0; index < std::min(lines.size(), maximumLines); ++index) {
        draw_text(renderer, x, y + static_cast<float>(index) * advance, lines[index], color, scale);
    }
}

[[nodiscard]] Color event_color(InspectorEventKind kind) noexcept {
    switch (kind) {
    case InspectorEventKind::GlobalBvhNode: return kBvh;
    case InspectorEventKind::BrickVisit: return kBrick;
    case InspectorEventKind::MicroBvhNode: return kMicro;
    case InspectorEventKind::VolumeSegment: return kVolume;
    case InspectorEventKind::TriangleHit: return kHit;
    }
    return kText;
}

[[nodiscard]] std::string_view event_name(InspectorEventKind kind) noexcept {
    switch (kind) {
    case InspectorEventKind::GlobalBvhNode: return "GLOBAL BVH NODE";
    case InspectorEventKind::BrickVisit: return "BRICK DDA STEP";
    case InspectorEventKind::MicroBvhNode: return "MICRO BVH NODE";
    case InspectorEventKind::VolumeSegment: return "VOLUME SAMPLE";
    case InspectorEventKind::TriangleHit: return "TRIANGLE HIT";
    }
    return "EVENT";
}

struct ControlLayout {
    UiRect play{};
    UiRect previous{};
    UiRect next{};
    UiRect reset{};
    UiRect refresh{};
    UiRect slower{};
    UiRect faster{};
    std::array<UiRect, 3> view{};
    UiRect status{};
    UiRect worldMap{};
    UiRect bvhDetails{};
    UiRect voxelDetails{};
    UiRect timeline{};
    UiRect explanation{};
};

[[nodiscard]] ControlLayout make_layout(float width, float height) {
    ControlLayout layout{};
    const float margin = 12.0F;
    float x = margin;
    float y = margin;
    auto button = [&](float buttonWidth) {
        UiRect rect{{x, y}, {buttonWidth, 38.0F}};
        x += buttonWidth + 6.0F;
        return rect;
    };
    layout.play = button(86.0F);
    layout.previous = button(72.0F);
    layout.next = button(72.0F);
    layout.reset = button(78.0F);
    layout.refresh = button(96.0F);

    x = margin;
    y += 44.0F;
    layout.slower = button(44.0F);
    layout.faster = button(44.0F);
#if EPOCH_VISUALIZER_USE_EPOCHGUI
    const std::array<float, 3> widths{88.0F, 78.0F, 96.0F};
    const gui::SegmentedControlLayoutOptions options{
        .position = {x + 8.0F, y}, .item_widths = widths, .height = 38.0F, .gap = 5.0F};
    for (u32 i = 0; i < 3; ++i) layout.view[i] = gui::segmented_control_item_layout(options, i);
#else
    x += 8.0F;
    layout.view[0] = {{x, y}, {88.0F, 38.0F}};
    layout.view[1] = {{x + 93.0F, y}, {78.0F, 38.0F}};
    layout.view[2] = {{x + 176.0F, y}, {96.0F, 38.0F}};
#endif

    layout.status = {{margin, y + 46.0F}, {width - margin * 2.0F, 66.0F}};
    const float mapTop = layout.status.position.y + layout.status.size.y + 8.0F;
    const float available = std::max(height - mapTop - margin, 300.0F);
    const float mapHeight = std::clamp(available * 0.32F, 205.0F, 285.0F);
    layout.worldMap = {{margin, mapTop}, {width - margin * 2.0F, mapHeight}};

    const float detailsTop = mapTop + mapHeight + 8.0F;
    const float detailsHeight = std::clamp(available * 0.205F, 128.0F, 155.0F);
    layout.bvhDetails = {{margin, detailsTop}, {width - margin * 2.0F, detailsHeight}};
    layout.voxelDetails = {{margin, detailsTop + detailsHeight + 8.0F}, {width - margin * 2.0F, detailsHeight}};

    const float timelineTop = layout.voxelDetails.position.y + layout.voxelDetails.size.y + 8.0F;
    const float timelineHeight = std::clamp(available * 0.12F, 76.0F, 100.0F);
    layout.timeline = {{margin, timelineTop}, {width - margin * 2.0F, timelineHeight}};
    const float explanationTop = timelineTop + timelineHeight + 8.0F;
    layout.explanation = {{margin, explanationTop},
                          {width - margin * 2.0F, std::max(70.0F, height - explanationTop - margin)}};
    return layout;
}

[[nodiscard]] UiVec2 project_xz(const SceneData& scene, UiRect viewport, Vec3 position) noexcept {
    const float worldWidth = static_cast<float>(scene.gridX) * scene.cellSize.x;
    const float worldDepth = static_cast<float>(scene.gridZ) * scene.cellSize.z;
    const float nx = std::clamp((position.x - scene.worldMinimum.x) / std::max(worldWidth, 0.001F), 0.0F, 1.0F);
    const float nz = std::clamp((position.z - scene.worldMinimum.z) / std::max(worldDepth, 0.001F), 0.0F, 1.0F);
    return {
        viewport.position.x + nx * std::max(viewport.size.x, 1.0F),
        viewport.position.y + (1.0F - nz) * std::max(viewport.size.y, 1.0F)};
}

[[nodiscard]] UiRect brick_rect(
    UiRect map,
    const BrickVariant& variant,
    u32 x,
    u32 z) noexcept {
    const float cellWidth = map.size.x / static_cast<float>(std::max(variant.dimensionX, 1u));
    const float cellHeight = map.size.y / static_cast<float>(std::max(variant.dimensionZ, 1u));
    return {{map.position.x + static_cast<float>(x) * cellWidth,
             map.position.y + static_cast<float>(variant.dimensionZ - 1u - z) * cellHeight},
            {std::max(cellWidth, 1.0F), std::max(cellHeight, 1.0F)}};
}

[[nodiscard]] const InspectorEvent* current_event(const InspectorCapture& capture, std::size_t visibleEvents) noexcept {
    if (capture.events.empty() || visibleEvents == 0) return nullptr;
    return &capture.events[std::min(visibleEvents - 1, capture.events.size() - 1)];
}

void draw_current_structure(
    SDL_Renderer* renderer,
    const SceneData& scene,
    UiRect map,
    const BrickVariant& variant,
    const InspectorEvent* event) {
    if (event == nullptr) return;
    if (event->kind == InspectorEventKind::BrickVisit) {
        outline_rect(renderer, brick_rect(map, variant, event->primary, event->tertiary), kText);
        return;
    }
    if (event->kind == InspectorEventKind::GlobalBvhNode && event->primary < scene.bvhNodes.size()) {
        const BvhNodeGpu& node = scene.bvhNodes[event->primary];
        const UiVec2 p0 = project_xz(scene, map, vec3(node.minimumPoint));
        const UiVec2 p1 = project_xz(scene, map, vec3(node.maximumPoint));
        outline_rect(renderer,
                     {{std::min(p0.x, p1.x), std::min(p0.y, p1.y)},
                      {std::max(std::abs(p1.x - p0.x), 1.0F), std::max(std::abs(p1.y - p0.y), 1.0F)}},
                     kBvh);
        return;
    }
    if (event->kind == InspectorEventKind::MicroBvhNode && event->primary < scene.microBvhNodes.size()) {
        const BvhNodeGpu& node = scene.microBvhNodes[event->primary];
        const UiVec2 p0 = project_xz(scene, map, vec3(node.minimumPoint));
        const UiVec2 p1 = project_xz(scene, map, vec3(node.maximumPoint));
        outline_rect(renderer,
                     {{std::min(p0.x, p1.x), std::min(p0.y, p1.y)},
                      {std::max(std::abs(p1.x - p0.x), 1.0F), std::max(std::abs(p1.y - p0.y), 1.0F)}},
                     kMicro);
        return;
    }
    if (event->kind == InspectorEventKind::TriangleHit && event->primary < scene.triangles.size()) {
        const TriangleGpu& triangle = scene.triangles[event->primary];
        const UiVec2 a = project_xz(scene, map, vec3(triangle.v0));
        const UiVec2 b = project_xz(scene, map, vec3(triangle.v1));
        const UiVec2 c = project_xz(scene, map, vec3(triangle.v2));
        draw_line(renderer, a.x, a.y, b.x, b.y, kHit);
        draw_line(renderer, b.x, b.y, c.x, c.y, kHit);
        draw_line(renderer, c.x, c.y, a.x, a.y, kHit);
    }
}

void draw_world_map(
    SDL_Renderer* renderer,
    const SceneData& scene,
    UiRect rect,
    const BrickVariant& variant,
    const InspectorCapture& bvh,
    const InspectorCapture& voxel,
    std::size_t visibleEvents,
    int viewMode) {
    fill_rect(renderer, rect, kPanel2);
    outline_rect(renderer, rect, kBorder);
    draw_text(renderer, rect.position.x + 10.0F, rect.position.y + 9.0F,
              "WORLD / BRICK MAP (TOP-DOWN, Y COLLAPSED)", kText, kHeadingTextScale);
    const std::string dimensions = std::format(
        "BRICK {}^3  GRID {}x{}x{}  {} OCCUPIED  {} VOLUME",
        variant.brickSize, variant.dimensionX, variant.dimensionY, variant.dimensionZ,
        variant.occupiedBricks, variant.volumeBricks);
    draw_text(renderer, rect.position.x + 10.0F, rect.position.y + 31.0F,
              dimensions, kMuted, kSmallTextScale);

    UiRect map{{rect.position.x + 10.0F, rect.position.y + 52.0F},
               {rect.size.x - 20.0F, rect.size.y - 64.0F}};
    fill_rect(renderer, map, Color{10, 13, 19, 255});

    for (u32 z = 0; z < variant.dimensionZ; ++z) {
        for (u32 x = 0; x < variant.dimensionX; ++x) {
            bool surface = false;
            bool volume = false;
            for (u32 y = 0; y < variant.dimensionY; ++y) {
                const u32 index = flatten_3d(x, y, z, variant.dimensionX, variant.dimensionY);
                const BrickHeaderGpu& header = scene.brickHeaders[variant.headerOffset + index];
                surface = surface || header.nodeCount > 0u;
                volume = volume || header.volume.w > 0.0F;
            }
            if (!surface && !volume) continue;
            UiRect cell = brick_rect(map, variant, x, z);
            cell.position.x += 0.5F;
            cell.position.y += 0.5F;
            cell.size.x = std::max(cell.size.x - 1.0F, 1.0F);
            cell.size.y = std::max(cell.size.y - 1.0F, 1.0F);
            fill_rect(renderer, cell, surface && volume ? kMixedCell : (surface ? kSurfaceCell : kVolumeCell));
        }
    }

    auto draw_visits = [&](const InspectorCapture& capture, Color color, bool enabled) {
        if (!enabled) return;
        const std::size_t count = std::min(visibleEvents, capture.events.size());
        for (std::size_t index = 0; index < count; ++index) {
            const InspectorEvent& event = capture.events[index];
            if (event.kind != InspectorEventKind::BrickVisit) continue;
            UiRect cell = brick_rect(map, variant, event.primary, event.tertiary);
            outline_rect(renderer, cell, color);
        }
    };
    draw_visits(bvh, kBvh, viewMode != 2);
    draw_visits(voxel, kBrick, viewMode != 1);

    float maximumT = 1.0F;
    for (const InspectorEvent& event : bvh.events) maximumT = std::max(maximumT, event.t);
    for (const InspectorEvent& event : voxel.events) maximumT = std::max(maximumT, event.t);
    const InspectorRay& ray = !voxel.events.empty() ? voxel.ray : bvh.ray;
    const UiVec2 rayStart = project_xz(scene, map, ray.origin);
    const UiVec2 rayEnd = project_xz(scene, map, ray.origin + ray.direction * maximumT);
    draw_line(renderer, rayStart.x, rayStart.y, rayEnd.x, rayEnd.y, kText);

    draw_current_structure(renderer, scene, map, variant, current_event(bvh, visibleEvents));
    draw_current_structure(renderer, scene, map, variant, current_event(voxel, visibleEvents));

    draw_text(renderer, map.position.x + 6.0F, map.position.y + 6.0F,
              "DARK=EMPTY  GREEN=SURFACE  CYAN=VOLUME  WHITE=CURRENT", kMuted, 1.25F);
}

[[nodiscard]] std::string describe_event_details(
    const SceneData& scene,
    const BrickVariant& variant,
    const InspectorEvent* event) {
    if (event == nullptr) return "NO STEP IS VISIBLE YET. PRESS PLAY OR NEXT.";
    switch (event->kind) {
    case InspectorEventKind::GlobalBvhNode: {
        if (event->primary >= scene.bvhNodes.size()) return "GLOBAL BVH NODE INDEX IS OUT OF RANGE.";
        const BvhNodeGpu& node = scene.bvhNodes[event->primary];
        return std::format(
            "NODE {} AT T {:.3f}. LEAF REFS {}. STACK AFTER POP {}. BOUNDS X {:.1f}..{:.1f}, Y {:.1f}..{:.1f}, Z {:.1f}..{:.1f}.",
            event->primary, event->t, node.meta[3], event->tertiary,
            node.minimumPoint.x, node.maximumPoint.x,
            node.minimumPoint.y, node.maximumPoint.y,
            node.minimumPoint.z, node.maximumPoint.z);
    }
    case InspectorEventKind::BrickVisit: {
        if (event->primary >= variant.dimensionX || event->secondary >= variant.dimensionY || event->tertiary >= variant.dimensionZ)
            return "BRICK COORDINATE IS OUT OF RANGE.";
        const u32 localIndex = flatten_3d(event->primary, event->secondary, event->tertiary,
                                          variant.dimensionX, variant.dimensionY);
        const BrickHeaderGpu& header = scene.brickHeaders[variant.headerOffset + localIndex];
        return std::format(
            "BRICK ({},{},{}) INDEX {} AT T {:.3f}. MICRO NODES {}. TRIANGLE REFS {}. VOLUME DENSITY {:.3f}.",
            event->primary, event->secondary, event->tertiary, localIndex, event->t,
            header.nodeCount, header.referenceCount, header.volume.w);
    }
    case InspectorEventKind::MicroBvhNode: {
        if (event->primary >= scene.microBvhNodes.size()) return "MICRO BVH NODE INDEX IS OUT OF RANGE.";
        const BvhNodeGpu& node = scene.microBvhNodes[event->primary];
        return std::format(
            "LOCAL NODE {} AT T {:.3f}. LEAF REFS {}. STACK AFTER POP {}. THIS NODE EXISTS ONLY INSIDE AN OCCUPIED BRICK.",
            event->primary, event->t, node.meta[3], event->tertiary);
    }
    case InspectorEventKind::VolumeSegment:
        return std::format(
            "VOLUME SAMPLE AT T {:.3f}. DENSITY {:.3f}. SAMPLE {}. SURFACE AND VOLUME CAN SHARE THIS BRICK WALK.",
            event->t, std::bit_cast<float>(event->primary), event->secondary);
    case InspectorEventKind::TriangleHit:
        return std::format(
            "EXACT TRIANGLE {} HIT AT T {:.3f}. NODE {} LEAF SLOT {}. THIS IS THE FINAL GEOMETRY TEST, NOT A BOX APPROXIMATION.",
            event->primary, event->t, event->secondary, event->tertiary);
    }
    return "UNKNOWN STEP.";
}

void draw_stage_pipeline(
    SDL_Renderer* renderer,
    float x,
    float y,
    float width,
    const std::array<std::string_view, 5>& stages,
    int activeStage) {
    const float gap = 5.0F;
    const float itemWidth = (width - gap * 4.0F) / 5.0F;
    for (int index = 0; index < 5; ++index) {
        UiRect box{{x + static_cast<float>(index) * (itemWidth + gap), y}, {itemWidth, 27.0F}};
        const bool active = index == activeStage;
        fill_rect(renderer, box, active ? Color{42, 93, 145, 255} : Color{31, 37, 49, 255});
        outline_rect(renderer, box, active ? kAccent : kBorder);
        draw_text(renderer, box.position.x + 5.0F, box.position.y + 8.0F,
                  stages[static_cast<std::size_t>(index)], active ? kText : kMuted, 1.25F);
    }
}

[[nodiscard]] int stage_for_event(const InspectorEvent* event, bool voxelPath) noexcept {
    if (event == nullptr) return 0;
    if (voxelPath) {
        switch (event->kind) {
        case InspectorEventKind::BrickVisit: return 1;
        case InspectorEventKind::MicroBvhNode: return 2;
        case InspectorEventKind::TriangleHit: return 3;
        case InspectorEventKind::VolumeSegment: return 4;
        case InspectorEventKind::GlobalBvhNode: return 0;
        }
    }
    switch (event->kind) {
    case InspectorEventKind::GlobalBvhNode: return 1;
    case InspectorEventKind::TriangleHit: return 2;
    case InspectorEventKind::BrickVisit: return 3;
    case InspectorEventKind::VolumeSegment: return 4;
    case InspectorEventKind::MicroBvhNode: return 0;
    }
    return 0;
}

void draw_detail_card(
    SDL_Renderer* renderer,
    const SceneData& scene,
    UiRect rect,
    std::string_view title,
    const InspectorCapture& capture,
    const BrickVariant& variant,
    std::size_t visibleEvents,
    bool voxelPath,
    bool enabled) {
    fill_rect(renderer, rect, enabled ? kPanel : kPanel2);
    outline_rect(renderer, rect, enabled ? kBorder : Color{44, 50, 62, 255});
    const InspectorEvent* event = current_event(capture, visibleEvents);
    draw_text(renderer, rect.position.x + 10.0F, rect.position.y + 8.0F,
              title, enabled ? kText : kMuted, kHeadingTextScale);
    const std::string metricLine = std::format(
        "STEPS {}/{}  NODES {}  BRICKS {}  MICRO {}  TRIS {}  VOL {}",
        std::min(visibleEvents, capture.events.size()), capture.events.size(),
        capture.metrics.bvhNodeTests, capture.metrics.brickSteps,
        capture.metrics.microBvhNodeTests, capture.metrics.triangleTests,
        capture.metrics.volumeSamples);
    draw_text(renderer, rect.position.x + 10.0F, rect.position.y + 30.0F,
              metricLine, kMuted, 1.3F);

    const std::array<std::string_view, 5> globalStages{"RAY", "BVH", "TRI", "BRICK", "VOLUME"};
    const std::array<std::string_view, 5> voxelStages{"RAY", "BRICK", "MICRO", "TRI", "VOLUME"};
    draw_stage_pipeline(renderer, rect.position.x + 10.0F, rect.position.y + 49.0F,
                        rect.size.x - 20.0F, voxelPath ? voxelStages : globalStages,
                        stage_for_event(event, voxelPath));

    if (event != nullptr) {
        draw_text(renderer, rect.position.x + 10.0F, rect.position.y + 82.0F,
                  std::format("{}  POS ({:.2f},{:.2f},{:.2f})",
                              event_name(event->kind), event->position.x, event->position.y, event->position.z),
                  event_color(event->kind), 1.45F);
    }
    draw_wrapped_text(renderer, rect.position.x + 10.0F, rect.position.y + 101.0F,
                      describe_event_details(scene, variant, event), enabled ? kText : kMuted,
                      1.25F, 62, 2);
}

[[nodiscard]] float event_row(InspectorEventKind kind) noexcept {
    switch (kind) {
    case InspectorEventKind::GlobalBvhNode: return 0.0F;
    case InspectorEventKind::BrickVisit: return 1.0F;
    case InspectorEventKind::MicroBvhNode: return 2.0F;
    case InspectorEventKind::VolumeSegment: return 3.0F;
    case InspectorEventKind::TriangleHit: return 4.0F;
    }
    return 0.0F;
}

void draw_timeline(
    SDL_Renderer* renderer,
    UiRect rect,
    const InspectorCapture& bvh,
    const InspectorCapture& voxel,
    std::size_t visibleEvents,
    int viewMode) {
    fill_rect(renderer, rect, kPanel2);
    outline_rect(renderer, rect, kBorder);
    const std::array<std::string_view, 5> labels{"BVH", "BRICK", "MICRO", "VOLUME", "HIT"};
    const float rowGap = std::max((rect.size.y - 18.0F) / 5.0F, 11.0F);
    for (u32 row = 0; row < labels.size(); ++row) {
        const float y = rect.position.y + 11.0F + static_cast<float>(row) * rowGap;
        draw_text(renderer, rect.position.x + 8.0F, y - 5.0F, labels[row], kMuted, 1.25F);
        draw_line(renderer, rect.position.x + 67.0F, y, rect.position.x + rect.size.x - 10.0F, y,
                  Color{43, 50, 62, 255});
    }
    float maxT = 1.0F;
    for (const InspectorEvent& event : bvh.events) maxT = std::max(maxT, event.t);
    for (const InspectorEvent& event : voxel.events) maxT = std::max(maxT, event.t);
    auto draw_events = [&](const InspectorCapture& capture, bool upperHalf, bool enabled) {
        if (!enabled) return;
        const std::size_t count = std::min(visibleEvents, capture.events.size());
        for (std::size_t i = 0; i < count; ++i) {
            const InspectorEvent& event = capture.events[i];
            const float x = rect.position.x + 69.0F + (event.t / maxT) * std::max(rect.size.x - 81.0F, 1.0F);
            const float rowY = rect.position.y + 11.0F + event_row(event.kind) * rowGap;
            const float y = rowY + (upperHalf ? -3.0F : 3.0F);
            fill_rect(renderer, {{x - 2.0F, y - 2.0F}, {5.0F, 5.0F}}, event_color(event.kind));
        }
    };
    draw_events(bvh, true, viewMode != 2);
    draw_events(voxel, false, viewMode != 1);
}

[[nodiscard]] std::string explain(const InspectorCapture& bvh, const InspectorCapture& voxel, u32 workloadMode) {
    const int triangleDelta = static_cast<int>(bvh.metrics.triangleTests) - static_cast<int>(voxel.metrics.triangleTests);
    const int bvhHierarchy = static_cast<int>(bvh.metrics.bvhNodeTests + bvh.metrics.brickSteps);
    const int voxelHierarchy = static_cast<int>(voxel.metrics.brickSteps + voxel.metrics.microBvhNodeTests);
    const int hierarchyDelta = voxelHierarchy - bvhHierarchy;
    if (triangleDelta > 0 && hierarchyDelta > 0) {
        return std::format(
            "THE VOXEL PATH REJECTED {} EXACT TRIANGLE TESTS, BUT PAID {} EXTRA BRICK OR MICRO-BVH OPERATIONS. THE WIN DEPENDS ON WHICH WORK IS CHEAPER ON THIS GPU AND HOW WELL NEARBY RAYS REUSE THE SAME BRICKS.",
            triangleDelta, hierarchyDelta);
    }
    if (triangleDelta > 0 && hierarchyDelta <= 0) {
        return std::format(
            "THE UNIFIED PATH DID LESS WORK ON THIS RAY: {} FEWER TRIANGLE TESTS AND {} FEWER HIERARCHY OPERATIONS. THIS IS THE BEST CASE FOR A BRICK-NATIVE WORLD.",
            triangleDelta, -hierarchyDelta);
    }
    if (triangleDelta <= 0 && hierarchyDelta > 0) {
        return std::format(
            "THE GLOBAL BVH HAS BETTER LOCALITY HERE. THE VOXEL PATH DID {} MORE TRIANGLE TESTS AND {} MORE HIERARCHY OPERATIONS. INCOHERENT RAYS OFTEN LOOK LIKE THIS.",
            -triangleDelta, hierarchyDelta);
    }
    if (workloadMode != 0u && voxel.metrics.volumeSamples > 0u) {
        return "THE VOXEL PATH REUSED ONE BRICK WALK FOR BOTH SURFACE AND VOLUME DATA. THE GLOBAL PATH FIRST SEARCHED ITS TRIANGLE BVH, THEN WALKED BRICKS SEPARATELY FOR FOG OR SMOKE.";
    }
    return "THE TWO METHODS TOUCHED SIMILAR WORK FOR THIS RAY. CHANGE BRICK SIZE, RAY COHERENCE, WORKLOAD, OR THE SELECTED PIXEL TO EXPOSE THE TRADEOFF.";
}

void draw_button(SDL_Renderer* renderer, UiRect rect, std::string_view label, bool active = false) {
    fill_rect(renderer, rect, active ? Color{46, 104, 166, 255} : kPanel);
    outline_rect(renderer, rect, active ? kAccent : kBorder);
    const float scale = label.size() <= 2 ? kHeadingTextScale : 1.5F;
    draw_text(renderer, rect.position.x + 8.0F, rect.position.y + 12.0F,
              label, active ? kText : kMuted, scale);
}

[[nodiscard]] u64 hash_float(u64 seed, float value) noexcept {
    return (seed ^ static_cast<u64>(std::bit_cast<u32>(value))) * 1099511628211ull;
}

[[nodiscard]] u64 camera_signature(const InspectorFrameInput& input) noexcept {
    u64 hash = 1469598103934665603ull;
    for (const float value : std::array<float, 12>{
             input.push.cameraPosition.x, input.push.cameraPosition.y, input.push.cameraPosition.z,
             input.push.cameraForward.x, input.push.cameraForward.y, input.push.cameraForward.z,
             input.push.cameraRight.x, input.push.cameraRight.y, input.push.cameraRight.z,
             input.push.cameraUp.x, input.push.cameraUp.y, input.push.cameraUp.z}) {
        hash = hash_float(hash, value);
    }
    return hash;
}

} // namespace

struct LiveRayInspector::Impl {
    SDL_Window* window{};
    SDL_Renderer* renderer{};
    SDL_Window* parent{};
    SDL_WindowID windowId{};
    InspectorCapture bvh{};
    InspectorCapture voxel{};
    bool playing{true};
    bool refreshRequested{true};
    float playbackRate{18.0F};
    float accumulator{};
    std::size_t visibleEvents{};
    int viewMode{}; // 0 both, 1 bvh, 2 voxel
    u64 selectionSignature{};
    u64 frozenCameraSignature{};
    UiVec2 mouse{};

    [[nodiscard]] std::size_t maximum_events() const noexcept {
        return std::max(bvh.events.size(), voxel.events.size());
    }

    void clamp_step() noexcept { visibleEvents = std::min(visibleEvents, maximum_events()); }
};

namespace {

void position_child_near_parent(SDL_Window* child, SDL_Window* parent) noexcept {
    if (child == nullptr || parent == nullptr) return;

    int parentX = 0;
    int parentY = 0;
    int parentWidth = kLiveRenderWidth;
    int parentHeight = kLiveRenderHeight;
    int childWidth = kLiveInspectorWindowWidth;
    int childHeight = kLiveInspectorWindowHeight;
    SDL_GetWindowPosition(parent, &parentX, &parentY);
    SDL_GetWindowSize(parent, &parentWidth, &parentHeight);
    SDL_GetWindowSize(child, &childWidth, &childHeight);

    // Place the owned tool window beside the main view when possible. It remains
    // independently movable and resizable after this initial placement.
    const int childX = parentX + std::max(parentWidth - childWidth - 32, 32);
    const int childY = parentY + std::max((parentHeight - childHeight) / 2, 32);
    SDL_SetWindowPosition(child, childX, childY);
}

} // namespace

LiveRayInspector::LiveRayInspector() : impl_(new Impl{}) {}
LiveRayInspector::~LiveRayInspector() { close(); delete impl_; }

bool LiveRayInspector::open(SDL_Window* parent) {
    if (impl_->window != nullptr) {
        SDL_ShowWindow(impl_->window);
        SDL_RaiseWindow(impl_->window);
        return true;
    }
    if (parent == nullptr) {
        SDL_SetError("The live ray inspector requires a valid parent window.");
        return false;
    }

    const SDL_PropertiesID properties = SDL_CreateProperties();
    if (properties == 0) return false;

    const bool propertiesReady =
        SDL_SetStringProperty(properties, SDL_PROP_WINDOW_CREATE_TITLE_STRING,
                              "Epoch traversal inspector") &&
        SDL_SetNumberProperty(properties, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER,
                              kLiveInspectorWindowWidth) &&
        SDL_SetNumberProperty(properties, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER,
                              kLiveInspectorWindowHeight) &&
        SDL_SetPointerProperty(properties, SDL_PROP_WINDOW_CREATE_PARENT_POINTER, parent) &&
        SDL_SetBooleanProperty(properties, SDL_PROP_WINDOW_CREATE_UTILITY_BOOLEAN, true) &&
        SDL_SetBooleanProperty(properties, SDL_PROP_WINDOW_CREATE_RESIZABLE_BOOLEAN, true) &&
        SDL_SetBooleanProperty(properties, SDL_PROP_WINDOW_CREATE_HIGH_PIXEL_DENSITY_BOOLEAN, true) &&
        SDL_SetBooleanProperty(properties, SDL_PROP_WINDOW_CREATE_FOCUSABLE_BOOLEAN, true) &&
        SDL_SetBooleanProperty(properties, SDL_PROP_WINDOW_CREATE_HIDDEN_BOOLEAN, true);

    if (!propertiesReady) {
        SDL_DestroyProperties(properties);
        return false;
    }

    impl_->window = SDL_CreateWindowWithProperties(properties);
    SDL_DestroyProperties(properties);
    if (impl_->window == nullptr) return false;

    impl_->parent = parent;
    if (SDL_GetWindowParent(impl_->window) != parent) {
        SDL_DestroyWindow(impl_->window);
        impl_->window = nullptr;
        impl_->parent = nullptr;
        SDL_SetError("SDL created the traversal inspector without the requested parent relationship.");
        return false;
    }

    SDL_SetWindowMinimumSize(
        impl_->window,
        kLiveInspectorMinimumWindowWidth,
        kLiveInspectorMinimumWindowHeight);

    impl_->renderer = SDL_CreateRenderer(impl_->window, "software");
    if (impl_->renderer == nullptr) {
        SDL_DestroyWindow(impl_->window);
        impl_->window = nullptr;
        impl_->parent = nullptr;
        return false;
    }

    SDL_SetRenderDrawBlendMode(impl_->renderer, SDL_BLENDMODE_BLEND);
    SDL_SetRenderVSync(impl_->renderer, 1);
    impl_->windowId = SDL_GetWindowID(impl_->window);
    impl_->refreshRequested = true;
    position_child_near_parent(impl_->window, parent);
    SDL_ShowWindow(impl_->window);
    SDL_RaiseWindow(impl_->window);
    return true;
}

void LiveRayInspector::close() noexcept {
    if (impl_ == nullptr) return;
    if (impl_->renderer != nullptr) SDL_DestroyRenderer(impl_->renderer);
    if (impl_->window != nullptr) SDL_DestroyWindow(impl_->window);
    impl_->renderer = nullptr;
    impl_->window = nullptr;
    impl_->parent = nullptr;
    impl_->windowId = 0;
}

bool LiveRayInspector::is_open() const noexcept { return impl_ != nullptr && impl_->window != nullptr; }
bool LiveRayInspector::is_owned_child() const noexcept {
    return impl_ != nullptr && impl_->window != nullptr && impl_->parent != nullptr &&
           SDL_GetWindowParent(impl_->window) == impl_->parent;
}

bool LiveRayInspector::handle_event(const SDL_Event& event) {
    if (!is_open()) return false;
    SDL_WindowID eventWindow = 0;
    switch (event.type) {
    case SDL_EVENT_WINDOW_CLOSE_REQUESTED: eventWindow = event.window.windowID; break;
    case SDL_EVENT_WINDOW_RESIZED:
    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED: eventWindow = event.window.windowID; break;
    case SDL_EVENT_MOUSE_MOTION: eventWindow = event.motion.windowID; break;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP: eventWindow = event.button.windowID; break;
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP: eventWindow = event.key.windowID; break;
    default: return false;
    }
    if (eventWindow != impl_->windowId) return false;
    if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
        close();
        return true;
    }
    if (event.type == SDL_EVENT_MOUSE_MOTION) {
        impl_->mouse = {event.motion.x, event.motion.y};
        return true;
    }
    if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) {
        switch (event.key.key) {
        case SDLK_SPACE:
            if (!impl_->playing && impl_->visibleEvents >= impl_->maximum_events()) impl_->visibleEvents = 0;
            impl_->playing = !impl_->playing;
            break;
        case SDLK_LEFT: if (impl_->visibleEvents > 0) --impl_->visibleEvents; impl_->playing = false; break;
        case SDLK_RIGHT: impl_->visibleEvents = std::min(impl_->visibleEvents + 1, impl_->maximum_events()); impl_->playing = false; break;
        case SDLK_HOME: impl_->visibleEvents = 0; impl_->playing = false; break;
        case SDLK_RETURN: impl_->refreshRequested = true; break;
        case SDLK_1: impl_->viewMode = 1; break;
        case SDLK_2: impl_->viewMode = 2; break;
        case SDLK_3: impl_->viewMode = 0; break;
        case SDLK_MINUS: impl_->playbackRate = std::max(1.0F, impl_->playbackRate * 0.75F); break;
        case SDLK_EQUALS: impl_->playbackRate = std::min(240.0F, impl_->playbackRate * 1.333333F); break;
        default: break;
        }
        return true;
    }
    if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && event.button.button == SDL_BUTTON_LEFT) {
        int width = 0, height = 0;
        SDL_GetWindowSize(impl_->window, &width, &height);
        const ControlLayout layout = make_layout(static_cast<float>(width), static_cast<float>(height));
        const UiVec2 point{event.button.x, event.button.y};
        if (ui_contains(layout.play, point)) {
            if (!impl_->playing && impl_->visibleEvents >= impl_->maximum_events()) impl_->visibleEvents = 0;
            impl_->playing = !impl_->playing;
        } else if (ui_contains(layout.previous, point)) {
            if (impl_->visibleEvents > 0) --impl_->visibleEvents;
            impl_->playing = false;
        } else if (ui_contains(layout.next, point)) {
            impl_->visibleEvents = std::min(impl_->visibleEvents + 1, impl_->maximum_events());
            impl_->playing = false;
        } else if (ui_contains(layout.reset, point)) {
            impl_->visibleEvents = 0;
            impl_->playing = false;
        } else if (ui_contains(layout.refresh, point)) {
            impl_->refreshRequested = true;
        } else if (ui_contains(layout.slower, point)) {
            impl_->playbackRate = std::max(1.0F, impl_->playbackRate * 0.75F);
        } else if (ui_contains(layout.faster, point)) {
            impl_->playbackRate = std::min(240.0F, impl_->playbackRate * 1.333333F);
        } else {
            for (int index = 0; index < 3; ++index) {
                if (ui_contains(layout.view[static_cast<std::size_t>(index)], point)) impl_->viewMode = index;
            }
            if (ui_contains(layout.timeline, point) && impl_->maximum_events() > 0) {
                const float fraction = std::clamp(
                    (point.x - layout.timeline.position.x) / std::max(layout.timeline.size.x, 1.0F), 0.0F, 1.0F);
                impl_->visibleEvents = static_cast<std::size_t>(fraction * static_cast<float>(impl_->maximum_events()));
                impl_->playing = false;
            }
        }
        impl_->clamp_step();
        return true;
    }
    return true;
}

void LiveRayInspector::update(const SceneData& scene, const InspectorFrameInput& input, float deltaSeconds) {
    if (!is_open() || scene.brickVariants.empty()) return;
    const u32 variantIndex = std::min(input.activeBrickVariant, static_cast<u32>(scene.brickVariants.size() - 1));
    const BrickVariant& variant = scene.brickVariants[variantIndex];
    const u64 selectionSignature =
        static_cast<u64>(input.selectedX) |
        (static_cast<u64>(input.selectedY) << 16u) |
        (static_cast<u64>(variantIndex) << 32u) |
        (static_cast<u64>(input.rayMode) << 40u) |
        (static_cast<u64>(input.workloadMode) << 41u);
    const bool selectionChanged = selectionSignature != impl_->selectionSignature;
    if (selectionChanged || impl_->refreshRequested || (impl_->bvh.events.empty() && impl_->voxel.events.empty())) {
        impl_->selectionSignature = selectionSignature;
        impl_->frozenCameraSignature = camera_signature(input);
        impl_->visibleEvents = 0;
        impl_->accumulator = 0.0F;
        impl_->bvh = capture_bvh(scene, variant, input);
        impl_->voxel = capture_voxel(scene, variant, input);
        impl_->refreshRequested = false;
    }
    impl_->clamp_step();
    if (impl_->playing && impl_->maximum_events() > 0) {
        impl_->accumulator += std::max(deltaSeconds, 0.0F) * impl_->playbackRate;
        while (impl_->accumulator >= 1.0F && impl_->visibleEvents < impl_->maximum_events()) {
            impl_->accumulator -= 1.0F;
            ++impl_->visibleEvents;
        }
        if (impl_->visibleEvents >= impl_->maximum_events()) impl_->playing = false;
    }
}

void LiveRayInspector::render(const SceneData& scene, const InspectorFrameInput& input) {
    if (!is_open() || scene.brickVariants.empty()) return;
    int width = 0, height = 0;
    SDL_GetWindowSize(impl_->window, &width, &height);
    if (width <= 0 || height <= 0) return;
    const ControlLayout layout = make_layout(static_cast<float>(width), static_cast<float>(height));
    set_color(impl_->renderer, kBackground);
    SDL_RenderClear(impl_->renderer);

    draw_button(impl_->renderer, layout.play, impl_->playing ? "PAUSE" : "PLAY", impl_->playing);
    draw_button(impl_->renderer, layout.previous, "PREV");
    draw_button(impl_->renderer, layout.next, "NEXT");
    draw_button(impl_->renderer, layout.reset, "RESET");
    draw_button(impl_->renderer, layout.refresh, "CAPTURE");
    draw_button(impl_->renderer, layout.slower, "-");
    draw_button(impl_->renderer, layout.faster, "+");
    draw_button(impl_->renderer, layout.view[0], "BOTH", impl_->viewMode == 0);
    draw_button(impl_->renderer, layout.view[1], "BVH", impl_->viewMode == 1);
    draw_button(impl_->renderer, layout.view[2], "VOXEL", impl_->viewMode == 2);

    fill_rect(impl_->renderer, layout.status, kPanel);
    outline_rect(impl_->renderer, layout.status, kBorder);
    const BrickVariant& activeVariant = scene.brickVariants[
        std::min(input.activeBrickVariant, static_cast<u32>(scene.brickVariants.size() - 1))];
    const std::string status = std::format(
        "PIXEL {}:{}  {}  {}  BRICK {}^3",
        input.selectedX, input.selectedY,
        input.rayMode == 0u ? "COHERENT" : "INCOHERENT",
        input.workloadMode == 0u ? "SURFACE" : "SURFACE+VOLUME",
        activeVariant.brickSize);
    draw_text(impl_->renderer, layout.status.position.x + 10.0F, layout.status.position.y + 8.0F,
              status, kText, 1.55F);
    const std::string playback = std::format(
        "{}  STEP {}/{}  {:.1f} STEPS/S",
        impl_->playing ? "PLAYING" : "PAUSED",
        impl_->visibleEvents, impl_->maximum_events(), impl_->playbackRate);
    draw_text(impl_->renderer, layout.status.position.x + 10.0F, layout.status.position.y + 29.0F,
              playback, kMuted, 1.35F);
    draw_text(impl_->renderer, layout.status.position.x + 10.0F, layout.status.position.y + 48.0F,
              "RAY = ORIGIN + DIRECTION * T. LEAF INDICES REFER TO TRIANGLES; THE RAY DOES NOT.", kMuted, 1.0F);

    draw_world_map(impl_->renderer, scene, layout.worldMap, activeVariant,
                   impl_->bvh, impl_->voxel, impl_->visibleEvents, impl_->viewMode);
    draw_detail_card(impl_->renderer, scene, layout.bvhDetails,
                     "GLOBAL BVH + SEPARATE VOLUME WALK", impl_->bvh, activeVariant,
                     impl_->visibleEvents, false, impl_->viewMode != 2);
    draw_detail_card(impl_->renderer, scene, layout.voxelDetails,
                     "BRICK DDA + LOCAL MICRO-BVH", impl_->voxel, activeVariant,
                     impl_->visibleEvents, true, impl_->viewMode != 1);
    draw_timeline(impl_->renderer, layout.timeline, impl_->bvh, impl_->voxel,
                  impl_->visibleEvents, impl_->viewMode);

    fill_rect(impl_->renderer, layout.explanation, kPanel);
    outline_rect(impl_->renderer, layout.explanation, kBorder);
    draw_wrapped_text(impl_->renderer, layout.explanation.position.x + 10.0F,
                      layout.explanation.position.y + 9.0F,
                      explain(impl_->bvh, impl_->voxel, input.workloadMode),
                      kText, 1.4F, 58, 4);
    const std::string gpu = std::format(
        "GPU SAMPLE: BVH {:.3f} MS  VOXEL {:.3f} MS. INSPECTOR CPU MIRROR IS OUTSIDE TIMESTAMPED GPU DISPATCH.",
        input.bvhGpuMs, input.voxelGpuMs);
    draw_wrapped_text(impl_->renderer, layout.explanation.position.x + 10.0F,
                      layout.explanation.position.y + std::max(layout.explanation.size.y - 39.0F, 30.0F),
                      gpu, kMuted, 1.25F, 64, 2);

    SDL_RenderPresent(impl_->renderer);
}

} // namespace epoch::voxel_demo
