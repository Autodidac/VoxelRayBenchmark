#pragma once

#include "benchmark_common.hpp"

#include <SDL3/SDL.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace epoch::voxel_demo {

inline constexpr int kLiveInspectorWindowWidth = 760;
inline constexpr int kLiveInspectorWindowHeight = 920;
inline constexpr int kLiveInspectorMinimumWindowWidth = 620;
inline constexpr int kLiveInspectorMinimumWindowHeight = 720;
inline constexpr int kLiveRenderWidth = 1280;
inline constexpr int kLiveRenderHeight = 720;

enum class InspectorEventKind : u32 {
    GlobalBvhNode = 0,
    BrickVisit = 1,
    MicroBvhNode = 2,
    VolumeSegment = 3,
    TriangleHit = 4,
};

struct InspectorEvent {
    InspectorEventKind kind{InspectorEventKind::GlobalBvhNode};
    Vec3 position{};
    float t{};
    u32 primary{};
    u32 secondary{};
    u32 tertiary{};
};

struct InspectorRay {
    Vec3 origin{};
    Vec3 direction{};
};

struct InspectorCapture {
    InspectorRay ray{};
    MetricsSnapshot metrics{};
    std::vector<InspectorEvent> events{};
    bool hit{};
    u32 triangleIndex{};
    float hitT{};
    bool truncated{};
};

struct InspectorFrameInput {
    PushConstants push{};
    u32 selectedX{};
    u32 selectedY{};
    u32 activeBrickVariant{};
    u32 rayMode{};
    u32 workloadMode{};
    double bvhGpuMs{};
    double voxelGpuMs{};
};

class LiveRayInspector {
public:
    LiveRayInspector();
    ~LiveRayInspector();

    LiveRayInspector(const LiveRayInspector&) = delete;
    LiveRayInspector& operator=(const LiveRayInspector&) = delete;

    [[nodiscard]] bool open(SDL_Window* parent);
    void close() noexcept;
    [[nodiscard]] bool is_open() const noexcept;
    [[nodiscard]] bool is_owned_child() const noexcept;
    [[nodiscard]] bool handle_event(const SDL_Event& event);
    void update(const SceneData& scene, const InspectorFrameInput& input, float deltaSeconds);
    void render(const SceneData& scene, const InspectorFrameInput& input);

private:
    struct Impl;
    Impl* impl_{};
};

} // namespace epoch::voxel_demo
