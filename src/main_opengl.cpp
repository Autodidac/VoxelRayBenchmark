#include <glad/glad.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include "benchmark_common.hpp"
#include "ray_inspector.hpp"

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ranges>

#ifndef EPOCH_VOXEL_SHADER_SOURCE
#error EPOCH_VOXEL_SHADER_SOURCE must point to shaders/trace.comp
#endif

namespace epoch::voxel_demo {

struct GlPrograms {
    GLuint bvhDebug{};
    GLuint voxelDebug{};
    GLuint bvhBenchmark{};
    GLuint voxelBenchmark{};
};

struct GlBuffers {
    GLuint triangles{};
    GLuint bvhNodes{};
    GLuint bvhReferences{};
    GLuint brickHeaders{};
    GLuint microNodes{};
    GLuint microReferences{};
    GLuint metrics{};
    GLuint debug{};
    GLuint traversalConfig{};
    GLuint pushConstants{};
};

struct BenchmarkStatsGl {
    std::string name;
    u32 rayMode{};
    u32 workloadMode{};
    u32 brickSize{};
    double structureBuildMs{};
    std::size_t structureBytes{};
    double medianMs{};
    double averageMs{};
    double p10Ms{};
    double p90Ms{};
    MetricsSnapshot metrics{};
};

[[nodiscard]] std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        fail(std::format("Failed to open shader: {}", path.string()));
    }
    return {std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
}

[[nodiscard]] std::string shader_source(bool voxel, bool instrument) {
    std::string source = read_text_file(EPOCH_VOXEL_SHADER_SOURCE);
    const std::size_t versionEnd = source.find('\n');
    if (versionEnd == std::string::npos) {
        fail("Shader has no #version line.");
    }

    const std::string defines = std::format(
        "#define EPOCH_OPENGL 1\n#define TRACE_BVH {}\n#define TRACE_VOXEL {}\n#define INSTRUMENT {}\n",
        voxel ? 0 : 1,
        voxel ? 1 : 0,
        instrument ? 1 : 0);
    source.insert(versionEnd + 1, defines);
    return source;
}

[[nodiscard]] GLuint compile_compute_program(bool voxel, bool instrument) {
    const std::string source = shader_source(voxel, instrument);
    const char* sourcePointer = source.c_str();

    const GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &sourcePointer, nullptr);
    glCompileShader(shader);

    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled != GL_TRUE) {
        GLint length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
        glGetShaderInfoLog(shader, length, nullptr, log.data());
        glDeleteShader(shader);
        fail(std::format("OpenGL compute shader compilation failed:\n{}", log));
    }

    const GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glDetachShader(program, shader);
    glDeleteShader(shader);

    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked != GL_TRUE) {
        GLint length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
        glGetProgramInfoLog(program, length, nullptr, log.data());
        glDeleteProgram(program);
        fail(std::format("OpenGL compute program link failed:\n{}", log));
    }
    return program;
}

class OpenGLDemo {
public:
    explicit OpenGLDemo(DemoOptions options) : options_(options) {}
    OpenGLDemo(const OpenGLDemo&) = delete;
    OpenGLDemo& operator=(const OpenGLDemo&) = delete;

    ~OpenGLDemo() { cleanup(); }

    int run() {
        initialize();
        if (options_.benchmarkOnStart) {
            benchmark_all();
            if (options_.benchmarkOnly) {
                return 0;
            }
        }
        main_loop();
        return 0;
    }

private:
    SDL_Window* window_{};
    SDL_GLContext context_{};
    bool running_{true};
    bool cameraPaused_{};
    bool benchmarkRequested_{};
    LiveRayInspector inspector_{};
    u32 traversalMode_{};
    u32 visualizationMode_{};
    u32 activeBrickVariant_{1};
    u32 rayMode_{};
    u32 workloadMode_{};
    u32 selectedX_{640};
    u32 selectedY_{360};
    u32 width_{static_cast<u32>(kLiveRenderWidth)};
    u32 height_{static_cast<u32>(kLiveRenderHeight)};
    u64 frameIndex_{};
    float cameraAngle_{};
    double lastGpuMs_{};
    double lastBvhGpuMs_{};
    double lastVoxelGpuMs_{};
    MetricsSnapshot lastMetrics_{};
    std::chrono::steady_clock::time_point lastFrameTime_{};
    std::chrono::steady_clock::time_point lastTitleUpdate_{};

    DemoOptions options_{};
    SceneData scene_{};
    GlPrograms programs_{};
    GlBuffers buffers_{};
    GLuint outputTexture_{};
    GLuint outputFramebuffer_{};
    GLuint timeQuery_{};

    struct ViewportRect {
        int x{};
        int y{};
        int width{};
        int height{};
    };

    [[nodiscard]] ViewportRect scene_viewport_logical() const noexcept {
        int windowWidth = 1;
        int windowHeight = 1;
        SDL_GetWindowSize(window_, &windowWidth, &windowHeight);
        const int availableWidth = std::max(windowWidth, 1);
        const float targetAspect = static_cast<float>(width_) / static_cast<float>(std::max(height_, 1u));
        int viewportWidth = availableWidth;
        int viewportHeight = static_cast<int>(static_cast<float>(viewportWidth) / targetAspect);
        if (viewportHeight > windowHeight) {
            viewportHeight = windowHeight;
            viewportWidth = static_cast<int>(static_cast<float>(viewportHeight) * targetAspect);
        }
        return {
            .x = std::max((availableWidth - viewportWidth) / 2, 0),
            .y = std::max((windowHeight - viewportHeight) / 2, 0),
            .width = std::max(viewportWidth, 1),
            .height = std::max(viewportHeight, 1)};
    }

    [[nodiscard]] ViewportRect scene_viewport_pixels() const noexcept {
        int logicalWidth = 1;
        int logicalHeight = 1;
        int pixelWidth = 1;
        int pixelHeight = 1;
        SDL_GetWindowSize(window_, &logicalWidth, &logicalHeight);
        SDL_GetWindowSizeInPixels(window_, &pixelWidth, &pixelHeight);
        const ViewportRect logical = scene_viewport_logical();
        const float scaleX = static_cast<float>(pixelWidth) / static_cast<float>(std::max(logicalWidth, 1));
        const float scaleY = static_cast<float>(pixelHeight) / static_cast<float>(std::max(logicalHeight, 1));
        return {
            .x = static_cast<int>(static_cast<float>(logical.x) * scaleX),
            .y = static_cast<int>(static_cast<float>(logical.y) * scaleY),
            .width = std::max(static_cast<int>(static_cast<float>(logical.width) * scaleX), 1),
            .height = std::max(static_cast<int>(static_cast<float>(logical.height) * scaleY), 1)};
    }

    void initialize() {
        if (!SDL_Init(SDL_INIT_VIDEO)) {
            fail(std::format("SDL_Init failed: {}", SDL_GetError()));
        }

        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 0);
        SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 0);

        const int initialWidth = kLiveRenderWidth;
        const int initialHeight = kLiveRenderHeight;
        window_ = SDL_CreateWindow(
            "Epoch OpenGL voxel benchmark",
            initialWidth,
            initialHeight,
            SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
        if (window_ == nullptr) {
            fail(std::format("SDL_CreateWindow failed: {}", SDL_GetError()));
        }
        if (!options_.benchmarkOnly) {
            SDL_SetWindowMinimumSize(window_, 640, 360);
        }

        context_ = SDL_GL_CreateContext(window_);
        if (context_ == nullptr) {
            fail(std::format("SDL_GL_CreateContext failed: {}", SDL_GetError()));
        }
        if (!SDL_GL_MakeCurrent(window_, context_)) {
            fail(std::format("SDL_GL_MakeCurrent failed: {}", SDL_GetError()));
        }
        SDL_GL_SetSwapInterval(0);

        if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(SDL_GL_GetProcAddress))) {
            fail("gladLoadGLLoader failed.");
        }
        if (GLVersion.major < 4 || (GLVersion.major == 4 && GLVersion.minor < 3)) {
            fail(std::format("OpenGL 4.3 compute is required; loaded {}.{}.", GLVersion.major, GLVersion.minor));
        }

        std::cout << std::format(
            "OpenGL {}.{} | {} | {}\n",
            GLVersion.major,
            GLVersion.minor,
            reinterpret_cast<const char*>(glGetString(GL_VENDOR)),
            reinterpret_cast<const char*>(glGetString(GL_RENDERER)));
        std::cout << std::format("Building sparse scene targeting {} triangles...\n", options_.targetTriangles);
        scene_ = build_scene(options_);
        print_scene_summary();

        std::cout << "Creating OpenGL 4.3 buffers...\n";
        create_buffers();
        std::cout << "Compiling OpenGL compute programs...\n";
        create_programs();
        std::cout << "Creating output image...\n";
        create_output();
        glGenQueries(1, &timeQuery_);
        check_gl("timer query creation");
        apply_traversal_config(activeBrickVariant_, rayMode_, workloadMode_);

        selectedX_ = width_ / 2u;
        selectedY_ = height_ / 2u;
        lastFrameTime_ = std::chrono::steady_clock::now();
        lastTitleUpdate_ = lastFrameTime_;

        if (!options_.benchmarkOnly) {
            if (!inspector_.open(window_)) {
                std::cerr << std::format("Live ray inspector could not open: {}\n", SDL_GetError());
            } else {
                cameraPaused_ = true;
            }
        }

        std::cout
            << "Controls:\n"
            << "  1 / 2  : global BVH or brick DDA -> micro-BVH\n"
            << "  [ / ]  : previous / next brick size\n"
            << "  R      : coherent / incoherent ray set\n"
            << "  V      : surface-only / surface + sparse smoke-fog\n"
            << "  F1-F4  : shaded / triangle tests / traversal steps / composite\n"
            << "  Mouse  : select debug ray\n"
            << "  D      : dump selected ray traversal\n"
            << "  I      : toggle owned floating traversal inspector\n"
            << "  B      : benchmark all variants, write CSV, then exit cleanly\n"
            << "  Space  : pause camera\n"
            << "  Escape : quit\n\n";
    }

    void print_scene_summary() const {
        const std::size_t triangleBytes = scene_.triangles.size() * sizeof(TriangleGpu);
        const std::size_t bvhBytes = scene_.bvhNodes.size() * sizeof(BvhNodeGpu) +
                                     scene_.bvhTriangleReferences.size() * sizeof(u32);
        const std::size_t hybridBytes = scene_.brickHeaders.size() * sizeof(BrickHeaderGpu) +
                                        scene_.microBvhNodes.size() * sizeof(BvhNodeGpu) +
                                        scene_.microTriangleReferences.size() * sizeof(u32);
        std::cout << std::format(
            "Scene: {} triangles | world {}x{}x{} | global BVH {} nodes\n"
            "Build: global BVH {:.3f} ms | all brick variants {:.3f} ms\n"
            "Data: triangles {:.2f} MiB | global BVH {:.2f} MiB | all hybrid variants {:.2f} MiB\n",
            scene_.triangles.size(), scene_.gridX, scene_.gridY, scene_.gridZ, scene_.bvhNodes.size(),
            scene_.bvhBuildMs, scene_.voxelBuildMs,
            static_cast<double>(triangleBytes) / (1024.0 * 1024.0),
            static_cast<double>(bvhBytes) / (1024.0 * 1024.0),
            static_cast<double>(hybridBytes) / (1024.0 * 1024.0));
        for (const BrickVariant& variant : scene_.brickVariants) {
            const double duplication = static_cast<double>(variant.referenceCount) /
                                       static_cast<double>(std::max<std::size_t>(scene_.triangles.size(), 1));
            std::cout << std::format(
                "  brick {:2}^3: {:6} occupied / {:6} total | {:6} volume | {:8} micro nodes | {:9} refs ({:.2f}x) | {:.2f} MiB | build {:.3f} ms\n",
                variant.brickSize, variant.occupiedBricks, variant.headerCount, variant.volumeBricks, variant.nodeCount,
                variant.referenceCount, duplication,
                static_cast<double>(variant.byte_size()) / (1024.0 * 1024.0), variant.buildMs);
        }
    }

    static void check_gl(std::string_view operation) {
        const GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            fail(std::format("OpenGL error 0x{:04X} during {}.", static_cast<unsigned>(error), operation));
        }
    }

    template <typename T>
    static GLuint upload_buffer(std::span<const T> values) {
        GLuint buffer = 0;
        glGenBuffers(1, &buffer);
        if (buffer == 0) {
            fail("glGenBuffers returned zero.");
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
        glBufferData(
            GL_SHADER_STORAGE_BUFFER,
            static_cast<GLsizeiptr>(values.size_bytes()),
            values.data(),
            GL_STATIC_DRAW);
        check_gl("static buffer upload");
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        return buffer;
    }

    static GLuint dynamic_buffer(std::size_t size) {
        GLuint buffer = 0;
        glGenBuffers(1, &buffer);
        if (buffer == 0) {
            fail("glGenBuffers returned zero.");
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
        glBufferData(
            GL_SHADER_STORAGE_BUFFER,
            static_cast<GLsizeiptr>(size),
            nullptr,
            GL_DYNAMIC_COPY);
        check_gl("dynamic buffer allocation");
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        return buffer;
    }

    void create_buffers() {
        buffers_.triangles = upload_buffer<TriangleGpu>(scene_.triangles);
        buffers_.bvhNodes = upload_buffer<BvhNodeGpu>(scene_.bvhNodes);
        buffers_.bvhReferences = upload_buffer<u32>(scene_.bvhTriangleReferences);
        buffers_.brickHeaders = upload_buffer<BrickHeaderGpu>(scene_.brickHeaders);
        buffers_.microNodes = upload_buffer<BvhNodeGpu>(scene_.microBvhNodes);
        buffers_.microReferences = upload_buffer<u32>(scene_.microTriangleReferences);
        buffers_.metrics = dynamic_buffer(sizeof(MetricsSnapshot));
        buffers_.debug = dynamic_buffer(sizeof(DebugBufferGpu));
        buffers_.traversalConfig = dynamic_buffer(sizeof(TraversalConfigGpu));
        buffers_.pushConstants = dynamic_buffer(sizeof(PushConstants));

        const std::array<GLuint, 8> ssbos{
            buffers_.triangles,
            buffers_.bvhNodes,
            buffers_.bvhReferences,
            buffers_.brickHeaders,
            buffers_.microNodes,
            buffers_.microReferences,
            buffers_.metrics,
            buffers_.debug,
        };
        for (GLuint binding = 0; binding < ssbos.size(); ++binding) {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, ssbos[binding]);
        }
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, buffers_.traversalConfig);
        glBindBufferBase(GL_UNIFORM_BUFFER, 10, buffers_.pushConstants);
    }

    void create_programs() {
        programs_.bvhDebug = compile_compute_program(false, true);
        programs_.voxelDebug = compile_compute_program(true, true);
        programs_.bvhBenchmark = compile_compute_program(false, false);
        programs_.voxelBenchmark = compile_compute_program(true, false);
    }

    void destroy_output() noexcept {
        if (outputFramebuffer_ != 0) glDeleteFramebuffers(1, &outputFramebuffer_);
        if (outputTexture_ != 0) glDeleteTextures(1, &outputTexture_);
        outputFramebuffer_ = 0;
        outputTexture_ = 0;
    }

    void create_output() {
        destroy_output();
        glGenTextures(1, &outputTexture_);
        glBindTexture(GL_TEXTURE_2D, outputTexture_);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, static_cast<GLsizei>(width_), static_cast<GLsizei>(height_));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glGenFramebuffers(1, &outputFramebuffer_);
        glBindFramebuffer(GL_FRAMEBUFFER, outputFramebuffer_);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, outputTexture_, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            fail("OpenGL output framebuffer is incomplete.");
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindImageTexture(8, outputTexture_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
        check_gl("output image creation");
        selectedX_ = std::min(selectedX_, width_ - 1u);
        selectedY_ = std::min(selectedY_, height_ - 1u);
    }

    void apply_traversal_config(u32 variantIndex, u32 rayMode, u32 workloadMode) {
        if (scene_.brickVariants.empty()) {
            fail("No brick variants were built.");
        }
        activeBrickVariant_ = std::min(variantIndex, static_cast<u32>(scene_.brickVariants.size() - 1));
        rayMode_ = rayMode & 1u;
        workloadMode_ = workloadMode & 1u;
        const BrickVariant& variant = scene_.brickVariants[activeBrickVariant_];
        TraversalConfigGpu config{};
        config.headerOffset = variant.headerOffset;
        config.brickSize = variant.brickSize;
        config.brickDimensionX = variant.dimensionX;
        config.brickDimensionY = variant.dimensionY;
        config.brickDimensionZ = variant.dimensionZ;
        config.rayMode = rayMode_;
        config.variantIndex = activeBrickVariant_;
        config.workloadMode = workloadMode_;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_.traversalConfig);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(config), &config);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        check_gl("traversal configuration upload");
    }

    [[nodiscard]] PushConstants make_push_constants() const {
        const float worldWidth = static_cast<float>(scene_.gridX) * scene_.cellSize.x;
        const float worldDepth = static_cast<float>(scene_.gridZ) * scene_.cellSize.z;
        const float orbitRadius = std::max(worldWidth, worldDepth) * 0.72F;
        const Vec3 target{0.0F, 12.0F, 0.0F};
        const Vec3 position{
            std::sin(cameraAngle_) * orbitRadius,
            72.0F + std::sin(cameraAngle_ * 0.37F) * 10.0F,
            std::cos(cameraAngle_) * orbitRadius};
        const Vec3 forward = normalize(target - position);
        const Vec3 right = normalize(cross(forward, {0.0F, 1.0F, 0.0F}));
        const Vec3 up = normalize(cross(right, forward));
        const float aspect = static_cast<float>(width_) / static_cast<float>(std::max(height_, 1u));
        const float tangent = std::tan(60.0F * 0.5F * 3.14159265359F / 180.0F);

        PushConstants constants{};
        constants.cameraPosition = to_vec4(position, 1.0F);
        constants.cameraForward = to_vec4(forward);
        constants.cameraRight = to_vec4(right * (tangent * aspect));
        constants.cameraUp = to_vec4(up * tangent);
        constants.worldMinimum = to_vec4(scene_.worldMinimum);
        constants.cellSize = to_vec4(scene_.cellSize);
        constants.imageMode = {width_, height_, visualizationMode_, static_cast<u32>(frameIndex_)};
        const u32 clampedX = std::min(selectedX_, width_ - 1u);
        const u32 clampedY = std::min(selectedY_, height_ - 1u);
        constants.gridSelected = {
            scene_.gridX, scene_.gridY, scene_.gridZ, (clampedY << 16u) | (clampedX & 0xffffu)};
        return constants;
    }

    [[nodiscard]] InspectorFrameInput make_inspector_input() const {
        return InspectorFrameInput{
            .push = make_push_constants(),
            .selectedX = selectedX_,
            .selectedY = selectedY_,
            .activeBrickVariant = activeBrickVariant_,
            .rayMode = rayMode_,
            .workloadMode = workloadMode_,
            .bvhGpuMs = lastBvhGpuMs_,
            .voxelGpuMs = lastVoxelGpuMs_};
    }

    void clear_instrumentation() const {
        const u32 zero = 0u;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_.metrics);
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_.debug);
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    [[nodiscard]] double dispatch(GLuint program, bool instrument, bool timed) {
        if (instrument) clear_instrumentation();
        const PushConstants constants = make_push_constants();
        glBindBuffer(GL_UNIFORM_BUFFER, buffers_.pushConstants);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(constants), &constants);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
        glUseProgram(program);
        glBindImageTexture(8, outputTexture_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

        if (timed) glBeginQuery(GL_TIME_ELAPSED, timeQuery_);
        glDispatchCompute((width_ + 7u) / 8u, (height_ + 7u) / 8u, 1u);
        if (timed) glEndQuery(GL_TIME_ELAPSED);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT |
                        GL_UNIFORM_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT);

        if (!timed) return 0.0;
        GLuint64 nanoseconds = 0;
        glGetQueryObjectui64v(timeQuery_, GL_QUERY_RESULT, &nanoseconds);
        return static_cast<double>(nanoseconds) / 1'000'000.0;
    }

    [[nodiscard]] MetricsSnapshot read_metrics() const {
        MetricsSnapshot metrics{};
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_.metrics);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(metrics), &metrics);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        return metrics;
    }

    void present() const {
        int pixelWidth = 1;
        int pixelHeight = 1;
        SDL_GetWindowSizeInPixels(window_, &pixelWidth, &pixelHeight);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glViewport(0, 0, pixelWidth, pixelHeight);
        glClearColor(0.015F, 0.020F, 0.030F, 1.0F);
        glClear(GL_COLOR_BUFFER_BIT);

        const ViewportRect viewport = scene_viewport_pixels();
        glBindFramebuffer(GL_READ_FRAMEBUFFER, outputFramebuffer_);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glBlitFramebuffer(
            0, 0, static_cast<GLint>(width_), static_cast<GLint>(height_),
            viewport.x,
            pixelHeight - (viewport.y + viewport.height),
            viewport.x + viewport.width,
            pixelHeight - viewport.y,
            GL_COLOR_BUFFER_BIT, GL_NEAREST);
        SDL_GL_SwapWindow(window_);
    }

    void render_frame() {
        const GLuint program = traversalMode_ == 0 ? programs_.bvhDebug : programs_.voxelDebug;
        lastGpuMs_ = dispatch(program, true, true);
        if (traversalMode_ == 0u) lastBvhGpuMs_ = lastGpuMs_;
        else lastVoxelGpuMs_ = lastGpuMs_;
        lastMetrics_ = read_metrics();
        present();
    }

    void update_window_title() {
        const auto now = std::chrono::steady_clock::now();
        if (now - lastTitleUpdate_ < std::chrono::milliseconds{250}) return;
        lastTitleUpdate_ = now;
        const double rays = static_cast<double>(std::max(lastMetrics_.rays, 1u));
        const double triangles = static_cast<double>(lastMetrics_.triangleTests) / rays;
        const double traversal = static_cast<double>(
            lastMetrics_.bvhNodeTests + lastMetrics_.brickSteps + lastMetrics_.microBvhNodeTests) / rays;
        const double volume = static_cast<double>(lastMetrics_.volumeSamples) / rays;
        const BrickVariant& variant = scene_.brickVariants[activeBrickVariant_];
        const std::string mode = traversalMode_ == 0
            ? (workloadMode_ == 0 ? "global BVH -> triangles" : "global BVH + separate volume DDA")
            : std::format("brick {}^3 DDA -> micro-BVH{}", variant.brickSize,
                          workloadMode_ == 0 ? " -> triangles" : " + unified volume");
        const std::string title = std::format(
            "OpenGL compute | {} | {} | {} | {:.3f} ms | {:.2f} tri/ray | {:.2f} trav/ray | {:.2f} vol/ray | {}x{}",
            mode,
            rayMode_ == 0 ? "coherent" : "incoherent",
            workloadMode_ == 0 ? "surface" : "surface+volume",
            lastGpuMs_, triangles, traversal, volume, width_, height_);
        SDL_SetWindowTitle(window_, title.c_str());
    }

    [[nodiscard]] MetricsSnapshot run_instrumented(GLuint program) {
        static_cast<void>(dispatch(program, true, false));
        glFinish();
        return read_metrics();
    }

    [[nodiscard]] double run_sample(GLuint program) {
        return dispatch(program, false, true);
    }

    [[nodiscard]] BenchmarkStatsGl benchmark_pipeline(
        std::string name,
        GLuint benchmarkProgram,
        GLuint debugProgram,
        u32 rayMode,
        u32 workloadMode,
        u32 brickSize,
        double structureBuildMs,
        std::size_t structureBytes) {
        for (u32 index = 0; index < kBenchmarkWarmups; ++index) {
            static_cast<void>(run_sample(benchmarkProgram));
        }
        std::vector<double> samples;
        samples.reserve(kBenchmarkSamples);
        for (u32 index = 0; index < kBenchmarkSamples; ++index) {
            samples.push_back(run_sample(benchmarkProgram));
        }
        std::ranges::sort(samples);
        const auto percentile = [&](double p) {
            const std::size_t index = static_cast<std::size_t>(
                std::clamp(p, 0.0, 1.0) * static_cast<double>(samples.size() - 1));
            return samples[index];
        };

        BenchmarkStatsGl stats{};
        stats.name = std::move(name);
        stats.rayMode = rayMode;
        stats.workloadMode = workloadMode;
        stats.brickSize = brickSize;
        stats.structureBuildMs = structureBuildMs;
        stats.structureBytes = structureBytes;
        stats.medianMs = percentile(0.5);
        stats.averageMs = std::accumulate(samples.begin(), samples.end(), 0.0) /
                          static_cast<double>(samples.size());
        stats.p10Ms = percentile(0.1);
        stats.p90Ms = percentile(0.9);
        stats.metrics = run_instrumented(debugProgram);
        return stats;
    }

    void write_csv(const std::vector<BenchmarkStatsGl>& results) const {
        constexpr std::string_view filename{"voxel_ray_benchmark_opengl.csv"};
        std::ofstream stream(filename.data(), std::ios::trunc);
        if (!stream) {
            std::cerr << "Failed to write " << filename << '\n';
            return;
        }
        stream << "backend,mode,workload,ray_set,brick_size,width,height,triangles,global_bvh_nodes,structure_bytes,structure_build_ms,median_gpu_ms,average_gpu_ms,p10_gpu_ms,p90_gpu_ms,ns_per_ray,rays,global_bvh_node_tests,brick_steps,micro_bvh_node_tests,occupied_bricks,triangle_tests,volume_samples,volume_bricks,hits,max_traversal_steps,traversal_per_ray,triangle_tests_per_ray,volume_samples_per_ray\n";
        for (const BenchmarkStatsGl& stats : results) {
            const double rays = static_cast<double>(std::max(stats.metrics.rays, 1u));
            const double traversal = static_cast<double>(
                stats.metrics.bvhNodeTests + stats.metrics.brickSteps + stats.metrics.microBvhNodeTests) / rays;
            stream << "opengl_compute," << stats.name << ','
                   << (stats.workloadMode == 0 ? "surface" : "surface_plus_volume") << ','
                   << (stats.rayMode == 0 ? "coherent" : "incoherent") << ','
                   << stats.brickSize << ',' << width_ << ',' << height_ << ','
                   << scene_.triangles.size() << ',' << scene_.bvhNodes.size() << ','
                   << stats.structureBytes << ',' << stats.structureBuildMs << ','
                   << stats.medianMs << ',' << stats.averageMs << ',' << stats.p10Ms << ',' << stats.p90Ms << ','
                   << stats.medianMs * 1'000'000.0 / rays << ',' << stats.metrics.rays << ','
                   << stats.metrics.bvhNodeTests << ',' << stats.metrics.brickSteps << ','
                   << stats.metrics.microBvhNodeTests << ',' << stats.metrics.occupiedBricks << ','
                   << stats.metrics.triangleTests << ',' << stats.metrics.volumeSamples << ','
                   << stats.metrics.volumeBricks << ',' << stats.metrics.hits << ','
                   << stats.metrics.maximumTraversal << ',' << traversal << ','
                   << static_cast<double>(stats.metrics.triangleTests) / rays << ','
                   << static_cast<double>(stats.metrics.volumeSamples) / rays << '\n';
        }
    }

    void benchmark_all() {
        glFinish();
        const u32 savedVariant = activeBrickVariant_;
        const u32 savedRayMode = rayMode_;
        const u32 savedWorkloadMode = workloadMode_;
        std::vector<BenchmarkStatsGl> results;
        results.reserve(2u * (1u + scene_.brickVariants.size()) +
                        4u * scene_.brickVariants.size());
        const std::size_t globalBvhBytes = scene_.bvhNodes.size() * sizeof(BvhNodeGpu) +
                                           scene_.bvhTriangleReferences.size() * sizeof(u32);

        std::cout << std::format(
            "\nOpenGL benchmark matrix: {} triangles, {}x{}, {} warmups + {} samples\n",
            scene_.triangles.size(), width_, height_, kBenchmarkWarmups, kBenchmarkSamples);
        std::cout << "Surface mode compares one global BVH against every brick size.\n"
                     "Volume mode compares global BVH + a second volume brick walk against one unified brick walk.\n";

        const auto print_stats = [](const BenchmarkStatsGl& stats, double speedRatio) {
            const double rays = static_cast<double>(std::max(stats.metrics.rays, 1u));
            const double traversal = static_cast<double>(
                stats.metrics.bvhNodeTests + stats.metrics.brickSteps + stats.metrics.microBvhNodeTests) / rays;
            const double volume = static_cast<double>(stats.metrics.volumeSamples) / rays;
            std::cout << std::format(
                "  {:42} median {:8.4f} ms | {:6.2f} tri/ray | {:6.2f} trav/ray | {:6.2f} vol/ray",
                stats.name,
                stats.medianMs,
                static_cast<double>(stats.metrics.triangleTests) / rays,
                traversal,
                volume);
            if (speedRatio > 0.0) std::cout << std::format(" | {:6.3f}x vs paired BVH", speedRatio);
            std::cout << '\n';
        };

        for (u32 rayMode = 0; rayMode < 2; ++rayMode) {
            const std::string label = rayMode == 0 ? "coherent" : "incoherent";
            std::cout << std::format("\nSurface-only / {} rays:\n", label);

            apply_traversal_config(0u, rayMode, 0u);
            BenchmarkStatsGl bvh = benchmark_pipeline(
                std::format("global_bvh_surface_{}", label),
                programs_.bvhBenchmark,
                programs_.bvhDebug,
                rayMode,
                0u,
                0u,
                scene_.bvhBuildMs,
                globalBvhBytes);
            print_stats(bvh, 0.0);
            const double bvhMedian = bvh.medianMs;
            results.push_back(std::move(bvh));

            for (u32 variantIndex = 0; variantIndex < scene_.brickVariants.size(); ++variantIndex) {
                apply_traversal_config(variantIndex, rayMode, 0u);
                const BrickVariant& variant = scene_.brickVariants[variantIndex];
                BenchmarkStatsGl hybrid = benchmark_pipeline(
                    std::format("brick_{}^3_micro_bvh_surface_{}", variant.brickSize, label),
                    programs_.voxelBenchmark,
                    programs_.voxelDebug,
                    rayMode,
                    0u,
                    variant.brickSize,
                    variant.buildMs,
                    variant.byte_size());
                print_stats(hybrid, hybrid.medianMs > 0.0 ? bvhMedian / hybrid.medianMs : 0.0);
                results.push_back(std::move(hybrid));
            }

            std::cout << std::format("\nSurface + sparse volume / {} rays:\n", label);
            for (u32 variantIndex = 0; variantIndex < scene_.brickVariants.size(); ++variantIndex) {
                apply_traversal_config(variantIndex, rayMode, 1u);
                const BrickVariant& variant = scene_.brickVariants[variantIndex];
                const std::size_t volumeIndexBytes = static_cast<std::size_t>(variant.headerCount) * sizeof(BrickHeaderGpu);

                BenchmarkStatsGl bvhVolume = benchmark_pipeline(
                    std::format("global_bvh_plus_volume_{}^3_{}", variant.brickSize, label),
                    programs_.bvhBenchmark,
                    programs_.bvhDebug,
                    rayMode,
                    1u,
                    variant.brickSize,
                    scene_.bvhBuildMs + variant.buildMs,
                    globalBvhBytes + volumeIndexBytes);
                print_stats(bvhVolume, 0.0);
                const double pairedMedian = bvhVolume.medianMs;
                results.push_back(std::move(bvhVolume));

                BenchmarkStatsGl unified = benchmark_pipeline(
                    std::format("brick_{}^3_unified_surface_volume_{}", variant.brickSize, label),
                    programs_.voxelBenchmark,
                    programs_.voxelDebug,
                    rayMode,
                    1u,
                    variant.brickSize,
                    variant.buildMs,
                    variant.byte_size());
                print_stats(unified, unified.medianMs > 0.0 ? pairedMedian / unified.medianMs : 0.0);
                results.push_back(std::move(unified));
            }
        }
        apply_traversal_config(savedVariant, savedRayMode, savedWorkloadMode);
        write_csv(results);
        std::cout << "\nWrote voxel_ray_benchmark_opengl.csv\n\n";
    }

    void dump_debug() {
        const GLuint program = traversalMode_ == 0 ? programs_.bvhDebug : programs_.voxelDebug;
        static_cast<void>(dispatch(program, true, false));
        glFinish();
        DebugBufferGpu debug{};
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_.debug);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(debug), &debug);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        const u32 count = std::min(debug.count, kMaxDebugEvents);
        std::cout << std::format(
            "\nSelected ray ({}, {}) | {} | {} events{}\n",
            selectedX_, selectedY_, traversalMode_ == 0 ? "BVH" : "voxel hybrid",
            count, debug.overflow != 0 ? " (truncated)" : "");
        for (u32 index = 0; index < count; ++index) {
            const DebugEventGpu& event = debug.events[index];
            const char* type = "unknown";
            switch (event.data[0]) {
                case 0: type = "BVH node"; break;
                case 1: type = "brick"; break;
                case 2: type = "micro-BVH"; break;
                case 3: type = "volume"; break;
                case 4: type = "triangle hit"; break;
                default: break;
            }
            std::cout << std::format(
                "{:3}: {:12} t={:9.4f} pos=({:8.3f},{:8.3f},{:8.3f}) data=({}, {}, {})\n",
                index, type, event.positionT.w, event.positionT.x, event.positionT.y, event.positionT.z,
                event.data[1], event.data[2], event.data[3]);
        }
        std::cout << '\n';
    }

    void handle_events() {
        SDL_Event event{};
        while (SDL_PollEvent(&event)) {
            if (inspector_.handle_event(event)) {
                continue;
            }
            const SDL_WindowID mainWindowId = SDL_GetWindowID(window_);
            switch (event.type) {
                case SDL_EVENT_QUIT: running_ = false; break;
                case SDL_EVENT_WINDOW_RESIZED:
                case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                    break;
                case SDL_EVENT_MOUSE_BUTTON_DOWN:
                    if (event.button.windowID == mainWindowId && event.button.button == SDL_BUTTON_LEFT) {
                        const ViewportRect viewport = scene_viewport_logical();
                        if (event.button.x >= static_cast<float>(viewport.x) &&
                            event.button.x < static_cast<float>(viewport.x + viewport.width) &&
                            event.button.y >= static_cast<float>(viewport.y) &&
                            event.button.y < static_cast<float>(viewport.y + viewport.height)) {
                            selectedX_ = static_cast<u32>(std::clamp(
                                (event.button.x - static_cast<float>(viewport.x)) * static_cast<float>(width_) /
                                    static_cast<float>(std::max(viewport.width, 1)),
                                0.0F, static_cast<float>(width_ - 1u)));
                            selectedY_ = static_cast<u32>(std::clamp(
                                (event.button.y - static_cast<float>(viewport.y)) * static_cast<float>(height_) /
                                    static_cast<float>(std::max(viewport.height, 1)),
                                0.0F, static_cast<float>(height_ - 1u)));
                        }
                    }
                    break;
                case SDL_EVENT_KEY_DOWN:
                    if (event.key.windowID != mainWindowId) break;
                    if (event.key.repeat) break;
                    switch (event.key.key) {
                        case SDLK_ESCAPE: running_ = false; break;
                        case SDLK_1: traversalMode_ = 0; break;
                        case SDLK_2: traversalMode_ = 1; break;
                        case SDLK_LEFTBRACKET:
                            activeBrickVariant_ = (activeBrickVariant_ + static_cast<u32>(scene_.brickVariants.size()) - 1u) %
                                                  static_cast<u32>(scene_.brickVariants.size());
                            apply_traversal_config(activeBrickVariant_, rayMode_, workloadMode_);
                            break;
                        case SDLK_RIGHTBRACKET:
                            activeBrickVariant_ = (activeBrickVariant_ + 1u) % static_cast<u32>(scene_.brickVariants.size());
                            apply_traversal_config(activeBrickVariant_, rayMode_, workloadMode_);
                            break;
                        case SDLK_R:
                            rayMode_ ^= 1u;
                            apply_traversal_config(activeBrickVariant_, rayMode_, workloadMode_);
                            break;
                        case SDLK_V:
                            workloadMode_ ^= 1u;
                            apply_traversal_config(activeBrickVariant_, rayMode_, workloadMode_);
                            break;
                        case SDLK_F1: visualizationMode_ = 0; break;
                        case SDLK_F2: visualizationMode_ = 1; break;
                        case SDLK_F3: visualizationMode_ = 2; break;
                        case SDLK_F4: visualizationMode_ = 3; break;
                        case SDLK_SPACE: cameraPaused_ = !cameraPaused_; break;
                        case SDLK_D: dump_debug(); break;
                        case SDLK_I:
                            if (inspector_.is_open()) inspector_.close();
                            else if (!inspector_.open(window_)) std::cerr << std::format("Live ray inspector could not open: {}\n", SDL_GetError());
                            else cameraPaused_ = true;
                            break;
                        case SDLK_B: benchmarkRequested_ = true; break;
                        default: break;
                    }
                    break;
                default: break;
            }
        }
    }

    void main_loop() {
        while (running_) {
            handle_events();
            if (!running_) break;
            if (benchmarkRequested_) {
                benchmarkRequested_ = false;
                cameraPaused_ = true;
                benchmark_all();
                glFinish();
                running_ = false;
                continue;
            }
            const auto now = std::chrono::steady_clock::now();
            const float delta = std::chrono::duration<float>(now - lastFrameTime_).count();
            lastFrameTime_ = now;
            if (!cameraPaused_) cameraAngle_ += std::min(delta, 0.1F) * 0.18F;
            render_frame();
            ++frameIndex_;
            update_window_title();
            if (inspector_.is_open()) {
                const InspectorFrameInput inspectorInput = make_inspector_input();
                inspector_.update(scene_, inspectorInput, delta);
                inspector_.render(scene_, inspectorInput);
                SDL_GL_MakeCurrent(window_, context_);
            }
        }
    }

    void cleanup() noexcept {
        inspector_.close();
        if (context_ != nullptr) {
            glFinish();
            if (timeQuery_ != 0) glDeleteQueries(1, &timeQuery_);
            destroy_output();
            const std::array<GLuint, 4> programs{
                programs_.bvhDebug, programs_.voxelDebug, programs_.bvhBenchmark, programs_.voxelBenchmark};
            for (GLuint program : programs) if (program != 0) glDeleteProgram(program);
            const std::array<GLuint, 10> buffers{
                buffers_.triangles, buffers_.bvhNodes, buffers_.bvhReferences, buffers_.brickHeaders,
                buffers_.microNodes, buffers_.microReferences, buffers_.metrics, buffers_.debug,
                buffers_.traversalConfig, buffers_.pushConstants};
            glDeleteBuffers(static_cast<GLsizei>(buffers.size()), buffers.data());
        }
        if (context_ != nullptr) SDL_GL_DestroyContext(context_);
        if (window_ != nullptr) SDL_DestroyWindow(window_);
        SDL_Quit();
        context_ = nullptr;
        window_ = nullptr;
    }
};

} // namespace epoch::voxel_demo

int main(int argc, char** argv) {
    try {
        const epoch::voxel_demo::DemoOptions options = epoch::voxel_demo::parse_options(argc, argv);
        epoch::voxel_demo::OpenGLDemo demo{options};
        return demo.run();
    } catch (const std::exception& exception) {
        std::cerr << "Fatal error: " << exception.what() << '\n';
        return 1;
    }
}
