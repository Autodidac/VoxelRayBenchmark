#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include "benchmark_common.hpp"
#include "ray_inspector.hpp"

#include <array>
#include <bit>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <ranges>

namespace epoch::voxel_demo {

void vk_check(VkResult result, std::string_view operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::format("{} failed with VkResult {}", operation, static_cast<int>(result)));
    }
}

struct Buffer {
    VkBuffer handle{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    VkDeviceSize size{};
    void* mapped{};
};

struct Image {
    VkImage handle{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    VkImageView view{VK_NULL_HANDLE};
};

struct PipelineSet {
    VkPipeline bvhDebug{VK_NULL_HANDLE};
    VkPipeline voxelDebug{VK_NULL_HANDLE};
    VkPipeline bvhBenchmark{VK_NULL_HANDLE};
    VkPipeline voxelBenchmark{VK_NULL_HANDLE};
};

struct BenchmarkStats {
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

class VulkanDemo {
public:
    explicit VulkanDemo(DemoOptions options) : options_(options) {}
    VulkanDemo(const VulkanDemo&) = delete;
    VulkanDemo& operator=(const VulkanDemo&) = delete;

    ~VulkanDemo() {
        cleanup();
    }

    int run() {
        initialize();
        if (options_.benchmarkOnStart) {
            benchmark_all();
            if (options_.benchmarkOnly) {
                vkDeviceWaitIdle(device_);
                return 0;
            }
        }
        main_loop();
        vkDeviceWaitIdle(device_);
        return 0;
    }

private:
    SDL_Window* window_{};
    bool running_{true};
    bool cameraPaused_{};
    bool framebufferResized_{};
    bool benchmarkRequested_{};
    LiveRayInspector inspector_{};
    u32 traversalMode_{}; // 0 = global BVH, 1 = brick DDA + micro-BVH
    u32 visualizationMode_{};
    u32 activeBrickVariant_{1};
    u32 rayMode_{}; // 0 = coherent primary rays, 1 = incoherent secondary-like rays
    u32 workloadMode_{}; // 0 = surface-only, 1 = surface + sparse smoke/fog
    u32 selectedX_{640};
    u32 selectedY_{360};
    u64 frameIndex_{};
    float cameraAngle_{};

    VkInstance instance_{VK_NULL_HANDLE};
    VkSurfaceKHR surface_{VK_NULL_HANDLE};
    VkPhysicalDevice physicalDevice_{VK_NULL_HANDLE};
    VkPhysicalDeviceProperties physicalProperties_{};
    VkDevice device_{VK_NULL_HANDLE};
    VkQueue queue_{VK_NULL_HANDLE};
    u32 queueFamilyIndex_{};
    u32 timestampValidBits_{};

    VkSwapchainKHR swapchain_{VK_NULL_HANDLE};
    VkFormat swapchainFormat_{VK_FORMAT_UNDEFINED};
    VkExtent2D swapchainExtent_{};
    VkExtent2D renderExtent_{static_cast<u32>(kLiveRenderWidth), static_cast<u32>(kLiveRenderHeight)};
    std::vector<VkImage> swapchainImages_;
    std::vector<bool> swapchainInitialized_;

    VkCommandPool commandPool_{VK_NULL_HANDLE};
    VkCommandBuffer commandBuffer_{VK_NULL_HANDLE};
    VkSemaphore imageAvailable_{VK_NULL_HANDLE};
    VkSemaphore renderFinished_{VK_NULL_HANDLE};
    VkFence frameFence_{VK_NULL_HANDLE};
    VkQueryPool timestampQueryPool_{VK_NULL_HANDLE};

    VkDescriptorSetLayout descriptorSetLayout_{VK_NULL_HANDLE};
    VkDescriptorPool descriptorPool_{VK_NULL_HANDLE};
    VkDescriptorSet descriptorSet_{VK_NULL_HANDLE};
    VkPipelineLayout pipelineLayout_{VK_NULL_HANDLE};
    PipelineSet pipelines_{};

    Image outputImage_{};
    Buffer triangleBuffer_{};
    Buffer bvhNodeBuffer_{};
    Buffer bvhReferenceBuffer_{};
    Buffer brickHeaderBuffer_{};
    Buffer microBvhNodeBuffer_{};
    Buffer microReferenceBuffer_{};
    Buffer metricBuffer_{};
    Buffer debugBuffer_{};
    Buffer traversalConfigBuffer_{};

    DemoOptions options_{};
    SceneData scene_{};
    MetricsSnapshot lastMetrics_{};
    double lastGpuMs_{};
    double lastBvhGpuMs_{};
    double lastVoxelGpuMs_{};
    bool previousFrameValid_{};
    u32 previousTraversalMode_{};
    std::chrono::steady_clock::time_point lastFrameTime_{};
    std::chrono::steady_clock::time_point lastTitleUpdate_{};

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
        const float targetAspect = static_cast<float>(renderExtent_.width) /
                                   static_cast<float>(std::max(renderExtent_.height, 1u));
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

        const int initialWidth = kLiveRenderWidth;
        const int initialHeight = kLiveRenderHeight;
        window_ = SDL_CreateWindow(
            "Epoch brick micro-BVH benchmark",
            initialWidth,
            initialHeight,
            SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
        if (window_ == nullptr) {
            fail(std::format("SDL_CreateWindow failed: {}", SDL_GetError()));
        }
        if (!options_.benchmarkOnly) {
            SDL_SetWindowMinimumSize(window_, 640, 360);
        }

        create_instance();
        if (!SDL_Vulkan_CreateSurface(window_, instance_, nullptr, &surface_)) {
            fail(std::format("SDL_Vulkan_CreateSurface failed: {}", SDL_GetError()));
        }

        select_physical_device();
        create_device();
        create_command_resources();
        create_swapchain();
        create_output_image();

        std::cout << std::format(
            "Building sparse benchmark scene targeting {} triangles...\n",
            options_.targetTriangles);
        scene_ = build_scene(options_);

        const std::size_t triangleBytes = scene_.triangles.size() * sizeof(TriangleGpu);
        const std::size_t bvhBytes = scene_.bvhNodes.size() * sizeof(BvhNodeGpu) +
                                     scene_.bvhTriangleReferences.size() * sizeof(u32);
        const std::size_t hybridBytes = scene_.brickHeaders.size() * sizeof(BrickHeaderGpu) +
                                        scene_.microBvhNodes.size() * sizeof(BvhNodeGpu) +
                                        scene_.microTriangleReferences.size() * sizeof(u32);

        std::cout << std::format(
            "Scene: {} triangles in a {}x{}x{} sparse world | global BVH {} nodes\n"
            "Build: global BVH {:.3f} ms | all brick variants {:.3f} ms\n"
            "GPU data: triangles {:.2f} MiB | global BVH {:.2f} MiB | all hybrid variants {:.2f} MiB\n",
            scene_.triangles.size(),
            scene_.gridX,
            scene_.gridY,
            scene_.gridZ,
            scene_.bvhNodes.size(),
            scene_.bvhBuildMs,
            scene_.voxelBuildMs,
            static_cast<double>(triangleBytes) / (1024.0 * 1024.0),
            static_cast<double>(bvhBytes) / (1024.0 * 1024.0),
            static_cast<double>(hybridBytes) / (1024.0 * 1024.0));

        for (const BrickVariant& variant : scene_.brickVariants) {
            const double duplication = static_cast<double>(variant.referenceCount) /
                                       static_cast<double>(std::max<std::size_t>(scene_.triangles.size(), 1));
            std::cout << std::format(
                "  brick {:2}^3: {:6} occupied / {:6} total | {:6} volume | {:8} micro nodes | {:9} refs ({:.2f}x) | "
                "{:.2f} MiB | build {:.3f} ms\n",
                variant.brickSize,
                variant.occupiedBricks,
                variant.headerCount,
                variant.volumeBricks,
                variant.nodeCount,
                variant.referenceCount,
                duplication,
                static_cast<double>(variant.byte_size()) / (1024.0 * 1024.0),
                variant.buildMs);
        }

        create_scene_buffers();
        create_descriptors();
        create_pipelines();
        create_query_and_sync_objects();
        apply_traversal_config(activeBrickVariant_, rayMode_, workloadMode_);

        selectedX_ = renderExtent_.width / 2;
        selectedY_ = renderExtent_.height / 2;
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

    void create_instance() {
        u32 extensionCount = 0;
        const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&extensionCount);
        if (extensions == nullptr || extensionCount == 0) {
            fail(std::format("SDL_Vulkan_GetInstanceExtensions failed: {}", SDL_GetError()));
        }

        VkApplicationInfo applicationInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        applicationInfo.pApplicationName = "Epoch voxel traversal benchmark";
        applicationInfo.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
        applicationInfo.pEngineName = "EpochEngine experiment";
        applicationInfo.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
        applicationInfo.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo createInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        createInfo.pApplicationInfo = &applicationInfo;
        createInfo.enabledExtensionCount = extensionCount;
        createInfo.ppEnabledExtensionNames = extensions;

        vk_check(vkCreateInstance(&createInfo, nullptr, &instance_), "vkCreateInstance");
    }

    [[nodiscard]] static bool supports_device_extension(VkPhysicalDevice device, std::string_view required) {
        u32 count = 0;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> extensions(count);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data());
        return std::ranges::any_of(extensions, [&](const VkExtensionProperties& extension) {
            return required == extension.extensionName;
        });
    }

    [[nodiscard]] std::optional<u32> find_usable_queue_family(VkPhysicalDevice device) const {
        u32 count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

        for (u32 index = 0; index < count; ++index) {
            const VkQueueFlags required = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
            if ((families[index].queueFlags & required) != required || families[index].timestampValidBits == 0) {
                continue;
            }

            VkBool32 presentSupported = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, index, surface_, &presentSupported);
            if (presentSupported == VK_TRUE) {
                return index;
            }
        }
        return std::nullopt;
    }

    void select_physical_device() {
        u32 deviceCount = 0;
        vk_check(vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr), "vkEnumeratePhysicalDevices(count)");
        if (deviceCount == 0) {
            fail("No Vulkan physical device was found.");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vk_check(vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data()), "vkEnumeratePhysicalDevices");

        int bestScore = std::numeric_limits<int>::lowest();
        for (VkPhysicalDevice candidate : devices) {
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(candidate, &properties);
            if (VK_API_VERSION_MAJOR(properties.apiVersion) < 1 ||
                (VK_API_VERSION_MAJOR(properties.apiVersion) == 1 && VK_API_VERSION_MINOR(properties.apiVersion) < 3)) {
                continue;
            }
            if (!supports_device_extension(candidate, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
                continue;
            }

            const auto family = find_usable_queue_family(candidate);
            if (!family.has_value()) {
                continue;
            }

            VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
            VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            features2.pNext = &features13;
            vkGetPhysicalDeviceFeatures2(candidate, &features2);
            if (features13.synchronization2 != VK_TRUE) {
                continue;
            }

            int score = 0;
            if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                score += 10'000;
            }
            score += static_cast<int>(properties.limits.maxComputeSharedMemorySize / 1024);

            if (score > bestScore) {
                bestScore = score;
                physicalDevice_ = candidate;
                physicalProperties_ = properties;
                queueFamilyIndex_ = *family;
            }
        }

        if (physicalDevice_ == VK_NULL_HANDLE) {
            fail("No Vulkan 1.3 device with graphics+compute+present, timestamps, synchronization2, and swapchain support was found.");
        }

        u32 queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueFamilyCount, queueFamilies.data());
        timestampValidBits_ = queueFamilies.at(queueFamilyIndex_).timestampValidBits;

        std::cout << std::format(
            "GPU: {} | Vulkan {}.{}.{} | timestamp period {:.3f} ns | {} valid bits\n",
            physicalProperties_.deviceName,
            VK_API_VERSION_MAJOR(physicalProperties_.apiVersion),
            VK_API_VERSION_MINOR(physicalProperties_.apiVersion),
            VK_API_VERSION_PATCH(physicalProperties_.apiVersion),
            physicalProperties_.limits.timestampPeriod,
            timestampValidBits_);
    }

    void create_device() {
        const float priority = 1.0F;
        VkDeviceQueueCreateInfo queueCreateInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queueCreateInfo.queueFamilyIndex = queueFamilyIndex_;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &priority;

        VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        features13.synchronization2 = VK_TRUE;

        const char* extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        VkDeviceCreateInfo createInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        createInfo.pNext = &features13;
        createInfo.queueCreateInfoCount = 1;
        createInfo.pQueueCreateInfos = &queueCreateInfo;
        createInfo.enabledExtensionCount = 1;
        createInfo.ppEnabledExtensionNames = extensions;

        vk_check(vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_), "vkCreateDevice");
        vkGetDeviceQueue(device_, queueFamilyIndex_, 0, &queue_);
    }

    void create_command_resources() {
        VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndex_;
        vk_check(vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_), "vkCreateCommandPool");

        VkCommandBufferAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocateInfo.commandPool = commandPool_;
        allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocateInfo.commandBufferCount = 1;
        vk_check(vkAllocateCommandBuffers(device_, &allocateInfo, &commandBuffer_), "vkAllocateCommandBuffers");
    }

    [[nodiscard]] VkSurfaceFormatKHR choose_surface_format(std::span<const VkSurfaceFormatKHR> formats) const {
        if (formats.empty()) {
            fail("The presentation surface reported no supported formats.");
        }
        if (formats.size() == 1 && formats.front().format == VK_FORMAT_UNDEFINED) {
            return {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
        }
        for (const auto& format : formats) {
            if (format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return format;
            }
        }
        return formats.front();
    }

    void create_swapchain() {
        VkSurfaceCapabilitiesKHR capabilities{};
        vk_check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice_, surface_, &capabilities),
                 "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
        if ((capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) == 0) {
            fail("The presentation surface does not support transfer-destination swapchain images.");
        }

        u32 formatCount = 0;
        vk_check(vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_, surface_, &formatCount, nullptr),
                 "vkGetPhysicalDeviceSurfaceFormatsKHR(count)");
        std::vector<VkSurfaceFormatKHR> formats(formatCount);
        vk_check(vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_, surface_, &formatCount, formats.data()),
                 "vkGetPhysicalDeviceSurfaceFormatsKHR");
        const VkSurfaceFormatKHR format = choose_surface_format(formats);
        VkFormatProperties swapchainFormatProperties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice_, format.format, &swapchainFormatProperties);
        if ((swapchainFormatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) == 0) {
            fail("The selected swapchain format does not support optimal-tiling blit destinations.");
        }

        int pixelWidth = 0;
        int pixelHeight = 0;
        SDL_GetWindowSizeInPixels(window_, &pixelWidth, &pixelHeight);
        VkExtent2D extent{};
        if (capabilities.currentExtent.width != std::numeric_limits<u32>::max()) {
            extent = capabilities.currentExtent;
        } else {
            extent.width = std::clamp(static_cast<u32>(std::max(pixelWidth, 1)),
                                      capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            extent.height = std::clamp(static_cast<u32>(std::max(pixelHeight, 1)),
                                       capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
        }

        u32 imageCount = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount > 0) {
            imageCount = std::min(imageCount, capabilities.maxImageCount);
        }

        VkSwapchainCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        createInfo.surface = surface_;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = format.format;
        createInfo.imageColorSpace = format.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.preTransform = capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        vk_check(vkCreateSwapchainKHR(device_, &createInfo, nullptr, &swapchain_), "vkCreateSwapchainKHR");
        swapchainFormat_ = format.format;
        swapchainExtent_ = extent;

        u32 actualCount = 0;
        vkGetSwapchainImagesKHR(device_, swapchain_, &actualCount, nullptr);
        swapchainImages_.resize(actualCount);
        vkGetSwapchainImagesKHR(device_, swapchain_, &actualCount, swapchainImages_.data());
        swapchainInitialized_.assign(actualCount, false);
    }

    [[nodiscard]] u32 find_memory_type(u32 typeBits, VkMemoryPropertyFlags requiredProperties) const {
        VkPhysicalDeviceMemoryProperties memoryProperties{};
        vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memoryProperties);

        for (u32 index = 0; index < memoryProperties.memoryTypeCount; ++index) {
            const bool supported = (typeBits & (1u << index)) != 0;
            const bool matches = (memoryProperties.memoryTypes[index].propertyFlags & requiredProperties) == requiredProperties;
            if (supported && matches) {
                return index;
            }
        }
        fail("No compatible Vulkan memory type was found.");
    }

    Buffer create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memoryProperties) {
        Buffer buffer{};
        buffer.size = size;

        VkBufferCreateInfo createInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        createInfo.size = size;
        createInfo.usage = usage;
        createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vk_check(vkCreateBuffer(device_, &createInfo, nullptr, &buffer.handle), "vkCreateBuffer");

        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, buffer.handle, &requirements);

        VkMemoryAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocateInfo.allocationSize = requirements.size;
        allocateInfo.memoryTypeIndex = find_memory_type(requirements.memoryTypeBits, memoryProperties);
        vk_check(vkAllocateMemory(device_, &allocateInfo, nullptr, &buffer.memory), "vkAllocateMemory(buffer)");
        vk_check(vkBindBufferMemory(device_, buffer.handle, buffer.memory, 0), "vkBindBufferMemory");

        if ((memoryProperties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
            vk_check(vkMapMemory(device_, buffer.memory, 0, size, 0, &buffer.mapped), "vkMapMemory");
        }
        return buffer;
    }

    void destroy_buffer(Buffer& buffer) noexcept {
        if (buffer.mapped != nullptr && device_ != VK_NULL_HANDLE) {
            vkUnmapMemory(device_, buffer.memory);
        }
        if (buffer.handle != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, buffer.handle, nullptr);
        }
        if (buffer.memory != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, buffer.memory, nullptr);
        }
        buffer = {};
    }

    void immediate_submit(const std::function<void(VkCommandBuffer)>& record) {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkCommandBufferAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocateInfo.commandPool = commandPool_;
        allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocateInfo.commandBufferCount = 1;
        vk_check(vkAllocateCommandBuffers(device_, &allocateInfo, &commandBuffer), "vkAllocateCommandBuffers(immediate)");

        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_check(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer(immediate)");
        record(commandBuffer);
        vk_check(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer(immediate)");

        VkCommandBufferSubmitInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        commandInfo.commandBuffer = commandBuffer;
        VkSubmitInfo2 submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &commandInfo;
        vk_check(vkQueueSubmit2(queue_, 1, &submitInfo, VK_NULL_HANDLE), "vkQueueSubmit2(immediate)");
        vk_check(vkQueueWaitIdle(queue_), "vkQueueWaitIdle(immediate)");
        vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer);
    }

    template <typename T>
    Buffer upload_vector(std::span<const T> values, VkBufferUsageFlags extraUsage = 0) {
        if (values.empty()) {
            fail("Attempted to upload an empty GPU vector.");
        }

        const VkDeviceSize byteSize = values.size_bytes();
        Buffer staging = create_buffer(
            byteSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        std::memcpy(staging.mapped, values.data(), values.size_bytes());

        Buffer destination = create_buffer(
            byteSize,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | extraUsage,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        immediate_submit([&](VkCommandBuffer commandBuffer) {
            VkBufferCopy copy{};
            copy.size = byteSize;
            vkCmdCopyBuffer(commandBuffer, staging.handle, destination.handle, 1, &copy);

            VkBufferMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = destination.handle;
            barrier.offset = 0;
            barrier.size = destination.size;

            VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.bufferMemoryBarrierCount = 1;
            dependency.pBufferMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(commandBuffer, &dependency);
        });
        destroy_buffer(staging);
        return destination;
    }

    void image_barrier(
        VkCommandBuffer commandBuffer,
        VkImage image,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        VkPipelineStageFlags2 sourceStage,
        VkAccessFlags2 sourceAccess,
        VkPipelineStageFlags2 destinationStage,
        VkAccessFlags2 destinationAccess) const {
        VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.srcStageMask = sourceStage;
        barrier.srcAccessMask = sourceAccess;
        barrier.dstStageMask = destinationStage;
        barrier.dstAccessMask = destinationAccess;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.imageMemoryBarrierCount = 1;
        dependency.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependency);
    }

    void create_output_image() {
        VkFormatProperties outputFormatProperties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice_, VK_FORMAT_R8G8B8A8_UNORM, &outputFormatProperties);
        constexpr VkFormatFeatureFlags requiredFeatures =
            VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_BLIT_SRC_BIT;
        if ((outputFormatProperties.optimalTilingFeatures & requiredFeatures) != requiredFeatures) {
            fail("VK_FORMAT_R8G8B8A8_UNORM lacks storage-image or blit-source support on this device.");
        }

        VkImageCreateInfo createInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        createInfo.imageType = VK_IMAGE_TYPE_2D;
        createInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        createInfo.extent = {renderExtent_.width, renderExtent_.height, 1};
        createInfo.mipLevels = 1;
        createInfo.arrayLayers = 1;
        createInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        createInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        createInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vk_check(vkCreateImage(device_, &createInfo, nullptr, &outputImage_.handle), "vkCreateImage(output)");

        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device_, outputImage_.handle, &requirements);
        VkMemoryAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocateInfo.allocationSize = requirements.size;
        allocateInfo.memoryTypeIndex = find_memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vk_check(vkAllocateMemory(device_, &allocateInfo, nullptr, &outputImage_.memory), "vkAllocateMemory(output)");
        vk_check(vkBindImageMemory(device_, outputImage_.handle, outputImage_.memory, 0), "vkBindImageMemory(output)");

        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = outputImage_.handle;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;
        vk_check(vkCreateImageView(device_, &viewInfo, nullptr, &outputImage_.view), "vkCreateImageView(output)");

        immediate_submit([&](VkCommandBuffer commandBuffer) {
            image_barrier(
                commandBuffer,
                outputImage_.handle,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_NONE,
                VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        });
    }

    void destroy_output_image() noexcept {
        if (outputImage_.view != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
            vkDestroyImageView(device_, outputImage_.view, nullptr);
        }
        if (outputImage_.handle != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
            vkDestroyImage(device_, outputImage_.handle, nullptr);
        }
        if (outputImage_.memory != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, outputImage_.memory, nullptr);
        }
        outputImage_ = {};
    }

    void create_scene_buffers() {
        triangleBuffer_ = upload_vector<TriangleGpu>(scene_.triangles);
        bvhNodeBuffer_ = upload_vector<BvhNodeGpu>(scene_.bvhNodes);
        bvhReferenceBuffer_ = upload_vector<u32>(scene_.bvhTriangleReferences);
        brickHeaderBuffer_ = upload_vector<BrickHeaderGpu>(scene_.brickHeaders);
        microBvhNodeBuffer_ = upload_vector<BvhNodeGpu>(scene_.microBvhNodes);
        microReferenceBuffer_ = upload_vector<u32>(scene_.microTriangleReferences);

        metricBuffer_ = create_buffer(
            sizeof(MetricsSnapshot),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        debugBuffer_ = create_buffer(
            sizeof(DebugBufferGpu),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        traversalConfigBuffer_ = create_buffer(
            sizeof(TraversalConfigGpu),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        std::memset(metricBuffer_.mapped, 0, static_cast<std::size_t>(metricBuffer_.size));
        std::memset(debugBuffer_.mapped, 0, static_cast<std::size_t>(debugBuffer_.size));
        std::memset(traversalConfigBuffer_.mapped, 0, static_cast<std::size_t>(traversalConfigBuffer_.size));
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
        std::memcpy(traversalConfigBuffer_.mapped, &config, sizeof(config));
    }

    void create_descriptors() {
        std::array<VkDescriptorSetLayoutBinding, 10> bindings{};
        for (u32 index = 0; index < 8; ++index) {
            bindings[index].binding = index;
            bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[index].descriptorCount = 1;
            bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        bindings[8].binding = 8;
        bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[8].descriptorCount = 1;
        bindings[8].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[9].binding = 9;
        bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[9].descriptorCount = 1;
        bindings[9].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = static_cast<u32>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        vk_check(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorSetLayout_),
                 "vkCreateDescriptorSetLayout");

        std::array<VkDescriptorPoolSize, 2> poolSizes{
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 9},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        };
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = static_cast<u32>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        vk_check(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_), "vkCreateDescriptorPool");

        VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocateInfo.descriptorPool = descriptorPool_;
        allocateInfo.descriptorSetCount = 1;
        allocateInfo.pSetLayouts = &descriptorSetLayout_;
        vk_check(vkAllocateDescriptorSets(device_, &allocateInfo, &descriptorSet_), "vkAllocateDescriptorSets");
        update_descriptors();
    }

    void update_descriptors() {
        const std::array<Buffer*, 9> buffers{
            &triangleBuffer_,
            &bvhNodeBuffer_,
            &bvhReferenceBuffer_,
            &brickHeaderBuffer_,
            &microBvhNodeBuffer_,
            &microReferenceBuffer_,
            &metricBuffer_,
            &debugBuffer_,
            &traversalConfigBuffer_,
        };
        constexpr std::array<u32, 9> bindingIndices{0, 1, 2, 3, 4, 5, 6, 7, 9};

        std::array<VkDescriptorBufferInfo, 9> bufferInfos{};
        std::array<VkWriteDescriptorSet, 10> writes{};
        for (u32 index = 0; index < static_cast<u32>(buffers.size()); ++index) {
            bufferInfos[index].buffer = buffers[index]->handle;
            bufferInfos[index].offset = 0;
            bufferInfos[index].range = buffers[index]->size;

            writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[index].dstSet = descriptorSet_;
            writes[index].dstBinding = bindingIndices[index];
            writes[index].descriptorCount = 1;
            writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[index].pBufferInfo = &bufferInfos[index];
        }

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageView = outputImage_.view;
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        writes[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[9].dstSet = descriptorSet_;
        writes[9].dstBinding = 8;
        writes[9].descriptorCount = 1;
        writes[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[9].pImageInfo = &imageInfo;

        vkUpdateDescriptorSets(device_, static_cast<u32>(writes.size()), writes.data(), 0, nullptr);
    }

    [[nodiscard]] std::vector<u32> read_spirv(const std::filesystem::path& path) const {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream) {
            fail(std::format("Failed to open shader: {}", path.string()));
        }
        const std::streamsize size = stream.tellg();
        if (size <= 0 || (size % 4) != 0) {
            fail(std::format("Invalid SPIR-V file size: {}", path.string()));
        }
        stream.seekg(0, std::ios::beg);
        std::vector<u32> words(static_cast<std::size_t>(size) / sizeof(u32));
        if (!stream.read(reinterpret_cast<char*>(words.data()), size)) {
            fail(std::format("Failed to read shader: {}", path.string()));
        }
        return words;
    }

    [[nodiscard]] VkPipeline create_compute_pipeline(std::string_view shaderName) {
        const std::filesystem::path path = std::filesystem::path{EPOCH_VOXEL_SHADER_DIRECTORY} /
                                           std::format("{}.spv", shaderName);
        const std::vector<u32> code = read_spirv(path);

        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = code.size() * sizeof(u32);
        moduleInfo.pCode = code.data();
        VkShaderModule module = VK_NULL_HANDLE;
        vk_check(vkCreateShaderModule(device_, &moduleInfo, nullptr, &module), "vkCreateShaderModule");

        VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = module;
        stageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = pipelineLayout_;

        VkPipeline pipeline = VK_NULL_HANDLE;
        const VkResult result = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
        vkDestroyShaderModule(device_, module, nullptr);
        vk_check(result, "vkCreateComputePipelines");
        return pipeline;
    }

    void create_pipelines() {
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &descriptorSetLayout_;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;
        vk_check(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &pipelineLayout_), "vkCreatePipelineLayout");

        pipelines_.bvhDebug = create_compute_pipeline("trace_bvh_debug");
        pipelines_.voxelDebug = create_compute_pipeline("trace_voxel_debug");
        pipelines_.bvhBenchmark = create_compute_pipeline("trace_bvh_benchmark");
        pipelines_.voxelBenchmark = create_compute_pipeline("trace_voxel_benchmark");
    }

    void create_query_and_sync_objects() {
        VkQueryPoolCreateInfo queryInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryInfo.queryCount = 2;
        vk_check(vkCreateQueryPool(device_, &queryInfo, nullptr, &timestampQueryPool_), "vkCreateQueryPool");

        VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        vk_check(vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &imageAvailable_), "vkCreateSemaphore(imageAvailable)");
        vk_check(vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &renderFinished_), "vkCreateSemaphore(renderFinished)");

        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vk_check(vkCreateFence(device_, &fenceInfo, nullptr, &frameFence_), "vkCreateFence");
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
        const float aspect = static_cast<float>(renderExtent_.width) /
                             static_cast<float>(std::max(renderExtent_.height, 1u));
        const float tangent = std::tan(60.0F * 0.5F * 3.14159265359F / 180.0F);

        PushConstants constants{};
        constants.cameraPosition = to_vec4(position, 1.0F);
        constants.cameraForward = to_vec4(forward);
        constants.cameraRight = to_vec4(right * (tangent * aspect));
        constants.cameraUp = to_vec4(up * tangent);
        constants.worldMinimum = to_vec4(scene_.worldMinimum);
        constants.cellSize = to_vec4(scene_.cellSize);
        constants.imageMode = {
            renderExtent_.width,
            renderExtent_.height,
            visualizationMode_,
            static_cast<u32>(frameIndex_)};
        const u32 clampedX = std::min(selectedX_, renderExtent_.width - 1);
        const u32 clampedY = std::min(selectedY_, renderExtent_.height - 1);
        constants.gridSelected = {
            scene_.gridX,
            scene_.gridY,
            scene_.gridZ,
            (clampedY << 16u) | (clampedX & 0xffffu)};
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

    void buffer_fill_to_compute_barrier(VkCommandBuffer commandBuffer) const {
        std::array<VkBufferMemoryBarrier2, 2> barriers{};
        const std::array<const Buffer*, 2> buffers{&metricBuffer_, &debugBuffer_};
        for (std::size_t index = 0; index < barriers.size(); ++index) {
            barriers[index].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            barriers[index].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            barriers[index].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barriers[index].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barriers[index].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            barriers[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[index].buffer = buffers[index]->handle;
            barriers[index].offset = 0;
            barriers[index].size = buffers[index]->size;
        }

        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.bufferMemoryBarrierCount = static_cast<u32>(barriers.size());
        dependency.pBufferMemoryBarriers = barriers.data();
        vkCmdPipelineBarrier2(commandBuffer, &dependency);
    }

    void clear_instrumentation_buffers(VkCommandBuffer commandBuffer) const {
        vkCmdFillBuffer(commandBuffer, metricBuffer_.handle, 0, metricBuffer_.size, 0);
        vkCmdFillBuffer(commandBuffer, debugBuffer_.handle, 0, debugBuffer_.size, 0);
        buffer_fill_to_compute_barrier(commandBuffer);
    }

    void record_compute_dispatch(VkCommandBuffer commandBuffer, VkPipeline pipeline) {
        const PushConstants constants = make_push_constants();
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipelineLayout_,
            0,
            1,
            &descriptorSet_,
            0,
            nullptr);
        vkCmdPushConstants(
            commandBuffer,
            pipelineLayout_,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(PushConstants),
            &constants);
        vkCmdDispatch(
            commandBuffer,
            (renderExtent_.width + 7u) / 8u,
            (renderExtent_.height + 7u) / 8u,
            1);
    }

    void compute_to_host_barrier(VkCommandBuffer commandBuffer) const {
        std::array<VkBufferMemoryBarrier2, 2> barriers{};
        const std::array<const Buffer*, 2> buffers{&metricBuffer_, &debugBuffer_};
        for (std::size_t index = 0; index < barriers.size(); ++index) {
            barriers[index].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            barriers[index].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barriers[index].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            barriers[index].dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
            barriers[index].dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
            barriers[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[index].buffer = buffers[index]->handle;
            barriers[index].offset = 0;
            barriers[index].size = buffers[index]->size;
        }

        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.bufferMemoryBarrierCount = static_cast<u32>(barriers.size());
        dependency.pBufferMemoryBarriers = barriers.data();
        vkCmdPipelineBarrier2(commandBuffer, &dependency);
    }

    [[nodiscard]] u64 timestamp_delta(u64 begin, u64 end) const noexcept {
        if (timestampValidBits_ >= 64u) {
            return end - begin;
        }
        const u64 mask = (u64{1} << timestampValidBits_) - 1u;
        return (end - begin) & mask;
    }

    void read_previous_frame() {
        if (!previousFrameValid_) {
            return;
        }

        std::array<u64, 2> timestamps{};
        vk_check(
            vkGetQueryPoolResults(
                device_,
                timestampQueryPool_,
                0,
                static_cast<u32>(timestamps.size()),
                sizeof(timestamps),
                timestamps.data(),
                sizeof(u64),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
            "vkGetQueryPoolResults(frame)");

        const u64 elapsedTicks = timestamp_delta(timestamps[0], timestamps[1]);
        lastGpuMs_ = static_cast<double>(elapsedTicks) *
                     static_cast<double>(physicalProperties_.limits.timestampPeriod) / 1'000'000.0;
        if (previousTraversalMode_ == 0u) lastBvhGpuMs_ = lastGpuMs_;
        else lastVoxelGpuMs_ = lastGpuMs_;
        std::memcpy(&lastMetrics_, metricBuffer_.mapped, sizeof(lastMetrics_));
        previousFrameValid_ = false;
    }

    bool render_frame() {
        vk_check(vkWaitForFences(device_, 1, &frameFence_, VK_TRUE, std::numeric_limits<u64>::max()),
                 "vkWaitForFences(frame)");
        read_previous_frame();

        u32 imageIndex = 0;
        const VkResult acquireResult = vkAcquireNextImageKHR(
            device_,
            swapchain_,
            std::numeric_limits<u64>::max(),
            imageAvailable_,
            VK_NULL_HANDLE,
            &imageIndex);
        if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
            recreate_swapchain();
            return false;
        }
        if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
            vk_check(acquireResult, "vkAcquireNextImageKHR");
        }

        vk_check(vkResetFences(device_, 1, &frameFence_), "vkResetFences(frame)");
        vk_check(vkResetCommandBuffer(commandBuffer_, 0), "vkResetCommandBuffer(frame)");

        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_check(vkBeginCommandBuffer(commandBuffer_, &beginInfo), "vkBeginCommandBuffer(frame)");

        vkCmdResetQueryPool(commandBuffer_, timestampQueryPool_, 0, 2);
        clear_instrumentation_buffers(commandBuffer_);
        vkCmdWriteTimestamp2(commandBuffer_, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, timestampQueryPool_, 0);
        const VkPipeline activePipeline = traversalMode_ == 0 ? pipelines_.bvhDebug : pipelines_.voxelDebug;
        record_compute_dispatch(commandBuffer_, activePipeline);
        vkCmdWriteTimestamp2(commandBuffer_, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, timestampQueryPool_, 1);
        compute_to_host_barrier(commandBuffer_);

        image_barrier(
            commandBuffer_,
            outputImage_.handle,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT);

        image_barrier(
            commandBuffer_,
            swapchainImages_[imageIndex],
            swapchainInitialized_[imageIndex] ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            swapchainInitialized_[imageIndex] ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_NONE,
            VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT);

        const VkClearColorValue clearColor{{0.015F, 0.020F, 0.030F, 1.0F}};
        const VkImageSubresourceRange clearRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1};
        vkCmdClearColorImage(
            commandBuffer_,
            swapchainImages_[imageIndex],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            &clearColor,
            1,
            &clearRange);

        const ViewportRect sceneViewport = scene_viewport_pixels();
        VkImageBlit blit{};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.layerCount = 1;
        blit.srcOffsets[1] = {
            static_cast<int>(renderExtent_.width),
            static_cast<int>(renderExtent_.height),
            1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.layerCount = 1;
        blit.dstOffsets[0] = {sceneViewport.x, sceneViewport.y, 0};
        blit.dstOffsets[1] = {
            sceneViewport.x + sceneViewport.width,
            sceneViewport.y + sceneViewport.height,
            1};
        vkCmdBlitImage(
            commandBuffer_,
            outputImage_.handle,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            swapchainImages_[imageIndex],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &blit,
            VK_FILTER_NEAREST);

        image_barrier(
            commandBuffer_,
            outputImage_.handle,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        image_barrier(
            commandBuffer_,
            swapchainImages_[imageIndex],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_NONE,
            VK_ACCESS_2_NONE);

        vk_check(vkEndCommandBuffer(commandBuffer_), "vkEndCommandBuffer(frame)");

        VkSemaphoreSubmitInfo waitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        waitInfo.semaphore = imageAvailable_;
        waitInfo.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        VkCommandBufferSubmitInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        commandInfo.commandBuffer = commandBuffer_;
        VkSemaphoreSubmitInfo signalInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        signalInfo.semaphore = renderFinished_;
        signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

        VkSubmitInfo2 submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submitInfo.waitSemaphoreInfoCount = 1;
        submitInfo.pWaitSemaphoreInfos = &waitInfo;
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &commandInfo;
        submitInfo.signalSemaphoreInfoCount = 1;
        submitInfo.pSignalSemaphoreInfos = &signalInfo;
        vk_check(vkQueueSubmit2(queue_, 1, &submitInfo, frameFence_), "vkQueueSubmit2(frame)");

        VkPresentInfoKHR presentInfo{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &renderFinished_;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain_;
        presentInfo.pImageIndices = &imageIndex;
        const VkResult presentResult = vkQueuePresentKHR(queue_, &presentInfo);
        swapchainInitialized_[imageIndex] = true;
        previousTraversalMode_ = traversalMode_;
        previousFrameValid_ = true;

        if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR || framebufferResized_) {
            framebufferResized_ = false;
            recreate_swapchain();
        } else if (presentResult != VK_SUCCESS) {
            vk_check(presentResult, "vkQueuePresentKHR");
        }

        return true;
    }

    void recreate_swapchain() {
        int width = 0;
        int height = 0;
        SDL_GetWindowSizeInPixels(window_, &width, &height);
        if (width <= 0 || height <= 0) {
            return;
        }

        vkDeviceWaitIdle(device_);
        destroy_output_image();
        if (swapchain_ != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(device_, swapchain_, nullptr);
            swapchain_ = VK_NULL_HANDLE;
        }
        create_swapchain();
        create_output_image();
        update_descriptors();
        selectedX_ = std::min(selectedX_, renderExtent_.width - 1);
        selectedY_ = std::min(selectedY_, renderExtent_.height - 1);
        previousFrameValid_ = false;
    }

    void update_window_title() {
        const auto now = std::chrono::steady_clock::now();
        if (now - lastTitleUpdate_ < std::chrono::milliseconds{250}) {
            return;
        }
        lastTitleUpdate_ = now;

        const double rays = static_cast<double>(std::max(lastMetrics_.rays, 1u));
        const double triangleTestsPerRay = static_cast<double>(lastMetrics_.triangleTests) / rays;
        const double traversalPerRay = static_cast<double>(
            lastMetrics_.bvhNodeTests + lastMetrics_.brickSteps + lastMetrics_.microBvhNodeTests) / rays;
        const double volumeSamplesPerRay = static_cast<double>(lastMetrics_.volumeSamples) / rays;
        const BrickVariant& variant = scene_.brickVariants[activeBrickVariant_];
        const std::string mode = traversalMode_ == 0
            ? (workloadMode_ == 0 ? "global BVH -> triangles" : "global BVH + separate volume DDA")
            : std::format("brick {}^3 DDA -> micro-BVH{}", variant.brickSize,
                          workloadMode_ == 0 ? " -> triangles" : " + unified volume");
        const std::string raysLabel = rayMode_ == 0 ? "coherent" : "incoherent";
        const std::string workloadLabel = workloadMode_ == 0 ? "surface" : "surface+volume";
        const std::string title = std::format(
            "Vulkan compute | {} | {} | {} | {:.3f} ms | {:.2f} tri/ray | {:.2f} trav/ray | {:.2f} vol/ray | {}x{}",
            mode,
            raysLabel,
            workloadLabel,
            lastGpuMs_,
            triangleTestsPerRay,
            traversalPerRay,
            volumeSamplesPerRay,
            renderExtent_.width,
            renderExtent_.height);
        SDL_SetWindowTitle(window_, title.c_str());
    }

    [[nodiscard]] MetricsSnapshot run_instrumented_dispatch(VkPipeline pipeline) {
        vk_check(vkDeviceWaitIdle(device_), "vkDeviceWaitIdle(instrumented)");

        VkFence fence = VK_NULL_HANDLE;
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        vk_check(vkCreateFence(device_, &fenceInfo, nullptr, &fence), "vkCreateFence(instrumented)");

        vk_check(vkResetCommandBuffer(commandBuffer_, 0), "vkResetCommandBuffer(instrumented)");
        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_check(vkBeginCommandBuffer(commandBuffer_, &beginInfo), "vkBeginCommandBuffer(instrumented)");

        image_barrier(
            commandBuffer_,
            outputImage_.handle,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        clear_instrumentation_buffers(commandBuffer_);
        record_compute_dispatch(commandBuffer_, pipeline);
        compute_to_host_barrier(commandBuffer_);
        vk_check(vkEndCommandBuffer(commandBuffer_), "vkEndCommandBuffer(instrumented)");

        VkCommandBufferSubmitInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        commandInfo.commandBuffer = commandBuffer_;
        VkSubmitInfo2 submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &commandInfo;
        vk_check(vkQueueSubmit2(queue_, 1, &submitInfo, fence), "vkQueueSubmit2(instrumented)");
        vk_check(vkWaitForFences(device_, 1, &fence, VK_TRUE, std::numeric_limits<u64>::max()),
                 "vkWaitForFences(instrumented)");
        vkDestroyFence(device_, fence, nullptr);

        MetricsSnapshot snapshot{};
        std::memcpy(&snapshot, metricBuffer_.mapped, sizeof(snapshot));
        return snapshot;
    }

    [[nodiscard]] double run_benchmark_sample(VkPipeline pipeline) {
        VkFence fence = VK_NULL_HANDLE;
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        vk_check(vkCreateFence(device_, &fenceInfo, nullptr, &fence), "vkCreateFence(benchmark)");

        vk_check(vkResetCommandBuffer(commandBuffer_, 0), "vkResetCommandBuffer(benchmark)");
        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_check(vkBeginCommandBuffer(commandBuffer_, &beginInfo), "vkBeginCommandBuffer(benchmark)");

        image_barrier(
            commandBuffer_,
            outputImage_.handle,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        vkCmdResetQueryPool(commandBuffer_, timestampQueryPool_, 0, 2);
        vkCmdWriteTimestamp2(commandBuffer_, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, timestampQueryPool_, 0);
        record_compute_dispatch(commandBuffer_, pipeline);
        vkCmdWriteTimestamp2(commandBuffer_, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, timestampQueryPool_, 1);
        vk_check(vkEndCommandBuffer(commandBuffer_), "vkEndCommandBuffer(benchmark)");

        VkCommandBufferSubmitInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        commandInfo.commandBuffer = commandBuffer_;
        VkSubmitInfo2 submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &commandInfo;
        vk_check(vkQueueSubmit2(queue_, 1, &submitInfo, fence), "vkQueueSubmit2(benchmark)");
        vk_check(vkWaitForFences(device_, 1, &fence, VK_TRUE, std::numeric_limits<u64>::max()),
                 "vkWaitForFences(benchmark)");
        vkDestroyFence(device_, fence, nullptr);

        std::array<u64, 2> timestamps{};
        vk_check(
            vkGetQueryPoolResults(
                device_,
                timestampQueryPool_,
                0,
                2,
                sizeof(timestamps),
                timestamps.data(),
                sizeof(u64),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
            "vkGetQueryPoolResults(benchmark)");
        return static_cast<double>(timestamp_delta(timestamps[0], timestamps[1])) *
               static_cast<double>(physicalProperties_.limits.timestampPeriod) / 1'000'000.0;
    }

    [[nodiscard]] BenchmarkStats benchmark_pipeline(
        std::string name,
        VkPipeline benchmarkPipeline,
        VkPipeline debugPipeline,
        u32 rayMode,
        u32 workloadMode,
        u32 brickSize,
        double structureBuildMs,
        std::size_t structureBytes) {
        for (u32 index = 0; index < kBenchmarkWarmups; ++index) {
            static_cast<void>(run_benchmark_sample(benchmarkPipeline));
        }

        std::vector<double> samples;
        samples.reserve(kBenchmarkSamples);
        for (u32 index = 0; index < kBenchmarkSamples; ++index) {
            samples.push_back(run_benchmark_sample(benchmarkPipeline));
        }

        std::ranges::sort(samples);
        const auto percentile = [&](double p) {
            const std::size_t index = static_cast<std::size_t>(
                std::clamp(p, 0.0, 1.0) * static_cast<double>(samples.size() - 1));
            return samples[index];
        };

        BenchmarkStats stats{};
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
        stats.metrics = run_instrumented_dispatch(debugPipeline);
        return stats;
    }

    void write_benchmark_csv(const std::vector<BenchmarkStats>& results) const {
        constexpr std::string_view filename{"voxel_ray_benchmark_vulkan.csv"};
        std::ofstream stream(filename.data(), std::ios::trunc);
        if (!stream) {
            std::cerr << "Failed to write " << filename << '\n';
            return;
        }

        stream << "backend,mode,workload,ray_set,brick_size,width,height,triangles,global_bvh_nodes,structure_bytes,structure_build_ms,median_gpu_ms,average_gpu_ms,p10_gpu_ms,p90_gpu_ms,ns_per_ray,rays,global_bvh_node_tests,brick_steps,micro_bvh_node_tests,occupied_bricks,triangle_tests,volume_samples,volume_bricks,hits,max_traversal_steps,traversal_per_ray,triangle_tests_per_ray,volume_samples_per_ray\n";
        for (const BenchmarkStats& stats : results) {
            const double rays = static_cast<double>(std::max(stats.metrics.rays, 1u));
            const double nanosecondsPerRay = stats.medianMs * 1'000'000.0 / rays;
            const double traversalPerRay = static_cast<double>(
                stats.metrics.bvhNodeTests + stats.metrics.brickSteps + stats.metrics.microBvhNodeTests) / rays;
            const double trianglesPerRay = static_cast<double>(stats.metrics.triangleTests) / rays;
            const double volumePerRay = static_cast<double>(stats.metrics.volumeSamples) / rays;

            stream << "vulkan_compute," << stats.name << ','
                   << (stats.workloadMode == 0 ? "surface" : "surface_plus_volume") << ','
                   << (stats.rayMode == 0 ? "coherent" : "incoherent") << ','
                   << stats.brickSize << ','
                   << renderExtent_.width << ','
                   << renderExtent_.height << ','
                   << scene_.triangles.size() << ','
                   << scene_.bvhNodes.size() << ','
                   << stats.structureBytes << ','
                   << stats.structureBuildMs << ','
                   << stats.medianMs << ','
                   << stats.averageMs << ','
                   << stats.p10Ms << ','
                   << stats.p90Ms << ','
                   << nanosecondsPerRay << ','
                   << stats.metrics.rays << ','
                   << stats.metrics.bvhNodeTests << ','
                   << stats.metrics.brickSteps << ','
                   << stats.metrics.microBvhNodeTests << ','
                   << stats.metrics.occupiedBricks << ','
                   << stats.metrics.triangleTests << ','
                   << stats.metrics.volumeSamples << ','
                   << stats.metrics.volumeBricks << ','
                   << stats.metrics.hits << ','
                   << stats.metrics.maximumTraversal << ','
                   << traversalPerRay << ','
                   << trianglesPerRay << ','
                   << volumePerRay << '\n';
        }
    }

    void benchmark_all() {
        vk_check(vkDeviceWaitIdle(device_), "vkDeviceWaitIdle(benchmark all)");
        previousFrameValid_ = false;

        const u32 savedVariant = activeBrickVariant_;
        const u32 savedRayMode = rayMode_;
        const u32 savedWorkloadMode = workloadMode_;
        std::vector<BenchmarkStats> results;
        results.reserve(2u * (1u + scene_.brickVariants.size()) +
                        4u * scene_.brickVariants.size());

        const std::size_t globalBvhBytes = scene_.bvhNodes.size() * sizeof(BvhNodeGpu) +
                                           scene_.bvhTriangleReferences.size() * sizeof(u32);

        std::cout << std::format(
            "\nVulkan benchmark matrix: {} triangles, {}x{}, {} warmups + {} timestamped samples each\n",
            scene_.triangles.size(),
            renderExtent_.width,
            renderExtent_.height,
            kBenchmarkWarmups,
            kBenchmarkSamples);
        std::cout << "Surface mode compares one global BVH against every brick size.\n"
                     "Volume mode compares global BVH + a second volume brick walk against one unified brick walk.\n";

        const auto print_stats = [](const BenchmarkStats& stats, double speedRatio) {
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
            if (speedRatio > 0.0) {
                std::cout << std::format(" | {:6.3f}x vs paired BVH", speedRatio);
            }
            std::cout << '\n';
        };

        for (u32 rayMode = 0; rayMode < 2; ++rayMode) {
            const std::string rayLabel = rayMode == 0 ? "coherent" : "incoherent";
            std::cout << std::format("\nSurface-only / {} rays:\n", rayLabel);

            apply_traversal_config(0u, rayMode, 0u);
            BenchmarkStats bvh = benchmark_pipeline(
                std::format("global_bvh_surface_{}", rayLabel),
                pipelines_.bvhBenchmark,
                pipelines_.bvhDebug,
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
                BenchmarkStats hybrid = benchmark_pipeline(
                    std::format("brick_{}^3_micro_bvh_surface_{}", variant.brickSize, rayLabel),
                    pipelines_.voxelBenchmark,
                    pipelines_.voxelDebug,
                    rayMode,
                    0u,
                    variant.brickSize,
                    variant.buildMs,
                    variant.byte_size());
                print_stats(hybrid, hybrid.medianMs > 0.0 ? bvhMedian / hybrid.medianMs : 0.0);
                results.push_back(std::move(hybrid));
            }

            std::cout << std::format("\nSurface + sparse volume / {} rays:\n", rayLabel);
            for (u32 variantIndex = 0; variantIndex < scene_.brickVariants.size(); ++variantIndex) {
                apply_traversal_config(variantIndex, rayMode, 1u);
                const BrickVariant& variant = scene_.brickVariants[variantIndex];
                const std::size_t volumeIndexBytes = static_cast<std::size_t>(variant.headerCount) * sizeof(BrickHeaderGpu);

                BenchmarkStats bvhVolume = benchmark_pipeline(
                    std::format("global_bvh_plus_volume_{}^3_{}", variant.brickSize, rayLabel),
                    pipelines_.bvhBenchmark,
                    pipelines_.bvhDebug,
                    rayMode,
                    1u,
                    variant.brickSize,
                    scene_.bvhBuildMs + variant.buildMs,
                    globalBvhBytes + volumeIndexBytes);
                print_stats(bvhVolume, 0.0);
                const double pairedMedian = bvhVolume.medianMs;
                results.push_back(std::move(bvhVolume));

                BenchmarkStats unified = benchmark_pipeline(
                    std::format("brick_{}^3_unified_surface_volume_{}", variant.brickSize, rayLabel),
                    pipelines_.voxelBenchmark,
                    pipelines_.voxelDebug,
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
        write_benchmark_csv(results);
        std::cout << "\nWrote voxel_ray_benchmark_vulkan.csv\n\n";
    }

    void dump_debug_traversal() {
        vk_check(vkWaitForFences(device_, 1, &frameFence_, VK_TRUE, std::numeric_limits<u64>::max()),
                 "vkWaitForFences(debug dump)");
        read_previous_frame();

        const auto* debug = static_cast<const DebugBufferGpu*>(debugBuffer_.mapped);
        const u32 count = std::min(debug->count, kMaxDebugEvents);
        std::cout << std::format(
            "\nSelected ray ({}, {}) | mode {} | {} events{}\n",
            selectedX_,
            selectedY_,
            traversalMode_ == 0 ? "BVH" : "voxel hybrid",
            count,
            debug->overflow != 0 ? " (truncated)" : "");

        for (u32 index = 0; index < count; ++index) {
            const DebugEventGpu& event = debug->events[index];
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
                index,
                type,
                event.positionT.w,
                event.positionT.x,
                event.positionT.y,
                event.positionT.z,
                event.data[1],
                event.data[2],
                event.data[3]);
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
                case SDL_EVENT_QUIT:
                    running_ = false;
                    break;
                case SDL_EVENT_WINDOW_RESIZED:
                case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                    if (event.window.windowID == mainWindowId) {
                        framebufferResized_ = true;
                    }
                    break;
                case SDL_EVENT_MOUSE_BUTTON_DOWN:
                    if (event.button.windowID == mainWindowId && event.button.button == SDL_BUTTON_LEFT) {
                        const ViewportRect viewport = scene_viewport_logical();
                        if (event.button.x >= static_cast<float>(viewport.x) &&
                            event.button.x < static_cast<float>(viewport.x + viewport.width) &&
                            event.button.y >= static_cast<float>(viewport.y) &&
                            event.button.y < static_cast<float>(viewport.y + viewport.height)) {
                            selectedX_ = static_cast<u32>(std::clamp(
                                (event.button.x - static_cast<float>(viewport.x)) * static_cast<float>(renderExtent_.width) /
                                    static_cast<float>(std::max(viewport.width, 1)),
                                0.0F,
                                static_cast<float>(renderExtent_.width - 1)));
                            selectedY_ = static_cast<u32>(std::clamp(
                                (event.button.y - static_cast<float>(viewport.y)) * static_cast<float>(renderExtent_.height) /
                                    static_cast<float>(std::max(viewport.height, 1)),
                                0.0F,
                                static_cast<float>(renderExtent_.height - 1)));
                        }
                    }
                    break;
                case SDL_EVENT_KEY_DOWN:
                    if (event.key.windowID != mainWindowId) break;
                    if (event.key.repeat) {
                        break;
                    }
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
                        case SDLK_D: dump_debug_traversal(); break;
                        case SDLK_I:
                            if (inspector_.is_open()) inspector_.close();
                            else if (!inspector_.open(window_)) std::cerr << std::format("Live ray inspector could not open: {}\n", SDL_GetError());
                            else cameraPaused_ = true;
                            break;
                        case SDLK_B: benchmarkRequested_ = true; break;
                        default: break;
                    }
                    break;
                default:
                    break;
            }
        }
    }

    void main_loop() {
        while (running_) {
            handle_events();
            if (!running_) {
                break;
            }

            if (benchmarkRequested_) {
                benchmarkRequested_ = false;
                cameraPaused_ = true;
                benchmark_all();
                vk_check(vkDeviceWaitIdle(device_), "vkDeviceWaitIdle(after interactive benchmark)");
                running_ = false;
                continue;
            }

            const auto now = std::chrono::steady_clock::now();
            const float deltaSeconds = std::chrono::duration<float>(now - lastFrameTime_).count();
            lastFrameTime_ = now;
            if (!cameraPaused_) {
                cameraAngle_ += std::min(deltaSeconds, 0.1F) * 0.18F;
            }

            int width = 0;
            int height = 0;
            SDL_GetWindowSizeInPixels(window_, &width, &height);
            if (width <= 0 || height <= 0) {
                SDL_Delay(16);
                continue;
            }

            if (render_frame()) {
                ++frameIndex_;
                update_window_title();
            }
            if (inspector_.is_open()) {
                const InspectorFrameInput inspectorInput = make_inspector_input();
                inspector_.update(scene_, inspectorInput, deltaSeconds);
                inspector_.render(scene_, inspectorInput);
            }
        }
    }

    void cleanup() noexcept {
        inspector_.close();
        if (device_ != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device_);
        }

        if (pipelines_.bvhDebug != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipelines_.bvhDebug, nullptr);
        if (pipelines_.voxelDebug != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipelines_.voxelDebug, nullptr);
        if (pipelines_.bvhBenchmark != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipelines_.bvhBenchmark, nullptr);
        if (pipelines_.voxelBenchmark != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipelines_.voxelBenchmark, nullptr);
        if (pipelineLayout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        if (descriptorPool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        if (descriptorSetLayout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);

        destroy_buffer(traversalConfigBuffer_);
        destroy_buffer(debugBuffer_);
        destroy_buffer(metricBuffer_);
        destroy_buffer(microReferenceBuffer_);
        destroy_buffer(microBvhNodeBuffer_);
        destroy_buffer(brickHeaderBuffer_);
        destroy_buffer(bvhReferenceBuffer_);
        destroy_buffer(bvhNodeBuffer_);
        destroy_buffer(triangleBuffer_);
        destroy_output_image();

        if (timestampQueryPool_ != VK_NULL_HANDLE) vkDestroyQueryPool(device_, timestampQueryPool_, nullptr);
        if (frameFence_ != VK_NULL_HANDLE) vkDestroyFence(device_, frameFence_, nullptr);
        if (renderFinished_ != VK_NULL_HANDLE) vkDestroySemaphore(device_, renderFinished_, nullptr);
        if (imageAvailable_ != VK_NULL_HANDLE) vkDestroySemaphore(device_, imageAvailable_, nullptr);
        if (commandPool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_, commandPool_, nullptr);
        if (swapchain_ != VK_NULL_HANDLE) vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        if (device_ != VK_NULL_HANDLE) vkDestroyDevice(device_, nullptr);

        if (surface_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE) {
            SDL_Vulkan_DestroySurface(instance_, surface_, nullptr);
        }
        if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
        if (window_ != nullptr) SDL_DestroyWindow(window_);
        SDL_Quit();

        window_ = nullptr;
        device_ = VK_NULL_HANDLE;
        instance_ = VK_NULL_HANDLE;
    }
};

} // namespace epoch::voxel_demo

int main(int argc, char** argv) {
    try {
        const epoch::voxel_demo::DemoOptions options = epoch::voxel_demo::parse_options(argc, argv);
        epoch::voxel_demo::VulkanDemo demo{options};
        return demo.run();
    } catch (const std::exception& exception) {
        std::cerr << "Fatal error: " << exception.what() << '\n';
        return 1;
    }
}
