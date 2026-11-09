// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <boost/container/small_vector.hpp>

#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/back_split_pool.h"
#include "video_core/buffer_cache/geometry_ranges.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"

namespace Vulkan {

// EDEN_NX2_BACK_SPLIT=<workers>: the thread that draws hands two pieces of each draw to a pool
// and does the rest of the draw meanwhile (docs/nx2/back-split.md):
// - geometry: the index buffer and vertex stream ranges, posted when the draw starts and taken
//   by UpdateGraphicsBuffers under the cache locks;
// - image descriptors: posted once the draw's views and samplers are resolved, taken by the
//   first stage's descriptor push, while the buffers are resolved, synchronized and bound.
// EDEN_NX2_BACK_SPLIT_JOBS=geometry or =images keeps only that job (default: both).
// EDEN_NX2_BACK_SPLIT_VERIFY=1 runs the serial path as well, uses its result and logs any
// difference.
class BackSplit {
public:
    static constexpr size_t NUM_STAGES = Tegra::Engines::Maxwell3D::Regs::MaxShaderStage;

    /// EDEN_NX2_BACK_SPLIT's worker count; nullopt when unset or 0.
    [[nodiscard]] static std::optional<u32> Workers();

    explicit BackSplit(u32 workers);
    ~BackSplit();

    BackSplit(const BackSplit&) = delete;
    BackSplit& operator=(const BackSplit&) = delete;

    /// Start of a draw, before anything changes the caches: posts the index and vertex stream
    /// ranges the buffer cache will resolve for it (those dirty now).
    void PostGeometry(Tegra::Engines::Maxwell3D& maxwell3d,
                      const Tegra::MemoryManager& gpu_memory, bool is_indexed);

    /// Under the cache locks, before Configure: joins the geometry job and returns what the
    /// buffer cache takes (UsePreparedGeometry), or nullptr.
    [[nodiscard]] VideoCommon::PreparedGeometry* TakeGeometry(
        Tegra::Engines::Maxwell3D& maxwell3d, const Tegra::MemoryManager& gpu_memory);

    /// After the buffer cache took the prepared geometry: counts and logs verify differences.
    void FinishGeometry(u64 vertex_shader);

    /// Joins the geometry job of a draw that ended before taking it.
    void JoinGeometry();

    /// Whether draws post their image descriptors (EDEN_NX2_BACK_SPLIT_JOBS).
    [[nodiscard]] bool PostsImages() const noexcept {
        return images_on;
    }

    /// Once the draw's views and samplers are resolved: posts its image descriptors. `infos`
    /// has nullptr for the stages the pipeline doesn't have.
    void PostImageDescriptors(TextureCache& texture_cache,
                              const std::array<const Shader::Info*, NUM_STAGES>& infos,
                              const VideoCommon::SamplerId* samplers,
                              const VideoCommon::ImageViewInOut* views);

    /// At the first stage's descriptor push: joins the image descriptor job. True when its
    /// descriptors replace PushImageDescriptors (`rescaling` assigned, written images marked).
    [[nodiscard]] bool TakeImageDescriptors(TextureCache& texture_cache,
                                            RescalingPushConstant& rescaling);

    [[nodiscard]] std::span<const DescriptorUpdateEntry> ImageDescriptors(size_t stage) const;

    /// Verify: compares the descriptors the serial path pushed for `stage` with the job's.
    void VerifyImageDescriptors(u64 pipeline_hash, size_t stage,
                                std::span<const DescriptorUpdateEntry> serial);

    /// Verify: compares the rescaling words once every stage pushed.
    void VerifyRescaling(u64 pipeline_hash, const RescalingPushConstant& serial);

    /// Joins the image descriptor job of a draw that ended before taking it.
    void JoinImageDescriptors();

    /// A fast draw delta draw: no image descriptors to build.
    void NoteDeltaDraw() noexcept {
        ++window.delta_draws;
    }

    /// End of each draw: logs the totals every 10 s with EDEN_NX2_GPU_PROFILE or the verify.
    void Report();

private:
    struct GeometryJob {
        VideoCommon::BackSplit::Job job;
        const Tegra::Engines::Maxwell3D::Regs* regs{};
        const Tegra::Engines::Maxwell3D::Regs::IndexBuffer* index_buffer{};
        const Tegra::MemoryManager* gpu_memory{};
        bool native{};
        bool index{};
        u32 vertex_mask{};
        u64 map_serial{};
        VideoCommon::PreparedGeometry prepared;
    };

    struct ImageJob {
        VideoCommon::BackSplit::Job job;
        TextureCache* texture_cache{};
        std::array<const Shader::Info*, NUM_STAGES> infos{};
        const VideoCommon::SamplerId* samplers{};
        const VideoCommon::ImageViewInOut* views{};
        std::vector<DescriptorUpdateEntry> entries;
        std::array<std::pair<u32, u32>, NUM_STAGES> ranges{};
        std::array<u32, NUM_TEXTURE_AND_IMAGE_SCALING_WORDS> rescaling{};
        boost::container::small_vector<VideoCommon::ImageId, 8> written;
        bool failed{};
        bool taken{};
        bool verify_failed{}; ///< Verify: a difference was logged for this draw already
    };

    /// Counted on the thread that draws, logged and reset by Report.
    struct Window {
        u64 draws{};
        u64 geometry_posted{};
        u64 geometry_none{};
        u64 geometry_used{};
        u64 geometry_map_changed{};
        u64 geometry_back_slots{};
        u64 images_posted{};
        u64 images_used{};
        u64 images_failed{};
        u64 delta_draws{};
        u64 back_jobs{};
        u64 joins_waited{};
        u64 wait_ticks{};
        u64 geometry_checked{};
        u64 geometry_mismatches{};
        u64 images_checked{};
        u64 images_mismatches{};
    };

    static void RunGeometry(void* argument);
    static void RunImages(void* argument);

    void Join(VideoCommon::BackSplit::Job& job);
    void LogMismatch(const std::string& text);

    bool verify{};
    bool measure{};
    bool geometry_on{};
    bool images_on{};
    Window window;
    u64 window_start{};
    u64 window_start_ticks{};
    std::optional<u64> window_cpu_ns;
    VideoCommon::BackSplit::PoolTotals window_pool{};
    u64 total_checked{};
    u64 total_mismatches{};
    u32 logged_mismatches{};
    GeometryJob geometry;
    ImageJob images;
    // Last: the workers stop before the jobs they may run are destroyed.
    VideoCommon::BackSplit::Pool pool;
};

} // namespace Vulkan
