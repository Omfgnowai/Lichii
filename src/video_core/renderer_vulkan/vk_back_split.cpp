// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <string>
#include <string_view>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "common/logging.h"
#include "common/thread_cpu_clock.h"
#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_profile.h"
#include "video_core/renderer_vulkan/vk_back_split.h"

namespace Vulkan {
namespace {

namespace Profile = VideoCommon::Nx2Profile;
using Flags = Tegra::Engines::Maxwell3D::DirtyState::Flags;

/// The vertex streams whose dirty flag is set.
u32 DirtyVertexStreams(const Flags& flags) {
    constexpr u64 streams_mask = VideoCommon::NUM_VERTEX_BUFFERS >= 32
                                     ? 0xffff'ffffULL
                                     : (u64{1} << VideoCommon::NUM_VERTEX_BUFFERS) - 1;
    return static_cast<u32>(((flags >> VideoCommon::Dirty::VertexBuffer0) & Flags{streams_mask})
                                .to_ullong());
}

/// Where a back split worker pushes a stage's image descriptors.
struct ImageDescriptorSink {
    std::vector<DescriptorUpdateEntry>& entries;

    void AddSampledImage(VkImageView image_view, VkSampler sampler) {
        entries.push_back(GuestDescriptorQueue::SampledImageEntry(image_view, sampler));
    }

    void AddImage(VkImageView image_view) {
        entries.push_back(GuestDescriptorQueue::ImageEntry(image_view));
    }
};

bool SameImageEntry(const DescriptorUpdateEntry& lhs, const DescriptorUpdateEntry& rhs) {
    return lhs.image.sampler == rhs.image.sampler && lhs.image.imageView == rhs.image.imageView &&
           lhs.image.imageLayout == rhs.image.imageLayout;
}

std::string FormatImageEntry(const DescriptorUpdateEntry& entry) {
    return fmt::format("view {:#x} sampler {:#x} layout {}",
                       reinterpret_cast<uintptr_t>(entry.image.imageView),
                       reinterpret_cast<uintptr_t>(entry.image.sampler),
                       static_cast<int>(entry.image.imageLayout));
}

std::string FormatRange(const VideoCommon::GeometryRange& range) {
    return range.bound ? fmt::format("{:#x}+{:#x}", range.device_addr, range.size)
                       : std::string{"null"};
}

} // Anonymous namespace

std::optional<u32> BackSplit::Workers() {
    const char* const value = std::getenv("EDEN_NX2_BACK_SPLIT");
    if (value == nullptr) {
        return std::nullopt;
    }
    const unsigned long workers = std::strtoul(value, nullptr, 10);
    if (workers == 0) {
        return std::nullopt;
    }
    return static_cast<u32>((std::min)(workers, 16UL));
}

BackSplit::BackSplit(u32 workers)
    : verify{[] {
          const char* const value = std::getenv("EDEN_NX2_BACK_SPLIT_VERIFY");
          return value != nullptr && value[0] == '1';
      }()},
      measure{Profile::Enabled()}, pool{workers, Profile::Enabled()} {
    const char* const jobs = std::getenv("EDEN_NX2_BACK_SPLIT_JOBS");
    const std::string_view list = jobs != nullptr ? jobs : "geometry,images";
    geometry_on = list.find("geometry") != std::string_view::npos;
    images_on = list.find("images") != std::string_view::npos;
    images.entries.reserve(256);
    LOG_INFO(Render_Vulkan, "Back split: {} workers, jobs:{}{}{}", workers,
             geometry_on ? " geometry" : "", images_on ? " images" : "",
             verify ? ", verify" : "");
}

BackSplit::~BackSplit() = default;

void BackSplit::PostGeometry(Tegra::Engines::Maxwell3D& maxwell3d,
                             const Tegra::MemoryManager& gpu_memory, bool is_indexed) {
    ++window.draws;
    if (!geometry_on) {
        return;
    }
    const Flags& flags = maxwell3d.dirty.flags;
    const auto& draw_state = maxwell3d.draw_manager.draw_state;
    geometry.index = is_indexed && flags[VideoCommon::Dirty::IndexBuffer] &&
                     draw_state.inline_index_draw_indexes.empty();
    geometry.vertex_mask =
        flags[VideoCommon::Dirty::VertexBuffers] ? DirtyVertexStreams(flags) : 0;
    if (!geometry.index && geometry.vertex_mask == 0) {
        ++window.geometry_none;
        return;
    }
    geometry.regs = &maxwell3d.regs;
    geometry.index_buffer = &draw_state.index_buffer;
    geometry.gpu_memory = &gpu_memory;
    geometry.native = maxwell3d.GetNativeMacroEngine().native_turing;
    geometry.map_serial = gpu_memory.MapSerial();
    ++window.geometry_posted;
    pool.Post(geometry.job, &RunGeometry, &geometry);
}

void BackSplit::RunGeometry(void* argument) {
    GeometryJob& job = *static_cast<GeometryJob*>(argument);
    VideoCommon::PreparedGeometry& prepared = job.prepared;
    prepared.Clear();
    if (job.index) {
        prepared.Set(VideoCommon::PreparedGeometry::IndexSlot,
                     VideoCommon::IndexBufferRange(*job.index_buffer, *job.gpu_memory));
    }
    for (u32 mask = job.vertex_mask; mask != 0; mask &= mask - 1) {
        const u32 slot = static_cast<u32>(std::countr_zero(mask));
        if (const auto range =
                VideoCommon::VertexStreamRange(*job.regs, job.native, *job.gpu_memory, slot)) {
            prepared.Set(slot, *range);
        }
    }
}

VideoCommon::PreparedGeometry* BackSplit::TakeGeometry(Tegra::Engines::Maxwell3D& maxwell3d,
                                                       const Tegra::MemoryManager& gpu_memory) {
    if (!geometry.job.Pending()) {
        return nullptr;
    }
    Join(geometry.job);
    // A map made by a guest core since the post may have changed a translation.
    if (gpu_memory.MapSerial() != geometry.map_serial) {
        ++window.geometry_map_changed;
        return nullptr;
    }
    ++window.geometry_used;
    const Flags& flags = maxwell3d.dirty.flags;
    if (flags[VideoCommon::Dirty::VertexBuffers]) {
        const u64 unprepared = DirtyVertexStreams(flags) & ~geometry.prepared.prepared_mask;
        window.geometry_back_slots += static_cast<u64>(std::popcount(unprepared));
    }
    VideoCommon::PreparedGeometry& prepared = geometry.prepared;
    prepared.verify = verify;
    prepared.mismatches = 0;
    if (verify) {
        ++window.geometry_checked;
    }
    return &prepared;
}

void BackSplit::FinishGeometry(u64 vertex_shader) {
    VideoCommon::PreparedGeometry& prepared = geometry.prepared;
    if (!prepared.verify || prepared.mismatches == 0) {
        return;
    }
    ++window.geometry_mismatches;
    const u32 shown = (std::min)(prepared.mismatches, static_cast<u32>(prepared.first_mismatches.size()));
    for (u32 index = 0; index < shown; ++index) {
        const auto& mismatch = prepared.first_mismatches[index];
        const std::string slot = mismatch.slot == VideoCommon::PreparedGeometry::IndexSlot
                                     ? std::string{"index buffer"}
                                     : fmt::format("vertex stream {}", mismatch.slot);
        LogMismatch(fmt::format("vertex shader {:016x} {}: parallel {}, serial {}",
                                vertex_shader, slot, FormatRange(mismatch.parallel),
                                FormatRange(mismatch.serial)));
    }
    prepared.mismatches = 0;
}

void BackSplit::JoinGeometry() {
    if (geometry.job.Pending()) {
        Join(geometry.job);
    }
    geometry.prepared.verify = false;
}

void BackSplit::PostImageDescriptors(TextureCache& texture_cache,
                                     const std::array<const Shader::Info*, NUM_STAGES>& infos,
                                     const VideoCommon::SamplerId* samplers,
                                     const VideoCommon::ImageViewInOut* views) {
    images.texture_cache = &texture_cache;
    images.infos = infos;
    images.samplers = samplers;
    images.views = views;
    images.taken = false;
    images.verify_failed = false;
    ++window.images_posted;
    pool.Post(images.job, &RunImages, &images);
}

void BackSplit::RunImages(void* argument) {
    ImageJob& job = *static_cast<ImageJob*>(argument);
    job.entries.clear();
    job.written.clear();
    job.failed = false;
    try {
        RescalingPushConstant rescaling;
        ImageDescriptorSink sink{job.entries};
        const VideoCommon::SamplerId* samplers = job.samplers;
        const VideoCommon::ImageViewInOut* views = job.views;
        for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
            const u32 first = static_cast<u32>(job.entries.size());
            if (job.infos[stage] != nullptr) {
                PushImageDescriptors(*job.texture_cache, sink, *job.infos[stage], rescaling,
                                     samplers, views, [&job](VideoCommon::ImageId image_id) {
                                         job.written.push_back(image_id);
                                     });
            }
            job.ranges[stage] = {first, static_cast<u32>(job.entries.size()) - first};
        }
        job.rescaling = rescaling.Data();
    } catch (...) {
        // The serial path repeats the push on the thread that draws, which reports the error.
        job.failed = true;
    }
}

bool BackSplit::TakeImageDescriptors(TextureCache& texture_cache,
                                     RescalingPushConstant& rescaling) {
    Join(images.job);
    images.taken = true;
    if (images.failed) {
        ++window.images_failed;
        return false;
    }
    if (verify) {
        return false;
    }
    for (const VideoCommon::ImageId image_id : images.written) {
        texture_cache.MarkModification(image_id);
    }
    rescaling.Assign(images.rescaling);
    ++window.images_used;
    return true;
}

std::span<const DescriptorUpdateEntry> BackSplit::ImageDescriptors(size_t stage) const {
    const auto [first, count] = images.ranges[stage];
    return {images.entries.data() + first, count};
}

void BackSplit::VerifyImageDescriptors(u64 pipeline_hash, size_t stage,
                                       std::span<const DescriptorUpdateEntry> serial) {
    if (!verify || !images.taken || images.failed || images.verify_failed) {
        return;
    }
    const std::span<const DescriptorUpdateEntry> parallel = ImageDescriptors(stage);
    if (parallel.size() != serial.size()) {
        images.verify_failed = true;
        LogMismatch(fmt::format("pipeline {:016x} stage {}: {} image descriptors in parallel, {} "
                                "serial",
                                pipeline_hash, stage, parallel.size(), serial.size()));
        return;
    }
    for (size_t entry = 0; entry < serial.size(); ++entry) {
        if (!SameImageEntry(parallel[entry], serial[entry])) {
            images.verify_failed = true;
            LogMismatch(fmt::format("pipeline {:016x} stage {} image descriptor {} of {}: "
                                    "parallel {}, serial {}",
                                    pipeline_hash, stage, entry, serial.size(),
                                    FormatImageEntry(parallel[entry]),
                                    FormatImageEntry(serial[entry])));
            return;
        }
    }
}

void BackSplit::VerifyRescaling(u64 pipeline_hash, const RescalingPushConstant& serial) {
    if (!verify || !images.taken || images.failed) {
        return;
    }
    ++window.images_checked;
    if (images.rescaling != serial.Data() && !images.verify_failed) {
        images.verify_failed = true;
        LogMismatch(fmt::format("pipeline {:016x}: rescaling words in parallel {:08x}, serial "
                                "{:08x}",
                                pipeline_hash, fmt::join(images.rescaling, " "),
                                fmt::join(serial.Data(), " ")));
    }
    if (images.verify_failed) {
        ++window.images_mismatches;
    }
}

void BackSplit::JoinImageDescriptors() {
    if (images.job.Pending()) {
        Join(images.job);
    }
}

void BackSplit::Join(VideoCommon::BackSplit::Job& job) {
    Profile::Scope scope{Profile::Zone::BackSplitJoin};
    const VideoCommon::BackSplit::JoinResult result = pool.Join(job);
    window.back_jobs += result.ran_here ? 1 : 0;
    if (result.wait_ticks != 0) {
        ++window.joins_waited;
        window.wait_ticks += result.wait_ticks;
    }
}

void BackSplit::LogMismatch(const std::string& text) {
    if (logged_mismatches >= 16) {
        return;
    }
    ++logged_mismatches;
    LOG_ERROR(Render_Vulkan, "Back split verify mismatch {}/16: {}", logged_mismatches, text);
}

void BackSplit::Report() {
    if ((window.draws & 1023) != 0 || (!measure && !verify)) {
        return;
    }
    const u64 now = Profile::Now();
    if (window_start == 0) {
        window_start = now;
        window_start_ticks = Profile::Ticks();
        window_cpu_ns = Common::CurrentThreadCpuTimeNs();
        window_pool = pool.Totals();
        window = {};
        return;
    }
    if (now - window_start < 10'000'000'000ULL) {
        return;
    }
    const u64 now_ticks = Profile::Ticks();
    const double seconds = static_cast<double>(now - window_start) / 1e9;
    const double ticks = static_cast<double>((std::max<u64>)(now_ticks - window_start_ticks, 1));
    const auto rate = [&](u64 count) { return static_cast<double>(count) / seconds; };
    const auto tick_ms = [&](u64 count) { return static_cast<double>(count) * 1000.0 / ticks; };
    const std::optional<u64> cpu_ns = Common::CurrentThreadCpuTimeNs();
    const VideoCommon::BackSplit::PoolTotals totals = pool.Totals();
    if (measure) {
        const double back_cpu_ms =
            cpu_ns && window_cpu_ns
                ? static_cast<double>(*cpu_ns - *window_cpu_ns) / 1e6 / seconds
                : 0.0;
        LOG_INFO(Render_Vulkan,
                 "Back split per second ({} workers): {:.0f} draws | geometry: {:.0f} prepared, "
                 "{:.0f} used, {:.0f} nothing to prepare, {:.0f} discarded (GPU map changed), "
                 "{:.0f} vertex streams resolved on the back | image descriptors: {:.0f} "
                 "prepared, {:.0f} used, {:.0f} failed, {:.0f} delta draws | jobs: {:.0f} by "
                 "workers, {:.0f} by the back | back waited {:.1f} ms in {:.0f} joins | workers: "
                 "{:.1f} ms in jobs, CPU {:.1f} ms | back CPU {:.1f} ms",
                 pool.Workers(), rate(window.draws), rate(window.geometry_posted),
                 rate(window.geometry_used), rate(window.geometry_none),
                 rate(window.geometry_map_changed), rate(window.geometry_back_slots),
                 rate(window.images_posted), rate(window.images_used), rate(window.images_failed),
                 rate(window.delta_draws), rate(totals.worker_jobs - window_pool.worker_jobs),
                 rate(window.back_jobs), tick_ms(window.wait_ticks),
                 rate(window.joins_waited),
                 tick_ms(totals.worker_ticks - window_pool.worker_ticks),
                 static_cast<double>(totals.worker_cpu_ns - window_pool.worker_cpu_ns) / 1e6 /
                     seconds,
                 back_cpu_ms);
    }
    if (verify) {
        const u64 checked = window.geometry_checked + window.images_checked;
        const u64 mismatches = window.geometry_mismatches + window.images_mismatches;
        total_checked += checked;
        total_mismatches += mismatches;
        LOG_INFO(Render_Vulkan,
                 "Back split verify per second: {:.0f} checks ({:.0f} geometry, {:.0f} image "
                 "descriptors), {:.0f} mismatches ({:.0f} geometry, {:.0f} image descriptors) | "
                 "totals: {} checks, {} mismatches",
                 rate(checked), rate(window.geometry_checked), rate(window.images_checked),
                 rate(mismatches), rate(window.geometry_mismatches),
                 rate(window.images_mismatches), total_checked, total_mismatches);
    }
    window = {};
    window_start = now;
    window_start_ticks = now_ticks;
    window_cpu_ns = cpu_ns;
    window_pool = totals;
}

} // namespace Vulkan
