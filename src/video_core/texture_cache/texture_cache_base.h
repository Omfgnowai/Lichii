// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <type_traits>
// TODO: find out which don't require stable iters
#include <unordered_map>
#include "video_core/page_filter.h"
#include "common/container/unordered_map.h"
#include "common/container/unordered_set.h"
#include <vector>
#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>
#include <queue>

#include "common/common_types.h"
#include "common/spin_recursive_mutex.h"
#include "common/hash.h"
#include "common/literals.h"
#include "common/lru_cache.h"
#include <ranges>
#include "common/scratch_buffer.h"
#include "common/slot_vector.h"
#include "video_core/texture_cache/pending_download.h"
#include "common/thread_worker.h"
#include "video_core/compatible_formats.h"
#include "video_core/control/channel_state_cache.h"
#include "video_core/delayed_destruction_ring.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/game_dlss_status.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/descriptor_table.h"
#include "video_core/texture_cache/image_base.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_view_base.h"
#include "video_core/texture_cache/render_targets.h"
#include "video_core/texture_cache/sampled_views.h"
#include "video_core/texture_cache/stale_pages.h"
#include "video_core/texture_cache/types.h"
#include "video_core/textures/texture.h"
#include "video_core/textures/unswizzle_worker.h"

namespace Tegra {
namespace Control {
struct ChannelState;
}
} // namespace Tegra

namespace VideoCommon {

using Tegra::Texture::TICEntry;
using Tegra::Texture::TSCEntry;
using VideoCore::Surface::PixelFormat;
using namespace Common::Literals;

struct ImageViewInOut {
    u32 index{};
    bool blacklist{};
    ImageViewId id{};
};

/// A texture handle word a draw read, and where.
struct TextureHandleRead {
    GPUVAddr address{};
    u32 raw{};
};

/// What a draw's sampled textures were resolved from, kept by the pipeline that drew them. The
/// next draw with the same pipeline resolves its textures from it (ResolveMemoizedTextures) while
/// every byte they came from is the same: the handle words, the texture and sampler descriptors
/// and the tables, mappings and views behind them.
struct SampledTextureMemo {
    struct Entry {
        GPUVAddr handle_address{};
        const u8* handle_host{}; ///< The handle word in host memory
        const u8* tic_host{};
        const u8* tsc_host{};
        TICEntry tic{};
        TSCEntry tsc{};
        u32 raw{};        ///< The handle word
        u32 view_index{}; ///< The texture header index the handle gives
        ImageViewId view{};
        SamplerId sampler{};
    };
    std::vector<Entry> entries;
    const void* memory{};      ///< The address space the pointers belong to
    const void* image_table{}; ///< The channel's table the views were read from
    u64 host_serial{};         ///< MemoryManager::HostPointerSerial
    u64 image_changes{};       ///< DescriptorTable::changes of both tables
    u64 sampler_changes{};
    bool via_header_index{}; ///< The sampler binding the handles were split with
    bool valid{};
    u32 misses{}; ///< Draws in a row that did not find the memo good
};

struct AsyncDecodeContext {
    ImageId image_id;
    Common::ScratchBuffer<u8> decoded_data;
    boost::container::small_vector<BufferImageCopy, 16> copies;
    std::mutex mutex;
    std::atomic_bool complete;
    bool cancelled{}; ///< The image was deleted meanwhile: drop the result (GPU thread only).
};

using TextureCacheGPUMap = ::Common::unordered_map<u64, std::vector<ImageId>, Common::IdentityHash<u64>>;

class TextureCacheChannelInfo : public ChannelInfo {
public:
    TextureCacheChannelInfo() = delete;
    TextureCacheChannelInfo(Tegra::Control::ChannelState& state) noexcept;
    TextureCacheChannelInfo(const TextureCacheChannelInfo& state) = delete;
    TextureCacheChannelInfo& operator=(const TextureCacheChannelInfo&) = delete;

    DescriptorTable<TICEntry, ImageViewId> graphics_image_table;
    DescriptorTable<TSCEntry, SamplerId> graphics_sampler_table;
    DescriptorTable<TICEntry, ImageViewId> compute_image_table;
    DescriptorTable<TSCEntry, SamplerId> compute_sampler_table;

    // TODO: still relies on bad iterators :(
    std::unordered_map<TICEntry, ImageViewId> image_views;
    std::unordered_map<TSCEntry, SamplerId> samplers;

    TextureCacheGPUMap* gpu_page_table = nullptr;
    TextureCacheGPUMap* sparse_page_table = nullptr;
};

template <class P>
class TextureCache : public VideoCommon::ChannelSetupCaches<TextureCacheChannelInfo> {
    /// Address shift for caching images into a hash table
    static constexpr u64 YUZU_PAGEBITS = 20;

    /// Enables debugging features to the texture cache
    static constexpr bool ENABLE_VALIDATION = P::ENABLE_VALIDATION;
    /// Implement blits as copies between framebuffers
    static constexpr bool FRAMEBUFFER_BLITS = P::FRAMEBUFFER_BLITS;
    /// True when some copies have to be emulated
    static constexpr bool HAS_EMULATED_COPIES = P::HAS_EMULATED_COPIES;
    /// True when the API can provide info about the memory of the device.
    static constexpr bool HAS_DEVICE_MEMORY_INFO = P::HAS_DEVICE_MEMORY_INFO;
    /// True when the API can do asynchronous texture downloads.
    static constexpr bool IMPLEMENTS_ASYNC_DOWNLOADS = P::IMPLEMENTS_ASYNC_DOWNLOADS;

    static constexpr size_t UNSET_CHANNEL{(std::numeric_limits<size_t>::max)()};

#ifdef YUZU_LEGACY
    static constexpr s64 TARGET_THRESHOLD = 3_GiB;
#else
    static constexpr s64 TARGET_THRESHOLD = 4_GiB;
#endif

    // Periodic statistics (TickFrame).
    u64 gc_deletes{};
    u64 gc_passes[4]{};      ///< Collector passes: low priority, high priority, aggressive, out of memory
    u64 gc_deferred{};       ///< Low-priority passes skipped on frames that made images (TC_GC_DEFER)
    u64 gc_budget_stops{};   ///< Low-priority passes stopped by EDEN_NX2_TC_GC_BUDGET_US
    u32 frame_images_created{}; ///< Images made since the last TickFrame
    u32 gc_deferred_in_row{};
    u64 gc_costly_deletes{};
    u64 gc_download_deletes{}; ///< Evictions that downloaded the image first (a full GPU wait)
    u64 async_decodes_queued{};
    u64 images_created{};
    u64 images_uploaded{};   ///< Image contents uploaded from guest memory (last 600 frames)
    u64 uploaded_bytes{};    ///< Guest bytes those uploads read
    u64 largest_upload{};    ///< Largest of them
    u64 layer_uploads{};     ///< Of them, uploads of only the written layers
    bool native_thresholds_applied{};
    u64 native_critical_memory{};
    u64 native_expected_memory{};
    u64 budget_check_frames{};
    u64 logged_critical_memory{}; ///< The critical threshold of the last thresholds line logged
    u32 seen_pressure_events{};   ///< Allocator pressure events already answered
    u32 seen_failed_images{};     ///< Images the allocator could not make, already collected for
    u64 pressure_margin{};        ///< Extra room under the budget after images did not fit
    u64 pressure_quiet_frames{};  ///< Frames since the last such image
    u64 slack_margin{};           ///< Extra room for allocator block space the driver counts as used
    u64 astc_opaque_images{};
    u64 astc_alpha_images{};
    Common::ScratchBuffer<u8> astc_scan_buffer;
    u64 images_deleted{};
    u64 scale_epoch{};
    std::map<u64, std::pair<u32, GPUVAddr>> created_shapes;
    bool debug_stats{}; ///< Collect what only the debug statistics lines use
    /// EDEN_NX2_MEM_TRACE: when the last memory trace lines were logged.
    std::chrono::steady_clock::time_point mem_trace_time{};

    static constexpr s64 DEFAULT_EXPECTED_MEMORY = 1_GiB + 125_MiB;
    static constexpr s64 DEFAULT_CRITICAL_MEMORY = 1_GiB + 625_MiB;
    /// The most device memory the collector lets the cache fill before it collects in earnest on
    /// cards up to 8 GiB; larger ones 5/8 of the driver's budget (ApplyThresholdOverrides).
    static constexpr u64 MaxCriticalMemory = 5_GiB;
    static constexpr size_t GC_EMERGENCY_COUNTS = 2;

    using Runtime = typename P::Runtime;
    using Image = typename P::Image;
    using ImageAlloc = typename P::ImageAlloc;
    using ImageView = typename P::ImageView;
    using Sampler = typename P::Sampler;
    using Framebuffer = typename P::Framebuffer;
    using AsyncBuffer = typename P::AsyncBuffer;
    using BufferType = typename P::BufferType;

    static constexpr u32 LayerUploadMinSize = 64U << 20;
    static constexpr size_t LargeScratchCapacity = 64U << 20;

    /// Splits large 2D image uploads into smaller chunks, avoiding multi-hundred MiB staging buffers.
    bool UploadInChunks(Image& image);

    /// Large layered images whose refresh can decode only the layers written since the last one.
    [[nodiscard]] bool UploadsByLayer(const ImageBase& image) const noexcept;

    /// Records the layers a write of [cpu_addr, +size) touches.
    void MarkLayersWritten(Image& image, DAddr cpu_addr, size_t size);

    /// SparseTiles images: notes a written range, uploaded with the next refresh.
    void MarkRangeWritten(Image& image, DAddr cpu_addr, size_t size);

    /// SparseTiles images: binds and uploads the parts written since the last upload.
    void RefreshSparseTiles(Image& image);

    /// Image::BindTiles, counting what it binds in total_used_memory.
    bool BindSparseTiles(Image& image, std::span<const LevelRect> rects);

    /// Image::BackTilesWhole, counting what it binds in total_used_memory.
    bool BackSparseTilesWhole(Image& image);

    /// Uploads the image's dirty layers. False when it needs a full upload instead.
    bool UploadWrittenLayers(Image& image);

    /// Gives the written layers of a SparseLayers image memory and uploads them.
    void RefreshSparseLayers(Image& image);

    /// Image::BindLayers, with the memory it binds counted in total_used_memory.
    std::vector<u32> BindSparseLayers(Image& image, std::span<const u32> layers);

    /// Uploads `layers` of the image from guest memory.
    void UploadLayers(Image& image, std::span<const u32> layers);

    struct PendingUnswizzle {
        ImageId image_id;
        VideoCommon::ImageInfo info;
        size_t current_offset = 0;
        size_t total_size = 0;
        AsyncBuffer staging_buffer;
        size_t last_submitted_offset = 0;
        size_t bytes_per_slice;
        bool initialized = false;
    };

    struct BlitImages {
        ImageId dst_id;
        ImageId src_id;
        PixelFormat dst_format;
        PixelFormat src_format;
    };

public:
    /// Device pages this cache ever registered objects on (InnerInvalidation skips the rest).
    VideoCommon::PageFilter page_filter;
    /// Device pages that ever held a GPU-modified image: a guest CPU read elsewhere needs no
    /// flush from this cache, so GetFlushArea can skip the cache's lock (the GPU back thread
    /// holds it through image creation and garbage collection; guest reads waited on it).
    VideoCommon::PageFilter gpu_modified_filter;

    explicit TextureCache(Runtime&, Tegra::MaxwellDeviceMemoryManager&);

    /// Notify the cache that a new frame has been queued
    void TickFrame();

    /// Return a constant reference to the given image view id
    [[nodiscard]] const ImageView& GetImageView(ImageViewId id) const noexcept;

    /// Return a reference to the given image view id
    [[nodiscard]] ImageView& GetImageView(ImageViewId id) noexcept;

    /// Get the imageview from the graphics descriptor table in the specified index
    [[nodiscard]] ImageView& GetImageView(u32 index) noexcept;

    // Publish the CPU-readable collision image. Caller holds the cache mutex.
    bool PublishComputeCollisionImage(u32 index);

    /// Mark an image as modified from the GPU
    void MarkModification(ImageId id) noexcept;

    /// Fill image_view_ids with the graphics/compute images in indices
    void FillImageViews(std::span<ImageViewInOut> views, bool compute, bool blacklist = true);

    /// Fills views and samplers of a draw from the pipeline's memo when nothing they were
    /// resolved from has changed since (true), prepares the images as FillImageViews does.
    /// handle_addresses are where the draw reads its texture handles from, in binding order.
    [[nodiscard]] bool ResolveMemoizedTextures(SampledTextureMemo& memo,
                                               std::span<const GPUVAddr> handle_addresses,
                                               std::span<ImageViewInOut> views,
                                               std::span<SamplerId> samplers);

    /// Keeps what FillImageViews and GetSamplerId resolved the draw's textures from.
    void MemoizeTextures(SampledTextureMemo& memo, std::span<const TextureHandleRead> reads,
                         std::span<const ImageViewInOut> views, std::span<const SamplerId> samplers);

    /// Handle feedback loops during draws.
    void CheckFeedbackLoop(std::span<const ImageViewInOut> views);

    [[nodiscard]] bool SamplesRenderTarget(std::span<const ImageViewInOut> views);

    /// Whether sampling the views now needs PrepareImageView: an image with an upload pending or
    /// a newer alias to copy in (fast draw delta draws keep their views and prepare none).
    [[nodiscard]] bool SampledViewsChange(std::span<const ImageViewInOut> views) const noexcept {
        return VideoCommon::SampledViewsChange(views, slot_image_views, slot_images);
    }

    /// EDEN_NX2_TEXTURE_MEMO_VERIFY: PrepareImage adds the images it is asked to prepare to `log`
    /// (nullptr stops it) and, with `dry`, does nothing else.
    void LogPrepares(std::vector<ImageId>* log, bool dry) noexcept {
        prepare_log = log;
        prepare_log_dry = dry;
    }

    /// Get the sampler from the graphics/compute descriptor table in the specified index
    Sampler* GetSampler(u32 index, bool compute);

    /// Get the sampler id from the graphics/compute descriptor table in the specified index
    SamplerId GetSamplerId(u32 index, bool compute);

    /// Return a constant reference to the given sampler id
    [[nodiscard]] const Sampler& GetSampler(SamplerId id) const noexcept;

    /// Return a reference to the given sampler id
    [[nodiscard]] Sampler& GetSampler(SamplerId id) noexcept;

    /// Refresh the state for graphics/compute image view and sampler descriptors
    void SynchronizeDescriptors(bool compute);

    /// Updates the Render Targets if they can be rescaled
    /// @retval True if the Render Targets have been rescaled.
    bool RescaleRenderTargets();

    /// Update bound render targets and upload memory if necessary
    /// @param is_clear True when the render targets are being used for clears
    void UpdateRenderTargets(bool is_clear);

    /// Find a framebuffer with the currently bound render targets
    /// UpdateRenderTargets should be called before this
    Framebuffer* GetFramebuffer();

    [[nodiscard]] const RenderTargets& GetRenderTargets() const noexcept {
        return render_targets;
    }

    /// Mark images in a range as modified from the CPU
    void WriteMemory(DAddr cpu_addr, size_t size);

    /// Download contents of host images to guest memory in a region
    void DownloadMemory(DAddr cpu_addr, size_t size);

    std::optional<VideoCore::RasterizerDownloadArea> GetFlushArea(DAddr cpu_addr, u64 size);

    /// Remove images in a region
    void UnmapMemory(DAddr cpu_addr, size_t size);

    /// Remove images in a region
    void UnmapGPUMemory(size_t as_id, GPUVAddr gpu_addr, size_t size);

    /// Blit an image with the given parameters
    bool BlitImage(const Tegra::Engines::Fermi2D::Surface& dst,
                   const Tegra::Engines::Fermi2D::Surface& src,
                   const Tegra::Engines::Fermi2D::Config& copy);

    /// Stand-in for a DLSS evaluation: temporal upscale of src into dst (see the runtime's
    /// TemporalUpscale). False when unsupported; the caller retains its native dispatch.
    bool TemporalUpscale(const Tegra::Engines::Fermi2D::Surface& dst,
                         const Tegra::Engines::Fermi2D::Surface& src,
                         const Tegra::Engines::Fermi2D::Surface& motion_vectors,
                         const Tegra::Texture::TICEntry* depth, std::array<f32, 2> jitter,
                         std::array<f32, 2> motion_scale, bool use_history);

    /// Health check of a game's DLSS network output (dst) against its input (src); see the
    /// runtime's ProbeGameDlssOutput. Returns an earlier check's result when one is ready.
    std::optional<VideoCore::GameDlssProbe> ProbeGameDlssOutput(
        const Tegra::Engines::Fermi2D::Surface& dst, const Tegra::Engines::Fermi2D::Surface& src);

    /// Try to find a cached image view in the given CPU address
    [[nodiscard]] std::pair<ImageView*, bool> TryFindFramebufferImageView(
        const Tegra::FramebufferConfig& config, DAddr cpu_addr);

    /// Return true when there are uncommitted images to be downloaded
    [[nodiscard]] bool HasUncommittedFlushes() const noexcept;

    /// Whether CommitAsyncFlushes would commit anything (downloads or their buffers).
    [[nodiscard]] bool HasAsyncFlushWork() const noexcept;

    /// Return true when the caller should wait for async downloads
    [[nodiscard]] bool ShouldWaitAsyncFlushes() const noexcept;

    /// Commit asynchronous downloads
    void CommitAsyncFlushes();

    /// Pop asynchronous downloads
    void PopAsyncFlushes();

    [[nodiscard]] ImageId DmaImageId(const Tegra::DMA::ImageOperand& operand, bool is_upload);

    [[nodiscard]] std::pair<Image*, BufferImageCopy> DmaBufferImageCopy(
        const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::BufferOperand& buffer_operand,
        const Tegra::DMA::ImageOperand& image_operand, ImageId image_id, bool modifies_image);

    void DownloadImageIntoBuffer(Image* image, BufferType buffer, size_t buffer_offset,
                                 std::span<const VideoCommon::BufferImageCopy> copies,
                                 GPUVAddr address = 0, size_t size = 0);

    /// Return true when a CPU region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(DAddr addr, size_t size);

    [[nodiscard]] bool IsRescaling() const noexcept;

    [[nodiscard]] bool IsRescaling(const ImageViewBase& image_view) const noexcept;

    /// Counts InvalidateScale calls: an image changed its scale and lost its views.
    [[nodiscard]] u64 ScaleEpoch() const noexcept {
        return scale_epoch;
    }

    /// Create channel state.
    void CreateChannel(Tegra::Control::ChannelState& channel) final override;

    /// Prepare an image to be used
    void PrepareImage(ImageId image_id, bool is_modification, bool invalidate);

    Common::SpinRecursiveMutex mutex;

private:
    /// Iterate over all page indices in a range
    template <typename Func>
    static void ForEachCPUPage(DAddr addr, size_t size, Func&& func) {
        static constexpr bool RETURNS_BOOL = std::is_same_v<std::invoke_result<Func, u64>, bool>;
        const u64 page_end = (addr + size - 1) >> YUZU_PAGEBITS;
        for (u64 page = addr >> YUZU_PAGEBITS; page <= page_end; ++page) {
            if constexpr (RETURNS_BOOL) {
                if (func(page)) {
                    break;
                }
            } else {
                func(page);
            }
        }
    }

    template <typename Func>
    static void ForEachGPUPage(GPUVAddr addr, size_t size, Func&& func) {
        static constexpr bool RETURNS_BOOL = std::is_same_v<std::invoke_result<Func, u64>, bool>;
        const u64 page_end = (addr + size - 1) >> YUZU_PAGEBITS;
        for (u64 page = addr >> YUZU_PAGEBITS; page <= page_end; ++page) {
            if constexpr (RETURNS_BOOL) {
                if (func(page)) {
                    break;
                }
            } else {
                func(page);
            }
        }
    }

    void OnGPUASRegister(size_t map_id) final override;

    /// Runs the Garbage Collector.
    void RunGarbageCollector(bool out_of_memory);

    /// Find or create an image view in the guest descriptor table
    ImageViewId VisitImageView(u32 index, bool compute);

    /// Find or create a framebuffer with the given render target parameters
    FramebufferId GetFramebufferId(const RenderTargets& key);

    /// Refresh the contents (pixel data) of an image
    [[nodiscard]] bool IsDownloadable(const ImageBase& image) const noexcept;

    void RefreshContents(Image& image, ImageId image_id);

    /// Upload data from guest to an image
    template <typename StagingBuffer>
    void UploadImageContents(Image& image, StagingBuffer& staging_buffer);

    /// Create a new image view from a guest descriptor
    [[nodiscard]] ImageViewId CreateImageView(const TICEntry& config);

    /// Find or create an image from the given parameters
    [[nodiscard]] ImageId FindOrInsertImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                            RelaxedOptions options = RelaxedOptions{});

    /// Find an image from the given parameters
    [[nodiscard]] ImageId FindImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                    RelaxedOptions options);

    /// Create an image from the given parameters
    [[nodiscard]] ImageId InsertImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                      RelaxedOptions options);

    /// Create a new image and join perfectly matching existing images
    /// Remove joined images from the cache
    [[nodiscard]] ImageId JoinImages(const ImageInfo& info, GPUVAddr gpu_addr, DAddr cpu_addr);

    [[nodiscard]] ImageId FindDMAImage(const ImageInfo& info, GPUVAddr gpu_addr);

    /// Return a blit image pair from the given guest blit parameters
    [[nodiscard]] std::optional<BlitImages> GetBlitImages(
        const Tegra::Engines::Fermi2D::Surface& dst, const Tegra::Engines::Fermi2D::Surface& src,
        const Tegra::Engines::Fermi2D::Config& copy);

    /// Find or create a sampler from a guest descriptor sampler
    [[nodiscard]] SamplerId FindSampler(const TSCEntry& config, bool compute);

    /// Find or create an image view for the given color buffer index
    [[nodiscard]] ImageViewId FindColorBuffer(size_t index);

    /// Find or create an image view for the depth buffer
    [[nodiscard]] ImageViewId FindDepthBuffer();

    /// Find or create a view for a render target with the given image parameters
    [[nodiscard]] ImageViewId FindRenderTargetView(const ImageInfo& info, GPUVAddr gpu_addr);

    /// Iterates over all the images in a region calling func
    template <typename Func>
    void ForEachImageInRegion(DAddr cpu_addr, size_t size, Func&& func);

    template <typename Func>
    void ForEachImageInRegionGPU(size_t as_id, GPUVAddr gpu_addr, size_t size, Func&& func);

    template <typename Func>
    void ForEachSparseImageInRegion(size_t as_id, GPUVAddr gpu_addr, size_t size, Func&& func);

    /// Iterates over all the images in a region calling func
    template <typename Func>
    void ForEachSparseSegment(ImageBase& image, Func&& func);

    /// Find or create an image view in the given image with the passed parameters
    [[nodiscard]] ImageViewId FindOrEmplaceImageView(ImageId image_id, const ImageViewInfo& info);

    /// Register image in the page table
    void RegisterImage(ImageId image);

    /// Unregister image from the page table
    void UnregisterImage(ImageId image);

    /// Logs where UnregisterImage's image was registered when one of its pages is missing.
    void ReportMissingImagePage(ImageId image_id, const Image& image, u64 page);

    /// Track CPU reads and writes for image
    void TrackImage(ImageBase& image, ImageId image_id);

    /// Stop tracking CPU reads and writes for image
    void UntrackImage(ImageBase& image, ImageId image_id);

    /// Delete image from the cache
    void DeleteImage(ImageId image, bool immediate_delete = false);

    /// Remove image views references from the cache
    void RemoveImageViewReferences(std::span<const ImageViewId> removed_views);

    /// Per image view, the (channel, descriptor) entries of the channels' image_views maps that
    /// point to it. Deleting an image erased them by scanning every map (title 1: ~5,000 entries,
    /// 50-130 us for each deleted image, dozens a frame while the collector runs).
    std::unordered_map<ImageViewId, boost::container::small_vector<std::pair<size_t, TICEntry>, 2>>
        image_view_descriptors;

    /// Remove framebuffers using the given image views from the cache
    void RemoveFramebuffers(std::span<const ImageViewId> removed_views);

    /// Render target relocation (EDEN_NX2_EXP43, port of Fritchii EXP43 2026-11-08/09). Per
    /// shape (format, size, samples, levels, layers) and per address, sticky for the session: a
    /// shape or address whose GPU-written contents were read in a later frame (sampled, copied
    /// from, rendered to without a clear) or read by the CPU never relocates or retires early,
    /// since its predecessor's contents may still be wanted.
    enum RelocationGuard : u8 {
        GuardSampled = 1, ///< Sampled or copied from in a frame after the GPU wrote it
        GuardLoaded = 2,  ///< Rendered to without a clear in a frame after the GPU wrote it
        GuardCpuRead = 4, ///< Read by the CPU (GetFlushArea, DownloadMemory)
    };
    struct RelocationShape {
        u8 guard{};
        bool candidate_logged{};
        bool relocated_logged{};
        bool lineage_logged{};
    };
    /// EDEN_NX2_RELOC_LINEAGE=1: a cross-frame read of an image that did not drift (a target the
    /// game keeps at one address, like the main depth buffer) guards only its address, not its
    /// whole shape: one such buffer of a shape kept every drifting per-frame target of that shape
    /// from relocating (Kokiri Forest: 2560x1440 and 640x360 depth, 2560x1440 format 21).
    [[nodiscard]] static bool LineageGuard() {
        static const bool enabled = [] {
            const char* const value = std::getenv("EDEN_NX2_RELOC_LINEAGE");
            return value != nullptr && value[0] == '1';
        }();
        return enabled;
    }
    u64 relocation_shape_trips_skipped{}; ///< Shape trips the lineage guard kept to the address
    /// EDEN_NX2_RT_EARLY_FREE: 1 deletes the dead render targets a new cleared target covers
    /// whole (2: any it overlaps) when it is made, not when the collector gets to them, so the
    /// image recycler holds their host images for the next target of their shape.
    [[nodiscard]] static int EarlyFreeMode() {
        static const int mode = [] {
            const char* const value = std::getenv("EDEN_NX2_RT_EARLY_FREE");
            return value != nullptr ? std::atoi(value) : 0;
        }();
        return mode;
    }
    u64 early_freed{};       ///< Since the last relocation report
    u64 early_free_bytes{};  ///< Guest bytes of those
    u64 early_free_skipped{}; ///< Covered overlaps kept (used lately, bound, aliased, ...)
    std::vector<ImageId> early_free_ids;
    bool pending_churn_born{};
    ::Common::unordered_map<u64, RelocationShape> relocation_shapes;
    /// The guard by address too: OoT draws every dialogue box's text into a render target at one
    /// address, sized to the box, so per shape each new box size relocated once before its guard
    /// tripped and showed other images' contents (Fritchii b3396c01f).
    ::Common::unordered_map<GPUVAddr, u8> relocation_guarded_addresses;
    /// Since the last report (TickFrame, every 600 frames once a candidate was seen).
    u64 relocation_candidates{};
    u64 relocation_guarded{};
    u64 relocation_relocated{};
    u64 relocation_retired{};
    u64 relocation_crowded{};
    bool relocation_seen{};

    [[nodiscard]] static bool RelocationShaped(const ImageInfo& info) noexcept {
        return info.type == ImageType::e2D && info.resources.levels == 1 &&
               info.resources.layers == 1 && info.num_samples == 1;
    }
    [[nodiscard]] static u64 RelocationShapeKey(const ImageInfo& info) noexcept {
        return (u64{static_cast<u32>(info.format) & 0xFF} << 56) |
               (u64{info.num_samples & 0xF} << 52) |
               (u64{static_cast<u32>(info.resources.levels) & 0xF} << 48) |
               (u64{static_cast<u32>(info.resources.layers) & 0xFFF} << 36) |
               (u64{info.size.width & 0x3FFFF} << 18) | (info.size.height & 0x3FFFF);
    }
    /// The predecessor a new cleared render target could take the place of: among JoinImages'
    /// bad overlaps, the image of the same single-level, single-layer 2D shape and layout the
    /// GPU did not write in this frame (EXP43=3: the most recently modified of several), drifted
    /// by less than its size, with nothing unusual about it. Else none.
    [[nodiscard]] ImageId RelocationCandidate(const ImageInfo& new_info, GPUVAddr gpu_addr,
                                              DAddr cpu_addr);
    /// Sets `reason` on the image's shape and address (logs the shape's first trip).
    void NoteRelocationGuard(ImageBase& image, RelocationGuard reason);
    /// PrepareImage's cross-frame read check. Every texture, copy and render target use in this
    /// fork goes through PrepareImage (the sampled texture memo too); delta draws replay
    /// descriptors only within one render pass and submission, never across frames.
    void CheckCrossFrameUse(ImageBase& image, bool is_modification, bool invalidate) {
        if (image.last_write_frame == frame_tick || image.last_write_frame == ~u64{0} ||
            invalidate) {
            return;
        }
        const RelocationGuard reason = is_modification ? GuardLoaded : GuardSampled;
        if ((image.relocation_guard_noted & reason) == 0 && RelocationShaped(image.info)) {
            NoteRelocationGuard(image, reason);
        }
    }
    /// Moves a registered image to a new address, keeping its host image, views and
    /// framebuffers; its contents at the old address are dropped as the collector drops them.
    void RelocateImage(ImageId image_id, GPUVAddr gpu_addr, DAddr cpu_addr);
    /// Render target reuse (EDEN_NX2_RT_REUSE=1): Ocarina of Time's busiest per-frame targets
    /// (1280x720) drift ~10 MB a frame, past their own size, so they overlap nothing and EXP43
    /// never sees them. Cleared render targets of relocation shape go to a pool per shape; a new
    /// one that overlaps nothing takes a pooled image of exactly its shape that was not used in
    /// this frame or the previous one, under the same guard (shape, both addresses, the image's
    /// own guard bits) and exclusions as EXP43.
    [[nodiscard]] static bool ReuseEnabled() {
        static const bool enabled = [] {
            const char* const value = std::getenv("EDEN_NX2_RT_REUSE");
            return value != nullptr && value[0] == '1';
        }();
        return enabled;
    }
    [[nodiscard]] static bool SameRelocationShape(const ImageInfo& info,
                                                  const ImageInfo& new_info) noexcept {
        return RelocationShaped(info) && info.format == new_info.format &&
               info.size.width == new_info.size.width &&
               info.size.height == new_info.size.height &&
               info.size.depth == new_info.size.depth && info.block == new_info.block &&
               info.layer_stride == new_info.layer_stride &&
               info.maybe_unaligned_layer_stride == new_info.maybe_unaligned_layer_stride &&
               info.tile_width_spacing == new_info.tile_width_spacing &&
               info.native_tiling == new_info.native_tiling &&
               info.rescaleable == new_info.rescaleable &&
               info.downscaleable == new_info.downscaleable &&
               info.is_sparse == new_info.is_sparse && info.astc_opaque == new_info.astc_opaque;
    }
    /// A pooled image the new render target can take, or none.
    [[nodiscard]] ImageId ReuseCandidate(const ImageInfo& new_info, GPUVAddr gpu_addr,
                                         DAddr cpu_addr);
    ::Common::unordered_map<u64, std::vector<ImageId>> reuse_pool;
    u64 relocation_reused{};
    u64 reuse_misses{};

    /// Removes the image from its overlaps' overlap lists (DeleteImage, RelocateImage).
    void DropOverlapLinks(ImageId image_id);

    /// Drops a framebuffer from framebuffers_by_view and framebuffer_keys.
    void UnindexFramebuffer(FramebufferId framebuffer_id, const RenderTargets& key);

    /// The framebuffers each image view is in, and each framebuffer's key: RemoveFramebuffers
    /// takes a deleted image's framebuffers from here instead of scanning the whole map, which
    /// grows by one entry per drifted render target a frame (fable_answer_12 2E).
    ::Common::unordered_map<ImageViewId, boost::container::small_vector<FramebufferId, 4>>
        framebuffers_by_view;
    ::Common::unordered_map<FramebufferId, RenderTargets> framebuffer_keys;

    /// Mark an image as modified from the GPU
    void MarkModification(ImageBase& image) noexcept;

    /// Marks the image's device pages in gpu_modified_filter.
    void MarkGpuModifiedPages(ImageBase& image);

    /// Synchronize image aliases, copying data if needed
    void SynchronizeAliases(ImageId image_id);

    /// Prepare an image view to be used
    void PrepareImageView(ImageViewId image_view_id, bool is_modification, bool invalidate);

    /// Execute copies from one image to the other, even if they are incompatible
    void CopyImage(ImageId dst_id, ImageId src_id, std::vector<ImageCopy> copies);

    /// Bind an image view as render target, downloading resources preemtively if needed
    void BindRenderTarget(ImageViewId* old_id, ImageViewId new_id);

    /// Create a render target from a given image and image view parameters
    [[nodiscard]] std::pair<FramebufferId, ImageViewId> RenderTargetFromImage(
        ImageId, const ImageViewInfo& view_info);

    /// Returns true if the current clear parameters clear the whole image of a given image view
    [[nodiscard]] bool IsFullClear(ImageViewId id);

    /// Returns true if the current clear's render area and scissor cover a width x height image
    [[nodiscard]] bool ClearCovers(u32 width, u32 height) const;

    [[nodiscard]] std::pair<u32, u32> PrepareDmaImage(ImageId dst_id, GPUVAddr base_addr,
                                                      bool mark_as_modified);

    bool ImageCanRescale(ImageBase& image);
    void InvalidateScale(Image& image);
    bool ScaleUp(Image& image);
    bool ScaleDown(Image& image);
    u64 GetScaledImageSizeBytes(const ImageBase& image);

    /// The device memory an image counts for in total_used_memory between two driver readings:
    /// none for an unbacked image (too large to back, or over unwritten sparse blocks), what is
    /// bound to its layers for a SparseLayers one.
    u64 ImageMemoryBytes(const Image& image) const;

    void QueueAsyncDecode(Image& image, ImageId image_id);
    void TickAsyncDecode();

    void QueueAsyncUnswizzle(Image& image, ImageId image_id);
    void TickAsyncUnswizzle();

    Runtime& runtime;

    Tegra::MaxwellDeviceMemoryManager& device_memory;
    std::deque<TextureCacheGPUMap> gpu_page_table_storage;

    RenderTargets render_targets;
    u64 render_targets_serial = 0;
    u32 rt_active_mask = 0;
    std::array<ImageId, 8> rt_image_id{};
    ImageId rt_depth_image_id{};
    u64 texture_bindings_serial = 0;
    u64 last_feedback_loop_serial = 0;
    u64 last_feedback_texture_serial = 0;
    bool last_feedback_loop_result = false;
    bool feedback_depth_dirty = true;
    FramebufferId last_framebuffer_id{};
    u64 last_framebuffer_serial = 0;

    ::Common::unordered_map<RenderTargets, FramebufferId> framebuffers;
    ::Common::unordered_map<u64, std::vector<ImageMapId>, Common::IdentityHash<u64>> page_table;
    ::Common::unordered_map<ImageId, boost::container::small_vector<ImageViewId, 16>> sparse_views;

    DAddr virtual_invalid_space{};

    /// Applies EDEN_NX2_TC_EXPECTED_MIB / EDEN_NX2_TC_CRITICAL_MIB and logs the thresholds.
    void ApplyThresholdOverrides(bool log = true);
    void LogThresholds() const;

    /// A GPU-written image dropped without being written back leaves its
    /// guest memory stale (the GPU's data is gone). Its whole pages are marked here and kept
    /// tracked, so CPU writes, unmaps and write-backs clear them. An image made entirely on marked
    /// pages starts without an upload: the upload would only copy stale bytes the GPU overwrites.
    /// Switch 2 titles only: the buffer cache feeds Nx2MayBeGpuWritten for them alone.
    void MarkStale(const ImageBase& image);
    void ClearStale(DAddr addr, size_t size);
    void ClearStaleImage(const ImageBase& image);
    /// Clears every device range [gpu_addr, gpu_addr + size) maps to.
    void ClearStaleGpu(GPUVAddr gpu_addr, size_t size);
    StalePages stale_pages;
    u64 tc_stale_skips{};        ///< Uploads skipped (last 600 frames)
    u64 tc_stale_skip_bytes{};
    u64 tc_stale_gpu_written{};  ///< Stale but buffer-written pages: uploaded after all

    /// The CPU address an image at gpu_addr is keyed by: its backing, or a fixed stand-in.
    [[nodiscard]] std::optional<DAddr> ImageCpuAddress(GPUVAddr gpu_addr, size_t size) const;

    bool has_deleted_images = false;
    std::vector<ImageId>* prepare_log{}; ///< LogPrepares
    bool prepare_log_dry{};
    /// True while a clear binds its render targets (UpdateRenderTargets(true)), so JoinImages can
    /// skip the upload of a new target the clear overwrites whole.
    bool clearing_render_targets = false;
    bool is_rescaling = false;
    u64 total_used_memory = 0;
    u64 minimum_memory;
    u64 expected_memory;
    u64 critical_memory;
    size_t gpu_unswizzle_maxsize = 0;
    size_t swizzle_chunk_size = 0;
    u32 swizzle_slices_per_batch = 0;

    struct BufferDownload {
        GPUVAddr address;
        size_t size;
    };

    using PendingDownload = PendingImageDownload<Image>;

    Common::SlotVector<Image> slot_images;
    Common::SlotVector<ImageMapView> slot_map_views;
    Common::SlotVector<ImageView> slot_image_views;
    Common::SlotVector<ImageAlloc> slot_image_allocs;
    Common::SlotVector<Sampler> slot_samplers;
    Common::SlotVector<Framebuffer> slot_framebuffers;
    Common::SlotVector<BufferDownload> slot_buffer_downloads;

    // TODO: This data structure is not optimal and it should be reworked

    std::vector<PendingDownload> uncommitted_downloads;
    std::deque<std::vector<PendingDownload>> committed_downloads;
    std::vector<AsyncBuffer> uncommitted_async_buffers;
    std::deque<std::vector<AsyncBuffer>> async_buffers;
    std::deque<AsyncBuffer> async_buffers_death_ring;

    struct LRUItemParams {
        using ObjectType = ImageId;
        using TickType = u64;
    };
    Common::LeastRecentlyUsedCache<LRUItemParams> lru_cache;

 #ifdef YUZU_LEGACY
    static constexpr size_t TICKS_TO_DESTROY = 6;
 #else
    static constexpr size_t TICKS_TO_DESTROY = 8;
#endif
    DelayedDestructionRing<Image, TICKS_TO_DESTROY> sentenced_images;
    /// Device memory of the images in each slot of sentenced_images (same index): deleted, but
    /// counted in the driver's usage until the ring destroys them.
    std::array<u64, TICKS_TO_DESTROY> sentenced_image_bytes{};
    size_t sentenced_image_index = 0;
    /// Deleted images on runtimes that report GPU progress (CurrentTick, IsFree): destroyed, with
    /// their device memory, as soon as the GPU is past them rather than TICKS_TO_DESTROY frames
    /// later. While the collector deletes dozens a frame, eight frames of them held 1-1.5 GB of
    /// VRAM above its threshold (title 1 on an 8 GB card: 6.6-7.0 GB in use against 5 GiB).
    struct GpuSentencedImage {
        u64 tick;
        u64 bytes;
        Image image;
    };
    std::deque<GpuSentencedImage> gpu_sentenced_images;
    u64 gpu_sentenced_bytes{};

    /// Bytes of images per device memory block, for the collector (refreshed every 60 frames).
    std::unordered_map<u64, u64> block_image_bytes;
    u64 block_image_bytes_frame{~0ULL};
    DelayedDestructionRing<ImageView, TICKS_TO_DESTROY> sentenced_image_view;
    DelayedDestructionRing<Framebuffer, TICKS_TO_DESTROY> sentenced_framebuffers;

    ::Common::unordered_map<GPUVAddr, ImageAllocId> image_allocs_table;

    Common::ScratchBuffer<u8> swizzle_data_buffer;
    Common::ScratchBuffer<u8> unswizzle_data_buffer;
    Tegra::Texture::UnswizzleWorker unswizzle_worker;

    u64 modification_tick = 0;
    u64 frame_tick = 0;

    Common::ThreadWorker texture_decode_worker{1, "TextureDecoder", {},
                                               Common::ThreadPlacement::Efficiency};
    std::vector<std::unique_ptr<AsyncDecodeContext>> async_decodes;
    /// Guest memory images were uploaded from since the last cold hint (device address, size).
    std::vector<std::pair<DAddr, u64>> uploaded_ranges;

    std::deque<PendingUnswizzle> unswizzle_queue;
    u8 current_unswizzle_frame;

    // Join caching
    boost::container::small_vector<ImageId, 4> join_overlap_ids;
    ::Common::unordered_set<ImageId> join_overlaps_found;
    boost::container::small_vector<ImageId, 4> join_left_aliased_ids;
    boost::container::small_vector<ImageId, 4> join_right_aliased_ids;
    ::Common::unordered_set<ImageId> join_ignore_textures;
    boost::container::small_vector<ImageId, 4> join_bad_overlap_ids;
    struct JoinCopy {
        bool is_alias;
        ImageId id;
    };
    boost::container::small_vector<JoinCopy, 4> join_copies_to_do;
    ::Common::unordered_map<ImageId, size_t> join_alias_indices;
};

} // namespace VideoCommon
