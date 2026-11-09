// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif

#include "common/common_types.h"

namespace Tegra {
class MemoryManager;
}

// EDEN_NX2_STAGE_PROBE=1 (test only, EGLA-91 stage 0): measures, on the thread that draws (the
// split's back), what a second drawing stage would take over and how often the two stages would
// have to wait for each other. Every 10 s it logs one "Stage probe per second (10 s)" line
// (docs/nx2/stage-probe.md). EDEN_NX2_STAGE_PROBE_SPIN=1 starts one thread that only spins.
namespace VideoCommon::Nx2StageProbe {

[[nodiscard]] inline bool EnvOn(const char* name) {
    const char* const value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

// A plain flag: the hooks run several times per draw.
inline bool g_enabled = EnvOn("EDEN_NX2_STAGE_PROBE");

/// Called by VideoCore::CreateGPU (after nx2_switches.txt is loaded); starts the spinner.
void ReadSwitch();

[[nodiscard]] inline bool Enabled() {
    return g_enabled;
}

/// The cycle counter: rdtsc on x86, cntvct_el0 on arm64, else steady_clock nanoseconds.
inline u64 Ticks() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    return __rdtsc();
#elif defined(_MSC_VER) && defined(_M_ARM64)
    return static_cast<u64>(_ReadStatusReg(ARM64_SYSREG(3, 3, 14, 0, 2)));
#elif defined(__x86_64__) || defined(__i386__)
    return __rdtsc();
#elif defined(__aarch64__)
    u64 value;
    asm volatile("mrs %0, cntvct_el0" : "=r"(value));
    return value;
#else
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
#endif
}

/// Where the thread that draws spends its time. The first five are timed at every switch; the
/// rest only inside the sampled draws (1 in 8), and their time is scaled to all draws.
enum class Region : u8 {
    Outside, ///< Methods and everything else outside the regions below
    Wait,    ///< The split's back waits for the front (the GPU thread: for commands)
    Draw,    ///< Draw, DrawIndirect, DrawTexture; in sampled draws: their stage A rest
    Compute, ///< DispatchCompute
    Clear,   ///< Clear
    // Stage B
    IndexVertex,  ///< Index, vertex and transform feedback buffer resolve
    Storage,      ///< Storage and texel buffer resolve
    Sync,         ///< SynchronizeBuffer and its uploads, and the barrier after them
    GeometryBind, ///< BindHostGeometryBuffers (without the syncs)
    StageBind,    ///< BindHostStageBuffers without the uniform buffers (and the syncs)
    Descriptors,  ///< Descriptor acquire, image descriptor push, commit and pipeline bind
    RenderPass,   ///< Render pass request (scheduler)
    DynamicState, ///< UpdateDynamicStates
    DrawRecord,   ///< Recording the draw
    // Stage A, within draws
    Uniform, ///< Uniform buffer resolve, binds and copies (their syncs included)
    Count,
};

inline constexpr size_t RegionCount = static_cast<size_t>(Region::Count);
inline constexpr size_t FirstB = static_cast<size_t>(Region::IndexVertex);
inline constexpr size_t LastB = static_cast<size_t>(Region::DrawRecord);

/// Guest memory writes from the method stream that can overlap what stage B reads.
enum class Site : u8 {
    Dma,    ///< MaxwellDMA copies and clears
    Inline, ///< I2M / inline data (3D, compute and the inline-to-memory engine)
    CbData, ///< CB_DATA
    Query,  ///< Query, report and semaphore releases (RasterizerVulkan::Query)
    Count,
};

inline constexpr size_t SiteCount = static_cast<size_t>(Site::Count);

enum class Counter : u8 {
    Computes,
    Clears,
    TextureBufferBinds,   ///< BufferCache::BindGraphicsTextureBuffer (stage A resolves them)
    DmaViaBufferCache,    ///< DMA copies and clears the buffer cache made itself
    InlineViaBufferCache, ///< Inline writes uploaded into buffer cache buffers in place
    TextureToBuffer,      ///< Texture cache paths into the buffer cache (DMA image<->buffer)
    DmaImageJoins,        ///< Of TextureToBuffer: DMA launches not already counted as overlaps
    Count,
};

inline constexpr size_t CounterCount = static_cast<size_t>(Counter::Count);

/// Read slots of the overlap ring, so a binding that stays the same costs one compare.
inline constexpr u32 IndexSlot = 0;
inline constexpr u32 SlotCount = 1 + 32 + 5 * 64 + 5 * 32;
[[nodiscard]] constexpr u32 VertexSlot(u32 index) {
    return 1 + index;
}
[[nodiscard]] constexpr u32 StorageSlot(size_t stage, u32 index) {
    return 33 + static_cast<u32>(stage) * 64 + index;
}
[[nodiscard]] constexpr u32 TexelSlot(size_t stage, u32 index) {
    return 33 + 5 * 64 + static_cast<u32>(stage) * 32 + index;
}

/// The device ranges stage B read in the last Window draws: index buffers, vertex streams,
/// storage and texel buffers. A 64 KiB page filter keeps the write checks cheap.
class OverlapRing {
public:
    static constexpr u64 Window = 64;
    static constexpr size_t Capacity = 1024;

    OverlapRing();

    /// Draw number draw (the draws completed before it) read [begin, begin + size).
    void Read(u32 slot, u64 begin, u64 size, u64 draw);

    /// Whether [begin, begin + size) overlaps a range one of the last Window draws read,
    /// draws draws having completed.
    [[nodiscard]] bool Overlaps(u64 begin, u64 size, u64 draws);

    /// Drops ranges no draw of the window read and rebuilds the filter.
    void Compact(u64 draws);

    [[nodiscard]] size_t Live(u64 draws) const;

    u64 reads{};     ///< Read calls
    u64 scans{};     ///< Writes the filter couldn't rule out
    u64 scanned{};   ///< Ranges those compared
    u64 overflows{}; ///< Reads not kept: more than Capacity ranges in the window

private:
    struct Entry {
        u64 begin;
        u64 end;
        u64 last;
    };
    struct Memo {
        u64 begin{};
        u64 end{};
        u32 entry{};
    };

    static constexpr size_t IndexSize = 2 * Capacity;
    static constexpr u32 FilterBits = 18;

    void Mark(u64 begin, u64 end);
    [[nodiscard]] size_t HashSlot(u64 begin, u64 end) const;

    std::vector<Entry> entries;
    std::vector<u32> index;  ///< Entry + 1, or 0
    std::vector<u64> filter; ///< One bit per hashed 64 KiB page
    std::vector<Memo> memo;
    bool filter_all{}; ///< A range too large to mark is in the table
};

/// Totals of the thread that draws, reset every report.
struct Totals {
    std::array<u64, RegionCount> coarse_ticks{}; ///< Outside to Clear
    std::array<u64, RegionCount> fine_ticks{};   ///< Draw (stage A rest) to Uniform, sampled
    u64 fine_draw_ticks{};
    u64 draws{};
    u64 fine_draws{};
    u64 timer_calls{};
    std::array<u64, SiteCount> writes{};
    std::array<u64, SiteCount> overlaps{};
    std::array<u64, CounterCount> counters{};
    u64 packet_bytes{};
    u64 packet_max{};
};

/// Counted on any thread.
struct SharedTotals {
    std::array<u64, 32> split_waits{}; ///< By DrainSite
    u64 guest_reads_flushing{};        ///< GPU::OnCPURead that asked for a flush
    u64 guest_reads_in_flight{};       ///< ...that found the downloads already under way
    u64 buffer_flushes{};              ///< RasterizerVulkan::FlushRegion of the buffer cache
    u64 buffer_flushes_modified{};     ///< ...over GPU-modified buffer data
};

struct ThreadData {
    Region coarse{Region::Outside};
    u64 coarse_start{};
    u64 draw_start{};
    bool fine{};
    Region fine_current{Region::Draw};
    u64 fine_start{};
    u32 rng{0x9e3779b9U};
    bool in_dma{};
    bool dma_joined{};
    u32 packet_bytes{};
    u64 draws{}; ///< Draws completed since the start (the ring's clock)
    OverlapRing ring;
    Totals totals;
    // GPU to device translation of the last write's 4 KiB page
    const Tegra::MemoryManager* translated_memory{};
    u64 translated_serial{};
    u64 translated_page{~u64{0}};
    u64 translated_base{};
    // Report window
    u64 window_ticks{};
    std::chrono::steady_clock::time_point window_time{};
    double ticks_per_ns{};
    double switch_ns{};

    void Switch(Region region) {
        const u64 now = Ticks();
        totals.fine_ticks[static_cast<size_t>(fine_current)] += now - fine_start;
        fine_start = now;
        fine_current = region;
        ++totals.timer_calls;
    }

    void Read(u32 slot, u64 begin, u64 size, u32 bytes) {
        packet_bytes += bytes;
        ring.Read(slot, begin, size, draws);
    }

    void Write(Site site, u64 begin, u64 size);
};

/// Only the thread that draws has data: its first draw makes it.
inline thread_local ThreadData* t_data{};

/// Coarse regions (Draw, Compute, Clear, Wait, Outside), on the thread that draws.
class CoarseScope {
public:
    explicit CoarseScope(Region region) {
        if (g_enabled) {
            Begin(region);
        }
    }
    ~CoarseScope() {
        if (data != nullptr) {
            End();
        }
    }
    CoarseScope(const CoarseScope&) = delete;
    CoarseScope& operator=(const CoarseScope&) = delete;

private:
    void Begin(Region region);
    void End();

    ThreadData* data{};
    Region previous{};
};

/// A stage B or uniform region inside a sampled draw. when = false: skip it (an empty stage).
class Scope {
public:
    explicit Scope(Region region, bool when = true) {
        if (!g_enabled) {
            return;
        }
        ThreadData* const thread = t_data;
        if (thread == nullptr || !thread->fine || !when || thread->fine_current == region ||
            (region == Region::Sync && thread->fine_current == Region::Uniform)) {
            return;
        }
        previous = thread->fine_current;
        thread->Switch(region);
        data = thread;
    }
    ~Scope() {
        if (data != nullptr) {
            data->Switch(previous);
        }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    ThreadData* data{};
    Region previous{};
};

/// Bytes the draw's packet would carry for this (descriptors, bindings, offsets).
inline void AddPacketBytes(u32 bytes) {
    if (g_enabled && t_data != nullptr) {
        t_data->packet_bytes += bytes;
    }
}

/// Stage B reads this device range for the current draw (bytes: its packet entry, if any).
inline void NoteRead(u32 slot, u64 begin, u64 size, u32 bytes) {
    if (g_enabled && t_data != nullptr) {
        t_data->Read(slot, begin, size, bytes);
    }
}

/// A method writes guest memory at this device range.
inline void NoteWrite(Site site, u64 begin, u64 size) {
    if (g_enabled && t_data != nullptr) {
        t_data->Write(site, begin, size);
    }
}

void NoteGpuWriteImpl(ThreadData& thread, Site site, const Tegra::MemoryManager& memory,
                      GPUVAddr address, u64 size);

/// A method writes guest memory at this GPU range.
inline void NoteGpuWrite(Site site, const Tegra::MemoryManager& memory, GPUVAddr address,
                         u64 size) {
    if (g_enabled && t_data != nullptr) {
        NoteGpuWriteImpl(*t_data, site, memory, address, size);
    }
}

inline void Count(Counter counter) {
    if (g_enabled && t_data != nullptr) {
        ++t_data->totals.counters[static_cast<size_t>(counter)];
    }
}

/// The bytes a MaxwellDMA launch writes from regs.offset_out (block linear: width x height x
/// depth x bytes per pixel, without the block padding).
template <typename DmaRegs>
[[nodiscard]] u64 DmaDestinationBytes(const DmaRegs& regs) {
    const auto& launch = regs.launch_dma;
    using Layout = typename std::remove_cvref_t<decltype(launch)>::MemoryLayout;
    const u64 pixel = launch.remap_enable != 0
                          ? u64{regs.remap_const.num_dst_components_minus_one + 1U} *
                                (regs.remap_const.component_size_minus_one + 1U)
                          : 1;
    const u64 line = u64{regs.line_length_in} * pixel;
    if (launch.multi_line_enable == 0) {
        return line;
    }
    if (launch.dst_memory_layout == Layout::PITCH) {
        const u64 lines = regs.line_count != 0 ? regs.line_count : 1;
        const s64 pitch = static_cast<s64>(regs.pitch_out);
        return (lines - 1) * static_cast<u64>(pitch < 0 ? -pitch : pitch) + line;
    }
    const auto& dst = regs.dst_params;
    return u64{dst.width} * (dst.height != 0 ? dst.height : 1U) *
           (dst.depth != 0 ? dst.depth : 1U) * pixel;
}

void BeginDma(ThreadData& thread, const Tegra::MemoryManager& memory, GPUVAddr address, u64 size);

/// A MaxwellDMA launch starts.
template <typename DmaRegs>
void NoteDma(const Tegra::MemoryManager& memory, const DmaRegs& regs) {
    if (g_enabled && t_data != nullptr) {
        BeginDma(*t_data, memory, regs.offset_out, DmaDestinationBytes(regs));
    }
}

/// AccelerateDMA::BufferCopy or BufferClear; cached: the buffer cache made it.
inline void NoteDmaBufferCache(bool cached) {
    if (g_enabled && t_data != nullptr && cached) {
        ++t_data->totals.counters[static_cast<size_t>(Counter::DmaViaBufferCache)];
    }
}

/// A texture cache path that calls into the buffer cache (AccelerateDMA image<->buffer copies).
inline void NoteTextureToBufferCache() {
    if (g_enabled && t_data != nullptr) {
        ThreadData& thread = *t_data;
        ++thread.totals.counters[static_cast<size_t>(Counter::TextureToBuffer)];
        if (thread.in_dma && !thread.dma_joined) {
            thread.dma_joined = true;
            ++thread.totals.counters[static_cast<size_t>(Counter::DmaImageJoins)];
        }
    }
}

struct SharedCounters {
    std::array<std::atomic<u64>, 32> split_waits{};
    std::atomic<u64> guest_reads_flushing{};
    std::atomic<u64> guest_reads_in_flight{};
    std::atomic<u64> buffer_flushes{};
    std::atomic<u64> buffer_flushes_modified{};
};

inline SharedCounters g_shared;

/// The split's front waits for its back (a drain or a wait for a position), by DrainSite.
inline void NoteSplitWait(u32 site) {
    if (g_enabled) {
        g_shared.split_waits[site & 31].fetch_add(1, std::memory_order_relaxed);
    }
}

/// GPU::OnCPURead: a guest read of GPU-written memory; flushing: it asked for a flush.
inline void NoteGuestRead(bool flushing) {
    if (g_enabled) {
        (flushing ? g_shared.guest_reads_flushing : g_shared.guest_reads_in_flight)
            .fetch_add(1, std::memory_order_relaxed);
    }
}

/// RasterizerVulkan::FlushRegion of the buffer cache (any thread); modified: over GPU-modified
/// buffer data. Called only while enabled.
inline void NoteBufferFlush(bool modified) {
    g_shared.buffer_flushes.fetch_add(1, std::memory_order_relaxed);
    if (modified) {
        g_shared.buffer_flushes_modified.fetch_add(1, std::memory_order_relaxed);
    }
}

/// Fixed packet part: sequence, draw parameters, pipeline handle, flags (the render target key
/// is added to it).
inline constexpr u32 PacketHeaderBytes = 48;
/// Per entry: a vertex stream, the index buffer, a storage or texel buffer (address, size,
/// format or flags); a uniform stream offset; an image descriptor (view and sampler).
inline constexpr u32 PacketRangeBytes = 16;
inline constexpr u32 PacketUniformBytes = 8;
inline constexpr u32 PacketImageBytes = 16;

/// The report line from a window's totals (exposed for the unit test).
[[nodiscard]] std::string FormatLine(const Totals& totals, const SharedTotals& shared,
                                     double seconds, double ticks_per_ns, double switch_ns,
                                     size_t live_ranges, u64 ring_overflows, u64 reads,
                                     u64 scans, u64 scanned);

} // namespace VideoCommon::Nx2StageProbe
