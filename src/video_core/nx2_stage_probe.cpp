// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <fmt/format.h>

#include "common/logging.h"
#include "common/thread.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_stage_probe.h"
#include "video_core/texture_cache/render_targets.h"

namespace VideoCommon::Nx2StageProbe {
namespace {

constexpr u64 Golden = 0x9E3779B97F4A7C15ULL;

void Pause() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(_MSC_VER) && defined(_M_ARM64)
    __yield();
#elif defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

void StartSpinner() {
    static std::once_flag once;
    std::call_once(once, [] {
        static std::jthread spinner{[](std::stop_token stop) {
            Common::SetCurrentThreadName("Nx2ProbeSpin");
            while (!stop.stop_requested()) {
                for (u32 spin = 0; spin < 4096; ++spin) {
                    Pause();
                }
            }
        }};
        LOG_INFO(HW_GPU, "Stage probe: EDEN_NX2_STAGE_PROBE_SPIN started one spinning thread");
    });
}

thread_local std::unique_ptr<ThreadData> t_owner;

ThreadData* Attach() {
    t_owner = std::make_unique<ThreadData>();
    ThreadData& thread = *t_owner;
    // What one region switch costs, the clock read and the bookkeeping, for the overhead estimate.
    {
        auto scratch = std::make_unique<ThreadData>();
        scratch->fine_start = Ticks();
        constexpr u32 switches = 4096;
        const u64 start = Ticks();
        for (u32 count = 0; count < switches; ++count) {
            scratch->Switch(count & 1 ? Region::Draw : Region::Sync);
        }
        thread.switch_ns = static_cast<double>(Ticks() - start) / switches; // ticks until calibrated
    }
    thread.window_time = std::chrono::steady_clock::now();
    thread.window_ticks = Ticks();
    thread.coarse_start = thread.window_ticks;
    t_data = &thread;
    LOG_INFO(HW_GPU, "Stage probe: measuring on this thread (EDEN_NX2_STAGE_PROBE)");
    return &thread;
}

std::optional<u64> Translate(ThreadData& thread, const Tegra::MemoryManager& memory,
                             GPUVAddr address) {
    const u64 page = address >> 12;
    const u64 serial = memory.MapSerial();
    if (thread.translated_memory == &memory && thread.translated_serial == serial &&
        thread.translated_page == page) {
        return thread.translated_base + (address & 0xfff);
    }
    const std::optional<DAddr> device = memory.GpuToCpuAddress(address);
    if (!device) {
        return std::nullopt;
    }
    thread.translated_memory = &memory;
    thread.translated_serial = serial;
    thread.translated_page = page;
    thread.translated_base = *device - (address & 0xfff);
    return *device;
}

void MaybeReport(ThreadData& thread) {
    const auto now_time = std::chrono::steady_clock::now();
    const auto elapsed = now_time - thread.window_time;
    if (elapsed < std::chrono::seconds{10}) {
        return;
    }
    const u64 now = Ticks();
    ++thread.totals.timer_calls;
    Totals& totals = thread.totals;
    totals.coarse_ticks[static_cast<size_t>(thread.coarse)] += now - thread.coarse_start;
    thread.coarse_start = now;
    const double ns = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
    if (thread.ticks_per_ns == 0.0) {
        // Calibrated once, over the first window.
        thread.ticks_per_ns = static_cast<double>(now - thread.window_ticks) / ns;
        thread.switch_ns /= thread.ticks_per_ns;
    }
    SharedTotals shared;
    for (size_t site = 0; site < shared.split_waits.size(); ++site) {
        shared.split_waits[site] = g_shared.split_waits[site].exchange(0, std::memory_order_relaxed);
    }
    shared.guest_reads_flushing = g_shared.guest_reads_flushing.exchange(0, std::memory_order_relaxed);
    shared.guest_reads_in_flight =
        g_shared.guest_reads_in_flight.exchange(0, std::memory_order_relaxed);
    shared.buffer_flushes = g_shared.buffer_flushes.exchange(0, std::memory_order_relaxed);
    shared.buffer_flushes_modified =
        g_shared.buffer_flushes_modified.exchange(0, std::memory_order_relaxed);
    OverlapRing& ring = thread.ring;
    LOG_INFO(HW_GPU, "{}",
             FormatLine(totals, shared, ns / 1e9, thread.ticks_per_ns, thread.switch_ns,
                        ring.Live(thread.draws), ring.overflows, ring.reads, ring.scans,
                        ring.scanned));
    totals = {};
    ring.reads = 0;
    ring.scans = 0;
    ring.scanned = 0;
    ring.overflows = 0;
    thread.window_time = now_time;
    thread.window_ticks = now;
}

} // Anonymous namespace

void ReadSwitch() {
    g_enabled = EnvOn("EDEN_NX2_STAGE_PROBE");
    if (EnvOn("EDEN_NX2_STAGE_PROBE_SPIN")) {
        StartSpinner();
    }
}

OverlapRing::OverlapRing()
    : index(IndexSize), filter((size_t{1} << FilterBits) / 64), memo(SlotCount) {
    entries.reserve(Capacity);
}

size_t OverlapRing::HashSlot(u64 begin, u64 end) const {
    const u64 hash = (begin * Golden) ^ (end * 0xC2B2AE3D27D4EB4FULL);
    return static_cast<size_t>(hash >> 32) & (IndexSize - 1);
}

void OverlapRing::Mark(u64 begin, u64 end) {
    const u64 first = begin >> 16;
    const u64 last = (end - 1) >> 16;
    if (last - first >= 4096) {
        filter_all = true;
        return;
    }
    for (u64 page = first; page <= last; ++page) {
        const u64 bit = (page * Golden) >> (64 - FilterBits);
        filter[bit >> 6] |= u64{1} << (bit & 63);
    }
}

void OverlapRing::Read(u32 slot, u64 begin, u64 size, u64 draw) {
    ++reads;
    if (size == 0 || begin == 0) {
        return;
    }
    const u64 end = begin + size;
    Memo& slot_memo = memo[(std::min)(slot, SlotCount - 1)];
    if (slot_memo.begin == begin && slot_memo.end == end && slot_memo.entry < entries.size()) {
        Entry& entry = entries[slot_memo.entry];
        if (entry.begin == begin && entry.end == end) {
            entry.last = draw;
            return;
        }
    }
    size_t hash = HashSlot(begin, end);
    for (; index[hash] != 0; hash = (hash + 1) & (IndexSize - 1)) {
        const u32 stored = index[hash] - 1;
        Entry& entry = entries[stored];
        if (entry.begin == begin && entry.end == end) {
            entry.last = draw;
            slot_memo = {begin, end, stored};
            return;
        }
    }
    if (entries.size() == Capacity) {
        Compact(draw);
        if (entries.size() == Capacity) {
            ++overflows;
            return;
        }
        hash = HashSlot(begin, end);
        while (index[hash] != 0) {
            hash = (hash + 1) & (IndexSize - 1);
        }
    }
    entries.push_back({begin, end, draw});
    index[hash] = static_cast<u32>(entries.size());
    Mark(begin, end);
    slot_memo = {begin, end, static_cast<u32>(entries.size() - 1)};
}

bool OverlapRing::Overlaps(u64 begin, u64 size, u64 draws) {
    if (entries.empty() || size == 0) {
        return false;
    }
    const u64 end = begin + size;
    const u64 first = begin >> 16;
    const u64 last = (end - 1) >> 16;
    if (!filter_all && last - first < 64) {
        bool marked = false;
        for (u64 page = first; page <= last && !marked; ++page) {
            const u64 bit = (page * Golden) >> (64 - FilterBits);
            marked = (filter[bit >> 6] >> (bit & 63)) & 1;
        }
        if (!marked) {
            return false;
        }
    }
    ++scans;
    for (const Entry& entry : entries) {
        ++scanned;
        if (draws - entry.last <= Window && entry.begin < end && begin < entry.end) {
            return true;
        }
    }
    return false;
}

void OverlapRing::Compact(u64 draws) {
    std::erase_if(entries, [draws](const Entry& entry) { return draws - entry.last > Window; });
    std::ranges::fill(index, 0U);
    std::ranges::fill(filter, u64{0});
    filter_all = false;
    for (size_t position = 0; position < entries.size(); ++position) {
        const Entry& entry = entries[position];
        size_t hash = HashSlot(entry.begin, entry.end);
        while (index[hash] != 0) {
            hash = (hash + 1) & (IndexSize - 1);
        }
        index[hash] = static_cast<u32>(position + 1);
        Mark(entry.begin, entry.end);
    }
}

size_t OverlapRing::Live(u64 draws) const {
    return static_cast<size_t>(std::ranges::count_if(
        entries, [draws](const Entry& entry) { return draws - entry.last <= Window; }));
}

void ThreadData::Write(Site site, u64 begin, u64 size) {
    const size_t index = static_cast<size_t>(site);
    ++totals.writes[index];
    if (ring.Overlaps(begin, size, draws)) {
        ++totals.overlaps[index];
    }
}

void NoteGpuWriteImpl(ThreadData& thread, Site site, const Tegra::MemoryManager& memory,
                      GPUVAddr address, u64 size) {
    const std::optional<u64> device = Translate(thread, memory, address);
    if (!device) {
        ++thread.totals.writes[static_cast<size_t>(site)];
        return;
    }
    thread.Write(site, *device, size);
}

void BeginDma(ThreadData& thread, const Tegra::MemoryManager& memory, GPUVAddr address,
              u64 size) {
    const size_t index = static_cast<size_t>(Site::Dma);
    ++thread.totals.writes[index];
    const std::optional<u64> device = Translate(thread, memory, address);
    const bool overlap = device && thread.ring.Overlaps(*device, size, thread.draws);
    thread.totals.overlaps[index] += overlap ? 1 : 0;
    thread.in_dma = true;
    thread.dma_joined = overlap;
}

void CoarseScope::Begin(Region region) {
    ThreadData* thread = t_data;
    if (thread == nullptr) {
        if (region != Region::Draw) {
            return;
        }
        thread = Attach();
    }
    if (thread->coarse == region) {
        return;
    }
    const u64 now = Ticks();
    Totals& totals = thread->totals;
    ++totals.timer_calls;
    totals.coarse_ticks[static_cast<size_t>(thread->coarse)] += now - thread->coarse_start;
    thread->coarse_start = now;
    previous = thread->coarse;
    thread->coarse = region;
    data = thread;
    switch (region) {
    case Region::Draw: {
        thread->draw_start = now;
        thread->packet_bytes =
            PacketHeaderBytes + static_cast<u32>(sizeof(VideoCommon::RenderTargets));
        u32 rng = thread->rng;
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        thread->rng = rng;
        thread->fine = (rng & 7) == 0;
        if (thread->fine) {
            thread->fine_current = Region::Draw;
            thread->fine_start = now;
        }
        break;
    }
    case Region::Compute:
        ++totals.counters[static_cast<size_t>(Counter::Computes)];
        break;
    case Region::Clear:
        ++totals.counters[static_cast<size_t>(Counter::Clears)];
        break;
    default:
        break;
    }
}

void CoarseScope::End() {
    ThreadData& thread = *data;
    Totals& totals = thread.totals;
    const u64 now = Ticks();
    ++totals.timer_calls;
    totals.coarse_ticks[static_cast<size_t>(thread.coarse)] += now - thread.coarse_start;
    thread.coarse_start = now;
    const Region ended = thread.coarse;
    thread.coarse = previous;
    if (ended != Region::Draw) {
        return;
    }
    if (thread.fine) {
        totals.fine_ticks[static_cast<size_t>(thread.fine_current)] += now - thread.fine_start;
        totals.fine_draw_ticks += now - thread.draw_start;
        ++totals.fine_draws;
        thread.fine = false;
    }
    ++totals.draws;
    totals.packet_bytes += thread.packet_bytes;
    totals.packet_max = (std::max)(totals.packet_max, u64{thread.packet_bytes});
    ++thread.draws;
    if ((thread.draws & 4095) == 0) {
        thread.ring.Compact(thread.draws);
    }
    if ((totals.draws & 255) == 0) {
        MaybeReport(thread);
    }
}

std::string FormatLine(const Totals& totals, const SharedTotals& shared, double seconds,
                       double ticks_per_ns, double switch_ns, size_t live_ranges,
                       u64 ring_overflows, u64 reads, u64 scans, u64 scanned) {
    const auto rate = [seconds](u64 count) { return static_cast<double>(count) / seconds; };
    const auto ms = [&](double ticks) { return ticks / ticks_per_ns / 1e6 / seconds; };
    const auto coarse = [&](Region region) {
        return static_cast<double>(totals.coarse_ticks[static_cast<size_t>(region)]);
    };
    double wall_ticks = 0.0;
    for (const u64 ticks : totals.coarse_ticks) {
        wall_ticks += static_cast<double>(ticks);
    }
    const double busy = ms(wall_ticks - coarse(Region::Wait));
    const double draw = ms(coarse(Region::Draw));
    const double compute = ms(coarse(Region::Compute));
    const double clear = ms(coarse(Region::Clear));
    // The sampled draws' regions, scaled by the share of draw time those draws took.
    const double scale = totals.fine_draw_ticks != 0
                             ? coarse(Region::Draw) / static_cast<double>(totals.fine_draw_ticks)
                             : 0.0;
    const auto fine = [&](size_t region) {
        return ms(static_cast<double>(totals.fine_ticks[region]) * scale);
    };
    double b_total = 0.0;
    std::string b_regions;
    constexpr std::array<const char*, LastB - FirstB + 1> b_names{
        "index/vertex",    "storage",     "sync",          "geometry bind", "stage bind",
        "descriptors",     "render pass", "dynamic state", "draw record",
    };
    for (size_t region = FirstB; region <= LastB; ++region) {
        const double value = fine(region);
        b_total += value;
        b_regions += fmt::format("{}{} {:.1f}", region == FirstB ? "" : " + ",
                                 b_names[region - FirstB], value);
    }
    const auto share = [](double part, double whole) {
        return whole > 0.0 ? part * 100.0 / whole : 0.0;
    };
    const auto site = [&](Site value) { return static_cast<size_t>(value); };
    const auto counter = [&](Counter value) {
        return totals.counters[static_cast<size_t>(value)];
    };
    const u64 dma_joins = totals.overlaps[site(Site::Dma)] + counter(Counter::DmaImageJoins);
    const u64 flush_joins = shared.guest_reads_flushing + shared.buffer_flushes_modified;
    const size_t ring_full = static_cast<size_t>(DrainSite::RingFull);
    u64 split_joins = 0;
    std::string split_sites;
    for (size_t index = 0; index < static_cast<size_t>(DrainSite::Count); ++index) {
        const u64 count = shared.split_waits[index];
        if (count == 0 || index == ring_full) {
            continue;
        }
        split_joins += count;
        split_sites += fmt::format("{}{} {:.0f}", split_sites.empty() ? " (" : ", ",
                                   drain_site_names[index], rate(count));
    }
    if (!split_sites.empty()) {
        split_sites += ')';
    }
    const u64 joins = dma_joins + totals.overlaps[site(Site::Inline)] +
                      totals.overlaps[site(Site::CbData)] + totals.overlaps[site(Site::Query)] +
                      flush_joins + split_joins;
    const double draws = static_cast<double>(totals.draws);
    const double overhead = static_cast<double>(totals.timer_calls) * switch_ns / 1e6 / seconds;
    return fmt::format(
        "Stage probe per second (10 s): back busy {:.1f} ms (split wait {:.1f} ms) | draws {:.0f} "
        "({:.0f} timed in detail), draw {:.1f} ms, outside draws {:.1f} ms, compute {:.1f} ms "
        "({:.0f}), clear {:.1f} ms ({:.0f}) | B {:.1f} ms = {} | uniform (A) {:.1f} ms, rest of "
        "draw (A) {:.1f} ms | B share {:.1f}% of busy, {:.1f}% of draw, {:.1f}% of busy with "
        "compute and clears | joins {:.2f} per 100 draws ({:.0f}): DMA {:.0f} of {:.0f} ({:.0f} "
        "overlap, {:.0f} image<->buffer), I2M {:.0f} of {:.0f}, CB_DATA {:.0f} of {:.0f}, query "
        "{:.0f} of {:.0f}, flush reads {:.0f} (guest reads {:.0f} flushing, {:.0f} in flight; "
        "buffer flushes {:.0f} of {:.0f} over GPU-written data), split waits {:.0f}{} | not joins: compute {:.0f}, clears {:.0f}, texture "
        "buffer binds {:.0f}, DMA via buffer cache {:.0f}, I2M via buffer cache {:.0f}, "
        "texture->buffer cache calls {:.0f}, ring-full waits {:.0f} | packet avg {:.0f} B, max {} "
        "B, {:.1f} MB | ring {} ranges, {:.0f} reads, {:.0f} scans ({:.0f} ranges compared), {} "
        "overflows | "
        "probe {:.0f} timer calls, est. {:.2f} ms ({:.2f}% of busy)",
        busy, ms(coarse(Region::Wait)), rate(totals.draws), rate(totals.fine_draws), draw,
        ms(coarse(Region::Outside)), compute, rate(counter(Counter::Computes)), clear,
        rate(counter(Counter::Clears)), b_total, b_regions, fine(static_cast<size_t>(Region::Uniform)),
        fine(static_cast<size_t>(Region::Draw)), share(b_total, busy), share(b_total, draw),
        share(b_total + compute + clear, busy), draws > 0.0 ? static_cast<double>(joins) * 100.0 / draws : 0.0,
        rate(joins), rate(dma_joins), rate(totals.writes[site(Site::Dma)]),
        rate(totals.overlaps[site(Site::Dma)]), rate(counter(Counter::DmaImageJoins)),
        rate(totals.overlaps[site(Site::Inline)]), rate(totals.writes[site(Site::Inline)]),
        rate(totals.overlaps[site(Site::CbData)]), rate(totals.writes[site(Site::CbData)]),
        rate(totals.overlaps[site(Site::Query)]), rate(totals.writes[site(Site::Query)]),
        rate(flush_joins), rate(shared.guest_reads_flushing), rate(shared.guest_reads_in_flight),
        rate(shared.buffer_flushes_modified), rate(shared.buffer_flushes), rate(split_joins),
        split_sites,
        rate(counter(Counter::Computes)), rate(counter(Counter::Clears)),
        rate(counter(Counter::TextureBufferBinds)), rate(counter(Counter::DmaViaBufferCache)),
        rate(counter(Counter::InlineViaBufferCache)), rate(counter(Counter::TextureToBuffer)),
        rate(shared.split_waits[ring_full]),
        draws > 0.0 ? static_cast<double>(totals.packet_bytes) / draws : 0.0, totals.packet_max,
        static_cast<double>(totals.packet_bytes) / 1e6 / seconds, live_ranges, rate(reads), rate(scans),
        rate(scanned), ring_overflows, rate(totals.timer_calls), overhead, share(overhead, busy));
}

} // namespace VideoCommon::Nx2StageProbe
