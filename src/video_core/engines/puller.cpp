// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>

#include "common/assert.h"
#include "common/logging.h"
#include "common/settings.h"
#include "core/core.h"
#include <optional>

#include "video_core/nx2_profile.h"
#include "video_core/control/channel_state.h"
#include "video_core/dma_pusher.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/kepler_memory.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/engines/puller.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_acquire_test.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_perf.h"
#include "video_core/rasterizer_interface.h"

namespace Tegra::Engines {

namespace {

constexpr u32 Gf100BindClassMask = 0xffff;
constexpr u32 Gf100BindValidMask =  0x1f0000 | Gf100BindClassMask;

// Reads a semaphore word as a GPU acquire sees it, flushing a GPU-written copy out of the buffer
// cache first. A mirrored release the guest memory does not hold counts (Nx2MirrorShadow).
u32 ReadSemaphoreWord(DmaPusher& dma_pusher, GPUVAddr address) {
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address);
    if (device && dma_pusher.rasterizer->MustFlushRegion(*device, sizeof(u32),
                                                         VideoCommon::CacheType::BufferCache)) {
        VideoCommon::GpuSplit::Get().Run(
            [rasterizer = dma_pusher.rasterizer, device = *device] {
                rasterizer->FlushRegion(device, sizeof(u32), VideoCommon::CacheType::BufferCache);
            },
            VideoCommon::DrainSite::SemaphoreFlush);
    }
    const u32 guest = dma_pusher.memory_manager.Read<u32>(address);
    if (!device) {
        return guest;
    }
    return Nx2MirrorShadow::Get().Read(*device, guest).value_or(guest);
}

// With the GPU thread split (gpu_split.h) the rasterizer belongs to the back thread: work that
// only feeds it runs there in order; reads of what it wrote first wait for it (DrainBack).
template <typename Func>
void OnBack(const Func& func,
            const std::source_location location = std::source_location::current()) {
    if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
        split.Defer(func, 0, location);
    } else {
        func();
    }
}

void DrainBack() {
    VideoCommon::GpuSplit::Get().Drain(VideoCommon::DrainSite::Acquire);
}

/// EDEN_NX2_REPORT_ACQUIRE=1 turns it on. Off by default since build-19: in Ocarina of Time it
/// never applied (run 12: 0 of ~1.3 M acquires) and its writer wait cost the front time; the
/// acquire line then counts why it bails out.
bool ReportAcquireEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_REPORT_ACQUIRE");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

/// With the GPU thread split and the macros on the front: whether the semaphore word at
/// `address`, as the back will see it once it ran what is queued, is known without draining it
/// and meets `met` (#956: guest memory with the queued writes the front knows on top, after the
/// queued compute and copy writers ran, for report memory only). For the acquires
/// CheckWithoutDrain sends to the drained path, mostly equal and mask ones (Ocarina of Time: over
/// 99% of those that drain). Words the semaphore mirror shadows and fence releases the back ran
/// that wait for their host fence take the drained path. EDEN_NX2_PERF_VERIFY drains anyway and
/// reports an acquire the drained read doesn't meet (EDEN_NX2_ACQUIRE_VERIFY counts it).
template <typename Met>
bool AcquireMetQueued(DmaPusher& dma_pusher, GPUVAddr address, const Met& met) {
    auto& split = VideoCommon::GpuSplit::Get();
    if (!ReportAcquireEnabled() || !split.Active() || !split.FrontMacros()) {
        return false;
    }
    auto& bails = VideoCommon::g_acquire_stats.report_bails;
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address);
    if (!device || Nx2MirrorShadow::Get().Diverts(*device) ||
        Nx2SemaphoreWaiters::Get().HasDelayedRelease(*device)) {
        VideoCommon::Count(bails[6]);
        return false;
    }
    u32 word{};
    if (u32 bail = 0; !ReadReportWordsQueued(dma_pusher.memory_manager, split, address, &word, 1,
                                             VideoCommon::DrainSite::Acquire, &bail)) {
        VideoCommon::Count(bails[(std::min)(bail, 5U)]);
        return false;
    }
    if (!met(word)) {
        VideoCommon::Count(bails[7]);
        return false;
    }
    if (const auto shadowed = Nx2MirrorShadow::Get().Read(*device, word);
        shadowed && !met(*shadowed)) {
        VideoCommon::Count(bails[8]);
        return false;
    }
    if (VideoCommon::Nx2Perf::Verify()) {
        DrainBack();
        if (const u32 drained = ReadSemaphoreWord(dma_pusher, address); !met(drained)) {
            static std::atomic<u32> reports{};
            if (reports.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_ERROR(HW_GPU,
                          "Nx2Perf acquire: semaphore {:#x} read {:#x} without draining, {:#x} "
                          "drained",
                          address, word, drained);
            }
            return false;
        }
    }
    // A drain would have run the work queued before the acquire: GPU-written command lists
    // fetched after it wait for that work (WaitForSync) instead.
    split.NoteSync();
    VideoCommon::Count(VideoCommon::g_acquire_stats.by_report_read);
    return true;
}

// The fence manager records into the scheduler and downloads the CPU's writes over GPU data:
// the back's state, which a drained back still uses from its idle hook (it submits).
void ReleaseFences(DmaPusher& dma_pusher) {
    VideoCommon::GpuSplit::Get().Run(
        [rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); },
        VideoCommon::DrainSite::FenceRelease);
}

// A release was just sent to the back: tell the front what it writes (CheckWithoutDrain).
void NoteRelease(DmaPusher& dma_pusher, GPUVAddr address, u32 payload, bool four_words) {
    auto& split = VideoCommon::GpuSplit::Get();
    if (!split.Active()) {
        return;
    }
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address);
    if (!device) {
        return;
    }
    split.NoteSemaphoreWrite(*device, payload, true);
    if (four_words) {
        // The payload as 64 bits (a timestamp follows).
        split.NoteSemaphoreWrite(*device + 4, 0, true);
    }
}

// Whether the back writes a payload release to memory when it runs it (CounterReport): below
// High GPU accuracy, with fences that run their operations at once (FenceManager::SignalFence).
bool ReleasesLandAtOnce() {
    const bool delay_fence =
        Settings::IsGPUFenceBehaviorDefault()
            ? Settings::IsGPULevelHigh()
            : Settings::IsGPUFenceBehaviorBalanced() || Settings::IsGPUFenceBehaviorAccurate();
    return !delay_fence && !Settings::IsGPULevelHigh();
}

enum class AcquireCheck {
    Met,         ///< By the word in memory
    MetByQueued, ///< By a release sent to the word that the back has not run yet
    NotMet,      ///< Nothing sent to the word can meet it yet
    Drain,       ///< Drain the split and read the word as before
};

// Checks a greater-or-equal acquire without draining the split. The words Switch 2 syncs on only
// advance (NVN2 even releases QMD completions as maxima, KeplerCompute::ProcessNativeMethod), so
// a value that meets one now still does once the back ran the queued work. Only the last release
// the front sent to the word decides instead, when the back may not have run it yet.
// EDEN_NX2_EQUAL_ACQUIRE (on by default since build-19; =0 turns it off): equal and mask acquires
// take the same
// check. The front's queued-release tracking says what the word holds once the back ran the work
// sent before the acquire, whatever the operation; only the ">= stays met" argument is specific to
// greater-or-equal. In Ocarina of Time over 99% of the acquires that still drain are equal or mask
// (Castle Town: ~300 ms of every second of the split's front). Check with EDEN_NX2_ACQUIRE_VERIFY=1
// (run 12c, build-18 in Castle Town: 0 of ~500k acquires passed that a drain doesn't meet).
// Read on first use: nx2_switches.txt sets the variable after static initialisation.
bool EqualAcquireFast() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_EQUAL_ACQUIRE");
        const bool on = value == nullptr || value[0] != '0';
        LOG_INFO(HW_GPU, "EDEN_NX2_EQUAL_ACQUIRE={}: equal and mask acquires {}", on ? 1 : 0,
                 on ? "are checked without draining the split when they can be"
                    : "drain the split");
        return on;
    }();
    return enabled;
}

bool SuspendNotMet() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_SUSPEND_NOT_MET");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

AcquireCheck CheckWithoutDrain(DmaPusher& dma_pusher, const Puller::PendingAcquire& acquire) {
    auto& split = VideoCommon::GpuSplit::Get();
    auto& stats = VideoCommon::g_acquire_stats;
    if (!acquire.WaitsToReach() && !EqualAcquireFast()) {
        VideoCommon::Count(stats.drain_operation);
        return AcquireCheck::Drain;
    }
    if (!split.Active() || !split.FrontMacros() || !ReleasesLandAtOnce()) {
        VideoCommon::Count(stats.drain_settings);
        return AcquireCheck::Drain;
    }
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(acquire.address);
    // A word the buffer cache may hold newer than memory is read through the back.
    if (!device || Nx2MayBeGpuWritten(*device, sizeof(u32))) {
        VideoCommon::Count(stats.drain_word);
        return AcquireCheck::Drain;
    }
    // A fence-flagged release the back ran but whose write waits for its fence: memory still
    // holds the old value, which an equal or mask acquire could take as met.
    if (!acquire.WaitsToReach() && Nx2SemaphoreWaiters::Get().HasDelayedRelease(*device)) {
        VideoCommon::Count(stats.drain_queued);
        return AcquireCheck::Drain;
    }
    if (const auto queued = split.QueuedSemaphoreWriteAt(*device)) {
        if (queued->known && acquire.Matches(queued->payload)) {
            return AcquireCheck::MetByQueued;
        }
        VideoCommon::Count(stats.drain_queued);
        return AcquireCheck::Drain;
    }
    return acquire.Matches(dma_pusher.memory_manager.Read<u32>(acquire.address))
               ? AcquireCheck::Met
               : AcquireCheck::NotMet;
}

// EDEN_NX2_ACQUIRE_VERIFY=1: drains anyway and counts where the word after the drain disagrees
// with what CheckWithoutDrain decided.
void VerifyWithoutDrain(DmaPusher& dma_pusher, const Puller::PendingAcquire& acquire,
                        AcquireCheck decided) {
    if (!VideoCommon::g_acquire_verify) {
        return;
    }
    DrainBack();
    const u32 word = ReadSemaphoreWord(dma_pusher, acquire.address);
    auto& stats = VideoCommon::g_acquire_stats;
    if (acquire.Matches(word)) {
        if (decided == AcquireCheck::NotMet) {
            VideoCommon::Count(stats.late);
        }
        return;
    }
    if (decided == AcquireCheck::MetByQueued) {
        // A fence-flagged release writes memory once the host GPU passed it.
        VideoCommon::Count(stats.unlanded);
        return;
    }
    if (decided != AcquireCheck::Met) {
        return;
    }
    VideoCommon::Count(stats.mismatches);
    if (static std::atomic<u32> logged{}; logged.fetch_add(1) < 16) {
        LOG_WARNING(HW_GPU,
                    "Acquire verify: channel {} passed semaphore {:#x} (op {}, {:#x}) without a "
                    "drain, the word is {:#x} after it",
                    dma_pusher.channel_state.bind_id, acquire.address, acquire.operation,
                    acquire.value, word);
    }
}

} // Anonymous namespace

bool Puller::PendingAcquire::Matches(u32 current) const {
    switch (static_cast<GpuSemaphoreOperation>(operation)) {
    case GpuSemaphoreOperation::AcquireEqual:
        return current == value;
    case GpuSemaphoreOperation::AcquireGequal:
        return current >= value;
    case GpuSemaphoreOperation::AcquireMask:
        return (current & value) != 0;
    default:
        return true;
    }
}

bool Puller::PendingAcquire::WaitsToReach() const {
    return static_cast<GpuSemaphoreOperation>(operation) == GpuSemaphoreOperation::AcquireGequal;
}

void Puller::NoteValveDrain(DmaPusher& dma_pusher, u32 before, u32 after) {
    static std::atomic<u64> drains{}, met_before{}, met_after{}, latched{}, unmet{};
    static std::atomic<u64> waited_us{};
    static std::atomic<u32> samples{};
    static std::mutex report_mutex;
    static auto next_report = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    const bool was_met = pending_acquire->Matches(before);
    const bool is_met = pending_acquire->Matches(after);
    const bool is_latched = Nx2SemaphoreWaiters::Get().Latched(this);
    const auto waited = std::chrono::steady_clock::now() - pending_acquire->since;
    drains.fetch_add(1, std::memory_order_relaxed);
    (was_met ? met_before : is_met ? met_after : is_latched ? latched : unmet)
        .fetch_add(1, std::memory_order_relaxed);
    waited_us.fetch_add(
        static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(waited).count()),
        std::memory_order_relaxed);
    if (samples.fetch_add(1, std::memory_order_relaxed) < 48) {
        const auto device = dma_pusher.memory_manager.GpuToCpuAddress(pending_acquire->address);
        LOG_INFO(HW_GPU,
                 "Acquire valve drain: channel {} semaphore {:#x} (device {:#x}) op {} value {:#x}; "
                 "word {:#x} before the drain, {:#x} after, latched {}; waited {} us",
                 dma_pusher.channel_state.bind_id, pending_acquire->address, device.value_or(0),
                 pending_acquire->operation, pending_acquire->value, before, after, is_latched,
                 std::chrono::duration_cast<std::chrono::microseconds>(waited).count());
    }
    std::scoped_lock lock{report_mutex};
    if (const auto now = std::chrono::steady_clock::now(); now >= next_report) {
        next_report = now + std::chrono::seconds{10};
        const u64 total = drains.exchange(0);
        LOG_INFO(HW_GPU,
                 "Acquire valve drains in 10 s: {}; word already met before the drain {}, met "
                 "after it {}, latched {}, still unmet {}; average wait at the drain {} us",
                 total, met_before.exchange(0), met_after.exchange(0), latched.exchange(0),
                 unmet.exchange(0), total != 0 ? waited_us.exchange(0) / total : 0);
    }
}

bool Puller::CanResumeAcquire(DmaPusher& dma_pusher) {
    if (!pending_acquire) {
        return true;
    }
    bool valve_drain = false;
    auto& stats = VideoCommon::g_acquire_stats;
    switch (const AcquireCheck check = CheckWithoutDrain(dma_pusher, *pending_acquire); check) {
    case AcquireCheck::MetByQueued:
        // What the front reads after the acquire waits for the work sent before the release.
        VideoCommon::GpuSplit::Get().NoteSync();
        VideoCommon::Count(stats.by_queued);
        [[fallthrough]];
    case AcquireCheck::Met:
        VideoCommon::Count(stats.passed);
        VerifyWithoutDrain(dma_pusher, *pending_acquire, check);
        return true;
    case AcquireCheck::NotMet: {
        // A release the back ran latches the waiter. Otherwise poll again later, without waiting
        // for the back: until it runs one, nothing sent to the word meets it. Work only a drain
        // finishes (the back's idle hook submits what the host GPU writes) still lands: a waiting
        // acquire drains at most once per 1 ms (PantACRO4life's valve, b837e742c4), and the drain
        // reports waits over 2 s.
        if (Nx2SemaphoreWaiters::Get().Latched(this)) {
            VideoCommon::Count(stats.latched);
            return true;
        }
        // EDEN_NX2_ACQUIRE_VALVE_US: how long an unmet acquire polls before it drains (1000).
        static const auto valve = [] {
            const char* value = std::getenv("EDEN_NX2_ACQUIRE_VALVE_US");
            const long long us = value != nullptr ? std::strtoll(value, nullptr, 10) : 1000;
            return std::chrono::microseconds{std::clamp<long long>(us, 0, 100'000)};
        }();
        const auto now = std::chrono::steady_clock::now();
        if (now - pending_acquire->last_drain < valve) {
            VideoCommon::Count(stats.polls);
            VerifyWithoutDrain(dma_pusher, *pending_acquire, check);
            return false;
        }
        pending_acquire->last_drain = now;
        VideoCommon::Count(stats.valve_drains);
        valve_drain = true;
        break;
    }
    case AcquireCheck::Drain:
        if (AcquireMetQueued(dma_pusher, pending_acquire->address,
                             [&](u32 word) { return pending_acquire->Matches(word); })) {
            VideoCommon::Count(stats.passed);
            VerifyWithoutDrain(dma_pusher, *pending_acquire, AcquireCheck::Met);
            return true;
        }
        break;
    }
    VideoCommon::Count(stats.drained);
    // EDEN_NX2_ACQUIRE_VALVE_LOG=1: what the valve's drains find (the word before and after).
    static const bool valve_log = [] {
        const char* value = std::getenv("EDEN_NX2_ACQUIRE_VALVE_LOG");
        return value != nullptr && value[0] == '1';
    }();
    u32 before_drain = 0;
    if (valve_log && valve_drain) {
        before_drain = dma_pusher.memory_manager.Read<u32>(pending_acquire->address);
    }
    // The release may still be queued for the split GPU thread's back half (and the caches
    // ReadSemaphoreWord asks are its state).
    DrainBack();
    const u32 current = ReadSemaphoreWord(dma_pusher, pending_acquire->address);
    if (valve_log && valve_drain) {
        NoteValveDrain(dma_pusher, before_drain, current);
    }
    if (Nx2SemaphoreWaiters::Get().Latched(this)) {
        if (static std::atomic<u32> rescued{};
            !pending_acquire->Matches(current) && rescued.fetch_add(1) < 64) {
            LOG_INFO(HW_GPU, "Channel {} met semaphore {:#x} value {:#x} at its release (now {:#x})",
                     dma_pusher.channel_state.bind_id, pending_acquire->address,
                     pending_acquire->value, current);
        }
        return true;
    }
    if (pending_acquire->Matches(current)) {
        return true;
    }
    // With delayed fence operations (Balanced or Accurate fences, accurate GPU), a release
    // queued after the acquire released the fences waits in the caches for the next fence. With
    // every channel waiting none comes: release them again now and then while waiting.
    const bool delayed = Settings::IsGPUFenceBehaviorDefault()
                             ? Settings::IsGPULevelHigh()
                             : Settings::IsGPUFenceBehaviorBalanced() ||
                                   Settings::IsGPUFenceBehaviorAccurate();
    if (const auto now = std::chrono::steady_clock::now();
        delayed && now - pending_acquire->last_fences >= std::chrono::milliseconds{1}) {
        pending_acquire->last_fences = now;
        ReleaseFences(dma_pusher);
        if (Nx2SemaphoreWaiters::Get().Latched(this) ||
            pending_acquire->Matches(ReadSemaphoreWord(dma_pusher, pending_acquire->address))) {
            return true;
        }
    }
    if (!pending_acquire->reported &&
        std::chrono::steady_clock::now() - pending_acquire->since > std::chrono::seconds{2}) {
        pending_acquire->reported = true;
        const auto device = dma_pusher.memory_manager.GpuToCpuAddress(pending_acquire->address);
        LOG_WARNING(HW_GPU, "Channel {} waits on semaphore {:#x} (op {}, {:#x}, now {:#x}) for 2 s; "
                    "device {:#x}, buffer cache holds GPU data: {}, words {:08x} {:08x} {:08x} {:08x}",
                    dma_pusher.channel_state.bind_id, pending_acquire->address,
                    pending_acquire->operation, pending_acquire->value, current, device.value_or(0),
                    device && dma_pusher.rasterizer->MustFlushRegion(*device, 16, VideoCommon::CacheType::All),
                    dma_pusher.memory_manager.Read<u32>(pending_acquire->address & ~GPUVAddr{0xf}),
                    dma_pusher.memory_manager.Read<u32>((pending_acquire->address & ~GPUVAddr{0xf}) + 4),
                    dma_pusher.memory_manager.Read<u32>((pending_acquire->address & ~GPUVAddr{0xf}) + 8),
                    dma_pusher.memory_manager.Read<u32>((pending_acquire->address & ~GPUVAddr{0xf}) + 12));
    }
    return false;
}

bool Puller::ResumeAcquire(DmaPusher& dma_pusher) {
    if (!CanResumeAcquire(dma_pusher)) {
        return false;
    }
    // Only a resume from a suspension counts: DmaPusher::Step calls this before every entry, and
    // counting those made the count odd ("suspended") at random, so submit-time snapshots were
    // thrown away for about half of all entries (CommandList::snapshot_suspensions).
    if (pending_acquire) {
        VideoCommon::Nx2AcquireTest::Stats::Get().Acquire(std::chrono::steady_clock::now() -
                                                          pending_acquire->since);
        pending_acquire.reset();
        dma_pusher.suspensions.fetch_add(1, std::memory_order_release);
        Nx2SemaphoreWaiters::Get().Remove(this);
    }
    return true;
}

void Puller::ProcessBindMethod(DmaPusher& dma_pusher, const MethodCall& method_call) {
    LOG_DEBUG(HW_GPU, "Binding subchannel {} to engine {:#x}", method_call.subchannel, method_call.argument);
    u32 engine = method_call.argument;
    if ((engine & ~Gf100BindClassMask) != 0 && (engine & ~Gf100BindValidMask) == 0) {
        engine &= Gf100BindClassMask;
    }
    // Switch 2 binds Ampere classes; run them on the Maxwell engines (same method layout for
    // most state). Class 0 unbinds the subchannel.
    switch (engine & 0xFFFF) {
    case 0xC797: // AMPERE_B
        engine = static_cast<u32>(EngineID::MAXWELL_B);
        break;
    case 0xC7C0: // AMPERE_COMPUTE_B
        engine = static_cast<u32>(EngineID::KEPLER_COMPUTE_B);
        break;
    case 0xC7B5: // AMPERE_DMA_COPY_B
        engine = static_cast<u32>(EngineID::MAXWELL_DMA_COPY_A);
        break;
    case 0x0:
        if (g_nx2_ampere) {
            LOG_DEBUG(HW_GPU, "Switch 2 unbinds subchannel {}", method_call.subchannel);
            return;
        }
        break;
    default:
        break;
    }

    const auto engine_id = static_cast<EngineID>(engine);
    bound_engines[method_call.subchannel] = engine_id;
    switch (engine_id) {
    case EngineID::FERMI_TWOD_A:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->fermi_2d, method_call.subchannel, EngineTypes::Fermi2D);
        break;
    case EngineID::MAXWELL_B:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->maxwell_3d, method_call.subchannel, EngineTypes::Maxwell3D);
        break;
    case EngineID::KEPLER_COMPUTE_B:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->kepler_compute, method_call.subchannel, EngineTypes::KeplerCompute);
        break;
    case EngineID::MAXWELL_DMA_COPY_A:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->maxwell_dma, method_call.subchannel, EngineTypes::MaxwellDMA);
        break;
    case EngineID::KEPLER_INLINE_TO_MEMORY_B:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->kepler_memory, method_call.subchannel, EngineTypes::KeplerMemory);
        break;
    case EngineID::NV01_TIMER:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->nv01_timer, method_call.subchannel, EngineTypes::Nv01Timer);
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented engine {:04X}", engine_id);
        break;
    }
}

void Puller::ProcessFenceActionMethod(DmaPusher& dma_pusher) {
    switch (regs.fence_action.op) {
    case Puller::FenceOperation::Acquire:
        // UNIMPLEMENTED_MSG("Channel Scheduling pending.");
        // WaitFence(regs.fence_action.syncpoint_id, regs.fence_value);
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); });
        VideoCommon::GpuSplit::Get().NoteSync();
        break;
    case Puller::FenceOperation::Increment:
        OnBack([rasterizer = dma_pusher.rasterizer, id = static_cast<u32>(regs.fence_action.syncpoint_id.Value())] {
            rasterizer->SignalSyncPoint(id);
        });
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented operation {}", regs.fence_action.op.Value());
        break;
    }
}


void Puller::ProcessSemaphoreTriggerMethod(DmaPusher& dma_pusher) {
    const auto semaphoreOperationMask = 0xF;
    const auto op = GpuSemaphoreOperation(regs.semaphore_trigger & semaphoreOperationMask);
    // EDEN_NX2_SEM_WATCH=<gpu address>: log every release/acquire on the 16 bytes there.
    static const GPUVAddr watch = [] {
        const char* value = std::getenv("EDEN_NX2_SEM_WATCH");
        return value ? GPUVAddr{std::strtoull(value, nullptr, 16)} : GPUVAddr{};
    }();
    if (watch != 0 && (regs.semaphore_address.SemaphoreAddress() & ~GPUVAddr{0xf}) == watch) {
        LOG_INFO(HW_GPU, "Semaphore watch: channel {} as {} device {:#x} op {} {:#x} value {:#x} (now {:#x})",
                 dma_pusher.channel_state.bind_id, dma_pusher.memory_manager.GetID(),
                 dma_pusher.memory_manager.GpuToCpuAddress(regs.semaphore_address.SemaphoreAddress()).value_or(0),
                 static_cast<u32>(op),
                 regs.semaphore_address.SemaphoreAddress(), regs.semaphore_sequence,
                 dma_pusher.memory_manager.Read<u32>(regs.semaphore_address.SemaphoreAddress()));
    }
    if (op == GpuSemaphoreOperation::WriteLong) {
        const GPUVAddr sequence_address{regs.semaphore_address.SemaphoreAddress()};
        const u32 payload = regs.semaphore_sequence;
        // RELEASE_SIZE (bit 24) set means a 4-byte release without the timestamp; writing 16
        // bytes there clobbers the words next to it (Switch 2 packs semaphores tightly).
        const auto properties = g_nx2_ampere && (regs.semaphore_trigger & (1U << 24)) != 0
                                    ? VideoCommon::QueryPropertiesFlags::IsAFence
                                    : VideoCommon::QueryPropertiesFlags::HasTimeout;
        const GPUVAddr mirror = g_nx2_ampere ? Nx2SemaphoreMirror(sequence_address) : 0;
        if (mirror != 0) {
            LOG_DEBUG(HW_GPU, "Switch 2 semaphore {:#x}={:#x} mirrored to {:#x}", sequence_address, payload, mirror);
        }
        OnBack([rasterizer = dma_pusher.rasterizer, sequence_address, mirror, properties, payload] {
            rasterizer->Query(sequence_address, VideoCommon::QueryType::Payload, properties, payload, 0);
            if (mirror != 0) {
                rasterizer->Query(mirror, VideoCommon::QueryType::Payload,
                                  properties | VideoCommon::QueryPropertiesFlags::Nx2Mirror, payload, 0);
            }
        });
        NoteRelease(dma_pusher, sequence_address, payload,
                    properties == VideoCommon::QueryPropertiesFlags::HasTimeout);
    } else if (g_nx2_ampere) {
        // Switch 2 waits on semaphores the CPU (or another channel) releases, e.g. once it has
        // written the commands this channel runs next: suspend the channel until the value is
        // met instead of reading ahead (approach from changes.patch).
        PendingAcquire acquire{
            .address = regs.semaphore_address.SemaphoreAddress(),
            .value = regs.semaphore_sequence,
            .operation = static_cast<u32>(op),
            .since = std::chrono::steady_clock::now(),
            .last_fences = std::chrono::steady_clock::now(),
            .last_drain = std::chrono::steady_clock::now(),
            .reported = false,
        };
        if (op != GpuSemaphoreOperation::AcquireEqual && op != GpuSemaphoreOperation::AcquireGequal &&
            op != GpuSemaphoreOperation::AcquireMask) {
            LOG_ERROR(HW_GPU, "Invalid semaphore operation {}", static_cast<u32>(op));
            return;
        }
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireChecks, 1);
        auto& stats = VideoCommon::g_acquire_stats;
        const AcquireCheck without_drain = CheckWithoutDrain(dma_pusher, acquire);
        if (without_drain == AcquireCheck::Met || without_drain == AcquireCheck::MetByQueued) {
            if (without_drain == AcquireCheck::MetByQueued) {
                // What the front reads after the acquire waits for the work sent before the
                // release (GpuSplit::NoteSync).
                VideoCommon::GpuSplit::Get().NoteSync();
                VideoCommon::Count(stats.by_queued);
            }
            VideoCommon::Count(stats.passed);
            VerifyWithoutDrain(dma_pusher, acquire, without_drain);
            return;
        } else if (without_drain == AcquireCheck::Drain &&
                   AcquireMetQueued(dma_pusher, acquire.address,
                                    [&](u32 word) { return acquire.Matches(word); })) {
            VideoCommon::Count(stats.passed);
            VerifyWithoutDrain(dma_pusher, acquire, AcquireCheck::Met);
            return;
        }
        // Not met by what the front knows of the word: suspend the channel at once, as a poll of
        // a pending acquire does (CanResumeAcquire: latched by the release, polled without a
        // drain, drained by the 1 ms valve), instead of draining the back first. With the
        // command-list writer wait narrowed (#805) the front runs further ahead, and these first
        // checks drained ~2,700 times a second in Castle Town (run 13b, ~430 ms/s of the front).
        // EDEN_NX2_SUSPEND_NOT_MET=0 drains first as before.
        if (without_drain == AcquireCheck::NotMet && SuspendNotMet()) {
            auto& waiters = Nx2SemaphoreWaiters::Get();
            if (const auto device = dma_pusher.memory_manager.GpuToCpuAddress(acquire.address);
                device && !waiters.Add(this, *device, acquire.value, acquire.operation)) {
                if (static std::atomic<u32> full{}; full.fetch_add(1) < 8) {
                    LOG_ERROR(HW_GPU, "Channel {} waits on semaphore {:#x} unlatched: {} channels wait",
                              dma_pusher.channel_state.bind_id, acquire.address,
                              Nx2SemaphoreWaiters::MaxWaiters);
                }
            }
            if (waiters.Latched(this)) {
                waiters.Remove(this);
                VideoCommon::Count(stats.latched);
                return;
            }
            VerifyWithoutDrain(dma_pusher, acquire, without_drain);
            VideoCommon::Count(stats.suspended);
            pending_acquire = acquire;
            dma_pusher.suspensions.fetch_add(1, std::memory_order_release);
            VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireWaits, 1);
            return;
        }
        VideoCommon::Count(stats.drained);
        DrainBack();
        if (acquire.Matches(ReadSemaphoreWord(dma_pusher, acquire.address))) {
            return;
        }
        // Registered before the fences run, so a release among them that meets the value counts
        // even if a later one overwrites it.
        auto& waiters = Nx2SemaphoreWaiters::Get();
        if (const auto device = dma_pusher.memory_manager.GpuToCpuAddress(acquire.address);
            device && !waiters.Add(this, *device, acquire.value, acquire.operation)) {
            if (static std::atomic<u32> full{}; full.fetch_add(1) < 8) {
                LOG_ERROR(HW_GPU, "Channel {} waits on semaphore {:#x} unlatched: {} channels wait",
                          dma_pusher.channel_state.bind_id, acquire.address,
                          Nx2SemaphoreWaiters::MaxWaiters);
            }
        }
        ReleaseFences(dma_pusher);
        if (!waiters.Latched(this) && !acquire.Matches(ReadSemaphoreWord(dma_pusher, acquire.address))) {
            pending_acquire = acquire;
            dma_pusher.suspensions.fetch_add(1, std::memory_order_release);
            VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireWaits, 1);
        } else {
            waiters.Remove(this);
        }
    } else {
        DrainBack();
        do {
            const u32 word = dma_pusher.memory_manager.Read<u32>(regs.semaphore_address.SemaphoreAddress());
            regs.acquire_source = true;
            regs.acquire_value = regs.semaphore_sequence;
            if (op == GpuSemaphoreOperation::AcquireEqual) {
                regs.acquire_active = true;
                regs.acquire_mode = false;
                if (word != regs.acquire_value) {
                    ReleaseFences(dma_pusher);
                    continue;
                }
            } else if (op == GpuSemaphoreOperation::AcquireGequal) {
                regs.acquire_active = true;
                regs.acquire_mode = true;
                if (word < regs.acquire_value) {
                    ReleaseFences(dma_pusher);
                    continue;
                }
            } else if (op == GpuSemaphoreOperation::AcquireMask) {
                if (word && regs.semaphore_sequence == 0) {
                    ReleaseFences(dma_pusher);
                    continue;
                }
            } else {
                LOG_ERROR(HW_GPU, "Invalid semaphore operation");
            }
        } while (false);
    }
}

void Puller::ProcessSemaphoreRelease(DmaPusher& dma_pusher) {
    const GPUVAddr sequence_address{regs.semaphore_address.SemaphoreAddress()};
    const u32 payload = regs.semaphore_release;
    OnBack([rasterizer = dma_pusher.rasterizer, sequence_address, payload] {
        rasterizer->Query(sequence_address, VideoCommon::QueryType::Payload,
                          VideoCommon::QueryPropertiesFlags::IsAFence, payload, 0);
    });
    NoteRelease(dma_pusher, sequence_address, payload, false);
}

void Puller::ProcessSemaphoreAcquire(DmaPusher& dma_pusher) {
    DrainBack();
    // The CPU may rewrite what the next draws read once it sees the release this waits for.
    VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Fence);
    u32 word = dma_pusher.memory_manager.Read<u32>(regs.semaphore_address.SemaphoreAddress());
    const auto value = regs.semaphore_acquire;
    while (word != value) {
        regs.acquire_active = true;
        regs.acquire_value = value;
        ReleaseFences(dma_pusher);
        word = dma_pusher.memory_manager.Read<u32>(regs.semaphore_address.SemaphoreAddress());
        // TODO(kemathe73) figure out how to do the acquire_timeout
        regs.acquire_mode = false;
        regs.acquire_source = false;
    }
}

// Ampere SEM_EXECUTE (method 0x1B): operation in bits 2:0 on the semaphore at
// SEM_ADDR_HI:SEM_ADDR_LO (methods 0x18/0x17) with payload SEM_PAYLOAD_LO (0x19).
void Puller::ProcessSemaphoreExecute(DmaPusher& dma_pusher, u32 argument) {
    const GPUVAddr address = (GPUVAddr{regs.reg_array[0x18] & 0x1FFFFFF} << 32) |
                             (regs.reg_array[0x17] & ~3U);
    const u32 payload = regs.reg_array[0x19];
    const u32 operation = argument & 7;
    if (operation == 1) { // RELEASE
        OnBack([rasterizer = dma_pusher.rasterizer, address, payload] {
            rasterizer->Query(address, VideoCommon::QueryType::Payload,
                              VideoCommon::QueryPropertiesFlags::IsAFence, payload, 0);
        });
        NoteRelease(dma_pusher, address, payload, false);
        return;
    }
    // Acquires: check once after flushing pending fences; never block the GPU thread on them.
    DrainBack();
    VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Fence);
    ReleaseFences(dma_pusher);
    u32 word = dma_pusher.memory_manager.Read<u32>(address);
    if (const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address)) {
        word = Nx2MirrorShadow::Get().Read(*device, word).value_or(word);
    }
    const bool ready = operation == 0   ? word == payload
                       : operation == 2 ? word >= payload
                       : operation == 3 ? static_cast<s32>(word - payload) >= 0
                       : operation == 4 ? (word & payload) != 0
                       : operation == 5 ? (word & payload) == 0
                                        : true;
    if (!ready) {
        static std::atomic<u32> reported{};
        if (reported.fetch_add(1) < 8) {
            LOG_WARNING(HW_GPU, "SEM_EXECUTE op {} at {:#x}: {:#x} vs {:#x} not met, continuing",
                        operation, address, word, payload);
        }
    }
}

// A semaphore release queued for the split's back: the front's macro DMA reads of those bytes
// wait for it (Maxwell3D::FrontDmaRead).
// A fence release (`fence_payload`): reads overlay its payload while it is queued.
static void NoteSemaphoreWrite(DmaPusher& dma_pusher, GPUVAddr address,
                               std::optional<u32> fence_payload) {
    auto& split = VideoCommon::GpuSplit::Get();
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address);
    if (!device) {
        return; // Queries write nothing at an unmapped address.
    } else if (fence_payload) {
        split.NoteFenceWrite(*device, *fence_payload);
    } else {
        // Written when its fence is signalled: reads wait until the back ran it and it was.
        split.NoteAsyncWrite(*device, 16);
        split.NoteMemoryWrite(*device, 16, "semaphore (timed)");
    }
}

/// Calls a GPU puller method.
void Puller::CallPullerMethod(DmaPusher& dma_pusher, const MethodCall& method_call) {
    static const bool nx2_trace = std::getenv("EDEN_NX2_PULLER_TRACE") != nullptr;
    if (nx2_trace) {
        LOG_INFO(HW_GPU, "nx2 puller sub={} method={:#x} arg={:#x}", method_call.subchannel,
                 method_call.method, method_call.argument);
    }
    regs.reg_array[method_call.method] = method_call.argument;
    const auto method = static_cast<BufferMethods>(method_call.method);

    switch (method) {
    case BufferMethods::BindObject: {
        ProcessBindMethod(dma_pusher, method_call);
        break;
    }
    case BufferMethods::Nop:
    case BufferMethods::SemaphoreAddressHigh:
    case BufferMethods::SemaphoreAddressLow:
    case BufferMethods::SemaphoreSequencePayload:
    case BufferMethods::SyncpointPayload:
    case BufferMethods::WrcacheFlush:
        break;
    case BufferMethods::RefCnt:
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->SignalReference(); });
        break;
    case BufferMethods::SyncpointOperation:
        ProcessFenceActionMethod(dma_pusher);
        break;
    case BufferMethods::WaitForIdle:
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->WaitForIdle(); });
        VideoCommon::GpuSplit::Get().NoteSync();
        break;
    case BufferMethods::SemaphoreOperation: {
        ProcessSemaphoreTriggerMethod(dma_pusher);
        if (static_cast<GpuSemaphoreOperation>(regs.semaphore_trigger & 0xF) ==
            GpuSemaphoreOperation::WriteLong) {
            const bool fence = g_nx2_ampere && (regs.semaphore_trigger & (1U << 24)) != 0;
            NoteSemaphoreWrite(dma_pusher, regs.semaphore_address.SemaphoreAddress(),
                               fence ? std::optional<u32>{regs.semaphore_sequence} : std::nullopt);
        }
        break;
    }
    case BufferMethods::NonStallInterrupt: {
        LOG_ERROR(HW_GPU, "Special puller engine method NonStallInterrupt not implemented");
        break;
    }
    case BufferMethods::MemOpA: {
        LOG_TRACE(HW_GPU, "Memory Operation A");
        break;
    }
    case BufferMethods::MemOpB: {
        // Implement this better.
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->InvalidateGPUCache(); });
        break;
    }
    case BufferMethods::MemOpC:
    case BufferMethods::MemOpD: {
        LOG_TRACE(HW_GPU, "Memory Operation C,D");
        break;
    }
    case BufferMethods::SemaphoreAcquire: {
        // On Ampere host classes 0x17..0x1A are SEM_ADDR_LO/HI and SEM_PAYLOAD_LO/HI.
        if (!g_nx2_ampere) {
            ProcessSemaphoreAcquire(dma_pusher);
        }
        break;
    }
    case BufferMethods::SemaphoreRelease: {
        if (g_nx2_ampere) {
            ProcessSemaphoreExecute(dma_pusher, method_call.argument);
        } else {
            ProcessSemaphoreRelease(dma_pusher);
        }
        if (g_nx2_ampere && (method_call.argument & 7) == 1) {
            NoteSemaphoreWrite(dma_pusher,
                               (GPUVAddr{regs.reg_array[0x18] & 0x1FFFFFF} << 32) |
                                   (regs.reg_array[0x17] & ~3U),
                               regs.reg_array[0x19]);
        } else if (!g_nx2_ampere) {
            NoteSemaphoreWrite(dma_pusher, regs.semaphore_address.SemaphoreAddress(),
                               std::nullopt);
        }
        break;
    }
    case BufferMethods::Yield: {
        // TODO(Kmather73): Research and implement this method.
        LOG_ERROR(HW_GPU, "Special puller engine method Yield not implemented");
        break;
    }
    default:
        LOG_ERROR(HW_GPU, "Special puller engine method {:X} not implemented", method);
        if (static bool dumped = false; g_nx2_ampere && !dumped) {
            dumped = true;
            dma_pusher.DumpCurrentEntry("unknown host method");
        }
        break;
    }
}

/// Calls a GPU engine method.
static bool Nx2MethodTrace() {
    static const bool trace = std::getenv("EDEN_NX2_METHOD_TRACE") != nullptr;
    return trace;
}

void Puller::CallEngineMethod(DmaPusher& dma_pusher, const MethodCall& method_call) {
    const EngineID engine = bound_engines[method_call.subchannel];
    if (Nx2MethodTrace()) {
        LOG_INFO(HW_GPU, "nx2 method sub={} engine={:04X} method={:#x} arg={:#x}",
                 method_call.subchannel, static_cast<u32>(engine), method_call.method,
                 method_call.argument);
    }
    switch (engine) {
    case EngineID::FERMI_TWOD_A:
        dma_pusher.channel_state.payload->fermi_2d.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::MAXWELL_B:
        dma_pusher.channel_state.payload->maxwell_3d.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::KEPLER_COMPUTE_B:
        dma_pusher.channel_state.payload->kepler_compute.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::MAXWELL_DMA_COPY_A:
        dma_pusher.channel_state.payload->maxwell_dma.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::KEPLER_INLINE_TO_MEMORY_B:
        dma_pusher.channel_state.payload->kepler_memory.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::NV01_TIMER:
        dma_pusher.channel_state.payload->nv01_timer.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented engine");
        break;
    }
}

/// Calls a GPU engine multivalue method.
void Puller::CallEngineMultiMethod(DmaPusher& dma_pusher, u32 method, u32 subchannel, const u32* base_start, u32 amount, u32 methods_pending) {
    const EngineID engine = bound_engines[subchannel];
    if (Nx2MethodTrace()) {
        LOG_INFO(HW_GPU, "nx2 multimethod sub={} engine={:04X} method={:#x} amount={} first={:#x}",
                 subchannel, static_cast<u32>(engine), method, amount, base_start[0]);
    }
    switch (engine) {
    case EngineID::FERMI_TWOD_A:
        dma_pusher.channel_state.payload->fermi_2d.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::MAXWELL_B:
        dma_pusher.channel_state.payload->maxwell_3d.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::KEPLER_COMPUTE_B:
        dma_pusher.channel_state.payload->kepler_compute.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::MAXWELL_DMA_COPY_A:
        dma_pusher.channel_state.payload->maxwell_dma.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::KEPLER_INLINE_TO_MEMORY_B:
        dma_pusher.channel_state.payload->kepler_memory.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::NV01_TIMER:
        dma_pusher.channel_state.payload->nv01_timer.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented engine");
        break;
    }
}

/// Calls a GPU method.
void Puller::CallMethod(DmaPusher& dma_pusher, const MethodCall& method_call) {
    LOG_TRACE(HW_GPU, "Processing method {:08X} on subchannel {}", method_call.method, method_call.subchannel);
    ASSERT(method_call.subchannel < bound_engines.size());

    if (ExecuteMethodOnEngine(dma_pusher, method_call.method)) {
        CallEngineMethod(dma_pusher, method_call);
    } else {
        CallPullerMethod(dma_pusher, method_call);
    }
}

/// Calls a GPU multivalue method.
void Puller::CallMultiMethod(DmaPusher& dma_pusher, u32 method, u32 subchannel, const u32* base_start, u32 amount, u32 methods_pending) {
    LOG_TRACE(HW_GPU, "Processing method {:08X} on subchannel {}", method, subchannel);
    ASSERT(subchannel < bound_engines.size());
    if (ExecuteMethodOnEngine(dma_pusher, method)) {
        CallEngineMultiMethod(dma_pusher, method, subchannel, base_start, amount, methods_pending);
    } else {
        for (u32 i = 0; i < amount; i++) {
            CallPullerMethod(dma_pusher, MethodCall{
                method,
                base_start[i],
                subchannel,
                methods_pending - i,
            });
        }
    }
}

/// Determines where the method should be executed.
[[nodiscard]] bool Puller::ExecuteMethodOnEngine(DmaPusher& dma_pusher, u32 method) {
    const auto buffer_method = BufferMethods(method);
    return buffer_method >= BufferMethods::NonPullerMethods;
}

} // namespace Tegra::Engines
