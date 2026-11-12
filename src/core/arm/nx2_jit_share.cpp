// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/arm/nx2_jit_share.h"

#include <chrono>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include "common/logging.h"

namespace Core::Nx2JitShare {
namespace {
thread_local Ring* t_ring = nullptr;
thread_local int t_precompiling = 0;

constexpr std::uint64_t SingleSteppingBit = std::uint64_t{1} << 57;

std::atomic<std::uint64_t> g_listed{};
std::atomic<std::uint64_t> g_seen{};
std::atomic<std::uint64_t> g_compiled{};
std::atomic<std::int64_t> g_next_log{};
} // Anonymous namespace

bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_JIT_SHARE");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

void Ring::Push(std::uint64_t location) noexcept {
    // One slot per compile; several cores may list at once, so the index is taken atomically.
    const std::uint64_t slot = head.fetch_add(1, std::memory_order_acq_rel);
    slots[slot & (Capacity - 1)].store(location, std::memory_order_release);
}

bool Ring::Next(std::uint64_t& cursor, std::uint64_t& location) const noexcept {
    const std::uint64_t end = head.load(std::memory_order_acquire);
    if (cursor >= end) {
        return false;
    }
    if (end - cursor > Capacity) {
        cursor = end - Capacity;
    }
    // A slot whose writer took its index but has not stored yet still holds an older location:
    // compiling that is harmless (usually already cached).
    location = slots[cursor & (Capacity - 1)].load(std::memory_order_acquire);
    ++cursor;
    return true;
}

Ring* RingFor(const void* process) {
    static std::mutex mutex;
    static auto* const rings = new std::unordered_map<const void*, std::unique_ptr<Ring>>;
    std::scoped_lock lock{mutex};
    auto& ring = (*rings)[process];
    if (!ring) {
        ring = std::make_unique<Ring>();
    }
    return ring.get();
}

void SetRunning(Ring* ring) noexcept {
    t_ring = ring;
}

PrecompileScope::PrecompileScope() noexcept {
    ++t_precompiling;
}

PrecompileScope::~PrecompileScope() {
    --t_precompiling;
}

void NoteBlock(std::uint64_t location) noexcept {
    Ring* const ring = t_ring;
    if (ring == nullptr || t_precompiling != 0 || (location & SingleSteppingBit) != 0) {
        return;
    }
    ring->Push(location);
    g_listed.fetch_add(1, std::memory_order_relaxed);
}

void Count(std::size_t seen, std::size_t compiled) noexcept {
    g_seen.fetch_add(seen, std::memory_order_relaxed);
    g_compiled.fetch_add(compiled, std::memory_order_relaxed);
    const std::int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
                                 std::chrono::steady_clock::now().time_since_epoch())
                                 .count();
    std::int64_t next = g_next_log.load(std::memory_order_relaxed);
    if (next == 0) {
        g_next_log.compare_exchange_strong(next, now + 10, std::memory_order_relaxed);
        return;
    }
    if (now < next || !g_next_log.compare_exchange_strong(next, now + 10, std::memory_order_relaxed)) {
        return;
    }
    LOG_INFO(Core_ARM, "JIT share last 10 s: {} blocks listed, {} seen by other cores, {} compiled ahead",
             g_listed.exchange(0, std::memory_order_relaxed),
             g_seen.exchange(0, std::memory_order_relaxed),
             g_compiled.exchange(0, std::memory_order_relaxed));
}

} // namespace Core::Nx2JitShare
