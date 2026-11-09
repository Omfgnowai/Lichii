// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Test build only (EGLA-86): switches that restore build 117's behaviour of single commits, and
// acquire/release timing stats. Not for nx2.

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "common/common_types.h"
#include "common/logging.h"

namespace VideoCommon::Nx2AcquireTest {

inline bool Env(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

// Read lazily (on first use) so values from nx2_switches.txt, which the GUI
// loads after static initialisation, are seen.
#define NX2_LAZY_SWITCH(fn, name)                                                                  \
    inline bool fn() {                                                                             \
        static const bool value = Env(name);                                                       \
        return value;                                                                              \
    }
NX2_LAZY_SWITCH(T337Off, "EDEN_NX2_T337_OFF")
NX2_LAZY_SWITCH(T752Off, "EDEN_NX2_T752_OFF")
NX2_LAZY_SWITCH(T687Off, "EDEN_NX2_T687_OFF")
NX2_LAZY_SWITCH(FixLatch, "EDEN_NX2_FIX_LATCH")
NX2_LAZY_SWITCH(StatsEnabled, "EDEN_NX2_ACQUIRE_STATS")
#undef NX2_LAZY_SWITCH
inline long long WarnMs() {
    static const long long value = [] {
        const char* env = std::getenv("EDEN_NX2_ACQUIRE_WARN_MS");
        return env != nullptr ? std::atoll(env) : 0LL;
    }();
    return value;
}

struct PendingRelease {
    u32 value;
    std::chrono::steady_clock::time_point at;
};

// Fence-flagged semaphore releases the back processed whose memory write still waits for its
// fence (#337 at GPU accuracy Low).
class PendingReleases {
public:
    static PendingReleases& Get() {
        static PendingReleases instance;
        return instance;
    }
    void Note(u64 address, u32 value) {
        std::scoped_lock lk{mutex};
        map[address] = {value, std::chrono::steady_clock::now()};
        count.store(map.size(), std::memory_order_release);
    }
    void Written(u64 address, u32 value) {
        if (count.load(std::memory_order_acquire) == 0) {
            return;
        }
        std::scoped_lock lk{mutex};
        if (const auto it = map.find(address); it != map.end() && it->second.value == value) {
            map.erase(it);
            count.store(map.size(), std::memory_order_release);
        }
    }
    std::optional<PendingRelease> Find(u64 address) {
        if (count.load(std::memory_order_acquire) == 0) {
            return std::nullopt;
        }
        std::scoped_lock lk{mutex};
        if (const auto it = map.find(address); it != map.end()) {
            return it->second;
        }
        return std::nullopt;
    }

private:
    std::mutex mutex;
    std::unordered_map<u64, PendingRelease> map;
    std::atomic<size_t> count{};
};

// Buckets: <1 ms, <10 ms, <100 ms, <500 ms, <2 s, >=2 s.
class Stats {
public:
    static Stats& Get() {
        static Stats instance;
        return instance;
    }
    void Acquire(std::chrono::steady_clock::duration wait) {
        Add(acquire, acquire_total_us, acquire_max_us, wait);
    }
    void Release(std::chrono::steady_clock::duration delay) {
        Add(release, release_total_us, release_max_us, delay);
    }

private:
    using Buckets = std::array<std::atomic<u64>, 6>;
    void Add(Buckets& buckets, std::atomic<u64>& total, std::atomic<u64>& max,
             std::chrono::steady_clock::duration duration) {
        if (!StatsEnabled()) {
            return;
        }
        const u64 us = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::microseconds>(duration).count());
        const size_t bucket = us < 1000 ? 0 : us < 10000 ? 1 : us < 100000 ? 2 : us < 500000 ? 3
                            : us < 2000000 ? 4 : 5;
        buckets[bucket].fetch_add(1, std::memory_order_relaxed);
        total.fetch_add(us, std::memory_order_relaxed);
        for (u64 seen = max.load(std::memory_order_relaxed);
             us > seen && !max.compare_exchange_weak(seen, us, std::memory_order_relaxed);) {
        }
        Report();
    }
    void Report() {
        const auto now = std::chrono::steady_clock::now();
        const s64 now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               now.time_since_epoch()).count();
        s64 last = last_ms.load(std::memory_order_relaxed);
        if (last == 0) {
            last_ms.compare_exchange_strong(last, now_ms);
            return;
        }
        if (now_ms - last < 10000 || !last_ms.compare_exchange_strong(last, now_ms)) {
            return;
        }
        const auto take = [](Buckets& b, std::atomic<u64>& total, std::atomic<u64>& max) {
            std::array<u64, 8> out{};
            for (size_t i = 0; i < 6; ++i) {
                out[i] = b[i].exchange(0, std::memory_order_relaxed);
            }
            out[6] = total.exchange(0, std::memory_order_relaxed);
            out[7] = max.exchange(0, std::memory_order_relaxed);
            return out;
        };
        const auto a = take(acquire, acquire_total_us, acquire_max_us);
        const auto r = take(release, release_total_us, release_max_us);
        LOG_INFO(HW_GPU,
                 "EGLA86 acquire waits <1/<10/<100/<500/<2000/>=2000 ms: {} {} {} {} {} {}, total "
                 "{} ms, max {} ms | fence-flagged release delays: {} {} {} {} {} {}, total {} ms, "
                 "max {} ms | t337_off={} t752_off={} t687_off={} fix_latch={}",
                 a[0], a[1], a[2], a[3], a[4], a[5], a[6] / 1000, a[7] / 1000, r[0], r[1], r[2],
                 r[3], r[4], r[5], r[6] / 1000, r[7] / 1000, T337Off(), T752Off(), T687Off(),
                 FixLatch());
    }

    Buckets acquire{};
    Buckets release{};
    std::atomic<u64> acquire_total_us{}, acquire_max_us{}, release_total_us{}, release_max_us{};
    std::atomic<s64> last_ms{};
};

} // namespace VideoCommon::Nx2AcquireTest
