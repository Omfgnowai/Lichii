// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Test build only (EGLA-86 A/B): acquire suspension histogram, off unless
// EDEN_NX2_ACQUIRE_STATS is set. Not for nx2.

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>

#include "common/common_types.h"
#include "common/logging.h"

namespace VideoCommon::Nx2AcquireTest {

inline const bool g_stats = [] {
    const char* value = std::getenv("EDEN_NX2_ACQUIRE_STATS");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}();

// Buckets: <1 ms, <10 ms, <100 ms, <500 ms, <2 s, >=2 s.
class Stats {
public:
    static Stats& Get() {
        static Stats instance;
        return instance;
    }
    void Acquire(std::chrono::steady_clock::duration wait) {
        if (!g_stats) {
            return;
        }
        const u64 us = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::microseconds>(wait).count());
        const size_t bucket = us < 1000 ? 0 : us < 10000 ? 1 : us < 100000 ? 2 : us < 500000 ? 3
                            : us < 2000000 ? 4 : 5;
        buckets[bucket].fetch_add(1, std::memory_order_relaxed);
        total_us.fetch_add(us, std::memory_order_relaxed);
        for (u64 seen = max_us.load(std::memory_order_relaxed);
             us > seen && !max_us.compare_exchange_weak(seen, us, std::memory_order_relaxed);) {
        }
        const s64 now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now().time_since_epoch())
                               .count();
        s64 last = last_ms.load(std::memory_order_relaxed);
        if (last == 0) {
            last_ms.compare_exchange_strong(last, now_ms);
            return;
        }
        if (now_ms - last < 10000 || !last_ms.compare_exchange_strong(last, now_ms)) {
            return;
        }
        std::array<u64, 6> b{};
        for (size_t i = 0; i < b.size(); ++i) {
            b[i] = buckets[i].exchange(0, std::memory_order_relaxed);
        }
        LOG_INFO(HW_GPU,
                 "EGLA86 acquire waits <1/<10/<100/<500/<2000/>=2000 ms: {} {} {} {} {} {}, total "
                 "{} ms, max {} ms",
                 b[0], b[1], b[2], b[3], b[4], b[5],
                 total_us.exchange(0, std::memory_order_relaxed) / 1000,
                 max_us.exchange(0, std::memory_order_relaxed) / 1000);
    }

private:
    std::array<std::atomic<u64>, 6> buckets{};
    std::atomic<u64> total_us{}, max_us{};
    std::atomic<s64> last_ms{};
};

} // namespace VideoCommon::Nx2AcquireTest
