// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/nx2_block_count.h"

namespace Common::Nx2BlockCount {
namespace {

struct Counter {
    u64 pc;
    u64 count; ///< Added to by JIT code, never by this file
    u64 written;
};

struct State {
    std::mutex mutex;
    std::deque<Counter> counters; ///< A deque: elements never move
    std::unordered_map<u64, Counter*> by_pc;
    std::atomic<u64> frames{};
    u64 window_frames{};
    u64 window_index{};
    std::chrono::steady_clock::time_point window_start{};
    u64 window_ns{5'000'000'000ULL};
    std::FILE* file{};
    bool failed{};
};

State& GetState() {
    static State state;
    return state;
}

} // namespace

bool Enabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_BLOCK_COUNT");
        const bool on = value != nullptr && value[0] != '\0' && value[0] != '0';
        if (on) {
            State& state = GetState();
            if (const char* const ms = std::getenv("EDEN_NX2_BLOCK_COUNT_MS")) {
                const long long parsed = std::strtoll(ms, nullptr, 10);
                if (parsed >= 100) {
                    state.window_ns = static_cast<u64>(parsed) * 1'000'000ULL;
                }
            }
            LOG_INFO(Core_ARM, "Nx2BlockCount: counting guest block runs, {} ms windows",
                     state.window_ns / 1'000'000ULL);
        }
        return on;
    }();
    return enabled;
}

u64* CounterFor(u64 pc) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    const auto [it, added] = state.by_pc.try_emplace(pc, nullptr);
    if (added) {
        it->second = &state.counters.emplace_back(Counter{pc, 0, 0});
    }
    return &it->second->count;
}

void OnFrame() {
    if (!Enabled()) {
        return;
    }
    State& state = GetState();
    const auto now = std::chrono::steady_clock::now();
    std::scoped_lock lock{state.mutex};
    ++state.window_frames;
    if (state.window_start == std::chrono::steady_clock::time_point{}) {
        state.window_start = now;
        state.window_frames = 0;
        return;
    }
    const u64 elapsed = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - state.window_start).count());
    if (elapsed < state.window_ns || state.failed) {
        return;
    }
    if (state.file == nullptr) {
        const auto path = Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir) /
                          "nx2_block_counts.bin";
#ifdef _WIN32
        state.file = _wfopen(path.c_str(), L"wb");
#else
        state.file = std::fopen(path.c_str(), "wb");
#endif
        if (state.file == nullptr) {
            state.failed = true;
            LOG_ERROR(Core_ARM, "Nx2BlockCount: could not create {}", path.string());
            return;
        }
        LOG_INFO(Core_ARM, "Nx2BlockCount: writing {}", path.string());
    }
    std::vector<u64> out;
    out.reserve(5 + state.counters.size() / 4);
    out.insert(out.end(), {0x31304B4C4232584EULL /* "NX2BLK01" */, state.window_index, elapsed,
                           state.window_frames, 0});
    u64 entries = 0;
    for (Counter& counter : state.counters) {
        // The JIT's adds race with this read; a count is only ever behind.
        const u64 count = std::atomic_ref<u64>{counter.count}.load(std::memory_order_relaxed);
        if (count != counter.written) {
            out.push_back(counter.pc);
            out.push_back(count - counter.written);
            counter.written = count;
            ++entries;
        }
    }
    out[4] = entries;
    std::fwrite(out.data(), sizeof(u64), out.size(), state.file);
    std::fflush(state.file);
    ++state.window_index;
    state.window_frames = 0;
    state.window_start = now;
}

} // namespace Common::Nx2BlockCount
