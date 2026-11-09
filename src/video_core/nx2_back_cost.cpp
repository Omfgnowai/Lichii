// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <mutex>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/engines/engine_interface.h"
#include "video_core/nx2_back_cost.h"

namespace VideoCommon::BackCost {
namespace {

std::mutex names_mutex;
std::unordered_map<const void*, std::string> thunk_names;

struct Cost {
    u64 nanoseconds{};
    u64 samples{};
    u64 words{};
};

struct Key {
    EntryKind kind;
    const void* engine;
    u64 what;
    bool operator==(const Key&) const = default;
};

struct KeyHash {
    size_t operator()(const Key& key) const {
        return std::hash<u64>{}(key.what * 31 + static_cast<u64>(key.kind)) ^
               std::hash<const void*>{}(key.engine);
    }
};

// Back thread only.
std::unordered_map<Key, Cost, KeyHash> costs;
std::unordered_map<const void*, std::string> engine_names;

std::string EngineName(const void* engine) {
    if (engine == nullptr) {
        return "?";
    }
    if (const auto it = engine_names.find(engine); it != engine_names.end()) {
        return it->second;
    }
    std::string name = typeid(*static_cast<const Tegra::Engines::EngineInterface*>(engine)).name();
    if (const size_t colon = name.rfind(':'); colon != std::string::npos) {
        name = name.substr(colon + 1);
    }
    engine_names.emplace(engine, name);
    return name;
}

std::string ThunkName(const void* thunk) {
    std::scoped_lock lock{names_mutex};
    if (const auto it = thunk_names.find(thunk); it != thunk_names.end()) {
        return it->second;
    }
    return fmt::format("call {}", thunk);
}

} // Anonymous namespace

void RegisterThunk(const void* thunk, const char* file, u32 line) {
    std::string name = file != nullptr ? file : "?";
    if (const size_t slash = name.find_last_of("/\\"); slash != std::string::npos) {
        name = name.substr(slash + 1);
    }
    std::scoped_lock lock{names_mutex};
    thunk_names.try_emplace(thunk, fmt::format("call {}:{}", name, line));
}

void Record(EntryKind kind, const void* engine, u64 what, u64 nanoseconds, size_t words) {
    Cost& cost = costs[Key{kind, kind == EntryKind::Call ? nullptr : engine, what}];
    cost.nanoseconds += nanoseconds;
    ++cost.samples;
    cost.words += words;
}

void Report() {
    static auto start = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - start).count();
    if (seconds < 10.0) {
        return;
    }
    start = now;
    if (costs.empty()) {
        return;
    }
    std::vector<std::pair<Key, Cost>> sorted(costs.begin(), costs.end());
    costs.clear();
    std::ranges::sort(sorted, [](const auto& a, const auto& b) {
        return a.second.nanoseconds > b.second.nanoseconds;
    });
    // One entry in eight is timed: scale to the whole.
    constexpr double Scale = 8.0;
    double total_ms = 0;
    for (const auto& [key, cost] : sorted) {
        total_ms += static_cast<double>(cost.nanoseconds) * Scale / 1e6 / seconds;
    }
    std::string line;
    const size_t shown = (std::min<size_t>)(sorted.size(), 24);
    for (size_t i = 0; i < shown; ++i) {
        const auto& [key, cost] = sorted[i];
        const double ms = static_cast<double>(cost.nanoseconds) * Scale / 1e6 / seconds;
        const double per_second = static_cast<double>(cost.samples) * Scale / seconds;
        const double ns_each =
            static_cast<double>(cost.nanoseconds) / static_cast<double>(cost.samples);
        std::string what;
        switch (key.kind) {
        case EntryKind::Method:
            what = fmt::format("{} {:#x}", EngineName(key.engine), key.what);
            break;
        case EntryKind::Multi:
            what = fmt::format("{} multi {:#x}", EngineName(key.engine), key.what);
            break;
        case EntryKind::Call:
            what = ThunkName(reinterpret_cast<const void*>(key.what));
            break;
        case EntryKind::Other:
            what = "other entry";
            break;
        }
        line += fmt::format(" | {} {:.1f} ms ({:.0f}/s, {:.0f} ns)", what, ms, per_second, ns_each);
    }
    LOG_INFO(HW_GPU, "Back cost per second (1 in 8 entries timed): {:.1f} ms in entries{}",
             total_ms, line);
}

} // namespace VideoCommon::BackCost
