// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <atomic>
#include <chrono>
#include <optional>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "video_core/back_split_pool.h"
#include "video_core/buffer_cache/geometry_ranges.h"

namespace {

using VideoCommon::GeometryRange;
using VideoCommon::PreparedGeometry;
using VideoCommon::BackSplit::Job;
using VideoCommon::BackSplit::Pool;

/// A job's per-draw record, prepared from the draw's input the way the geometry and image
/// descriptor jobs prepare theirs.
struct FakeJob {
    Job job;
    u64 input{};
    std::array<u64, 8> record{};
};

void Prepare(void* argument) {
    FakeJob& fake = *static_cast<FakeJob*>(argument);
    u64 value = fake.input;
    for (u64& word : fake.record) {
        value = value * 6364136223846793005ULL + 1442695040888963407ULL;
        word = value;
    }
}

/// A stream of draws as the thread that draws runs them: each posts two jobs, does serial work
/// between them in order and consumes the records in order. Without a pool it prepares inline.
std::vector<u64> RunStream(Pool* pool, u32 draws, bool let_workers_sleep) {
    std::vector<u64> out;
    out.reserve(static_cast<size_t>(draws) * 17);
    FakeJob geometry;
    FakeJob images;
    u64 serial = 1;
    const auto post = [pool](FakeJob& fake) {
        if (pool != nullptr) {
            pool->Post(fake.job, &Prepare, &fake);
        } else {
            Prepare(&fake);
        }
    };
    const auto join = [pool](FakeJob& fake) {
        if (pool != nullptr) {
            pool->Join(fake.job);
        }
    };
    for (u32 draw = 0; draw < draws; ++draw) {
        geometry.input = u64{draw} * 2;
        images.input = u64{draw} * 2 + 1;
        post(geometry);
        serial = serial * 31 + draw;
        post(images);
        if (let_workers_sleep && draw % 64 == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds{20});
        }
        join(geometry);
        out.insert(out.end(), geometry.record.begin(), geometry.record.end());
        out.push_back(serial);
        join(images);
        out.insert(out.end(), images.record.begin(), images.record.end());
    }
    return out;
}

} // Anonymous namespace

TEST_CASE("BackSplit pool: a draw stream prepared in parallel matches the serial one",
          "[video_core]") {
    const std::vector<u64> serial = RunStream(nullptr, 20000, false);
    for (const u32 workers : {0U, 1U, 2U, 4U}) {
        Pool pool{workers, true};
        REQUIRE(RunStream(&pool, 20000, false) == serial);
        REQUIRE(pool.Totals().worker_jobs <= 40000);
    }
}

TEST_CASE("BackSplit pool: joins finish while the workers sleep", "[video_core]") {
    Pool pool{2, false};
    std::this_thread::sleep_for(std::chrono::milliseconds{50});
    REQUIRE(RunStream(&pool, 512, true) == RunStream(nullptr, 512, true));
}

TEST_CASE("BackSplit pool: a post wakes a sleeping worker", "[video_core]") {
    Pool pool{1, true};
    // Long enough for the worker to stop spinning and sleep.
    std::this_thread::sleep_for(std::chrono::milliseconds{200});
    struct Started {
        Job job;
        std::atomic<bool> started{};
        std::atomic<bool> release{};
    } work;
    pool.Post(
        work.job,
        [](void* argument) {
            auto& self = *static_cast<Started*>(argument);
            self.started.store(true);
            while (!self.release.load()) {
                std::this_thread::yield();
            }
        },
        &work);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!work.started.load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    REQUIRE(work.started.load());
    work.release.store(true);
    const auto result = pool.Join(work.job);
    REQUIRE_FALSE(result.ran_here);
    REQUIRE_FALSE(work.job.Pending());
    REQUIRE(pool.Totals().worker_jobs == 1);
}

TEST_CASE("PreparedGeometry: prepared ranges, and verify keeps the serial one", "[video_core]") {
    const GeometryRange parallel{.device_addr = 0x1000, .size = 0x100, .bound = true};
    const GeometryRange other{.device_addr = 0x2000, .size = 0x100, .bound = true};
    const auto serial = [&] { return std::optional<GeometryRange>{other}; };
    PreparedGeometry prepared;
    prepared.Set(3, parallel);
    REQUIRE(prepared.Resolve(3, serial) == parallel);
    REQUIRE(prepared.Resolve(4, serial) == other);
    REQUIRE(prepared.mismatches == 0);

    prepared.verify = true;
    REQUIRE(prepared.Resolve(3, serial) == other);
    REQUIRE(prepared.mismatches == 1);
    REQUIRE(prepared.first_mismatches[0].slot == 3);
    REQUIRE(prepared.first_mismatches[0].parallel == parallel);
    REQUIRE(prepared.first_mismatches[0].serial == other);

    // The native size check the serial path throws on is a mismatch too.
    REQUIRE_FALSE(prepared.Resolve(3, [] { return std::optional<GeometryRange>{}; }));
    REQUIRE(prepared.mismatches == 2);

    prepared.Set(PreparedGeometry::IndexSlot, other);
    REQUIRE(prepared.Resolve(PreparedGeometry::IndexSlot, serial) == other);
    REQUIRE(prepared.mismatches == 2);

    prepared.Clear();
    REQUIRE_FALSE(prepared.Has(3));
    REQUIRE(prepared.mismatches == 0);
}
