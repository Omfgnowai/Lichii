// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <string>

#include <catch2/catch_test_macros.hpp>

#include "video_core/gpu_split.h"
#include "video_core/nx2_stage_probe.h"

namespace Probe = VideoCommon::Nx2StageProbe;

TEST_CASE("Nx2StageProbe: the ring keeps a range for the window's draws", "[video_core]") {
    Probe::OverlapRing ring;
    ring.Read(Probe::VertexSlot(0), 0x10000, 0x1000, 0);
    REQUIRE(ring.Overlaps(0x10800, 16, 1));
    REQUIRE(ring.Overlaps(0x0f000, 0x1001, 1));
    // Same 64 KiB page, past the end: the filter lets it through, the scan rules it out.
    const u64 scans = ring.scans;
    REQUIRE_FALSE(ring.Overlaps(0x11000, 16, 1));
    REQUIRE(ring.scans == scans + 1);
    // Another page: the filter rules it out without a scan.
    REQUIRE_FALSE(ring.Overlaps(0x900000, 16, 1));
    REQUIRE(ring.scans == scans + 1);
    // Read by draw 0: live until 64 draws have completed after it.
    REQUIRE(ring.Overlaps(0x10000, 4, Probe::OverlapRing::Window));
    REQUIRE_FALSE(ring.Overlaps(0x10000, 4, Probe::OverlapRing::Window + 1));
    // Reading it again (the slot's memo) makes it live again.
    ring.Read(Probe::VertexSlot(0), 0x10000, 0x1000, 100);
    REQUIRE(ring.Overlaps(0x10000, 4, 101));
    ring.Compact(101);
    REQUIRE(ring.Live(101) == 1);
    REQUIRE(ring.Overlaps(0x10000, 4, 101));
    ring.Compact(200);
    REQUIRE(ring.Live(200) == 0);
    REQUIRE_FALSE(ring.Overlaps(0x10000, 4, 200));
}

TEST_CASE("Nx2StageProbe: the ring finds ranges of other slots and large ranges",
          "[video_core]") {
    Probe::OverlapRing ring;
    ring.Read(Probe::IndexSlot, 0x200000, 0x100, 5);
    ring.Read(Probe::StorageSlot(4, 3), 0x400000, 0x10, 5);
    ring.Read(Probe::TexelSlot(4, 31), 0x1'0000'0000ULL, 512ULL << 20, 5);
    REQUIRE(ring.Overlaps(0x2000f0, 0x20, 6));
    REQUIRE(ring.Overlaps(0x40000c, 4, 6));
    REQUIRE(ring.Overlaps(0x1'0000'0000ULL + (300ULL << 20), 4, 6));
    REQUIRE_FALSE(ring.Overlaps(0x500000, 4, 6));
    // Empty and unbound ranges are not kept.
    ring.Read(Probe::VertexSlot(1), 0, 0x100, 6);
    ring.Read(Probe::VertexSlot(2), 0x600000, 0, 6);
    REQUIRE(ring.Live(7) == 3);
}

TEST_CASE("Nx2StageProbe: the ring counts the reads past its capacity", "[video_core]") {
    Probe::OverlapRing ring;
    for (u64 range = 0; range <= Probe::OverlapRing::Capacity; ++range) {
        ring.Read(Probe::StorageSlot(0, 0), 0x100000 + range * 0x100, 0x10, 0);
    }
    REQUIRE(ring.overflows == 1);
    REQUIRE(ring.Live(1) == Probe::OverlapRing::Capacity);
    // A later draw drops what the window no longer holds.
    ring.Read(Probe::StorageSlot(0, 0), 0x900000, 0x10, Probe::OverlapRing::Window + 1);
    REQUIRE(ring.overflows == 1);
    REQUIRE(ring.Live(Probe::OverlapRing::Window + 2) == 1);
}

TEST_CASE("Nx2StageProbe: the report line", "[video_core]") {
    Probe::Totals totals;
    const auto region = [](Probe::Region value) { return static_cast<size_t>(value); };
    totals.coarse_ticks[region(Probe::Region::Wait)] = 200'000'000;
    totals.coarse_ticks[region(Probe::Region::Draw)] = 500'000'000;
    totals.coarse_ticks[region(Probe::Region::Outside)] = 300'000'000;
    totals.fine_draw_ticks = 100'000'000;
    totals.fine_ticks[region(Probe::Region::IndexVertex)] = 20'000'000;
    totals.fine_ticks[region(Probe::Region::Uniform)] = 10'000'000;
    totals.fine_ticks[region(Probe::Region::Draw)] = 70'000'000;
    totals.draws = 1000;
    totals.fine_draws = 125;
    totals.timer_calls = 10'000;
    totals.writes[static_cast<size_t>(Probe::Site::CbData)] = 300;
    totals.overlaps[static_cast<size_t>(Probe::Site::CbData)] = 5;
    totals.packet_bytes = 400'000;
    totals.packet_max = 900;
    Probe::SharedTotals shared;
    shared.split_waits[static_cast<size_t>(VideoCommon::DrainSite::Acquire)] = 5;
    shared.split_waits[static_cast<size_t>(VideoCommon::DrainSite::RingFull)] = 7;
    const std::string line = Probe::FormatLine(totals, shared, 1.0, 1.0, 10.0, 3, 0, 0, 0, 0);
    INFO(line);
    REQUIRE(line.starts_with("Stage probe per second (10 s): back busy 800.0 ms (split wait "
                             "200.0 ms) | draws 1000 (125 timed in detail), draw 500.0 ms"));
    REQUIRE(line.find("B 100.0 ms = index/vertex 100.0 + storage 0.0 + sync 0.0") !=
            std::string::npos);
    REQUIRE(line.find("uniform (A) 50.0 ms, rest of draw (A) 350.0 ms") != std::string::npos);
    REQUIRE(line.find("B share 12.5% of busy, 20.0% of draw") != std::string::npos);
    REQUIRE(line.find("joins 1.00 per 100 draws (10)") != std::string::npos);
    REQUIRE(line.find("CB_DATA 5 of 300") != std::string::npos);
    REQUIRE(line.find("split waits 5 (acquire 5)") != std::string::npos);
    REQUIRE(line.find("ring-full waits 7") != std::string::npos);
    REQUIRE(line.find("packet avg 400 B, max 900 B, 0.4 MB") != std::string::npos);
    REQUIRE(line.find("probe 10000 timer calls, est. 0.10 ms (0.01% of busy)") !=
            std::string::npos);
}
