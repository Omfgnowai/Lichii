// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

// EDEN_NX2_BLOCK_COUNT=1 (diagnostic, off by default): every guest block the JIT compiles gets a
// counter its host code adds one to each time it runs. Each window (EDEN_NX2_BLOCK_COUNT_MS,
// default 5000) the counts made since the last one go to <log dir>/nx2_block_counts.bin with the
// number of frames presented, so a run at one frame rate can be compared with a run at another:
// code that runs per frame keeps its count per frame, code tied to real time keeps its count per
// second. Counts from several cores can lose increments (plain adds); the ratios don't need more.
//
// File: records of { u64 magic "NX2BLK01", u64 window index, u64 window ns, u64 frames,
// u64 entries, then entries x { u64 guest pc, u64 count } } with only the blocks that ran.
// Guest PCs are absolute: the log's "Module main loaded at" line gives the base.
namespace Common::Nx2BlockCount {

/// Read once, at the first call (the frontend has loaded nx2_switches.txt by then).
[[nodiscard]] bool Enabled();

/// The counter of the block at a guest PC, made on first use. Stays valid for the session.
/// Matches Dynarmic::BlockCounter.
[[nodiscard]] u64* CounterFor(u64 pc);

/// A frame was presented. Writes a window when one is due.
void OnFrame();

} // namespace Common::Nx2BlockCount
