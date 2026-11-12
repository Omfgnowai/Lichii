// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace Nx2Sampler {

// EDEN_NX2_SAMPLE=<thread names>: a sampling profiler for named host threads (Windows only).
// Names are comma separated; a name ending in '*' matches by prefix (CPUCore_*). From
// EDEN_NX2_SAMPLE_START seconds (default 60) for EDEN_NX2_SAMPLE_SECONDS (default 30), each
// thread is paused about 1000 times a second and its call stack recorded; the counts per
// function (where the thread is, and everything on its stack) go to EDEN_NX2_SAMPLE_OUT
// (default nx2_samples.txt).
void StartFromEnvironment();

/// Timeline recording, on whenever EDEN_NX2_SAMPLE is set: every sample also gets its time,
/// the thread's CPU cycles and (CPU cores) the guest thread on the core; game presents and JIT
/// blocks are logged so the analysis can split samples by frame time and name game code.
/// The report then also writes <out>.timeline.tsv, <out>.stacks.tsv and <out>.presents.tsv.
bool Recording();
/// The kernel scheduler switched core `core` to guest thread `thread_id`.
void NoteCoreThread(unsigned core, std::uint64_t thread_id);
/// The JIT emitted host code [start, start + size) for the guest code at `guest_pc`.
void NoteJitBlock(const void* start, std::size_t size, std::uint64_t guest_pc);
/// The game presented a frame.
void NotePresent();
/// QueryPerformanceCounter now (0 off Windows): the clock of every timeline record.
std::uint64_t NowQpc();
/// One host GPU submission: submitted at `submit_qpc`, ran on the GPU from `begin_ns` to
/// `end_ns` (GPU timestamp clock). Written to <out>.gpu.tsv.
void NoteGpuSubmission(std::uint64_t submit_qpc, std::uint64_t begin_ns, std::uint64_t end_ns);
/// A guest read of game data from `begin_qpc` to `end_qpc`: `bytes` from `path` (the first file
/// the read touches, empty if unknown). Written to <out>.reads.tsv.
void NoteFileRead(std::uint64_t begin_qpc, std::uint64_t end_qpc, std::uint64_t bytes,
                  const std::string& path);

/// Samples the named threads for `seconds` now (blocking), writes the report to `out` (and
/// the folded stacks to `out`.folded) and returns the report.
std::string Sample(const char* names, double seconds, const std::string& out);

} // namespace Nx2Sampler
