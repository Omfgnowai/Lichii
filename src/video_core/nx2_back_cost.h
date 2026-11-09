// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>

#include "common/common_types.h"

// EDEN_NX2_BACK_COST=1: where the split GPU thread's back spends its time, by ring entry. Times
// one entry in eight (methods by engine and method, multi-method runs, deferred calls by their
// source line) and logs the costliest every 10 s as an estimate of milliseconds per second.
// Measurement only: run 13b/14a (Ocarina of Time, Castle Town) had the back ~92% busy, mostly
// in zones the profiler can't split further ("method", "other").
namespace VideoCommon::BackCost {

inline bool Enabled() {
    // Read on first use: nx2_switches.txt sets the variable after static initialisation.
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_BACK_COST");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

/// Names a deferred call's thunk after the line that sent it (once per call site).
void RegisterThunk(const void* thunk, const char* file, u32 line);

enum class EntryKind : u32 { Method, Multi, Call, Other };

/// Back thread: one timed entry. `what` is the method (Method, Multi) or the thunk (Call).
void Record(EntryKind kind, const void* engine, u64 what, u64 nanoseconds, size_t words);

/// Back thread: logs and resets every 10 s.
void Report();

} // namespace VideoCommon::BackCost
