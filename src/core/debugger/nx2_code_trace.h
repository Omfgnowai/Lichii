// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace Kernel {
class KProcess;
}

/// Code hooks for the debug server: log every time chosen guest instructions run, with the
/// registers, without stopping the game.
///
/// A hook replaces one HINT-class instruction (NOP, PACIBSP, PACIASP, BTI, ...: the first
/// instruction of almost every function in a PAC build) with `SVC #TraceSvc`. The guest takes the
/// supervisor call, svc::Call hands it to HandleSvc, which records the registers and returns, and
/// the guest goes on at the next instruction. Dynarmic translates every HINT as a no-op, so
/// skipping the replaced instruction changes nothing. Removing a hook writes the original back.
/// Both invalidate the JIT's translation of that word on every core.
namespace Core::Nx2CodeTrace {

/// SVC immediate used by hooks. Real SVC numbers are 8-bit; this one is never issued by games.
constexpr u32 TraceSvc = 0xC0DE;

struct Hit {
    u64 seq;
    u64 present;     ///< g_nx2_present_count when it ran
    u64 address;     ///< the hooked instruction
    u64 thread_id;
    u64 sp;
    u64 lr;
    std::array<u64, 8> x;  ///< x0..x7: the arguments at a function's entry
};

struct Hook {
    u64 address;
    u32 original;
    u64 hits;
};

/// Called from svc::Call for every supervisor call. True when it was a hook (already handled).
bool HandleSvc(Core::System& system, u32 imm);

/// Installs a hook at address (4-aligned, mapped, holding a HINT-class instruction). Throws
/// std::runtime_error with the reason otherwise. Installing the same address again is a no-op.
Hook Set(Kernel::KProcess& process, u64 address);

/// Removes the hook at address (throws if there is none) or, with all, every hook.
void Clear(Kernel::KProcess& process, u64 address);
std::size_t ClearAll(Kernel::KProcess& process);

std::vector<Hook> Hooks(const Kernel::KProcess& process);

/// Hits with seq >= since, oldest first, at most max. next is the seq to pass next time; dropped
/// counts hits that were overwritten in the ring before they were read.
std::vector<Hit> Read(u64 since, std::size_t max, u64& next, u64& dropped);

} // namespace Core::Nx2CodeTrace
