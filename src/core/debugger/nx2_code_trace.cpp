// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <stdexcept>

#include <fmt/format.h>

#include "core/arm/arm_interface.h"
#include "core/core.h"
#include "core/debugger/nx2_code_trace.h"
#include "core/hardware_properties.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/svc_types.h"
#include "core/memory.h"

namespace Core::Nx2CodeTrace {
namespace {

constexpr std::size_t RingSize = 8192;
constexpr u32 SvcInstruction = 0xD4000001u | (TraceSvc << 5);

struct State {
    std::mutex mutex;
    const Kernel::KProcess* process = nullptr;  ///< the process the hooks were written into
    std::map<u64, Hook> hooks;
    std::array<Hit, RingSize> ring{};
    u64 next_seq = 0;
};

State& GetState() {
    static State state;
    return state;
}

bool IsHint(u32 instruction) {
    // HINT #imm: 1101 0101 0000 0011 0010 xxxx xxx1 1111 (NOP, YIELD, PAC*SP, AUT*SP, BTI, ...)
    return (instruction & 0xFFFFF01Fu) == 0xD503201Fu;
}

void Invalidate(Kernel::KProcess& process, u64 address) {
    for (std::size_t i = 0; i < Hardware::NUM_CPU_CORES; ++i) {
        if (auto* const interface = process.GetArmInterface(i)) {
            interface->InvalidateCacheRange(address, 4);
        }
    }
}

/// Forgets hooks written into another (earlier) process: its memory is gone.
void ForgetStale(State& state, const Kernel::KProcess& process) {
    if (state.process != &process) {
        state.hooks.clear();
        state.process = &process;
    }
}

} // Anonymous namespace

bool HandleSvc(Core::System& system, u32 imm) {
    if (imm != TraceSvc) {
        return false;
    }
    auto& kernel = system.Kernel();
    Kernel::KProcess* const process = GetCurrentProcessPointer(kernel);
    Kernel::Svc::ThreadContext context{};
    if (process != nullptr) {
        if (auto* const interface = process->GetArmInterface(kernel.CurrentPhysicalCoreIndex())) {
            interface->GetContext(context);
        }
    }
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    // The PC is past the SVC; fall back to it as is if that is not a hook (it always should be).
    u64 address = context.pc - 4;
    auto hook = state.hooks.find(address);
    if (hook == state.hooks.end()) {
        hook = state.hooks.find(context.pc);
        address = context.pc;
    }
    if (hook != state.hooks.end()) {
        ++hook->second.hits;
    }
    Hit& hit = state.ring[state.next_seq % RingSize];
    hit.seq = state.next_seq++;
    hit.present = g_nx2_present_count.load(std::memory_order_relaxed);
    hit.address = address;
    hit.thread_id = GetCurrentThread(kernel).GetThreadId();
    hit.sp = context.sp;
    hit.lr = context.lr;
    for (std::size_t i = 0; i < hit.x.size(); ++i) {
        hit.x[i] = context.r[i];
    }
    return true;
}

Hook Set(Kernel::KProcess& process, u64 address) {
    if (address % 4 != 0) {
        throw std::runtime_error{fmt::format("{:#x} is not 4-byte aligned", address)};
    }
    Memory::Memory& memory = process.GetMemory();
    if (!memory.IsValidVirtualAddressRange(address, 4)) {
        throw std::runtime_error{fmt::format("{:#x} is not mapped", address)};
    }
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    ForgetStale(state, process);
    if (const auto it = state.hooks.find(address); it != state.hooks.end()) {
        return it->second;
    }
    const u32 original = memory.Read32(address);
    if (!IsHint(original)) {
        throw std::runtime_error{fmt::format(
            "{:#x} holds {:#010x}, not a HINT-class instruction (NOP, PACIBSP, BTI, ...): hook a "
            "function's first instruction",
            address, original)};
    }
    memory.Write32(address, SvcInstruction);
    Invalidate(process, address);
    return state.hooks[address] = Hook{address, original, 0};
}

void Clear(Kernel::KProcess& process, u64 address) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    ForgetStale(state, process);
    const auto it = state.hooks.find(address);
    if (it == state.hooks.end()) {
        throw std::runtime_error{fmt::format("no hook at {:#x}", address)};
    }
    process.GetMemory().Write32(address, it->second.original);
    Invalidate(process, address);
    state.hooks.erase(it);
}

std::size_t ClearAll(Kernel::KProcess& process) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    ForgetStale(state, process);
    const std::size_t count = state.hooks.size();
    for (const auto& [address, hook] : state.hooks) {
        process.GetMemory().Write32(address, hook.original);
        Invalidate(process, address);
    }
    state.hooks.clear();
    return count;
}

std::vector<Hook> Hooks(const Kernel::KProcess& process) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    std::vector<Hook> out;
    if (state.process == &process) {
        for (const auto& [address, hook] : state.hooks) {
            out.push_back(hook);
        }
    }
    return out;
}

std::vector<Hit> Read(u64 since, std::size_t max, u64& next, u64& dropped) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    const u64 oldest = state.next_seq > RingSize ? state.next_seq - RingSize : 0;
    dropped = since < oldest ? oldest - since : 0;
    std::vector<Hit> out;
    u64 seq = std::max(since, oldest);
    for (; seq < state.next_seq && out.size() < max; ++seq) {
        out.push_back(state.ring[seq % RingSize]);
    }
    next = seq;
    return out;
}

} // namespace Core::Nx2CodeTrace
