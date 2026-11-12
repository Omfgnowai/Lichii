// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

/// EDEN_NX2_JIT_SHARE=1: every CPU core has its own JIT, so a block of guest code runs compiled
/// once per core that reaches it (Ocarina of Time area loads: 374k blocks compiled on all 6
/// cores, ~3.7 ms of compiling in each slow frame). Blocks a core compiles are listed in a ring
/// per process; the other cores compile them while idle (PhysicalCore::Idle), before their own
/// guest threads need them.
namespace Core::Nx2JitShare {

bool Enabled();

class Ring {
public:
    static constexpr std::uint64_t Capacity = std::uint64_t{1} << 18;

    void Push(std::uint64_t location) noexcept;

    std::uint64_t Head() const noexcept {
        return head.load(std::memory_order_acquire);
    }

    /// The next location after `cursor` (which it advances); false once caught up. A reader that
    /// fell more than Capacity behind skips to the oldest location kept.
    bool Next(std::uint64_t& cursor, std::uint64_t& location) const noexcept;

private:
    std::atomic<std::uint64_t> head{};
    std::unique_ptr<std::atomic<std::uint64_t>[]> slots =
        std::make_unique<std::atomic<std::uint64_t>[]>(Capacity);
};

/// The ring of a process (made on first use, kept for the life of the program).
Ring* RingFor(const void* process);

/// Blocks compiled on this host thread go to `ring` (nullptr: nowhere) until set again.
void SetRunning(Ring* ring) noexcept;

/// While alive, blocks compiled on this thread are not listed (precompiles of listed blocks).
class PrecompileScope {
public:
    PrecompileScope() noexcept;
    ~PrecompileScope();
};

/// From Dynarmic::g_block_hook: lists a block this thread just compiled.
void NoteBlock(std::uint64_t location) noexcept;

/// Statistics, logged every 10 s: listed blocks seen by a core and how many it compiled.
void Count(std::size_t seen, std::size_t compiled) noexcept;

} // namespace Core::Nx2JitShare
