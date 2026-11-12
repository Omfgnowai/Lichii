// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>

namespace Dynarmic {

/// Called for every block the x64 backend emits: host code [start, start + size) runs the guest
/// block whose location descriptor value is `location` (A64: guest PC in the low 56 bits). Set
/// before the first block is compiled. Eden's sampling profiler uses it to name JIT code.
using BlockHook = void (*)(const void* start, std::size_t size, std::uint64_t location);
inline BlockHook g_block_hook = nullptr;

} // namespace Dynarmic
