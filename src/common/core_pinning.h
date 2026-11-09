// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

#include "common/common_types.h"

namespace Common {

/// One logical processor as the host reports it.
struct LogicalProcessor {
    /// What the host's affinity call takes (a CPU set id on Windows).
    u32 id;
    /// The physical core: equal for SMT siblings.
    u32 core;
    /// Higher is faster. Equal on every processor of a CPU that is not hybrid.
    u8 efficiency_class;
};

/// Physical cores of the fastest class needed before the GPU threads are pinned.
constexpr std::size_t MinCoresForGpuThreadPinning = 6;

/// Logical processor ids for each pinned role. All three are empty when the host has fewer than
/// MinCoresForGpuThreadPinning performance cores.
struct GpuThreadPinning {
    std::size_t performance_cores{};
    std::vector<u32> front;
    std::vector<u32> back;
    std::vector<u32> guest;
};

/// The split GPU thread's front and back each get a whole physical performance core, so no SMT
/// sibling shares it, and the guest cores get the remaining performance cores. Core 0 takes most
/// interrupts, so the GPU threads get the next two.
[[nodiscard]] GpuThreadPinning PlanGpuThreadPinning(std::span<const LogicalProcessor> processors);

/// EDEN_NX2_PIN_PLAN: "F,B" or "F,B;G-list" moves the front and back to physical cores F and B
/// and, with a list, keeps the guest cores on those cores only (e.g. "9,10;8,11-15" for the second
/// CCD of a 16-core Ryzen). Cores count from 0 in host order, of the fastest class only. Returns
/// the plan unchanged when `text` is empty or malformed.
[[nodiscard]] GpuThreadPinning OverrideGpuThreadPinning(GpuThreadPinning plan,
                                                        std::span<const LogicalProcessor> processors,
                                                        std::string_view text);

/// The CPUs of a Linux sysfs CPU list such as "0-3,8,10-11", in order. Empty for an empty or
/// malformed list.
[[nodiscard]] std::vector<u32> ParseCpuList(std::string_view list);

} // namespace Common
