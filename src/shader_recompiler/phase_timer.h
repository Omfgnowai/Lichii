// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// EDEN_NX2_SHADER_PHASES=1 (measurement): where the GPU thread's time goes when it builds a
// pipeline at run time. Each thread adds to its own counters; the Vulkan pipeline cache logs the
// GPU thread's every 10 s beside its runtime pipeline totals.

#pragma once

#include <array>
#include <chrono>
#include <cstdlib>

#include "common/common_types.h"

namespace Shader::PhaseTimer {

enum class Phase : u8 {
    Environments, ///< GetGraphicsEnvironments: program analysis, native code lowering, hashes
    Cfg,          ///< Flow::CFG
    Decode,       ///< BuildASL: decoding and IR emission
    Ssa,
    ConstantPropagation,
    GlobalMemory, ///< GlobalMemoryToStorageBufferPass
    Texture,      ///< TexturePass
    DeadCode,
    ShaderInfo, ///< CollectShaderInfoPass
    Translate,  ///< all of TranslateProgram (the passes above are part of it)
    MergeVertex,
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Phase::Count)> Names{
    "environments", "cfg", "decode", "ssa", "constant propagation", "global memory",
    "texture", "dead code", "shader info", "translate total", "dual vertex merge",
};

/// Read on first use: nx2_switches.txt sets the variable after static initialisation.
inline bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_SHADER_PHASES");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

inline thread_local std::array<u64, static_cast<size_t>(Phase::Count)> t_ns{};

class Scope {
public:
    explicit Scope(Phase phase_) : phase{phase_}, enabled{Enabled()} {
        if (enabled) {
            start = std::chrono::steady_clock::now();
        }
    }
    ~Scope() {
        if (enabled) {
            t_ns[static_cast<size_t>(phase)] += static_cast<u64>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start)
                    .count());
        }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Phase phase;
    bool enabled;
    std::chrono::steady_clock::time_point start{};
};

} // namespace Shader::PhaseTimer
