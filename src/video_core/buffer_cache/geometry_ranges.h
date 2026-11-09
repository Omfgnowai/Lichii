// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <limits>
#include <optional>

#include "common/common_types.h"
#include "common/literals.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"

namespace VideoCommon {

/// The device range the index buffer or a vertex stream binds, before its buffer is found: what
/// BufferCache::UpdateIndexBuffer and UpdateVertexBuffer compute from the registers and the GPU
/// page table.
struct GeometryRange {
    DAddr device_addr{};
    u32 size{};
    bool bound{}; ///< False: the null binding

    bool operator==(const GeometryRange&) const = default;
};

/// The index buffer of a draw that doesn't use inline indexes.
inline GeometryRange IndexBufferRange(const Tegra::Engines::Maxwell3D::Regs::IndexBuffer& index_buffer,
                                      const Tegra::MemoryManager& gpu_memory) {
    const GPUVAddr gpu_addr_begin = index_buffer.StartAddress();
    const GPUVAddr gpu_addr_end = index_buffer.EndAddress();
    const std::optional<DAddr> device_addr = gpu_memory.GpuToCpuAddress(gpu_addr_begin);
    const u32 address_size = static_cast<u32>(gpu_addr_end - gpu_addr_begin);
    const u32 draw_size =
        (index_buffer.count + index_buffer.first) * u32(index_buffer.FormatSizeInBytes());
    const u32 size = (std::min)(address_size, draw_size);
    if (size == 0 || !device_addr) {
        return {};
    }
    return {.device_addr = *device_addr, .size = size, .bound = true};
}

/// Vertex stream `index`. nullopt: a native stream whose size doesn't fit 32 bits, which
/// UpdateVertexBuffer rejects.
inline std::optional<GeometryRange> VertexStreamRange(const Tegra::Engines::Maxwell3D::Regs& regs,
                                                      bool native,
                                                      const Tegra::MemoryManager& gpu_memory,
                                                      u32 index) {
    using namespace Common::Literals;
    const auto& array = regs.vertex_streams[index];
    const auto& limit = regs.vertex_stream_limits[index];
    const GPUVAddr gpu_addr_begin = array.Address();
    const u64 native_size = native ? regs.NativeVertexStreamSize(index) : 0;
    if (native && array.enable != 0 && native_size > (std::numeric_limits<u32>::max)()) {
        return std::nullopt;
    }
    const GPUVAddr gpu_addr_end = native ? gpu_addr_begin + native_size : limit.Address() + 1;
    const std::optional<DAddr> device_addr = gpu_memory.GpuToCpuAddress(gpu_addr_begin);
    u32 size = static_cast<u32>(gpu_addr_end - gpu_addr_begin);
    if (array.enable == 0 || size == 0 || !device_addr) {
        return GeometryRange{};
    }
    if (!gpu_memory.IsWithinGPUAddressRange(gpu_addr_end) || size >= 64_MiB) {
        size = static_cast<u32>(gpu_memory.MaxContinuousRange(gpu_addr_begin, size));
    }
    return GeometryRange{.device_addr = *device_addr, .size = size, .bound = true};
}

/// Index and vertex stream ranges a back split worker resolved for the current draw
/// (EDEN_NX2_BACK_SPLIT), which the buffer cache takes instead of resolving them itself. With
/// `verify` it resolves them anyway, takes its own and counts the differences.
struct PreparedGeometry {
    static constexpr u32 IndexSlot =
        static_cast<u32>(Tegra::Engines::Maxwell3D::Regs::NumVertexArrays);

    struct Mismatch {
        u32 slot; ///< A vertex stream, or IndexSlot
        GeometryRange parallel;
        GeometryRange serial;
    };

    std::array<GeometryRange, IndexSlot + 1> ranges{};
    u64 prepared_mask{}; ///< Bit per slot in ranges
    bool verify{};
    u32 mismatches{};
    std::array<Mismatch, 4> first_mismatches{};

    void Clear() noexcept {
        prepared_mask = 0;
        mismatches = 0;
    }

    void Set(u32 slot, const GeometryRange& range) noexcept {
        ranges[slot] = range;
        prepared_mask |= u64{1} << slot;
    }

    [[nodiscard]] bool Has(u32 slot) const noexcept {
        return ((prepared_mask >> slot) & 1) != 0;
    }

    /// The prepared range of `slot` when there is one, else `serial()`.
    template <typename Serial>
    std::optional<GeometryRange> Resolve(u32 slot, Serial&& serial) {
        if (!Has(slot)) {
            return serial();
        }
        if (!verify) {
            return ranges[slot];
        }
        const std::optional<GeometryRange> expected = serial();
        if (expected != ranges[slot]) {
            if (mismatches < first_mismatches.size()) {
                first_mismatches[mismatches] = {slot, ranges[slot],
                                                expected.value_or(GeometryRange{})};
            }
            ++mismatches;
        }
        return expected;
    }
};

} // namespace VideoCommon
