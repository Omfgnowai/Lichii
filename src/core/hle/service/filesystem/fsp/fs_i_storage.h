// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/file_sys/vfs/vfs.h"
#include "core/hle/service/cmif_types.h"
#include "core/hle/service/filesystem/filesystem.h"
#include "core/hle/service/service.h"

namespace Service::FileSystem {

class IStorage final : public ServiceFramework<IStorage> {
public:
    explicit IStorage(Core::System& system_, FileSys::VirtualFile backend_);

private:
    FileSys::VirtualFile backend;

    static constexpr s64 ReadBytesPerSecond = 40'000'000;
    static constexpr u64 AgeOfImprisonmentProgramId = 0x04003F40232DE000ULL;
    static constexpr s64 AgeOfImprisonmentReadBytesPerSecond = 100'000'000;
    static constexpr s64 MaxChargedReadSize = 64 * 1024;
    s64 bytes_per_second = 0;
    std::chrono::steady_clock::time_point next_read_tick{};

    /// EDEN_NX2_ROMFS_TRACE=<substring>[,<substring>...]: when the backend is a RomFS image, log
    /// "Nx2RomFS read <path>" for reads of files whose path contains one of the substrings (once
    /// per file per 5 s), e.g. "_Static.bcett" marks Ocarina's area loads in a benchmark log.
    struct TracedFile {
        u64 begin;
        u64 end;
        std::string path;
        bool logged; ///< matches EDEN_NX2_ROMFS_TRACE (else indexed only for the sampler)
    };
    /// With EDEN_NX2_SAMPLE every RomFS file is indexed and each read goes to the sampler's
    /// timeline (Nx2Sampler::NoteFileRead) with its time and the first file it touches.
    void TraceRomFSRead(s64 offset, s64 length, u64 begin_qpc);
    std::once_flag trace_index_once;
    std::vector<TracedFile> traced_files; ///< sorted by begin
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> trace_last_logged;
    std::mutex trace_mutex;

    Result Read(
        OutBuffer<BufferAttr_HipcMapAlias | BufferAttr_HipcMapTransferAllowsNonSecure> out_bytes,
        s64 offset, s64 length);
    Result GetSize(Out<u64> out_size);
};

} // namespace Service::FileSystem
