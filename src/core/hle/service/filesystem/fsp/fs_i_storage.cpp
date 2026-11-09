// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <thread>

#include <cstdlib>
#include <functional>
#include <string_view>

#include "common/logging.h"
#include "common/settings.h"
#include "core/file_sys/romfs.h"
#include "core/file_sys/vfs/vfs_offset.h"
#include "core/file_sys/errors.h"
#include "core/hle/service/cmif_serialization.h"
#include "core/hle/service/filesystem/fsp/fs_i_storage.h"
#include "core/hle/service/filesystem/fsp/fs_read_trace.h"

namespace Service::FileSystem {

IStorage::IStorage(Core::System& system_, FileSys::VirtualFile backend_)
    : ServiceFramework{system_, "IStorage"}, backend(std::move(backend_)) {
    static const FunctionInfo functions[] = {
        {0, D<&IStorage::Read>, "Read"},
        {1, nullptr, "Write"},
        {2, nullptr, "Flush"},
        {3, nullptr, "SetSize"},
        {4, D<&IStorage::GetSize>, "GetSize"},
        {5, nullptr, "OperateRange"},
    };
    RegisterHandlers(functions);
}

Result IStorage::Read(
    OutBuffer<BufferAttr_HipcMapAlias | BufferAttr_HipcMapTransferAllowsNonSecure> out_bytes,
    s64 offset, s64 length) {
    LOG_DEBUG(Service_FS, "called, offset={:#x}, length={}", offset, length);

    R_UNLESS(length >= 0, FileSys::ResultInvalidSize);
    R_UNLESS(offset >= 0, FileSys::ResultInvalidOffset);

    // The 40 MB/s throttle emulates a Switch 1 card (Eden cbd4cf2d8c, Mario Strikers' intro).
    // Switch 2 storage is many times faster, and a Switch 2 title that streams a movie with
    // hundreds of reads a second (Ocarina of Time's opening: 300-460) sleeps most of each second
    // on the thread that also hands decoder images back, so its video falls seconds behind.
    if (!Settings::CurrentProgramIsSwitch2()) {
        const auto now = std::chrono::steady_clock::now();
        if (now < next_read_tick) {
            std::this_thread::sleep_for(next_read_tick - now);
        }
        next_read_tick = std::chrono::steady_clock::now() +
                         std::chrono::nanoseconds{std::min(length, MaxChargedReadSize) *
                                                  1'000'000'000 / ReadBytesPerSecond};
    }

    // Read the data from the Storage backend
    const ReadTraceScope trace{ReadTraceSource::Storage, length};
    backend->Read(out_bytes.data(), length, offset);
    TraceRomFSRead(offset, length);

    R_SUCCEED();
}

namespace {
const std::vector<std::string>& RomFSTracePatterns() {
    static const std::vector<std::string> patterns = [] {
        std::vector<std::string> out;
        const char* text = std::getenv("EDEN_NX2_ROMFS_TRACE");
        if (text == nullptr) {
            return out;
        }
        std::string_view rest{text};
        while (!rest.empty()) {
            const auto comma = rest.find(',');
            const auto part = rest.substr(0, comma);
            if (!part.empty() && part != "0") {
                out.emplace_back(part);
            }
            if (comma == std::string_view::npos) {
                break;
            }
            rest.remove_prefix(comma + 1);
        }
        return out;
    }();
    return patterns;
}
} // namespace

void IStorage::TraceRomFSRead(s64 offset, s64 length) {
    const auto& patterns = RomFSTracePatterns();
    if (patterns.empty() || length <= 0) {
        return;
    }
    std::call_once(trace_index_once, [&] {
        const auto root = FileSys::ExtractRomFS(backend);
        if (root == nullptr) {
            return;
        }
        const std::function<void(const FileSys::VirtualDir&, const std::string&)> walk =
            [&](const FileSys::VirtualDir& dir, const std::string& prefix) {
                for (const auto& file : dir->GetFiles()) {
                    const std::string path = prefix + file->GetName();
                    const auto* view = dynamic_cast<const FileSys::OffsetVfsFile*>(file.get());
                    if (view == nullptr || file->GetSize() == 0) {
                        continue;
                    }
                    for (const auto& pattern : patterns) {
                        if (path.find(pattern) != std::string::npos) {
                            traced_files.push_back(
                                {view->GetOffset(), view->GetOffset() + file->GetSize(), path});
                            break;
                        }
                    }
                }
                for (const auto& sub : dir->GetSubdirectories()) {
                    walk(sub, prefix + sub->GetName() + "/");
                }
            };
        walk(root, "");
        std::sort(traced_files.begin(), traced_files.end(),
                  [](const TracedFile& a, const TracedFile& b) { return a.begin < b.begin; });
        LOG_INFO(Service_FS, "Nx2RomFS trace: {} files match", traced_files.size());
    });
    const u64 begin = static_cast<u64>(offset);
    const u64 end = begin + static_cast<u64>(length);
    auto it = std::upper_bound(traced_files.begin(), traced_files.end(), begin,
                               [](u64 value, const TracedFile& f) { return value < f.begin; });
    if (it != traced_files.begin()) {
        --it;
    }
    const auto now = std::chrono::steady_clock::now();
    for (; it != traced_files.end() && it->begin < end; ++it) {
        if (it->end <= begin) {
            continue;
        }
        std::scoped_lock lock{trace_mutex};
        auto& last = trace_last_logged[it->path];
        if (last.time_since_epoch().count() == 0 || now - last > std::chrono::seconds{5}) {
            last = now;
            LOG_INFO(Service_FS, "Nx2RomFS read {}", it->path);
        }
    }
}

Result IStorage::GetSize(Out<u64> out_size) {
    *out_size = backend->GetSize();

    LOG_DEBUG(Service_FS, "called, size={}", *out_size);

    R_SUCCEED();
}

} // namespace Service::FileSystem
