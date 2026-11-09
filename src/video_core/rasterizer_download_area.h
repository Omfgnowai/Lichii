// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace VideoCore {

enum class DownloadAreaSource : u8 { Other, Texture, Buffer };

struct RasterizerDownloadArea {
    VAddr start_address;
    VAddr end_address;
    bool preemtive;
    DownloadAreaSource source{DownloadAreaSource::Other};
    /// GPU data for the area is downloading with a fence that has not been released: the read
    /// waits for it (RasterizerInterface::WaitForPendingDownloads) and asks again.
    bool wait_downloads{};
};

} // namespace VideoCore
