// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "common/common_types.h"

// EDEN_NX2_BACK_SPLIT=<workers>: the thread that draws hands pieces of a draw's preparation to a
// few worker threads and joins them later in the same draw (docs/nx2/back-split.md). The pool
// runs each job once, on a worker or, when no worker has taken it by the join, on the thread
// that joins, so a join never waits for a worker that hasn't started.
namespace VideoCommon::BackSplit {

class Job {
public:
    using Func = void (*)(void*);

    Job() = default;
    Job(const Job&) = delete;
    Job& operator=(const Job&) = delete;

    /// Whether the job was posted and not joined yet.
    [[nodiscard]] bool Pending() const noexcept {
        return state.load(std::memory_order_relaxed) != Idle;
    }

private:
    friend class Pool;

    enum State : u32 {
        Idle,    ///< Not posted, or joined
        Posted,  ///< Waiting for a worker or the join
        Running, ///< Taken by a worker
        Done,    ///< Run by a worker, not joined yet
    };

    Func func{};
    void* argument{};
    std::atomic<u32> state{Idle};
};

struct JoinResult {
    bool ran_here{};    ///< No worker had taken the job: the joining thread ran it
    u64 wait_ticks{};   ///< Time the joining thread waited for the worker running it (Ticks)
};

/// Totals since the pool started; the reader takes differences.
struct PoolTotals {
    u64 worker_jobs{};       ///< Jobs workers ran
    u64 worker_ticks{};      ///< Their time in those jobs (Ticks)
    u64 worker_cpu_ns{};     ///< Worker CPU time, spinning for jobs included (last published)
};

class Pool {
public:
    /// `measure`: time the workers' jobs (EDEN_NX2_GPU_PROFILE).
    explicit Pool(u32 workers, bool measure);
    ~Pool();

    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    /// The posting thread only. The job must not be pending. Its argument must stay valid until
    /// Join returns.
    void Post(Job& job, Job::Func func, void* argument);

    /// The posting thread only: returns once the job ran, and runs it here if no worker took it.
    JoinResult Join(Job& job);

    [[nodiscard]] u32 Workers() const noexcept {
        return static_cast<u32>(threads.size());
    }

    [[nodiscard]] PoolTotals Totals() const;

private:
    static constexpr size_t Slots = 16;

    struct alignas(64) WorkerData {
        std::atomic<u64> jobs{};
        std::atomic<u64> ticks{};
        std::atomic<u64> cpu_ns{};
    };

    void WorkerLoop(std::stop_token stop, u32 index);
    void TryRun(Job& job, WorkerData& data);

    std::array<std::atomic<Job*>, Slots> ring{};
    alignas(64) std::atomic<u64> posted{};
    alignas(64) std::atomic<u32> sleepers{};
    bool measure{};
    std::unique_ptr<WorkerData[]> worker_data;
    std::vector<std::jthread> threads;
};

} // namespace VideoCommon::BackSplit
