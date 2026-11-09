// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <string>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "common/thread.h"
#include "common/thread_cpu_clock.h"
#include "video_core/back_split_pool.h"
#include "video_core/nx2_profile.h"

namespace VideoCommon::BackSplit {
namespace {

void Pause() {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

// Pauses a worker spins for a job before it sleeps: a few hundred microseconds, longer than
// the gap between two draws, so a worker sleeps only when the drawing thread stops drawing.
constexpr u32 SpinLimit = 1U << 14;
// Spins between two publications of the worker's CPU time.
constexpr u32 CpuPublishSpins = 1U << 12;

} // Anonymous namespace

Pool::Pool(u32 workers, bool measure_) : measure{measure_} {
    worker_data = std::make_unique<WorkerData[]>(workers);
    threads.reserve(workers);
    for (u32 index = 0; index < workers; ++index) {
        threads.emplace_back([this, index](std::stop_token stop) { WorkerLoop(stop, index); });
    }
}

Pool::~Pool() {
    for (auto& thread : threads) {
        thread.request_stop();
    }
    posted.fetch_add(1, std::memory_order_seq_cst);
    posted.notify_all();
    threads.clear();
}

void Pool::Post(Job& job, Job::Func func, void* argument) {
    job.func = func;
    job.argument = argument;
    job.state.store(Job::Posted, std::memory_order_release);
    const u64 sequence = posted.load(std::memory_order_relaxed);
    ring[sequence % Slots].store(&job, std::memory_order_release);
    posted.store(sequence + 1, std::memory_order_seq_cst);
    if (sleepers.load(std::memory_order_seq_cst) != 0) {
        posted.notify_all();
    }
}

JoinResult Pool::Join(Job& job) {
    u32 expected = Job::Posted;
    if (job.state.compare_exchange_strong(expected, Job::Running, std::memory_order_acquire,
                                          std::memory_order_acquire)) {
        job.func(job.argument);
        job.state.store(Job::Idle, std::memory_order_relaxed);
        return {.ran_here = true};
    }
    JoinResult result{};
    if (expected != Job::Done) {
        const u64 start = Nx2Profile::Ticks();
        while (job.state.load(std::memory_order_acquire) != Job::Done) {
            Pause();
        }
        result.wait_ticks = Nx2Profile::Ticks() - start;
    }
    job.state.store(Job::Idle, std::memory_order_relaxed);
    return result;
}

PoolTotals Pool::Totals() const {
    PoolTotals totals{};
    for (size_t index = 0; index < threads.size(); ++index) {
        const WorkerData& data = worker_data[index];
        totals.worker_jobs += data.jobs.load(std::memory_order_relaxed);
        totals.worker_ticks += data.ticks.load(std::memory_order_relaxed);
        totals.worker_cpu_ns += data.cpu_ns.load(std::memory_order_relaxed);
    }
    return totals;
}

void Pool::TryRun(Job& job, WorkerData& data) {
    u32 expected = Job::Posted;
    if (!job.state.compare_exchange_strong(expected, Job::Running, std::memory_order_acquire,
                                           std::memory_order_relaxed)) {
        return;
    }
    const u64 start = measure ? Nx2Profile::Ticks() : 0;
    job.func(job.argument);
    if (measure) {
        data.ticks.store(data.ticks.load(std::memory_order_relaxed) + Nx2Profile::Ticks() - start,
                         std::memory_order_relaxed);
    }
    data.jobs.store(data.jobs.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    job.state.store(Job::Done, std::memory_order_release);
}

void Pool::WorkerLoop(std::stop_token stop, u32 index) {
    Common::SetCurrentThreadName(("GPUBackSplit" + std::to_string(index)).c_str());
    WorkerData& data = worker_data[index];
    const auto publish_cpu = [&] {
        if (const auto cpu_ns = Common::CurrentThreadCpuTimeNs()) {
            data.cpu_ns.store(*cpu_ns, std::memory_order_relaxed);
        }
    };
    u64 seen = posted.load(std::memory_order_acquire);
    u32 spins = 0;
    while (!stop.stop_requested()) {
        const u64 now_posted = posted.load(std::memory_order_acquire);
        if (now_posted == seen) {
            ++spins;
            if (measure && spins % CpuPublishSpins == 0) {
                publish_cpu();
            }
            if (spins < SpinLimit) {
                Pause();
                continue;
            }
            if (measure) {
                publish_cpu();
            }
            sleepers.fetch_add(1, std::memory_order_seq_cst);
            if (posted.load(std::memory_order_seq_cst) == seen && !stop.stop_requested()) {
                posted.wait(seen, std::memory_order_seq_cst);
            }
            sleepers.fetch_sub(1, std::memory_order_seq_cst);
            spins = 0;
            continue;
        }
        spins = 0;
        // A worker that fell a ring behind skips what was overwritten: those jobs are joined
        // (and run, if no other worker took them) by the posting thread anyway.
        if (now_posted - seen > Slots) {
            seen = now_posted - Slots;
        }
        for (; seen != now_posted; ++seen) {
            if (Job* const job = ring[seen % Slots].load(std::memory_order_acquire)) {
                TryRun(*job, data);
            }
        }
    }
}

} // namespace VideoCommon::BackSplit
