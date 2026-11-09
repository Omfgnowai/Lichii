# Back split (`EDEN_NX2_BACK_SPLIT`)

A test switch for EGLA-91 rank 4: the thread that draws (GPUBack with the split GPU thread) hands
two pieces of each draw to a small pool of worker threads and does the rest of the draw meanwhile.
It is off unless `EDEN_NX2_BACK_SPLIT` is set, and it stays a test switch until the PC A/B shows
a gain (AGENTS.md, "Toggles and merging").

| Variable | Effect |
|---|---|
| `EDEN_NX2_BACK_SPLIT=<workers>` | Starts that many workers (1 to 16). Unset or 0: today's path, no threads. |
| `EDEN_NX2_BACK_SPLIT_JOBS=geometry` or `=images` | Keeps only that job (default: both), to tell their effects apart. |
| `EDEN_NX2_BACK_SPLIT_VERIFY=1` | Runs the serial path as well, uses the serial result and logs every difference. |
| `EDEN_NX2_GPU_PROFILE=1` | Adds the `Back split per second` line and the `back split join` zone. |

## Design

The split is a fork-join inside each draw, not a pipeline across draws:

- **Geometry** (`buffer_cache/geometry_ranges.h`). When `PrepareDraw` starts, the back posts
  the index buffer and vertex stream ranges the buffer cache will resolve, for the streams whose
  dirty flags are set now. A worker translates their GPU addresses and sizes the ranges, the code
  `UpdateIndexBuffer` and `UpdateVertexBuffer` ran, while the back runs `FlushWork`, the
  pipeline lookup, the fast-draw checks and the wait for the cache locks. Under the locks the
  back joins the job and the buffer cache takes the prepared ranges (`UsePreparedGeometry`). It
  still finds the buffers, writes the bindings and clears the dirty flags itself.
- **Image descriptors** (`renderer_vulkan/vk_back_split.cpp`). Once a full draw's views and
  samplers are resolved, the back posts its sampled and storage image descriptors. A worker
  builds them, with `PushImageDescriptors` into its own array, while the back resolves,
  synchronizes and binds the buffers and copies the first stage's uniforms. The first stage's
  descriptor push joins the job; each stage then appends the worker's entries, and the back
  marks the written storage images and takes the rescaling words.

A join runs the job on the joining thread when no worker has taken it yet, so the back never
waits for a worker that is asleep or descheduled. It waits only for a worker that is still
running the job, and the profile line shows that time.

### Why not prepare draws ahead

The brief's design has the pool prepare draws N+1 to N+k while the back records draw N. That
does not work with how the back runs:

- Draw N+1's registers exist only after the back has run the methods between the two draws, so
  nothing can prepare N+1 while the back is still on N. Snapshotting the registers per draw
  would cost about as much as the work it moves.
- Deferring draw N's recording until its record is ready puts N+1's serial work ahead of it on
  the scheduler: uploads from buffer sync, uniform copies, render pass changes, queries. Draw N
  would then read N+1's data. To fix that, every scheduler use in the caches' runtimes would
  have to record into per-draw command lists. That is the 3-6k line rebuild in plan.md row 4,
  not a prototype.
- Method execution writes the dirty flags (a `std::bitset`) while a tail running on another
  thread would clear them, so the two would race on the same words.

Inside one draw, the caches are locked and the registers don't change, so the pieces that only
read can run beside the rest of the draw. The cost is that only work that is independent within
a draw overlaps.

## Thread safety

| State a worker touches | Why it is safe |
|---|---|
| Maxwell3D registers (`vertex_streams`, `vertex_stream_limits`, `NativeVertexStreamSize`) and `draw_state.index_buffer` | Read-only for the whole window: the back writes registers only when it runs methods, and it doesn't run any during `PrepareDraw` before the join. The front writes its own copies. The one register a draw writes (`logic_op.enable`, AMD dynamic state) is written after the join and is not one the job reads. |
| Dirty flags | Not touched. The back copies the index and vertex flags into the job when it posts it. Streams dirtied later (vertex repacking) are resolved by the buffer cache itself. |
| GPU page table (`GpuToCpuAddress`, `MaxContinuousRange`) | Read-only, as for the back. Maps through `MapInOrder` run on the GPU thread in stream order, not inside a draw. A map made directly on a guest core (nvhost_as_gpu's other paths) races with the back's own reads today. The job adds no new kind of race, and a map between post and take changes `MapSerial`, which discards the prepared ranges. `CurrentSnapshot` is per thread with its own lock. |
| Buffer cache | Not touched by the job. It reads only the record the back hands it, under the locks, after the join. |
| `views` and `samplers` arrays (`ConfigureImpl` locals) | Per draw. The back doesn't change them between post and join, and the join happens before they are used again (`CheckFeedbackLoop`) or freed (a scope guard joins on every exit). |
| Texture cache slot vectors (`GetImageView`, `GetSampler`, `IsRescaling`) | Read-only during the window: from the post to the join the back runs only buffer cache and descriptor queue code, which never calls the texture cache, so no view, image or sampler is created or deleted. The back holds the texture cache lock throughout, so guest cores can't change them either. |
| `ImageView::StorageView`, `Sampler::HandleFor` (lazy Vulkan views and sampler variants) | Changes to those objects alone. Nothing else touches them in the window. Vulkan object creation is thread safe. The device-wide border colour budget is atomic. |
| `TextureCache::MarkModification` (image flags, modification ticks) | Not called by the worker. It lists the images, and the back marks them in the same order after the join. |
| Descriptor queue, scheduler, state tracker, uniform stream buffers, memory tracker | Not touched by workers. |
| Job records | Owned by the job from `Post` to `Join`: `Post` publishes with a release store and the worker claims the job with an acquire CAS. `Done` is a release store and the join reads it with an acquire load. |

No lock's hold time on the guest cores changes. The back holds the buffer and texture cache
locks exactly as before, and the workers take no lock.

## What stays serial

- Everything outside these two jobs: flush work, uniform copies, buffer sync, handles, image
  views, render targets, queries, dynamic state, descriptor commits and recording.
- Geometry falls back to the buffer cache's own path for a stream when:
  - nothing was dirty at the post ("nothing to prepare");
  - the draw uses inline indexes;
  - a stream became dirty after the post;
  - a native stream's size doesn't fit 32 bits, where the serial path throws as before;
  - the GPU map changed between post and take ("discarded").
  Transform feedback and indirect buffers are always resolved by the buffer cache.
- Image descriptors are not posted for fast-draw delta draws, which replay their kept
  descriptors, or for pipelines without textures or images. If the worker throws, the back runs
  `PushImageDescriptors` itself, and the error is reported there as before ("failed").

## Lines to read on the PC

With `EDEN_NX2_GPU_PROFILE=1`, every 10 s, all per second:

```
Back split per second (2 workers): 164000 draws | geometry: 120000 prepared, 119990 used, 44000
nothing to prepare, 10 discarded (GPU map changed), 0 vertex streams resolved on the back | image
descriptors: 60000 prepared, 60000 used, 0 failed, 90000 delta draws | jobs: 175000 by workers,
5000 by the back | back waited 4.1 ms in 9000 joins | workers: 38.0 ms in jobs, CPU 1990.0 ms |
back CPU 810.0 ms
```

(These numbers show the format only; they are not a measurement.)

- **jobs by workers / by the back:** "by the back" are jobs no worker had started by the join.
- **back waited:** time the back spun at joins for a worker still running the job. The
  `back split join` zone in the `GPU back thread per second` line covers every join, jobs
  the back ran itself included.
- **workers ms in jobs:** the work moved off the back. **CPU** includes the workers' spinning
  while they wait for jobs, about a core each while draws come.
- **back CPU:** the back thread's CPU time, as `host_cpu_busy_ms_per_s` in the zone line.

With `EDEN_NX2_BACK_SPLIT_VERIFY=1`, every 10 s:

```
Back split verify per second: 180000 checks (120000 geometry, 60000 image descriptors), 0
mismatches (0 geometry, 0 image descriptors) | totals: 1800000 checks, 0 mismatches
```

The first 16 mismatches are logged in full as `Back split verify mismatch n/16: ...`, with the
vertex shader or pipeline hash, the slot or stage and entry, and both values. Verify runs both
paths, so its frame rate says nothing about the gain.

## Running the A/B

```sh
bash C:/lx/build-fix.sh egla91/back-split back-split "Bananza - back split" \
  "EDEN_NX2_DEBUG_SERVER=127.0.0.1:6545 EDEN_NX2_BACK_SPLIT=2 EDEN_NX2_GPU_PROFILE=1" "*:Info"
```

Alternate launches with `EDEN_NX2_BACK_SPLIT=0` and `2` (and `3` or 4, or one job via
`EDEN_NX2_BACK_SPLIT_JOBS`) at the Ingot Isle gate. Compare frame rate and the back's CPU per
frame without the profile, then use one profiled launch per setting for the lines above. Run
OoT at Kokiri and MKW Free Roam once each with verify, and check that there are 0 mismatches.
