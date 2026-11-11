Lichii OoT performance test build (Omfgnowai/Lichii fork)
==========================================================

What this is
------------
A test build of Lichii (zortos293/Lichii, nx2 branch) with extra performance work for the
Ocarina of Time remake (title 0400486024DEC000). It is meant for normal play on other hardware so
we can see whether the gains hold beyond the machine it was tuned on.

It contains no game mods (no randomizer, no benchmark). Diagnostics are built in but all off
unless you turn them on.

On the development PC (Ryzen 9 7950X3D, RTX 4090, 1x resolution, frame rate limit 120), Castle Town
went from ~55-62 fps to ~68-72 fps compared with the previous test build. Results on other
hardware are exactly what we want to learn.

What's in it (on top of upstream nx2 as of 2026-11-09, plus upstream perf merges of 2026-11-11)
----------------------------------------------------------------------------------------------
- GPU thread: fewer front/back round trips (equal/mask semaphore acquires without a drain, macro
  DMA reads on the front, synced list writers, suspend unmet acquires), more draws on the fast
  path (vertex layout, vertex_id_base, query methods), batched fence release, image recycler,
  skip of the once-a-frame native uniform bank upload in OoT, vertex buffer bind skips,
  per-thread GPU page translation cache, hot-page write protection skip, batched uploads.
- Upstream 2026-11-11: 3D state registers written in one call (#1095), split GpuSplit cache
  lines (#1332), dynarmic plain loads / no DMB (#1327), exclusive reservation cache lines (#1331),
  skip empty FlushCommands submits (#1329), guard house texture pool patch (#1313).
- Shaders: Switch 2 pipelines are never evicted; precompiled pipelines from a recipe (#896)
  if you have one.
- Fixes: SMMU device-space leak after ~1 h of play, Pickup Counter Fix built in (item count-up
  stalled above 60 fps), Hold 60 FPS hotkey (Ctrl+6, or Home+ZR on a controller).

Recommended settings
--------------------
- Graphics API: Vulkan. GPU accuracy: Low (shown as "FAST" in the status bar). "High" turns off
  the split GPU thread and costs a lot of fps.
- Pin GPU threads: on. Docked mode. VRAM usage: Aggressive.
- Per-game "Frame rate limit (Switch 2)": 120 (or your refresh rate). Frame generation off while
  you judge native fps.
- NVIDIA: set "Shader Cache Size" to Unlimited in the NVIDIA Control Panel.
- Logging: set the log filter to *:Warning (or *:Error) for play; *:Info costs some CPU.

Known issues above 60 fps (game-side, from FPS Unlock)
------------------------------------------------------
- Some cutscenes run too fast (the opening on a new save), the Deku Tree web doesn't deform, and
  water ripples don't animate. Press Ctrl+6 (Home+ZR) to hold 60 fps for those moments, press
  again to release.
- Occasional white glowing blobs / stretched planes after a while in a session: upstream issue
  #867, not fixed yet. Restarting clears it.

What to report
--------------
- CPU, GPU, RAM, OS, driver version, resolution scale and frame rate limit.
- Rough fps in a few places (Castle Town market square, Hyrule Field, Kokiri Forest, Lake Hylia).
- Any crash or visual problem, with the log: %APPDATA%\eden\log\eden_log.txt (copy it before
  starting Lichii again; it is overwritten at launch).
