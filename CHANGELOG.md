# Changelog

Changes to the team 048 solution, grouped by topic. Run labels (v5–v11b) name our recorded simulator runs; see
[README.en.md](README.en.md#simulator-runs) for their results.

## Perception
- Table plane by RANSAC on the head camera point cloud, objects as coloured segments above it, identified by size.
- Trays found by colour and depth; the long-axis sign is canonical (`trays.canonical_axis`), so a 0°/90° tie
  no longer depends on the last bit of rounding.
- `bgr8` frames are converted to RGB before colour checks.
- An empty depth frame is rejected before it reaches the Rust kernel.
- A camera frame is reused only if it is at most two camera periods old.

## Grasping and hand control
- Hold rule from finger positions and effort after closing; the older finger criteria are only logged.
- Guarded approach: when an open finger is off its command by more than `GUARD_DEV` (a contact), the approach
  stops without closing and the scene is observed again.
- Finger targets are clamped to the closed target and never exceed 1.0; jammed fingers keep the closed target.
- Grasp centre shifted so that the open hand (point cloud from the hand meshes) does not cut the measured object box.
- Rim pinch only when the object's weight torque on the rim is at most `RIM_TORQUE_MAX` (0.17 N·m); no rim pinch on a
  closed top.
- A dish taken by the rim is placed by its centre in the slot.
- Finger force 0.3–0.5 (rule #648) is sent to both hands and retried until each hand answers; a hand without a
  confirmed force request is not used. `TEAM048_HAND_FORCE` is clamped to 0.3–0.5. Thumb rotation force stays 1.0
  (see [docs/COMPLIANCE.md](docs/COMPLIANCE.md)).

## Planning and motion
- Start posture reached by a certified escape path; the table model starts at the measured near edge; arm-to-torso
  clearance is part of the collision model.
- Light objects first; time per cycle includes the lift and the measured settling time per arm move.
- Before a lift the remaining time must cover the worst-case holding time.
- The wait for `/joint_states` is bounded by the time budget; a joint missing from `/joint_states` stops the arm
  instead of being read as 0 rad.

## Robustness
- Camera pair buffer copied under a lock (callbacks run in another thread).
- An exception during a pick opens the hands and returns the arms home; the run continues.
- Setup errors and the end of the run always publish `/sim/task/done`.
- A failed observation is retried once.
- Hand state history is replaced atomically; tray components are sorted with a stable sort.

## Rust acceleration
- Optional `team048_accel` module (PyO3, abi3) for kinematics, collisions, perception and grasp kernels, with a NumPy
  fallback for the rest of the process on any error. A prebuilt wheel with its SHA256 is in `vendor/`.
- On 62 recorded frames the discrete results (object, grasp type, tray, slot, collision verdicts) match NumPy in all
  5358 records.

## Tests and documentation
- 237 offline tests; 16 of them need recorded frames (`TEAM048_V7_TRACE`), 8 more need the Rust module.
- Test files are named by topic; fixtures describe the recorded run and log line they come from.
- Comments explain why a rule exists instead of naming the development step that added it.
- Docs reference code by file and line; the references and the parameter table are checked against the code.
