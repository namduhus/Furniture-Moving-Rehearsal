# Regression configuration fixtures

`configs/` here contains earlier controller-transfer experiments and a carry-training comparison used by the offline tests. They are not selected by the current rehearsal or training entry points.

The JSON files were moved unchanged from the repository's top-level `configs/`. Their `base_profile`, controller and asset references remain relative to the repository root, as the existing loaders expect. Tests explicitly select these fixtures; there is no runtime path fallback.

- `isaac_g1_deployment*`, `isaac_g1_contact*`, `isaac_g1_arm_hold*`, `isaac_g1_crouch_start*`: standing/forward/turn regression cases for successive transfer changes.
- `isaac_g1_diagnostics.json`, `isaac_g1_placement.json`, `isaac_g1_palm.json`: diagnostic and physical-property comparisons.
- `isaac_g1_room_carry_resume.json`: the earlier continued-training comparison; the active carry settings remain in the top-level `configs/`.

Historical run directories retain their original snapshots and paths. To inspect one of these profiles explicitly, pass its new `tests/fixtures/configs/...` path to the corresponding runner; do not change the recorded run artifacts.
