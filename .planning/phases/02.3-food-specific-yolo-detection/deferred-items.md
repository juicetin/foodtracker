# Deferred Items - Phase 02.3

## modelLoader.ts pack path priority bug

**Found during:** Plan 03, Task 2 (human verification)
**Severity:** Medium (workaround: full uninstall+reinstall)
**Description:** `modelLoader.ts` prioritizes pack paths over bundled models. When the `installed_packs` DB has stale entries from prior phases (e.g., old COCO model path), the loader loads the old model instead of the new bundled model.
**Root cause:** No version check or migration logic when bundled models are updated between app versions.
**Suggested fix:** Add version metadata to model packs and a migration step in modelLoader that invalidates stale pack entries when the bundled model version changes.
**Workaround:** Full uninstall+reinstall clears the DB and forces the bundled model to load.

## GGCD composite dish recognition

**Found during:** Plan 03, Task 2 (human verification)
**Severity:** Low (expected model behavior, not a bug)
**Description:** GGCD YOLOv8n does not recognize composite dishes like ramen as a single entity. Instead, it detects individual components (e.g., "fried eggs" in ramen). This is a training data limitation.
**Suggested fix:** Include composite dish labels in future training data expansion (Phase 2.4).
