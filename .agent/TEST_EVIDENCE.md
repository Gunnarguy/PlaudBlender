# TEST EVIDENCE

## Local Test Run
* Local `.venv` python virtual environment symlink is currently broken due to pyenv 3.12.9 version deletion on the host. Recreating virtual environment is deferred to implementation phase.

## Remote Pi Test Run
* **Command**: `ssh gunnarhostetler@10.0.0.170 "cd /home/gunnarhostetler/PlaudBlender && venv/bin/pytest"`
* **Exit Code**: `1`
* **Summary**: `1 failed, 356 passed, 1 warning in 35.05s`
* **Warnings**: Pydantic Deprecated copy method warning in `test_database_models.py::test_upsert_recording_inserts_and_updates`.
* **Failed Test**:
  ```text
  FAILED tests/test_api.py::TestSettings::test_get_settings_exposes_autosync_controls
  E   assert 500 == 200
  E    +  where 500 = <Response [500 Internal Server Error]>.status_code
  ```
  * *Cause*: Missing fields in SimpleNamespace mock: `chronos_index_events_per_limit`, `chronos_autosync_index_timeout`, and `chronos_stats_enable_plaud_cloud` causing AttributeError during serialization.
