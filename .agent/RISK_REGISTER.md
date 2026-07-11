# RISK REGISTER

## Active Risks
1. **SQLite Database Foreign Key Violations**
   * *Description*: SQLite integrity check is OK, but foreign key check returns 16 violations (12 in `chronos_execution_spans` pointing to non-existent recordings/runs, 4 in `chronos_events` pointing to non-existent recordings). This is because `PRAGMA foreign_keys = ON;` is never executed on connection initialization.
   * *Impact*: Low database drift/orphaned records.
   * *Mitigation*: Register a connect listener on the SQLAlchemy Engine to enforce `PRAGMA foreign_keys = ON;` and write a cleanup script to delete orphaned records transactionally.

2. **Swap / Memory Pressure under Pipeline Execution**
   * *Description*: The Pi has 4GB memory and 1.8GB swap. The swap is already 1.3GB used. Concurrent execution of pipeline work, Qdrant indexing, and other servers (`jobscoutos` container consumes 477MB RSS) may cause OOM kills or severe disk/swap contention.
   * *Impact*: Starvation of FastAPI/UI responsive threads.
   * *Mitigation*: Limit the autosync process limits, adjust cgroup memory bounds, and ensure `chronos-pipeline` has low CPU/IO weight scheduling.
