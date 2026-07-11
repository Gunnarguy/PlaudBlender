# BACKUP AND ROLLBACK PLAN

## Pre-Deployment Backup Steps
1. **SQLite Database Backup**
   * Execute online SQLite backup to ensure WAL consistency:
     `sqlite3 /home/gunnarhostetler/PlaudBlender/data/brain.db ".backup '/home/gunnarhostetler/PlaudBlender/data/backups/brain_backup_$(date +%F).db'"`
2. **Qdrant Collection Snapshots**
   * Trigger a Qdrant Snapshot via curl:
     `curl -X POST http://localhost:6333/collections/chronos_events_openai_v1/snapshots`
3. **Environment & Systemd Backups**
   * Copy active systemd services:
     `cp /etc/systemd/system/chronos-* /home/gunnarhostetler/PlaudBlender/data/backups/`

## Rollback Steps
1. **Code Rollback**
   * Checkout original HEAD commit `a1ad35ba023f344a4ee2be47b80a63435f2b72b3` on main.
2. **Database Rollback**
   * Stop services: `sudo systemctl stop chronos-*`
   * Replace `data/brain.db` with the pre-deployment backup copy.
   * Start services: `sudo systemctl start chronos-api.service chronos-ui.service chronos-auto-sync.service`
3. **Qdrant Rollback**
   * Restore the collection snapshot using Qdrant snapshot REST API if vectors are invalidated.
