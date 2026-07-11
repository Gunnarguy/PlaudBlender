# PROCESS LIFECYCLE MAP

## Services Startup & Lifecycle
* All Chronos services are configured with `Restart=always` and `RestartSec` backoffs:
  * `chronos-api.service`: `RestartSec=15`
  * `chronos-auto-sync.service`: `RestartSec=15`
  * `chronos-ui.service`: `RestartSec=10`
  * `chronos-ngrok.service`: `RestartSec=20`
* Service timers regulate periodic tasks:
  * `chronos-watchdog.timer`: Runs every 5 minutes, triggers watchdog check of service endpoints.
  * `chronos-auto-update.timer`: Runs every 10 minutes.
  * `chronos-pipeline.timer`: Runs every 6 hours, triggers full ingestion pipeline.
  * `chronos-system-update.timer`: Runs daily at 2:30 AM.
