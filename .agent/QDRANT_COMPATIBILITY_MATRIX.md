# QDRANT COMPATIBILITY MATRIX

## Collection Configuration
* **Active Collection**: `chronos_events_openai_v1`
  * *Point count*: 10,850 points
  * *Vector dimension*: 768
  * *Distance metric*: Cosine
  * *HNWS configuration*: `on_disk` = false (HNSW indexing in memory; uses ~250 MB peak)
  * *Payload schema index fields*: `hour_of_day`, `day_of_week`, `category`, `timestamp`
* **Star Charts Collection**: `star_charts` (5 points, vector size 4, Dot distance)
* **Legacy Collection**: `chronos_events` (2,579 points, size 768, Cosine distance)
* **Transcripts Collection**: `transcripts` (1,251 points, size 768, Cosine distance)

## Indexing & Write Wear
* Qdrant payload has `on_disk_payload=true` which reduces memory footprints on the Pi.
* Heavy indexing spikes Docker memory up to 400 MB. Compaction runs in the background.
