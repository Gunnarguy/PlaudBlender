from google.cloud import monitoring_v3
from google.protobuf.timestamp_pb2 import Timestamp
import time
import os

client = monitoring_v3.MetricServiceClient()
project_name = f"projects/composed-facet-360120"

now = time.time()
start = Timestamp()
start.FromSeconds(int(now - 30 * 24 * 3600))
end = Timestamp()
end.FromSeconds(int(now))

interval = monitoring_v3.TimeInterval({"start_time": start, "end_time": end})
try:
    results = client.list_time_series(
        request={
            "name": project_name,
            "filter": 'metric.type="serviceruntime.googleapis.com/api/request_count"',
            "interval": interval,
            "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.HEADERS,
        }
    )
    from collections import defaultdict

    counts = defaultdict(int)
    for series in results:
        svc = (
            series.resource.labels.get("project_id")
            + " : "
            + series.resource.labels.get("service", "unknown")
        )
        counts[svc] += 1
    print("Services with traffic in last 30 days:")
    for s in counts.keys():
        print(f"- {s}")
except Exception as e:
    print("Failed:", e)
