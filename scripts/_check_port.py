"""Quick port check."""

import socket

s = socket.socket()
s.settimeout(3)
try:
    s.connect(("127.0.0.1", 8050))
    print("APP IS UP on port 8050")
except Exception as e:
    print(f"NOT UP: {e}")
finally:
    s.close()
