from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

def test_health():
    """Test the basic health endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"
