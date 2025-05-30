import pytest
from fastapi.testclient import TestClient
from main_simple import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["service"] == "LeMCS API"


def test_upload_document():
    test_file_content = b"This is a test legal document with some sample text."
    files = {"files": ("test.txt", test_file_content, "text/plain")}

    response = client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "Successfully uploaded 1 documents" in data["message"]
    assert len(data["documents"]) == 1
    assert data["documents"][0]["filename"] == "test.txt"


def test_upload_invalid_file():
    test_file_content = b"This is a test file."
    files = {"files": ("test.xyz", test_file_content, "text/plain")}

    response = client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 400
    assert "unsupported format" in response.json()["detail"]


def test_get_document():
    response = client.get("/api/v1/documents/test-123")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "test-123"
    assert "metadata" in data
