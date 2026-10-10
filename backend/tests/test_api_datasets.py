import pytest
from fastapi.testclient import TestClient
import sys
import os
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.main import app

client = TestClient(app)

def test_upload_dataset():
    csv_content = b"col1,col2\n1,2\n3,4"
    file = io.BytesIO(csv_content)
    file.name = "test.csv"
    
    response = client.post(
        "/datasets/upload",
        files={"file": ("test.csv", file, "text/csv")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "saved_filename" in data
    assert "profile" in data
    
    saved_name = data["saved_filename"]
    
    # Test preview
    preview_res = client.get(f"/datasets/{saved_name}/preview?page=1&size=10")
    assert preview_res.status_code == 200
    assert len(preview_res.json()["data"]) == 2
    
    # Test profile
    profile_res = client.get(f"/datasets/{saved_name}/profile")
    assert profile_res.status_code == 200
    assert "profile" in profile_res.json()
