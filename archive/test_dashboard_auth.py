import os
import pytest
import main

@pytest.fixture
def client():
    main.app.config['TESTING'] = True
    with main.app.test_client() as client:
        yield client

def test_public_health_endpoint(client):
    """Verify GET /api/health is accessible without API key."""
    response = client.get('/api/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data.get("status") == "ok"

def test_unauthenticated_killswitch(client, monkeypatch):
    """Verify POST /killswitch requires X-API-KEY when DASHBOARD_API_KEY is set."""
    monkeypatch.setenv("DASHBOARD_API_KEY", "secret_test_key_123")
    
    # Request without X-API-KEY header should fail with 401
    response = client.post('/killswitch')
    assert response.status_code == 401
    data = response.get_json()
    assert data.get("error") == "Unauthorized"

def test_authenticated_killswitch(client, monkeypatch):
    """Verify POST /killswitch succeeds with valid X-API-KEY header."""
    monkeypatch.setenv("DASHBOARD_API_KEY", "secret_test_key_123")
    monkeypatch.setattr("dashboard_routes.trigger_emergency_kill_switch", lambda *args, **kwargs: None)
    
    # Request with valid X-API-KEY header
    headers = {"X-API-KEY": "secret_test_key_123"}
    response = client.post('/killswitch', headers=headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data.get("status") == "KILL_SWITCH_ACTIVATED"
