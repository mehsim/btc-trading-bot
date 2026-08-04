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

def test_public_status_endpoint(client):
    """Verify GET /api/status is accessible without API key for frontend monitoring."""
    response = client.get('/api/status')
    assert response.status_code == 200

def test_unconfigured_key_fails_closed(client, monkeypatch):
    """Verify POST /killswitch fails closed with 401 when DASHBOARD_API_KEY is unset/empty."""
    monkeypatch.setenv("DASHBOARD_API_KEY", "")
    response = client.post('/killswitch')
    assert response.status_code == 401
    data = response.get_json()
    assert data.get("error") == "Unauthorized"

def test_unauthenticated_killswitch(client, monkeypatch):
    """Verify POST /killswitch requires X-API-KEY when DASHBOARD_API_KEY is set."""
    monkeypatch.setenv("DASHBOARD_API_KEY", "secret_test_key_123")
    
    # Request without X-API-KEY header should fail with 401
    response = client.post('/killswitch')
    assert response.status_code == 401
    data = response.get_json()
    assert data.get("error") == "Unauthorized"

def test_query_string_auth_rejected(client, monkeypatch):
    """Verify query-string credentials (?api_key=...) are rejected (header-only requirement)."""
    monkeypatch.setenv("DASHBOARD_API_KEY", "secret_test_key_123")
    
    response = client.post('/killswitch?api_key=secret_test_key_123')
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

def test_all_admin_endpoints_require_header_auth(client, monkeypatch):
    """Verify all admin POST endpoints fail with 401 when no X-API-KEY header is provided."""
    monkeypatch.setenv("DASHBOARD_API_KEY", "secret_test_key_123")
    
    admin_routes = [
        '/api/terminate',
        '/api/retrain',
        '/api/close_trade',
        '/api/partial_exit_trade',
        '/api/close_all_trades',
        '/api/toggle_bot',
        '/api/reset_circuit_breaker',
        '/api/clear_history',
        '/api/test_email'
    ]
    
    for route in admin_routes:
        res = client.post(route)
        assert res.status_code == 401, f"Route {route} did not return 401 Unauthorized when unauthenticated"
