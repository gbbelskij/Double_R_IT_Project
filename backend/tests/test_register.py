from unittest.mock import patch, MagicMock

def test_revoked_token(client):
    with patch('backend.handlers.personal_account.TokenBlockList') as mock_model:
        mock_model.query.filter_by.return_value.first.return_value = MagicMock()
        response = client.get('/personal_account/',
                            headers={'Authorization': 'Bearer revoked-token'})
        assert response.status_code == 401

def test_missing_token(client):
    response = client.get('/personal_account/')
    assert response.status_code == 401
