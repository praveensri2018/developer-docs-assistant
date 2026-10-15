from fastapi.testclient import TestClient
from app import app
from core import Assistant
client = TestClient(app)
def test_supported_question_has_real_source():
    r = client.post('/ask', json={'question': 'What happens when tokens expire?'}).json()
    assert r['status'] == 'excerpts_only'
    assert r['sources'][0]['source'] == 'orders.md'
    assert '401' in r['answer']
    assert r['trace'] == ['retrieve', 'generate']
def test_unknown_question_abstains():
    r = client.post('/ask', json={'question': 'Who won the Wimbledon tennis championship?'}).json()
    assert r['status'] == 'insufficient_evidence'
    assert r['trace'] == ['retrieve', 'abstain']
def test_uploaded_document_and_reset():
    try:
        r = client.post('/upload', files={'files': ('guide.md', b'Backups are retained for thirty days.', 'text/plain')})
        assert r.status_code == 200
        assert 'thirty' in client.post('/ask', json={'question': 'How long are backups retained?'}).json()['answer']
        assert client.post('/upload', files={'files': ('bad.exe', b'bad')}).status_code == 400
    finally:
        client.post('/reset')
def test_empty_document():
    a = Assistant()
    import pytest
    with pytest.raises(ValueError):
        a.ingest([('empty.md', 1, '')])
def test_provider_error(monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'openai')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    assert client.post('/ask', json={'question': 'How do tokens expire?'}).json()['status'] == 'provider_error'
