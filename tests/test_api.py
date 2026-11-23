import pytest
from fastapi.testclient import TestClient

from docqa.api.app import create_app
from docqa.engine import DocQAEngine
from docqa.schemas import ChatResponse
from tests.conftest import FakeEmbedding, FakeLLM


class StubChat:
    async def chat(self, message, session_id=None):
        return ChatResponse(session_id=session_id or "new", answer=f"echo: {message}")


@pytest.fixture
def make_client(settings):
    def factory(**overrides):
        engine = DocQAEngine(
            settings.with_overrides(**overrides), llm=FakeLLM(), embed_model=FakeEmbedding()
        )
        return TestClient(create_app(lambda: engine, chat_provider=StubChat))

    return factory


def test_health_and_stats(make_client):
    client = make_client()
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/stats").json()["documents"] == 2


def test_query_endpoint(make_client):
    client = make_client()
    body = client.post(
        "/query", json={"question": "Mật khẩu phải có ít nhất bao nhiêu ký tự?"}
    ).json()

    assert body["status"] == "ok"
    assert body["sources"][0]["file"] == "bao_mat.md"


def test_query_validation(make_client):
    assert make_client().post("/query", json={"question": ""}).status_code == 422


def test_chat_endpoint(make_client):
    body = make_client().post("/chat", json={"message": "xin chào", "session_id": "s1"}).json()
    assert body["session_id"] == "s1" and body["answer"] == "echo: xin chào"
    assert body["sources"] == [] and body["steps"] == [] and body["llm_calls"] == 0


def test_upload_list_delete(make_client):
    client = make_client()
    client.get("/stats")  # build the initial index first
    files = {
        "file": ("gui_xe.md", "Phí gửi xe máy là 100.000 đồng mỗi tháng.".encode(), "text/markdown")
    }

    assert client.post("/documents", files=files).json()["added"] == ["uploads/gui_xe.md"]
    assert "uploads/gui_xe.md" in [d["source"] for d in client.get("/documents").json()]
    assert client.delete("/documents/uploads/gui_xe.md").json()["deleted"] == ["uploads/gui_xe.md"]
    assert client.delete("/documents/uploads/gui_xe.md").status_code == 404


def test_upload_rejects_bad_files(make_client):
    client = make_client(max_upload_mb=1)
    bad_type = client.post("/documents", files={"file": ("x.exe", b"MZ")})
    too_big = client.post("/documents", files={"file": ("big.txt", b"a" * (1024 * 1024 + 10))})
    assert bad_type.status_code == 400
    assert too_big.status_code == 400


def test_mutating_endpoints_require_api_key_when_configured(make_client):
    client = make_client(api_key="secret")

    assert client.post("/index/sync").status_code == 401
    assert client.post("/index/sync", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/index/sync", headers={"X-API-Key": "secret"}).status_code == 200
    assert (
        client.post("/query", json={"question": "mật khẩu"}).status_code == 200
    )  # reads stay open
