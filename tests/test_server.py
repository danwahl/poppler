import pytest
from mcp import Client

from poppler.server import mcp

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def test_every_parameter_is_described():
    async with Client(mcp) as client:
        tools = (await client.list_tools()).tools
    assert {t.name for t in tools} == {
        "gpu_status",
        "submit",
        "wait_job",
        "job_log",
        "list_jobs",
        "cancel_job",
    }
    for tool in tools:
        assert tool.description
        for name, prop in tool.input_schema.get("properties", {}).items():
            assert prop.get("description"), f"{tool.name}.{name}"


async def test_submit_wait_and_read_log(tmp_path):
    async with Client(mcp) as client:
        submitted = await client.call_tool(
            "submit",
            {"command": "echo hello", "owner": "test", "cwd": str(tmp_path), "qos": "scavenger"},
        )
        job_id = submitted.structured_content["id"]
        finished = await client.call_tool("wait_job", {"job_id": job_id, "timeout_s": 5})
        log = await client.call_tool("job_log", {"job_id": job_id})
    assert finished.structured_content["state"] == "COMPLETED"
    assert finished.structured_content["owner"] == "test"
    assert finished.structured_content["qos"] == "scavenger"
    assert log.content[0].text == "hello\n"


async def test_cancel(tmp_path):
    async with Client(mcp) as client:
        submitted = await client.call_tool("submit", {"command": "sleep 30", "cwd": str(tmp_path)})
        job_id = submitted.structured_content["id"]
        cancelled = await client.call_tool("cancel_job", {"job_id": job_id})
    assert cancelled.structured_content["state"] == "CANCELLED"
