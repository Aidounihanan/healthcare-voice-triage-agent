# healthcare_mcp_server/server.py
import asyncio
import json
from datetime import datetime, timedelta

from mcp.server import Server, NotificationOptions
from mcp.types import Tool, TextContent
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

from llamaindex_kb import get_triage_answer

server = Server("healthcare-mcp-server")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="triage_patient",
            description=(
                "Analyze the patient's profile (symptoms, risk factors, etc.) "
                "and return an urgency level + recommendations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "age": {"type": "integer"},
                    "symptoms": {"type": "string"},
                    "duration": {"type": "string"},
                    "risk_factors": {"type": "string"},
                    "other_context": {"type": "string"},
                },
                "required": ["age", "symptoms", "duration"],
            },
        ),
        Tool(
            name="schedule_appointment",
            description=(
                "Propose a simulated appointment slot depending on urgency level."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "urgency_level": {
                        "type": "string",
                        "enum": ["critical", "high", "moderate", "low"],
                    },
                    "speciality": {"type": "string"},
                },
                "required": ["urgency_level"],
            },
        ),
        Tool(
            name="notify_team",
            description=(
                "Notify the medical team or emergency department with a summary."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "urgency_level": {"type": "string"},
                    "patient_summary": {"type": "string"},
                    "appointment_slot": {"type": "string"},
                },
                "required": ["urgency_level", "patient_summary"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "triage_patient":
        try:
            result = await handle_triage_patient(arguments)
            payload = {"ok": True, **result}
        except Exception as e:
            payload = {
                "ok": False,
                "error": f"Triage error: {e.__class__.__name__}",
                "details": str(e),
            }
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    elif name == "schedule_appointment":
        try:
            result = await handle_schedule_appointment(arguments)
            payload = {"ok": True, **result}
        except Exception as e:
            payload = {
                "ok": False,
                "error": f"Schedule error: {e.__class__.__name__}",
                "details": str(e),
            }
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    elif name == "notify_team":
        try:
            result = await handle_notify_team(arguments)
            payload = {"ok": True, **result}
        except Exception as e:
            payload = {
                "ok": False,
                "error": f"Notify error: {e.__class__.__name__}",
                "details": str(e),
            }
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    else:
        payload = {"ok": False, "error": f"Unknown tool: {name}"}
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]


async def handle_triage_patient(args: dict) -> dict:
    age = args.get("age")
    symptoms = args.get("symptoms", "")
    duration = args.get("duration", "")
    risk_factors = args.get("risk_factors", "")
    other = args.get("other_context", "")

    query = (
        f"Patient aged {age} years old. Symptoms: {symptoms}. Duration: {duration}. "
        f"Risk factors: {risk_factors}. Other context: {other}. "
        "According to the guidelines, what urgency level is appropriate "
        "(critical / high / moderate / low) and what basic recommendations apply?"
    )
    kb_answer = get_triage_answer(query)

    txt = kb_answer.lower()
    urgency_level = "moderate"
    if "emergency" in txt or "life-threatening" in txt or "call 911" in txt:
        urgency_level = "critical"
    elif "emergency department" in txt or "urgent evaluation" in txt:
        urgency_level = "high"
    elif "monitor at home" in txt or "mild" in txt:
        urgency_level = "low"

    return {
        "urgency_level": urgency_level,
        "guidelines_answer": kb_answer,
    }


async def handle_schedule_appointment(args: dict) -> dict:
    level = args.get("urgency_level", "moderate")
    speciality = args.get("speciality", "general practitioner")

    now = datetime.utcnow()
    if level == "critical":
        slot = "IMMEDIATE (emergency department recommended)"
    elif level == "high":
        slot = (now + timedelta(hours=2)).isoformat() + "Z"
    elif level == "moderate":
        slot = (now + timedelta(days=1)).isoformat() + "Z"
    else:
        slot = (now + timedelta(days=3)).isoformat() + "Z"

    return {
        "selected_slot": slot,
        "speciality": speciality,
        "note": "Simulated appointment slot for demo purposes.",
    }


async def handle_notify_team(args: dict) -> dict:
    urgency_level = args.get("urgency_level", "moderate")
    summary = args.get("patient_summary", "")
    slot = args.get("appointment_slot", "")

    message = (
        f"[MOCK NOTIFICATION] Urgency: {urgency_level.upper()} | "
        f"Appointment: {slot} | Patient summary: {summary[:200]}..."
    )
    return {
        "status": "sent",
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

async def main():
    """
    Run the MCP server over stdio so that the client in app.py can connect.
    This follows the official MCP quickstart pattern.
    """
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="healthcare-mcp-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())