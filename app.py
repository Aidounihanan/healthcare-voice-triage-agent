# app.py
import os
import json
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

# MCP client (new API)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

# 👉 Charger .env tout de suite
load_dotenv()

# Local modules
from stt_elevenlabs_client import transcribe_audio_file
from tts_elevenlabs_client import tts_to_wav

# ----------------------
#   ENV & CONFIG
# ----------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

PROJECT_ROOT = Path(__file__).parent

# ----------------------
#   MCP – CALL TOOL
# ----------------------
async def _call_mcp_tool_async(tool_name: str, args: dict) -> dict:
    """
    Start the MCP server as a subprocess, initialize the session,
    call a tool, then shut it down.
    Simple & robust for hackathon / demo.
    """
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "healthcare_mcp_server.server"],
        cwd=str(PROJECT_ROOT),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1) initialize
            await session.initialize()

            # 2) call tool
            result = await session.call_tool(tool_name, args)

            if not result.content:
                return {
                    "error": "MCP tool returned no content",
                    "tool_name": tool_name,
                }

            for c in result.content:
                if isinstance(c, TextContent) or getattr(c, "type", None) == "text":
                    raw = (c.text or "").strip()
                    if not raw:
                        return {
                            "error": "MCP tool returned empty text content",
                            "tool_name": tool_name,
                        }
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError as e:
                        return {
                            "error": "Invalid JSON from MCP tool",
                            "tool_name": tool_name,
                            "raw_text": raw,
                            "json_error": str(e),
                        }

            # no text content
            return {
                "error": "No TextContent from MCP tool",
                "tool_name": tool_name,
                "content": [str(c) for c in result.content],
            }


def call_mcp_tool(tool_name: str, args: dict) -> dict:
    """Synchronous wrapper for Gradio callbacks."""
    return asyncio.run(_call_mcp_tool_async(tool_name, args))


# ----------------------
#   OPENAI – CHAT & EXTRACTION
# ----------------------
def openai_chat(messages):
    """Standard chat completion for the dialog."""
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
    )
    return resp.choices[0].message.content


def extract_patient_profile(conv_text: str) -> dict:
    """
    Use OpenAI to extract a structured patient profile from the full conversation.
    Output must be valid JSON with:
      - age (int)
      - symptoms (string)
      - duration (string)
      - risk_factors (string)
      - other_context (string)
    """
    system = {
        "role": "system",
        "content": (
            "You are a medical assistant. From the following conversation between an agent "
            "and a patient, extract a STRICT JSON object with the keys: "
            "age (int), symptoms (string), duration (string), "
            "risk_factors (string), other_context (string). "
            "If information is missing, fill with an empty string or a reasonable default."
        ),
    }
    user = {"role": "user", "content": conv_text}

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[system, user],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


# ----------------------
#   GLOBAL CONVERSATION STATE
# ----------------------
# We keep messages in OpenAI format: [{"role": "user"/"assistant", "content": "..."}]
conversation_history: list[dict] = []


# ----------------------
#   AGENT LOGIC – TEXT + VOICE
# ----------------------
def _agent_reply_from_text(user_text: str) -> str:
    """Core logic: add user text, call OpenAI, get assistant reply (in English)."""
    # Add user message to history
    conversation_history.append({"role": "user", "content": user_text})

    # Build full messages
    messages = [
        {
            "role": "system",
            "content": (
                "You are a warm, empathetic medical intake agent speaking ENGLISH. "
                "Your job is to ask clear questions about the patient's symptoms, "
                "duration, risk factors, and any relevant context. "
                "Keep answers short, natural, and friendly, like a real call center nurse."
            ),
        }
    ]
    messages.extend(conversation_history)

    # Call OpenAI
    assistant_reply = openai_chat(messages)

    # Add assistant reply to history
    conversation_history.append({"role": "assistant", "content": assistant_reply})

    return assistant_reply


def handle_audio_input(audio):
    """
    Gradio audio callback:
    - Receive microphone input (filepath)
    - STT via ElevenLabs
    - Add to conversation
    - LLM reply
    - TTS via ElevenLabs
    Returns: chatbot messages, agent audio filepath
    """
    if audio is None:
        return [], None

    # Handle different formats from Gradio
    if isinstance(audio, dict) and "filepath" in audio:
        file_path = audio["filepath"]
    elif isinstance(audio, str):
        file_path = audio
    else:
        raise RuntimeError(
            "Unexpected audio format. Please configure gr.Audio(type='filepath')."
        )

    # --- STT (ElevenLabs) ---
    try:
        user_text = transcribe_audio_file(file_path, language="en") or "[Empty transcription]"
    except Exception as e:
        user_text = f"[STT error: {e}]"

    # --- LLM reply ---
    assistant_reply = _agent_reply_from_text(user_text)

    # --- TTS (ElevenLabs) ---
    agent_audio_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    try:
        tts_to_wav(assistant_reply, agent_audio_path)
    except Exception as e:
        # If TTS fails, we still return the text in the chat
        print("[TTS ERROR]", e)
        agent_audio_path = None

    # We simply return the raw messages for Gradio Chatbot
    return conversation_history, agent_audio_path


def handle_text_input(user_text: str):
    """
    Optional: text chat entry (for debugging or fallback).
    Returns: chatbot messages (no audio).
    """
    if not user_text.strip():
        return [], ""

    assistant_reply = _agent_reply_from_text(user_text)

    # Build chatbot display
    # Return messages directly for Chatbot
    return conversation_history, ""


def end_call_and_generate_report():
    """
    End the call:
      - Build full text conversation
      - Extract patient profile with OpenAI JSON
      - Call MCP tools:
          triage_patient
          schedule_appointment
          notify_team
      - Return final READABLE report
    """
    if not conversation_history:
        return "No conversation found."

    # 1) Build plain-text transcript
    conv_text = ""
    for msg in conversation_history:
        prefix = "Patient: " if msg["role"] == "user" else "Agent: "
        conv_text += prefix + msg["content"] + "\n"

    # 2) Extract patient profile
    try:
        patient_profile = extract_patient_profile(conv_text)
    except Exception as e:
        return "⚠️ Error while extracting patient profile:\n" + str(e)

    # 3) MCP triage
    triage_result = call_mcp_tool("triage_patient", patient_profile)
    if "error" in triage_result:
        return (
            "❌ MCP error (triage_patient):\n"
            + json.dumps(triage_result, indent=2, ensure_ascii=False)
        )

    urgency = triage_result.get("urgency_level", "moderate")

    # 4) MCP schedule_appointment
    schedule_result = call_mcp_tool(
        "schedule_appointment",
        {
            "urgency_level": urgency,
            "speciality": "general practitioner",
        },
    )
    if "error" in schedule_result:
        return (
            "❌ MCP error (schedule_appointment):\n"
            + json.dumps(schedule_result, indent=2, ensure_ascii=False)
        )

    # 5) MCP notify_team
    notif_result = call_mcp_tool(
        "notify_team",
        {
            "urgency_level": urgency,
            "patient_summary": conv_text,
            "appointment_slot": schedule_result.get("selected_slot", ""),
        },
    )
    if "error" in notif_result:
        return (
            "❌ MCP error (notify_team):\n"
            + json.dumps(notif_result, indent=2, ensure_ascii=False)
        )

    # 6) Generate readable report
    report = f"""
MEDICAL TRIAGE REPORT: VOICE INTAKE
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PATIENT INFORMATION:
Age:              {patient_profile.get('age', 'Not provided')} years
Symptoms:         {patient_profile.get('symptoms', 'Not provided')}
Duration:         {patient_profile.get('duration', 'Not provided')}
Risk Factors:     {patient_profile.get('risk_factors', 'None mentioned')}
Other Context:    {patient_profile.get('other_context', 'None')}

TRIAGE ASSESSMENT:
Urgency Level:    {urgency.upper()}

Clinical Recommendation:
{triage_result.get('guidelines_answer', 'No recommendation available')}

APPOINTMENT SCHEDULED:
Slot:             {schedule_result.get('selected_slot', 'Not scheduled')}
Specialty:        {schedule_result.get('speciality', 'General')}
Note:             {schedule_result.get('note', '')}

TEAM NOTIFICATION:
Status:           {notif_result.get('status', 'unknown').upper()}
Timestamp:        {notif_result.get('timestamp', 'N/A')}

CONVERSATION TRANSCRIPT:
{conv_text}
"""
    
    return report
# ----------------------
#   GRADIO UI
# ----------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🏥 Healthcare Voice Triage Agent (MCP + OpenAI + ElevenLabs)")
    gr.Markdown(
        "English-speaking voice agent for patient intake.\n\n"
        "- Uses **ElevenLabs STT** to transcribe the patient's voice.\n"
        "- Uses **OpenAI** to drive the conversation and extract a structured patient profile.\n"
        "- Uses **MCP tools** (`triage_patient`, `schedule_appointment`, `notify_team`) to simulate triage and care coordination.\n"
        "- Uses **ElevenLabs TTS** to answer the patient with natural speech.\n"
    )

    with gr.Row():
        audio_in = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="🎙️ Speak here (patient)",
        )
        audio_out = gr.Audio(label="🔊 Agent voice reply")

    chatbot = gr.Chatbot(label="Conversation transcript (patient ↔ agent)")
    text_input = gr.Textbox(
        label="Optional text input (for testing)",
        placeholder="Type a message instead of speaking (English)...",
    )
    end_btn = gr.Button("✅ End call and generate medical report")
    report_box = gr.Textbox(
        label="Final medical triage report (JSON)",
        lines=20,
    )

    # Voice input
    audio_in.change(
        fn=handle_audio_input,
        inputs=audio_in,
        outputs=[chatbot, audio_out],
    )

    # Text input (optional)
    text_input.submit(
        fn=handle_text_input,
        inputs=text_input,
        outputs=[chatbot, text_input],
    )

    # End call => MCP report
    end_btn.click(
        fn=end_call_and_generate_report,
        inputs=[],
        outputs=[report_box],
    )

if __name__ == "__main__":
    demo.launch()
