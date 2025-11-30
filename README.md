---
title: Healthcare Voice Triage Agent
emoji: 🏥🔊
colorFrom: gray
colorTo: pink
sdk: gradio
sdk_version: 6.0.1
app_file: app.py
pinned: false
tags:
  - mcp-in-action-track-enterprise
  - openai
  - elevenlabs
  - llamaindex
---

# 🏥 Healthcare Voice Triage Agent  
### **Real-time AI voice assistant for medical triage, appointment booking, and emergency routing.**  
**Built for the MCP 1st Birthday Hackathon – Track 2: MCP in Action (Enterprise).**

This project demonstrates how MCP tools, real-time speech interfaces, and LLM reasoning can power **enterprise-grade healthcare workflows** — from patient intake to triage, appointment scheduling, and team notifications.

---

## 🚀 Features

### 🔊 **Real-Time Voice Interaction**
- **ElevenLabs STT** for accurate English medical transcription  
- **OpenAI GPT-4.1 Mini** for natural medical conversation  
- **ElevenLabs TTS** for real-time voice responses  
- Zero GPU required (fully cloud-based)

### 🩺 **MCP-Powered Medical Tools**
The MCP server exposes 3 autonomous medical tools:

| Tool | Purpose |
|------|---------|
| `triage_patient` | Analyze symptoms & risk factors → determine urgency |
| `schedule_appointment` | Book the appropriate medical appointment |
| `notify_team` | Notify medical staff (mock notification via MCP) |

Each tool is independently callable and fully encapsulated.

### 📚 **Medical Knowledge Base (LlamaIndex)**
A lightweight medical triage knowledge base (local embeddings) helps estimate urgency levels and provide safe-triage recommendations.

### 🏢 **Enterprise Workflow Example**
This demo simulates a real hospital workflow:
1. Patient calls  
2. Voice interaction + symptom collection  
3. Automatic triage  
4. Appointment scheduling  
5. Team notification  
6. Final medical report generation  

---

## 🛠️ Technologies & Sponsors Used

- **OpenAI API** → conversational reasoning   
- **ElevenLabs** → real-time speech recognition + TTS  
- **LlamaIndex** → medical triage knowledge base  
- **MCP (Model Context Protocol)** → autonomous reasoning and tool calls  
- **Gradio** → browser-based voice UI  

---

## 🎬 Demo Video:
👉 

---

## 🔗 Social Media Post:
👉 https://x.com/Aidouni79030/status/1995254093749633152?s=20

---

## 🏁 How to Use
1. Click **"Record"** and speak normally (English).  
2. The agent transcribes, reasons, and responds by voice.  
3. Press **"End Call & Generate Report"** to trigger:
   - Patient profile extraction  
   - MCP triage  
   - Appointment scheduling  
   - Team notification  
4. The final JSON medical report is displayed.

---

## 📦 Project Structure
/app.py # Voice interface + LLM + MCP client
/healthcare_mcp_server/ # MCP tools server
server.py
llamaindex_kb.py # Medical KB with local embeddings
/stt_elevenlabs_client.py # ElevenLabs STT wrapper
/tts_elevenlabs_client.py # ElevenLabs TTS wrapper

## 💡 Why This Project?
Healthcare requires:
- safety  
- accuracy  
- rapid triage  
- automated decision making  

This demo shows how MCP can power **real-time medical workflows**,  
using multiple external tools and reasoning components — **all inside a single unified agent**.

Perfect example of **MCP in Action** for the **Enterprise Track**.

---

## 👥 Team
Solo project by: **Hanan Aidouni**  
*HuggingFace username: aidouni*

---

## 📝 Notice
This application is a **demo only** and **not a medical device**.  
It should not be used for real medical diagnosis or emergency decisions.

---