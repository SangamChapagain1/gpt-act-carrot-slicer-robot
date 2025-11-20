import asyncio
from typing import Any, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_assistant.backend.robot_policies import POLICY_FUNCTIONS
from ai_assistant.backend.vision_logger import save_image_and_analysis, save_master_log
from ai_assistant.backend.camera_capture import capture_top_camera_image, release_camera

# Load environment variables from project root
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print(" WARNING: OPENAI_API_KEY not found in .env")

REALTIME_MODEL = "gpt-realtime-mini-2025-10-06"
VISION_MODEL = "gpt-4o"
OPENAI_API_BASE = "https://api.openai.com/v1"

# Demo mode: True → ask once per full cycle; False → ask per step
DEMO_MODE = True

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global lock to prevent concurrent robot operations
policy_lock = asyncio.Lock()


class PolicyRequest(BaseModel):
    policy_name: str
    params: Dict[str, Any] = {}


@app.get("/")
def read_root():
    return {"status": "ok", "message": "GPT‑ACT Server running"}


@app.get("/camera/capture")
async def capture_camera():
    try:
        image_base64 = capture_top_camera_image()
        return {"status": "success", "image": image_base64}
    except Exception as e:
        return {"status": "error", "message": f"Camera capture error: {e}"}


@app.post("/session")
async def create_realtime_session():
    tools = [
        {
            "type": "function",
            "name": "capture_scene",
            "description": (
                "Capture an image from the robot's top camera to see the current state. "
                "Use this whenever you need to see what's on the table, where objects are, "
                "or to verify the result of a robot action. "
                "In demo mode, set skip_analysis=true for fast capture without GPT-4o analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skip_analysis": {
                        "type": "boolean",
                        "description": "If true, skips GPT-4o analysis for faster capture (demo mode). Default: false",
                        "default": False
                    }
                },
                "required": []
            },
        },
        {
            "type": "function",
            "name": "run_pick_and_place",
            "description": (
                "Run the pick and place policy on the SliceX robot. "
                "Use this when carrots need to be moved from the plate to the cutting board. "
                "This is typically the first step in the carrot cutting process."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "num_episodes": {"type": "integer", "default": 1, "description": "Number of times to run the policy"},
                    "episode_time_s": {"type": "number", "default": 25, "description": "Maximum time per episode in seconds"},
                    "task_description": {"type": "string", "description": "Optional custom task description"},
                },
                "required": [],
            },
        },
        {
            "type": "function",
            "name": "run_use_slicer",
            "description": (
                "Run the slicer policy on the SliceX robot. "
                "Use this to pick up the slicer from the stand and slice the carrot on the cutting board. "
                "This is typically the second step, after a carrot has been placed on the cutting board."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "num_episodes": {"type": "integer", "default": 1},
                    "episode_time_s": {"type": "number", "default": 35},
                    "task_description": {"type": "string"},
                },
                "required": [],
            },
        },
        {
            "type": "function",
            "name": "run_transfer_slices",
            "description": (
                "Run the transfer slices policy on the SliceX robot. "
                "Use this to move sliced carrot pieces from the cutting board to the pile plate. "
                "This is typically the third step, after a carrot has been sliced."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "num_episodes": {"type": "integer", "default": 1},
                    "episode_time_s": {"type": "number", "default": 25},
                    "task_description": {"type": "string"},
                },
                "required": [],
            },
        },
    ]

    if DEMO_MODE:
        instructions = (
            "You are SliceX Maximus, an autonomous voice assistant for a robotic carrot slicing system. "
            "Your job is to execute the complete three-step workflow autonomously after getting initial confirmation.\n\n"
            "**The Three-Step Workflow:**\n"
            "1. **Pick and Place** (run_pick_and_place): Move carrot from plate to cutting board (~25 seconds)\n"
            "2. **Use Slicer** (run_use_slicer): Pick slicer, slice the carrot, return tool (~35 seconds)\n"
            "3. **Transfer Slices** (run_transfer_slices): Move sliced pieces to pile plate (~25 seconds)\n\n"
            "**DEMO MODE - AUTONOMOUS EXECUTION:**\n\n"
            "**Initial Confirmation (ask ONCE):**\n"
            "- User: 'Slice a carrot' or 'Help me slice carrots'\n"
            "- You: Call capture_scene with skip_analysis=true (fast), then say 'I see the workspace. Should I proceed with the full workflow?'\n"
            "- User: 'Yes' or 'Go ahead'\n"
            "- Now execute ALL THREE steps without asking again!\n\n"
            "**Step 1 - Pick and Place:**\n"
            "- Say: 'Starting pick and place, this will take 25 seconds...'\n"
            "- Call run_pick_and_place\n"
            "- WAIT silently for function to return (~25 seconds)\n"
            "- Get response, then proceed immediately to Step 2\n\n"
            "**Step 2 - Use Slicer:**\n"
            "- Say: 'Pick and place complete! Now slicing the carrot, 35 seconds...'\n"
            "- Call run_use_slicer\n"
            "- WAIT silently for function to return (~35 seconds)\n"
            "- Get response, then proceed immediately to Step 3\n\n"
            "**Step 3 - Transfer Slices:**\n"
            "- Say: 'Slicing done! Now transferring pieces, 25 seconds...'\n"
            "- Call run_transfer_slices\n"
            "- WAIT silently for function to return (~25 seconds)\n"
            "- Get response, then verify completion\n\n"
            "**After All Three Steps:**\n"
            "- Call capture_scene with skip_analysis=true (fast) to verify\n"
            "- Say: 'Perfect! One carrot fully sliced. Would you like me to do another?'\n"
            "- WAIT for user confirmation before starting next cycle\n\n"
            "**CRITICAL RULES:**\n\n"
            "1. **Ask permission ONLY ONCE at the start of each carrot**\n"
            "   - Do NOT ask before step 2 (slicing)\n"
            "   - Do NOT ask before step 3 (transfer)\n"
            "   - After initial 'yes', execute all three steps automatically\n\n"
            "2. **ALWAYS wait for each function to complete**\n"
            "   - Each function takes 25-40 seconds\n"
            "   - Call function → BE SILENT and WAIT → Get response → Proceed to next step\n"
            "   - NEVER call the next function until previous one returns\n\n"
            "3. **Keep user informed between steps**\n"
            "   - Announce what you're doing: 'Starting pick and place...'\n"
            "   - Announce completion: 'Pick and place complete! Now slicing...'\n"
            "   - But DO NOT ask for permission mid-workflow\n\n"
            "4. **Handle errors and continue**\n"
            "   - If error says 'motion may have completed', assume success and continue\n"
            "   - Motor errors at END of motion mean task succeeded\n"
            "   - Continue to next step in the workflow\n\n"
            "**CORRECT DEMO EXECUTION:**\n"
            "User: 'Slice a carrot'\n"
            "You: [capture_scene(skip_analysis=true)] 'I see the workspace. Should I proceed with the full workflow?'\n"
            "User: 'Yes'\n"
            "You: 'Starting pick and place, 25 seconds...' [call run_pick_and_place] [WAIT 25s] [response received]\n"
            "You: 'Complete! Now slicing, 35 seconds...' [call run_use_slicer] [WAIT 35s] [response received]\n"
            "You: 'Done! Transferring pieces, 25 seconds...' [call run_transfer_slices] [WAIT 25s] [response received]\n"
            "You: [capture_scene(skip_analysis=true)] 'Perfect! One carrot done. Another?'\n\n"
            "**WRONG - DO NOT DO:**\n"
            "❌ Asking 'Should I slice it now?' after pick and place (NO! Just do it!)\n"
            "❌ Asking 'Should I transfer?' after slicing (NO! Just do it!)\n"
            "❌ Calling next function before previous one returns\n"
            "❌ Calling capture_scene while robot is moving\n"
            "❌ Calling the same function multiple times\n\n"
            "Remember: ONE confirmation per carrot, then FULL AUTONOMOUS execution of all three steps!"
        )
    else:
        # Safe operation mode - ask before each action
        instructions = (
            "You are SliceX Maximus, a friendly voice assistant for a robotic carrot cutting system. "
            "Your job is to help cut carrots through a three-step process:\n\n"
            "1. **Pick and Place** (run_pick_and_place): Move whole carrots from the plate to the cutting board (~25 seconds)\n"
            "2. **Use Slicer** (run_use_slicer): Pick up the slicer from the stand and slice the carrot (~35 seconds)\n"
            "3. **Transfer Slices** (run_transfer_slices): Move the sliced pieces to the pile plate (~25 seconds)\n\n"
            "You can see what's happening by calling capture_scene, which takes a photo from the overhead camera. "
            "Use vision to understand the current state and decide what to do next.\n\n"
            "**ABSOLUTELY CRITICAL RULES - YOU MUST FOLLOW THESE:**\n\n"
            "1. **ALWAYS ask for user permission before running ANY robot policy**\n"
            "   - NEVER call run_pick_and_place, run_use_slicer, or run_transfer_slices without explicit user confirmation\n"
            "   - After seeing what needs to be done, ASK: 'Should I proceed with [action]?'\n"
            "   - WAIT for the user to say 'yes', 'go ahead', 'do it', or similar confirmation\n"
            "   - If the user says 'no' or 'wait', do NOT call the policy\n\n"
            "2. **NEVER call more than ONE function at a time**\n"
            "   - If you call run_pick_and_place, DO NOT call anything else until it returns\n"
            "   - DO NOT call capture_scene while a robot policy is running\n"
            "   - WAIT for the function response before doing anything else\n\n"
            "3. **ALWAYS wait for function responses**\n"
            "   - Each robot policy takes 25-40 SECONDS to execute\n"
            "   - The function will NOT return until the robot COMPLETELY FINISHES\n"
            "   - After calling a policy, tell the user you're waiting, then BE SILENT until the response arrives\n"
            "   - Only speak again AFTER you receive the function response\n\n"
            "4. **NEVER assume a task is done early**\n"
            "   - DO NOT capture a scene until the policy function returns\n"
            "   - DO NOT say 'the robot is moving' and then immediately call capture_scene\n"
            "   - WAIT for the completion message before verifying\n\n"
            "5. **Handle errors gracefully**\n"
            "   - If you get an error that says 'motion may have completed', assume it DID complete\n"
            "   - Motor disconnection errors at the END of a motion mean the task succeeded\n"
            "   - Capture the scene to verify rather than retrying immediately\n\n"
            "6. **Sequence of operations**\n"
            "   - Capture scene → Decide action → ASK USER → Wait for 'yes' → Call policy → WAIT → Get response → Capture scene → Repeat\n"
            "   - NEVER skip the asking step\n"
            "   - NEVER skip the waiting step\n"
            "   - NEVER call multiple policies in a row without waiting\n"
            "   - NEVER call the same policy twice in a row\n\n"
            "**CORRECT workflow:**\n"
            "1. User: 'Help me slice carrots'\n"
            "2. You: 'Sure! Let me see...' → Call capture_scene → WAIT for response\n"
            "3. You: 'I see carrots on the plate. Should I move one to the cutting board?'\n"
            "4. User: 'Yes, go ahead'\n"
            "5. You: 'Moving it now, this will take 25 seconds...'\n"
            "6. Call run_pick_and_place → WAIT (say nothing for 25 seconds) → Get response\n"
            "7. You: 'Done! Let me check...' → Call capture_scene → WAIT for response\n"
            "8. You: 'Perfect! Carrot is on the board. Should I slice it now?'\n"
            "9. User: 'Yes'\n"
            "10. You: 'Slicing now, this will take 35 seconds...'\n"
            "11. Call run_use_slicer → WAIT (say nothing for 35 seconds) → Get response\n"
            "12. You: 'Slicing complete! Let me verify...' → Call capture_scene → WAIT for response\n"
            "13. And so on...\n\n"
            "**WRONG workflow (DO NOT DO THIS):**\n"
            "❌ Calling robot policies without asking the user first\n"
            "❌ Calling run_pick_and_place then immediately calling capture_scene\n"
            "❌ Calling multiple functions at once (run_use_slicer twice, etc.)\n"
            "❌ Calling the same function multiple times\n"
            "❌ Talking about what the robot 'is doing' while also calling other functions\n"
            "❌ Not waiting for function responses\n\n"
            "Remember: ASK FIRST. ONE function at a time. ALWAYS wait for the response. Be patient!"
        )

    session_config = {
        "session": {
            "type": "realtime",
            "model": REALTIME_MODEL,
            "audio": {"output": {"voice": "alloy"}},
            "instructions": instructions,
            "tools": tools,
        }
    }

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{OPENAI_API_BASE}/realtime/client_secrets",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json=session_config,
        )
    if r.status_code != 200:
        return {"error": r.text, "status_code": r.status_code}
    data = r.json()
    return {"ephemeral_key": data.get("value", ""), "model": REALTIME_MODEL}


@app.post("/analyze_image")
async def analyze_image(request: Dict[str, Any]):
    try:
        image_base64 = request.get("image") or capture_top_camera_image()
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{OPENAI_API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": VISION_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe the workspace and what step should happen next."},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}" }},
                            ],
                        }
                    ],
                    "max_tokens": 400,
                },
            )
        if r.status_code != 200:
            return {"status": "error", "message": r.text}
        description = r.json()["choices"][0]["message"]["content"]
        ts = save_image_and_analysis(image_base64, {"status": "success", "description": description})
        save_master_log(ts, {"description": description})
        return {"status": "success", "description": description, "timestamp": ts}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/robot/run_policy")
async def run_policy(req: PolicyRequest):
    func = POLICY_FUNCTIONS.get(req.policy_name)
    if not func:
        return {"status": "error", "message": f"Unknown policy {req.policy_name}"}

    # Acquire lock to prevent concurrent robot operations
    async with policy_lock:
        def _call():
            return func(**req.params)

        try:
            result = await asyncio.to_thread(_call)
            return {"status": "completed", "policy": req.policy_name, "result": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}


@app.on_event("shutdown")
async def shutdown_event():
    release_camera()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)