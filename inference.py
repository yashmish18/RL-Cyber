import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from cyber_openenv_rl.client import CyberEnv
from cyber_openenv_rl.models import CyberAction

# Environment Variables
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or "sk-dummy"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
MAX_STEPS = 20
TEMPERATURE = 0.0

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a Security Operations Center (SOC) analyst managing a network under attack.
    Your goal is to detect and contain threats while minimizing collateral damage (host isolation).
    
    Available Actions:
    - block_ip(source_ip): Block a suspicious IP address.
    - isolate_node(target_host): Disconnect a compromised host from the network.
    - scan_host(target_host): Perform a deep scan to confirm compromise.
    - patch_service(target_host, target_service): Patch a vulnerable service.
    - restore_backup(target_host): Restore a host from a clean backup.
    - ignore(target_host): Take no action.
    
    You must respond with a JSON-formatted action:
    {"action_type": "...", "target_host": "...", "source_ip": "...", "target_service": "..."}
    
    Actor is always "defender".
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def build_user_prompt(obs) -> str:
    alerts = "\n".join(obs.ids_alerts) if obs.ids_alerts else "No alerts"
    incidents = "\n".join([str(inc) for inc in obs.active_incidents]) if obs.active_incidents else "No active incidents"
    
    return textwrap.dedent(
        f"""
        Current Observation:
        - Task ID: {obs.task_id}
        - Host Status (Compromised): {obs.host_compromise}
        - Host Status (Isolated): {obs.host_isolation}
        - Traffic Anomaly Score: {obs.traffic_anomaly_score:.2f}
        - IDS Alerts:
        {alerts}
        - Active Incidents:
        {incidents}
        - Step Budget Remaining: {obs.step_budget_remaining}
        
        Provide your next action as a JSON object.
        """
    ).strip()

def get_model_action(client: OpenAI, obs) -> CyberAction:
    user_prompt = build_user_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"}
        )
        import json
        content = (completion.choices[0].message.content or "").strip()
        action_data = json.loads(content)
        action_data["actor"] = "defender"
        return CyberAction(**action_data)
    except Exception as exc:
        # Fallback action
        return CyberAction(actor="defender", action_type="ignore", target_host=obs.task_id)

async def run_task(task_id: str):
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # We assume the server is running locally for the inference script if PING_URL is not set
    base_url = os.getenv("PING_URL", "http://0.0.0.0:8000")
    env = CyberEnv(base_url=base_url)

    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=task_id, env="cyber_openenv_rl", model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_model_action(client, obs)
            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=str(action.action_type), reward=reward, done=done, error=None)

            if done:
                break

        # Calculate score based on normalized reward or specific success criteria
        # For cybersecurity, we want high cumulative reward or low compromise count.
        # Let's say score is normalized against a max expected reward (approximate).
        max_possible = 2.0 * steps_taken # Approximate max reward per step
        score = max(0.0, min(1.0, sum(rewards) / max(1.0, max_possible)))
        success = score > 0.5 # Arbitrary threshold for "success"

    except Exception as e:
        print(f"[DEBUG] Error during task {task_id}: {e}")
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main():
    tasks = ["easy", "medium", "hard"]
    for task in tasks:
        await run_task(task)

if __name__ == "__main__":
    asyncio.run(main())
