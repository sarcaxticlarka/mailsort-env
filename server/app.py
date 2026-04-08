"""
FastAPI application for the MailSort OpenEnv environment.

Exposes the standard OpenEnv HTTP endpoints:
  POST /reset  — start a new episode
  POST /step   — submit an action
  GET  /state  — get current episode state
  GET  /health — liveness probe

When openenv-core is installed, create_app() provides additional
WebSocket support and a built-in web UI.
"""

from __future__ import annotations

import sys
import os

# Ensure root is on path so `models` is importable inside the server package
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models import MailSortAction, MailSortObservation, MailSortState
from server.environment import MailSortEnvironment

# ---------------------------------------------------------------------------
# Web UI HTML (served at GET /)
# ---------------------------------------------------------------------------

MAILSORT_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MailSort — Email Triage</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}
header{background:linear-gradient(135deg,#1e40af,#7c3aed);padding:20px 28px}
header h1{font-size:1.5rem;font-weight:700;color:#fff}
header p{color:#bfdbfe;font-size:.85rem;margin-top:4px}
.wrap{max-width:960px;margin:0 auto;padding:20px 16px}
.tabs{display:flex;gap:6px;margin-bottom:18px}
.tab{padding:8px 18px;border-radius:6px;cursor:pointer;font-size:.85rem;font-weight:600;border:1px solid #334155;color:#94a3b8;background:#1e293b}
.tab.on{background:#6366f1;color:#fff;border-color:#6366f1}
.card{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:18px;margin-bottom:16px}
.card h2{font-size:.75rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.task-btn{padding:9px 16px;border-radius:7px;border:1px solid #334155;background:#0f172a;color:#e2e8f0;cursor:pointer;font-size:.85rem;font-weight:500;transition:all .15s}
.task-btn:hover,.task-btn.on{border-color:#6366f1;background:#1e1b4b;color:#fff}
.diff{font-size:.65rem;font-weight:700;padding:2px 7px;border-radius:4px;margin-left:5px}
.e{background:#064e3b;color:#6ee7b7}.m{background:#78350f;color:#fcd34d}.h{background:#7f1d1d;color:#fca5a5}
.btn{padding:9px 20px;border-radius:7px;border:none;font-weight:600;font-size:.88rem;cursor:pointer;transition:all .15s}
.btn-p{background:#6366f1;color:#fff}.btn-p:hover{background:#4f46e5}
.btn-s{background:#334155;color:#e2e8f0}.btn-s:hover{background:#475569}
.email{background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:12px;margin-bottom:8px}
.email .eid{font-size:.7rem;color:#6366f1;font-weight:700;margin-bottom:3px}
.email .subj{font-weight:600;font-size:.9rem;color:#e2e8f0}
.email .from{font-size:.76rem;color:#94a3b8;margin:3px 0}
.email .body{font-size:.79rem;color:#cbd5e1;line-height:1.5;margin-top:5px;max-height:68px;overflow:hidden;position:relative}
.email .body::after{content:'';position:absolute;bottom:0;left:0;right:0;height:22px;background:linear-gradient(transparent,#0f172a)}
textarea{width:100%;background:#020617;border:1px solid #334155;border-radius:7px;color:#e2e8f0;padding:11px;font-family:monospace;font-size:.8rem;resize:vertical;min-height:110px}
textarea:focus{outline:none;border-color:#6366f1}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px}
.stat{background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:10px 14px}
.stat .lbl{font-size:.7rem;color:#64748b;text-transform:uppercase}
.stat .val{font-size:1.2rem;font-weight:700;color:#e2e8f0;margin-top:2px}
.fb{background:#0f172a;border-left:3px solid #6366f1;padding:10px 14px;border-radius:0 7px 7px 0;font-size:.84rem;color:#94a3b8;margin-top:12px;line-height:1.6}
.rbar{height:7px;background:#1e293b;border-radius:4px;margin-top:8px;overflow:hidden}
.rfill{height:100%;border-radius:4px;background:linear-gradient(90deg,#6366f1,#06b6d4);transition:width .4s}
.banner{margin-top:14px;padding:13px 18px;border-radius:8px;font-weight:600;font-size:.95rem}
.ok{background:#064e3b;color:#6ee7b7}.fail{background:#7f1d1d;color:#fca5a5}
.dots{display:flex;gap:6px;align-items:center}
.dot{width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.72rem;font-weight:700;background:#1e293b;color:#64748b;border:1px solid #334155}
.dot.on{background:#6366f1;color:#fff;border-color:#6366f1}
.dot.done{background:#064e3b;color:#6ee7b7;border-color:#065f46}
pre{white-space:pre-wrap;word-break:break-all;font-size:.78rem;color:#94a3b8;line-height:1.7}
.code{background:#020617;border:1px solid #1e293b;border-radius:7px;padding:14px;margin-bottom:14px}
.code .lbl{font-size:.72rem;color:#6ee7b7;margin-bottom:7px;font-weight:700}
#page-api{display:none}
</style>
</head>
<body>
<header>
  <h1>&#128231; MailSort <span style="background:rgba(255,255,255,.15);color:#fff;padding:2px 10px;border-radius:20px;font-size:.72rem;font-weight:700;margin-left:8px">OpenEnv</span></h1>
  <p>Enterprise Email Triage &mdash; 3 tasks &middot; easy to hard &middot; deterministic grading</p>
</header>

<div class="wrap">
  <div class="tabs">
    <div class="tab on" onclick="showPage('ui',this)">&#128187; Interactive</div>
    <div class="tab" onclick="showPage('api',this)">&#9881; API / Terminal</div>
  </div>

  <!-- UI PAGE -->
  <div id="page-ui">
    <div class="card">
      <h2>1 &mdash; Choose Task</h2>
      <div class="row">
        <button class="task-btn on" onclick="pickTask('email_classify',this)">Email Classify <span class="diff e">EASY</span></button>
        <button class="task-btn" onclick="pickTask('email_rank',this)">Email Rank <span class="diff m">MEDIUM</span></button>
        <button class="task-btn" onclick="pickTask('email_triage',this)">Email Triage <span class="diff h">HARD</span></button>
      </div>
      <div id="task-desc" style="margin-top:10px;font-size:.83rem;color:#94a3b8"></div>
    </div>

    <div class="stats">
      <div class="stat"><div class="lbl">Task</div><div class="val" id="s-task">&#8212;</div></div>
      <div class="stat"><div class="lbl">Step</div><div class="val" id="s-step">&#8212;</div></div>
      <div class="stat"><div class="lbl">Reward</div><div class="val" id="s-rew">&#8212;</div></div>
      <div class="stat"><div class="lbl">Score</div><div class="val" id="s-score">&#8212;</div></div>
    </div>

    <div class="card">
      <h2>2 &mdash; Start Episode</h2>
      <div class="row">
        <button class="btn btn-p" onclick="doReset()">&#9654; Start / Reset</button>
        <div class="dots" id="dots"></div>
      </div>
    </div>

    <div class="card" id="emails-card" style="display:none">
      <h2>3 &mdash; Emails to Triage</h2>
      <div id="emails-list"></div>
    </div>

    <div class="card" id="action-card" style="display:none">
      <h2>4 &mdash; Submit Action (JSON)</h2>
      <div id="step-hint" style="font-size:.8rem;color:#6366f1;font-weight:500;margin-bottom:8px;white-space:pre-wrap"></div>
      <textarea id="action-box" rows="6"></textarea>
      <div class="row" style="margin-top:10px">
        <button class="btn btn-p" id="submit-btn" onclick="doStep()">&#9654; Submit Action</button>
        <button class="btn btn-s" onclick="autoFill()">&#128161; Auto-fill template</button>
      </div>
      <div id="fb-wrap" style="display:none">
        <div class="fb" id="fb-text"></div>
        <div class="rbar"><div class="rfill" id="rbar" style="width:0%"></div></div>
      </div>
      <div id="banner"></div>
    </div>
  </div>

  <!-- API PAGE -->
  <div id="page-api">
    <div class="card">
      <h2>API Reference</h2>
      <p style="color:#94a3b8;font-size:.83rem;margin-bottom:16px">Base URL: <strong id="base-url" style="color:#6366f1"></strong></p>
      <div class="code"><div class="lbl">POST /reset</div><pre>curl -X POST BASE_URL/reset \\
  -H "Content-Type: application/json" \\
  -d '{"task": "email_classify"}'</pre></div>
      <div class="code"><div class="lbl">POST /step</div><pre>curl -X POST BASE_URL/step \\
  -H "Content-Type: application/json" \\
  -d '{"classifications":[{"email_id":"e1_01","category":"urgent","priority":"critical"}]}'</pre></div>
      <div class="code"><div class="lbl">GET /health</div><pre>curl BASE_URL/health</pre></div>
      <div class="code"><div class="lbl">GET /tasks</div><pre>curl BASE_URL/tasks</pre></div>
      <div class="code"><div class="lbl">Run inference.py locally</div><pre>git clone https://github.com/sarcaxticlarka/mailsort-env
cd mailsort-env
python3 -m venv venv && source venv/bin/activate
pip install openai httpx pydantic fastapi uvicorn

export HF_TOKEN="your_token"
export ENV_BASE_URL="BASE_URL"
python3 inference.py</pre></div>
    </div>
  </div>
</div>

<script>
const BASE = window.location.origin;
document.getElementById('base-url').textContent = BASE;
document.querySelectorAll('pre').forEach(p => { p.innerHTML = p.innerHTML.replace(/BASE_URL/g, BASE); });

const TASKS = {
  email_classify:{label:'Classify',steps:1,desc:'Classify a single email into one of 6 categories and assign a priority level.'},
  email_rank:    {label:'Rank',    steps:1,desc:'Rank 5 emails from most to least urgent, and classify each one.'},
  email_triage:  {label:'Triage', steps:3,desc:'3 steps: classify + detect phishing, route to departments, draft a response.'}
};

let task='email_classify', stepCount=0, maxSteps=1, totalReward=0, emails=[];

function showPage(p,el){
  document.getElementById('page-ui').style.display  = p==='ui'?'block':'none';
  document.getElementById('page-api').style.display = p==='api'?'block':'none';
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  el.classList.add('on');
}

function pickTask(t,el){
  task=t;
  document.querySelectorAll('.task-btn').forEach(b=>b.classList.remove('on'));
  el.classList.add('on');
  document.getElementById('task-desc').textContent=TASKS[t].desc;
  document.getElementById('emails-card').style.display='none';
  document.getElementById('action-card').style.display='none';
  document.getElementById('dots').innerHTML='';
  ['s-task','s-step','s-rew','s-score'].forEach(id=>document.getElementById(id).textContent='—');
}

function ss(id,v){document.getElementById(id).textContent=v;}

function renderDots(cur,max){
  if(max<=1){document.getElementById('dots').innerHTML='';return;}
  document.getElementById('dots').innerHTML=
    Array.from({length:max},(_,i)=>'<div class="dot '+(i+1<cur?'done':i+1===cur?'on':'')+'">'+(i+1)+'</div>').join('');
}

function renderEmails(list){
  emails=list;
  document.getElementById('emails-list').innerHTML=list.map(e=>
    '<div class="email"><div class="eid">'+e.id+'</div><div class="subj">'+e.subject+'</div>'+
    '<div class="from">From: '+e.sender_email+'</div><div class="body">'+e.body+'</div></div>'
  ).join('');
  document.getElementById('emails-card').style.display='block';
}

function autoFill(){
  let t={};
  if(task==='email_classify'){
    t={classifications:emails.map(e=>({email_id:e.id,category:'urgent',priority:'high'}))};
  }else if(task==='email_rank'){
    t={rankings:emails.map(e=>e.id),classifications:emails.map(e=>({email_id:e.id,category:'routine',priority:'medium'}))};
  }else{
    if(stepCount===0) t={classifications:emails.map(e=>({email_id:e.id,category:'routine',priority:'medium',is_phishing:false}))};
    else if(stepCount===1) t={routings:emails.map(e=>({email_id:e.id,dept:'support'}))};
    else t={response_draft:'Thank you for your urgent escalation. We have received your message and our team is actively investigating. We will provide a full update shortly.'};
  }
  document.getElementById('action-box').value=JSON.stringify(t,null,2);
}

async function doReset(){
  try{
    const r=await fetch(BASE+'/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task})});
    const d=await r.json();
    const obs=d.observation||d;
    stepCount=0; maxSteps=obs.max_steps||1; totalReward=0;
    document.getElementById('banner').innerHTML='';
    document.getElementById('fb-wrap').style.display='none';
    document.getElementById('action-card').style.display='block';
    document.getElementById('submit-btn').disabled=false;
    document.getElementById('step-hint').textContent=obs.step_description||'';
    ss('s-task',TASKS[task].label); ss('s-step','0/'+maxSteps); ss('s-rew','—'); ss('s-score','—');
    renderDots(1,maxSteps);
    renderEmails((obs.emails||[]).slice(0,task==='email_triage'?3:99));
    autoFill();
  }catch(e){alert('Reset failed: '+e.message);}
}

async function doStep(){
  const raw=document.getElementById('action-box').value.trim();
  let parsed; try{parsed=JSON.parse(raw);}catch(e){alert('Invalid JSON: '+e.message);return;}
  try{
    const r=await fetch(BASE+'/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:parsed})});
    const d=await r.json();
    const obs=d.observation||{};
    const reward=parseFloat(d.reward)||0;
    const done=!!d.done;
    stepCount++; totalReward+=reward;
    const score=totalReward/stepCount;
    ss('s-step',stepCount+'/'+maxSteps);
    ss('s-rew',reward.toFixed(2));
    ss('s-score',score.toFixed(2));
    renderDots(done?maxSteps+1:stepCount+1,maxSteps);
    const fb=obs.feedback||obs.last_action_error||'';
    if(fb){
      document.getElementById('fb-text').textContent=fb;
      document.getElementById('fb-wrap').style.display='block';
      document.getElementById('rbar').style.width=Math.round(reward*100)+'%';
    }
    if(done){
      document.getElementById('submit-btn').disabled=true;
      document.getElementById('banner').innerHTML=
        '<div class="banner '+(score>=.5?'ok':'fail')+'">'+(score>=.5?'&#10003; Success':'&#10007; Episode ended')+
        ' &nbsp;&middot;&nbsp; Score: '+score.toFixed(2)+'/1.00 &nbsp;&middot;&nbsp; '+stepCount+' step'+(stepCount>1?'s':'')+'</div>';
    }else{
      document.getElementById('step-hint').textContent=obs.step_description||'';
      autoFill();
    }
  }catch(e){alert('Step failed: '+e.message);}
}

document.getElementById('task-desc').textContent=TASKS[task].desc;
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# Try to use openenv-core's create_app for full spec compliance.
# Fall back to a hand-rolled FastAPI app if the package is not installed.
# ---------------------------------------------------------------------------

try:
    from openenv.core.env_server import create_app

    app = create_app(
        MailSortEnvironment,        # factory — instantiated per WebSocket session
        MailSortAction,             # action type
        MailSortObservation,        # observation type
        env_name="mailsort-env",
        max_concurrent_envs=10,
    )

    # Add UI route on top of the openenv-core app
    from fastapi.responses import HTMLResponse as _HTMLResponse

    @app.get("/")
    async def _root():
        return _HTMLResponse(content=MAILSORT_UI_HTML)

except ImportError:
    # ---------------------------------------------------------------------------
    # Fallback: minimal FastAPI implementation that satisfies the OpenEnv HTTP
    # contract (POST /reset, POST /step, GET /state, POST /state).
    # The validator script only checks /reset for a 200 response.
    # ---------------------------------------------------------------------------

    import json
    from typing import Any, Dict, Optional

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    app = FastAPI(
        title="MailSort — Enterprise Email Triage",
        description=(
            "OpenEnv environment for enterprise email triage. "
            "Tasks: email_classify (easy), email_rank (medium), email_triage (hard)."
        ),
        version="1.0.0",
    )

    # Single shared environment instance for the fallback HTTP server.
    # Production deployments should use openenv-core's WebSocket concurrency.
    _env = MailSortEnvironment()

    # --- Request / response models ---

    class ResetRequest(BaseModel):
        model_config = {"extra": "allow"}
        task: Optional[str] = None
        seed: Optional[int] = None
        episode_id: Optional[str] = None

    class StepRequest(BaseModel):
        model_config = {"extra": "allow"}
        classifications: Optional[list] = None
        rankings: Optional[list] = None
        routings: Optional[list] = None
        response_draft: Optional[str] = None
        metadata: Optional[Dict[str, Any]] = None

    def _obs_to_dict(obs: MailSortObservation) -> Dict[str, Any]:
        if hasattr(obs, "model_dump"):
            return obs.model_dump()
        return obs.dict()

    def _state_to_dict(state: MailSortState) -> Dict[str, Any]:
        if hasattr(state, "model_dump"):
            return state.model_dump()
        return state.dict()

    # --- Endpoints ---

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok", "env": "mailsort-env"}

    @app.post("/reset")
    async def reset(req: ResetRequest = ResetRequest()) -> JSONResponse:
        obs = _env.reset(
            seed=req.seed,
            episode_id=req.episode_id,
            task=req.task,
        )
        state = _env.state
        return JSONResponse(
            content={
                "observation": _obs_to_dict(obs),
                "reward": 0.0,
                "done": False,
                "state": _state_to_dict(state),
            }
        )

    @app.post("/step")
    async def step(request: Request) -> JSONResponse:
        body = await request.json()
        # Support both {"action": {...}} (openenv-core style) and flat {...}
        if "action" in body and isinstance(body["action"], dict):
            body = body["action"]
        action = MailSortAction(
            classifications=body.get("classifications"),
            rankings=body.get("rankings"),
            routings=body.get("routings"),
            response_draft=body.get("response_draft"),
            metadata=body.get("metadata") or {},
        )
        obs = _env.step(action)
        state = _env.state
        return JSONResponse(
            content={
                "observation": _obs_to_dict(obs),
                "reward": obs.reward,
                "done": obs.done,
                "state": _state_to_dict(state),
            }
        )

    @app.get("/state")
    @app.post("/state")
    async def get_state() -> JSONResponse:
        return JSONResponse(content=_state_to_dict(_env.state))

    @app.get("/tasks")
    async def list_tasks() -> JSONResponse:
        from server.tasks import TASK_REGISTRY
        return JSONResponse(
            content={
                tid: {
                    "id": cfg["id"],
                    "name": cfg["name"],
                    "description": cfg["description"],
                    "difficulty": cfg["difficulty"],
                    "max_steps": cfg["max_steps"],
                }
                for tid, cfg in TASK_REGISTRY.items()
            }
        )

    @app.get("/")
    async def root():
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=MAILSORT_UI_HTML)



# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
