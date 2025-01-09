import time
import os
import json
import torch
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from datetime import datetime
from model import Brain, DQN, GRP
from engine import MortalEngine
from common import filtered_trimmed_lines
from libriichi.mjai import Bot
from libriichi.dataset import Grp
from config import config
from typing import List, Optional

app = FastAPI()


# 全局变量存储模型和引擎
mortal = None
dqn = None
engine = None
bot = None
review_mode = os.environ.get('MORTAL_REVIEW_MODE', '0') == '1'
device = torch.device('cuda:0')  # 或 'cpu'

@app.on_event("startup")
async def startup_event():
    global mortal, dqn, engine, bot

    try:
        player_id = 2 #int(os.environ.get('PLAYER_ID', '0'))
        assert player_id in range(4)
    except:
        raise HTTPException(status_code=400, detail="Invalid player ID. Must be an integer within [0, 3].")

    state = torch.load(config['control']['state_file'], weights_only=True, map_location=device)
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    num_blocks = cfg['resnet']['num_blocks']
    conv_channels = cfg['resnet']['conv_channels']

    if 'tag' in state:
        tag = state['tag']
    else:
        timestr = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        tag = f'mortal{version}-b{num_blocks}c{conv_channels}-t{timestr}'

    mortal = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels).eval()
    dqn = DQN(version=version).eval()
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])

    engine = MortalEngine(
        mortal,
        dqn,
        version=version,
        is_oracle=False,
        device=device,
        enable_amp=False,
        enable_quick_eval=not review_mode,
        enable_rule_based_agari_guard=True,
        name='mortal',
    )
    bot = Bot(engine, player_id)

class Event(BaseModel):
    type: str
    actor: Optional[int]= None
    pai: Optional[str]= None
    tsumogiri: Optional[bool]= None
    bakaze: Optional[str]= None
    dora_marker: Optional[str]= None
    kyoku: Optional[int]= None
    honba: Optional[int]= None
    kyotaku: Optional[int]= None
    oya: Optional[int]= None
    scores: Optional[List]= None
    tehais:Optional[List]= None
    target:Optional[int]= None
    consumed:Optional[List]= None
    
    def toJson(self):
        return self.model_dump_json()

class InferenceRequest(BaseModel):
    input_data: List[Event]


@app.post("/infer/")
async def infer(request: InferenceRequest):
    global bot, review_mode, device

    if review_mode:
        logs = []

    # 非复盘模式下的时间统计
    if not review_mode:
        start_time = time.time()
        reaction_times = []

    # 主推理循环
    eventList = request.input_data
    # print(eventList)
    #lines = filtered_trimmed_lines(eventList.splitlines())
    results = []

    for event in eventList:
        try:
            line = event.toJson()
            #print(line)
        except json.JSONDecodeError:
            results.append(f"Invalid JSON line: {line}")
            continue

        if not review_mode:
            iter_start = time.time()

        if reaction := bot.react(line):
            #results.append(reaction)

            if not review_mode:
                reaction_time = time.time() - iter_start
                reaction_times.append(reaction_time)


    # 非复盘模式下的时间统计
    if not review_mode: 
        total_time = time.time() - start_time
        avg_reaction_time = sum(reaction_times) / len(reaction_times) if reaction_times else 0
        results.append(f"\nTotal time: {total_time:.3f}s")
        results.append(f"Average decision time: {avg_reaction_time:.3f}s")
        results.append(f"Total decisions: {len(reaction_times)}")

    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

   
    #pip install uvicorn
    #uvicorn mortal_api_server:app --workers 8
    