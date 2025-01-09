import time
import os
import sys
import json
import torch
from datetime import datetime
from model import Brain, DQN, GRP
from engine import MortalEngine
from common import filtered_trimmed_lines
from libriichi.mjai import Bot
from libriichi.dataset import Grp
from config import config

USAGE = '''Usage: python mortal.py <ID>

ARGS:
    <ID>    The player ID, an integer within [0, 3].'''

def main():
    try:
        player_id = int(sys.argv[-1])
        assert player_id in range(4)
    except:
        print(USAGE, file=sys.stderr)
        sys.exit(1)
    review_mode = os.environ.get('MORTAL_REVIEW_MODE', '0') == '1'
    print(f'review_mode:{review_mode}')

    device = torch.device('cuda:0') #cpu
    state = torch.load(config['control']['state_file'], weights_only=True, map_location=torch.device('cuda:0')) #cpu
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    num_blocks = cfg['resnet']['num_blocks']
    conv_channels = cfg['resnet']['conv_channels']
    #print(f'state: {state}')
    if 'tag' in state:
        tag = state['tag']
    else:
        timestr= datetime.now().strftime('%Y-%m-%d %H:%M:%S')  #datetime.fromtimestamp(state['timestamp'], tz=timezone.utc).strftime('%y%m%d%H')
        tag = f'mortal{version}-b{num_blocks}c{conv_channels}-t{timestr}'

    mortal = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels).eval()
    dqn = DQN(version=version).eval()
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])

    engine = MortalEngine(
        mortal,
        dqn,
        version = version,
        is_oracle = False,
        device = device,
        enable_amp = False,
        enable_quick_eval = not review_mode,
        enable_rule_based_agari_guard = True,
        name = 'mortal',
    )
    bot = Bot(engine, player_id)

    if review_mode:
        logs = []

    # 非复盘模式下的时间统计
    if not review_mode:
        start_time = time.time()
        reaction_times = []

    # 主推理循环
    for line in filtered_trimmed_lines(sys.stdin):
        if review_mode:
            logs.append(line)

        if not review_mode:
            iter_start = time.time()

        if reaction := bot.react(line):
            #print(reaction, flush=True)

            if not review_mode:
                reaction_time = time.time() - iter_start
                reaction_times.append(reaction_time)
                #print(f"Decision time: {reaction_time:.3f}s", file=sys.stderr)

        elif review_mode:
            print('{"type":"none","meta":{"mask_bits":0}}', flush=True)

    if review_mode:
        grp = GRP(**config['grp']['network'])
        grp_state = torch.load(config['grp']['state_file'], weights_only=True, map_location=torch.device('cuda:0'))
        grp.load_state_dict(grp_state['model'])

        ins = Grp.load_log('\n'.join(logs))
        feature = ins.take_feature()
        seq = list(map(
            lambda idx: torch.as_tensor(feature[:idx+1], device=device),
            range(len(feature)),
        ))

        with torch.inference_mode():
            logits = grp(seq)
        matrix = grp.calc_matrix(logits)
        extra_data = {
            'model_tag': tag,
            'phi_matrix': matrix.tolist(),
        }
        print(json.dumps(extra_data), flush=True)
    else: # 非复盘模式下的时间统计
        total_time = time.time() - start_time
        avg_reaction_time = sum(reaction_times) / len(reaction_times) if reaction_times else 0
        print(f"\nTotal time: {total_time:.3f}s", file=sys.stderr)
        print(f"Average decision time: {avg_reaction_time:.3f}s", file=sys.stderr)
        print(f"Total decisions: {len(reaction_times)}", file=sys.stderr)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
