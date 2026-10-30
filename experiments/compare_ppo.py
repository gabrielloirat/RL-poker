"""Final held-out comparison and optional promotion of a selected candidate."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from agents.ppo import PPOAgent
from poker_rl.evaluation import evaluate_agents, attach_validation
from poker_rl.game import PokerGame
from poker_rl.hand_analysis import compact_report
from poker_rl.plugins import AgentRegistry
from poker_rl.records import SessionRecorder, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--reference-model',required=True,help='Identifiant courant ou version immuable servant de référence')
    parser.add_argument('--reward-scale',type=float,help="Unité d'entraînement si un ancien checkpoint expérimental ne l'enregistre pas")
    parser.add_argument('--deterministic',action='store_true')
    parser.add_argument('--hands',type=int,default=3000)
    parser.add_argument('--seed',type=int,default=3_000_000_007)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-report',type=Path)
    parser.add_argument('--promote',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    registry=AgentRegistry()
    reference=registry.load(args.reference_model)
    candidate=PPOAgent(checkpoint=args.checkpoint,autoload=False,deterministic=args.deterministic)
    if args.reward_scale is not None:
        if not np.isfinite(args.reward_scale) or args.reward_scale<=0:
            raise ValueError('--reward-scale doit être positif et fini.')
        candidate.model.poker_reward_scale=args.reward_scale
    elif not hasattr(candidate.model,'poker_reward_scale'):
        raise ValueError("Ce checkpoint ne documente pas ses unités ; précisez --reward-scale.")
    reports={}
    for label,agent in [('before',reference),('after',candidate)]:
        print('Final validation:',label,flush=True)
        if label=='before' and args.reference_report:
            report=json.loads(args.reference_report.read_text(encoding='utf-8'))
            assert report['checkpoint_sha256']==registry.resolve_model(args.reference_model)['checkpoint_sha256']
            assert report['protocol']['seed']==args.seed and report['protocol']['hands_per_opponent']==args.hands
        else:
            report=evaluate_agents([agent,None],(0,),seed=args.seed,hands_per_opponent=args.hands,
                on_progress=lambda seat,opponent,done,total: print(f'{label} {opponent}: {done}/{total}',flush=True)
                    if done%500==0 else None)
        if report['status']!='completed':
            raise RuntimeError(report)
        write_json(args.output/f'{label}.json',report)
        reports[label]=report
        print(label,[(b['opponent'],b['bb_per_100']) for b in report['agents'][0]['benchmarks']],flush=True)
    comparison={'seed':args.seed,'hands_per_opponent':args.hands,'checkpoint':str(args.checkpoint.resolve()),
        'deterministic':args.deterministic,'before':compact_report(reports['before']),
        'after':compact_report(reports['after']),'differences':[]}
    for before,after in zip(reports['before']['agents'][0]['benchmarks'],reports['after']['agents'][0]['benchmarks']):
        old,new=before['hand_analysis']['hands'],after['hand_analysis']['hands']
        assert len(old)==len(new)==args.hands
        assert all(a['seed']==b['seed'] and a['bot_seat']==b['bot_seat'] for a,b in zip(old,new))
        deltas=np.array([b['net_bb']-a['net_bb'] for a,b in zip(old,new)])
        mean=float(deltas.mean()*100)
        margin=float(1.96*deltas.std(ddof=1)/len(deltas)**.5*100)
        comparison['differences'].append({'opponent':before['opponent'],'before_bb_per_100':before['bb_per_100'],
            'after_bb_per_100':after['bb_per_100'],'gain_bb_per_100':mean,'paired_ci95':[mean-margin,mean+margin]})
    old=[b['bb_per_100'] for b in reports['before']['agents'][0]['benchmarks']]
    new=[b['bb_per_100'] for b in reports['after']['agents'][0]['benchmarks']]
    comparison['improved_balanced_benchmark']=min(new)>min(old) and np.mean(new)>np.mean(old) and min(new)>0
    write_json(args.output/'comparison.json',comparison)
    print(json.dumps(comparison['differences'],indent=2),flush=True)
    if args.promote:
        if not comparison['improved_balanced_benchmark']:
            raise RuntimeError('Le candidat ne satisfait pas le critère de promotion sur ce test indépendant.')
        options={'autoload':False,'deterministic':candidate.deterministic,
                 'seed':candidate.model.seed,'poker_features':candidate.poker_features}
        recorder=SessionRecorder([('ppo',options),('mixed',{'seed':candidate.model.seed+1})],kind='train',
            training_seats=(0,),seed=candidate.model.seed,game_kwargs=reports['after']['protocol']['environment'],
            continuous=False,root=registry.records_dir,registry=registry)
        try:
            game=PokerGame(**reports['after']['protocol']['environment'])
            recorder.manifest.update(observation_fields=game.observation_fields,observation_version=game.observation_version,
                history_recorded=False,experiment_checkpoint=str(args.checkpoint.resolve()),
                reference_model=args.reference_model,training_protocol={'independent_hands':True,
                'normalize_rewards':True,'augment_suits':True,'reward_unit':'initial_stack'},
                comparison=comparison['differences'])
            game.close()
            attach_validation(recorder,reports['after'])
            stats={'timesteps':candidate.model.num_timesteps,'training_history_recorded':False}
            training_file=args.checkpoint.parent/'training.json'
            if training_file.exists():
                training=json.loads(training_file.read_text(encoding='utf-8'))
                # End-of-run stats only belong to that final checkpoint.
                if args.checkpoint.name=='final.checkpoint':
                    stats.update(training['stats'])
            recorder.finish(stats,agents=[candidate,registry.load('mixed',{'seed':candidate.model.seed+1})])
            write_json(recorder.path/'comparison.json',comparison)
            write_json(args.output/'promoted.json',{'archive':str(recorder.path),'model_id':recorder.manifest['agents'][0]['model_id']})
            print('Promoted to current bot:',recorder.path,flush=True)
        finally:
            if not recorder.closed:
                recorder.finish({},status='failed')


if __name__=='__main__':
    main()
