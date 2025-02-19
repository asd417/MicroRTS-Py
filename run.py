
# Dependencies required are
import argparse
import torch # with cuda

import numpy as np
# if you want to record videos, install stable-baselines3 and use its `VecVideoRecorder`
from stable_baselines3.common.vec_env import VecVideoRecorder
from experiments.ssvd_trainer import fitness, fitness_mcts, SSVDVariable, run_test_es,run_test_ga,run_test_gam
from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env_custom import MicroRTSGridModeVecEnv
from gym_microrts.envs.vec_mcts_env import MicroRTSMCTSEnv
from setuptools._distutils.util import strtobool

def parse_args():
    parser = argparse.ArgumentParser(
                    prog='SSVD Trainer',
                    description='Trains a ssvd model on a microrts environment')
    # training type
    parser.add_argument('--mcts', help='train mcts model',type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--mcts-time', help='time in milliseconds used by mcts models per turn',type=int, default=300)

    # Environment Settings
    parser.add_argument('--cuda', help='use cuda',type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='whether to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--headless', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='whether to display environment')
    parser.add_argument('--exp-name', type=str, default="",
        help='the name of this experiment')
    parser.add_argument('--override-exp', help='set to override existing exp log and population',type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--pre-s-weights', type=int, default=1,
        help='the number of weight matrices before sigma multiplication')
    parser.add_argument('--post-s-weights', type=int, default=1,
        help='the number of weight matrices after sigma multiplication')
    parser.add_argument('--num-bot-envs', type=int, default=5,
        help='the number of bot game environment; 16 bot envs means 16 games in parallel')
    parser.add_argument('--max-steps', type=int, default=3000,
        help='the maximum number of steps per game environment')

    # Genetic Algorithm settings
    parser.add_argument('--population', type=int, default=50,
        help='the size of population')
    parser.add_argument('--max-gen', type=int, default=1000,
        help='maximum generation')
    parser.add_argument('--elitism', type=float, default=0.1,
        help='elitism used by GA and GAM')
    parser.add_argument('--optimizer', type=str, default="",
        help='the name of the optimizer. Choose between GA, GAM, OPENAIES')
    

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    env_num = args.num_bot_envs
    pop = args.population
    max_gen = args.max_gen
    elitism = args.elitism
    maxstep = args.max_steps
    # Win/Loss ResourceGather ProduceWorker ProduceBuilding AttackReward ProduceCombatUnit
    reward_weight_original = [10.0, 1.0, 1.0, 0.2, 1.0, 4.0]
    reward_weight_new = [10.0, 1.0, 1.0, 1.0, 5.0, 4.0]
    if not args.mcts:
        envs = MicroRTSGridModeVecEnv(
            num_selfplay_envs=0,
            num_bot_envs=env_num,
            max_steps=maxstep,
            render_theme=2,
            ai2s=[microrts_ai.coacAI for _ in range(env_num)],
            map_paths=["maps/16x16/basesWorkers16x16.xml"],
            reward_weight=np.array(reward_weight_new),
        )
        fitness_f = fitness
        input_h = envs.height
        input_w = envs.width
        #actionSpace = envs.height * envs.width + 6 # board + unit type count
        actionSpace = 13
    else:
        envs = MicroRTSMCTSEnv(
            num_selfplay_envs=0,
            num_bot_envs=env_num,
            max_steps=maxstep,
            render_theme=2,
            ai2s=[microrts_ai.coacAI for _ in range(env_num)],
            map_paths=["maps/16x16/basesWorkers16x16.xml"],
            reward_weight=np.array(reward_weight_new),
        )
        fitness_f = fitness_mcts
        input_h = envs.height
        input_w = envs.width
        actionSpace = 1
    if args.capture_video:
        envs = VecVideoRecorder(envs, "videos", record_video_trigger=lambda x: x % 4000 == 0, video_length=2000)

    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    print(f"Using device: {device}")
    print(f"Observation Space Height: {input_h}")
    print(f"Observation Space Width: {input_w}")
    print(f"Action Space size: {actionSpace}")
    ssvd = SSVDVariable(input_h, input_w, actionSpace, [args.pre_s_weights,args.post_s_weights])
    print(f"Attempting to optimize {ssvd.get_chromosome_size()} weights")

    if args.optimizer == "GA":
        testname = f"GA_{env_num}_{pop}_{int(elitism * 100)}%" if args.exp_name == "" else args.exp_name
        print(f"Set experiment name as {testname}")
        run_test_ga(ssvd, envs, pop, max_gen, device, fitness_f, name=testname, override=args.override_exp, elitism=elitism, maxstep=maxstep, render=(not args.headless), record=args.capture_video)
    if args.optimizer == "GAM":
        testname = f"GAM_{env_num}_{pop}_{int(elitism * 100)}%" if args.exp_name == "" else args.exp_name
        print(f"Set experiment name as {testname}")
        run_test_gam(ssvd, envs, pop, max_gen, device, fitness_f, name=testname, override=args.override_exp, elitism=elitism, maxstep=maxstep, render=(not args.headless), record=args.capture_video)
    if args.optimizer == "OPENAIES":
        testname = f"OpenAIES_{env_num}_{pop}_{int(elitism * 100)}%" if args.exp_name == "" else args.exp_name
        print(f"Set experiment name as {testname}")
        run_test_es(ssvd, envs, pop, max_gen, device, fitness_f, name=testname, override=args.override_exp, maxstep=maxstep, render=(not args.headless), record=args.capture_video)