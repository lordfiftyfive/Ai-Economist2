from ray.air.config import RunConfig, ScalingConfig
from ray.train.rl import RLTrainer
from gymnasium.spaces import Dict, Discrete, MultiDiscrete, Tuple,Box

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="QMIX", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument("--num-cpus", type=int, default=16)
#parser.add_argument("--num-gpus", type=int, default=1)
parser.add_argument("--num-gpus", type=float, default=0)#0.25)

parser.add_argument("--num-workers", type=int, default=6)
parser.add_argument("--num-gpus-per-worker", type=float, default=0)
#parser.add_argument("render_mode", type=int, default=1)
parser.add_argument(
    "--mixer",
    type=str,
    default="qmix",
    choices=["qmix", "vdn", "none"],
    help="The mixer model to use.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=200, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=70000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=9.0, help="Reward at which we stop training."
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)
if __name__ == "__main__":
    
    #def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            #if agent_id.startswith("low_level_"):
                #return "low_level_policy"
            #else:
                #return "high_level_policy"
    args = parser.parse_args()

    ray.init(num_cpus=16, num_gpus=1,local_mode=args.local_mode)
    
    """
    we are going to first experiment with determining whether the interactions between the agents in group 1
    can be captured and visualized. An agent group is a list of agent IDs that are mapped to a single
    logical agent. All these agents must act at the same time 
    
    
    """
    
    grouping = {
        "group_1": [0,1],
    }
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        #if agent_id.startswith("low_level_"):
        if agent_id.startswith(0):
            return "low_level_policy"
        else:
            return "high_level_policy"
    obs_space = Tuple(
        [
            Dict(
                {
                    "obs": MultiDiscrete([2, 2, 2, 3]),
                    ENV_STATE: MultiDiscrete([2, 2, 2]),
                }
            ),
            Dict(
                {
                    "obs": MultiDiscrete([2, 2, 2, 3]),
                    ENV_STATE: MultiDiscrete([2, 2, 2]),
                }
            ),
        ]
    )
    act_space = Tuple(
        [
            Discrete(2),#TwoStepGame.action_space,
            Discrete(2),#TwoStepGame.action_space,
        ]
    )
    register_env(
        "grouped_twostep",
        lambda config: TwoStepGame(config).with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space
        ),
    )
    """
    grouping = {
        "group_1": [0],
        "group_2": [0,1],#2
        #"group_3": [0]
    }


    
    from ray.tune import register_env
    from ray.rllib.algorithms.dqn import DQN 
    YourExternalEnv = ... 
    register_env("my_env", 
        lambda config: YourExternalEnv(config))
    trainer = DQN(env="my_env") 
    while True: 
        print(trainer.train()) 

    """


    config = (
        get_trainable_cls(args.run)
        .get_default_config()
        .environment(TwoStepGame)#
        .framework(args.framework)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=args.num_gpus,num_gpus_per_worker=args.num_gpus_per_worker,)
        #.reset_config(reuse_actors=True)
    )
    if args.run == "MADDPG":
        obs_space = Discrete(6)
        act_space = TwoStepGame.action_space
        (
            config.framework("tf")
            .environment(env_config={"actions_are_logits": True})
            .training(num_steps_sampled_before_learning_starts=100)
            .multi_agent(
                policies={
                    "high_level_policy": PolicySpec(
                        observation_space=obs_space,
                        action_space=act_space,
                        config=config.overrides(agent_id=0),
                    ),
                    "low_level_policy": PolicySpec(
                        observation_space=obs_space,
                        action_space=act_space,
                        config=config.overrides(agent_id=1),
                    ),
                },
                policy_mapping_fn=policy_mapping_fn#lambda agent_id, episode, worker, **kwargs: "pol2"
                #if agent_id
                #else "pol1",
            )
        )
    elif args.run == "QMIX":
        (
        config.framework("torch")
        .training(mixer=args.mixer, train_batch_size=32)
            
        .rollouts(rollout_fragment_length=4)
        .exploration(
            exploration_config={
                "final_epsilon": 0.0,
            }
        )
        .environment(
            env="grouped_twostep",
            env_config={
                "separate_state_space": True,
                "one_hot_state_encoding": True,
            },
        )
    )

    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
    }
    from ray.tune.schedulers import PopulationBasedTraining

    pbt_scheduler = PopulationBasedTraining(
        time_attr='training_iteration',
        metric='episode_reward_mean',#'loss',
        mode='min',
        perturbation_interval=1,
        hyperparam_mutations={
            #"lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "alpha": tune.uniform(0.0, 1.0),
        }
    )
    trainer = RLTrainer(
        "QMIX",#args.run,
        run_config=air.RunConfig(stop=stop, verbose=2),#RunConfig(stop={"training_iteration": 5}),
        scaling_config=ScalingConfig(num_workers=10,use_gpu=True),
        #algorithm="QMIX",
        config=config.to_dict()
        
    ).fit()
    traine = RLTrainer(
        args.run,
        run_config=air.RunConfig(stop=stop, verbose=2),#RunConfig(stop={"training_iteration": 5}),
        scaling_config=ScalingConfig(num_workers=10,use_gpu=True),
        #algorithm="QMIX",
        config=config.to_dict()
        
    )
