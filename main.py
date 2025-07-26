import ray
import os
import gin
import argparse
from D2MAV_A.agent import Agent
from D2MAV_A.runner import Runner
from copy import deepcopy
import time
import platform
from datetime import datetime
import numpy as np
import gc
import tensorflow as tf
import random

os.environ["PYTHONPATH"] = os.getcwd()

import logging
import json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.FATAL)

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


parser = argparse.ArgumentParser()
parser.add_argument("--cluster", action="store_true")
parser.add_argument("--learn_action", action="store_true")
parser.add_argument("--debug", action="store_true")

args = parser.parse_args()


@gin.configurable
class Driver:
    def __init__(
        self,
        cluster=False,
        run_name=None,
        scenario_file=None,
        config_file=None,
        num_workers=1,
        iterations=1000,
        simdt=1,
        max_steps=1024,
        speeds=[0, 0, 84],
        alt_level_separation=500,
        LOS=10,
        dGoal=100,
        intruderThreshold=750,
        altChangePenalty=0.05,
        stepPenalty=0,
        clearancePenalty=0.005,
        gui=False,
        non_coop_tag=0,
        max_alt = 3000,
        min_alt = 1000,
        weighting_factor_noise=[0, 0.5, 1],
        weighting_factor_energy=[1, 0.5, 0],
        weighting_factor_separation=[0, 0, 0],
        weights_file=None,
        run_type="train",
        traffic_manager_active=True,
        n_waypoints=2,
    ):
        self.cluster = cluster
        self.run_name = run_name
        self.run_type = run_type
        self.num_workers = num_workers
        self.simdt = simdt
        self.iterations = iterations
        self.max_steps = max_steps
        self.speeds = speeds

        self.alt_level_separation = alt_level_separation
        self.max_alt = max_alt
        self.min_alt = min_alt
        self.weighting_factor_noise=weighting_factor_noise
        self.weighting_factor_energy = weighting_factor_energy
        self.weighting_factor_separation = weighting_factor_separation

        self.LOS = LOS
        self.dGoal = dGoal
        self.intruderThreshold = intruderThreshold
        self.altChangePenalty = altChangePenalty
        self.stepPenalty = stepPenalty
        self.clearancePenalty = clearancePenalty
        self.scenario_file = scenario_file
        self.config_file = config_file
        self.weights_file = weights_file
        self.gui = gui

        # === MODIFIED: Adjusted State and Action Dimension ===
        self.action_dim = 3
        self.observation_dim = 5
        self.context_dim = 3
        # === END OF MODIFIED SEGMENT ===


        self.working_directory = os.getcwd()
        self.non_coop_tag = non_coop_tag
        self.traffic_manager_active = traffic_manager_active
        self.n_waypoints = n_waypoints


        if self.run_name is None:
            path_results = "results"
            path_models = "models"
        else:
            path_results = f"results/{self.run_name}"
            path_models = f"models/{self.run_name}"

        os.makedirs(path_results, exist_ok=True)
        os.makedirs(path_models, exist_ok=True)

        self.path_models = path_models
        self.path_results = path_results

    def train(self):
        # Start simulations on actors
        current_time = time.time()
        testing_params= {}
        base_seed = 42   
        for run_idx in range(len(self.weighting_factor_noise)):
            seed = base_seed + run_idx
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            # print("Reached Here")
            tf.keras.backend.clear_session()
            # Force garbage collection
            gc.collect()
            self.agent = Agent()
            self.agent_template = deepcopy(self.agent)
            self.agent.initialize(tf, self.observation_dim, self.context_dim, self.action_dim)

            workers = {
                i: Runner.remote(
                    i,
                    self.agent_template,
                    scenario_file=self.scenario_file,
                    config_file=self.config_file,
                    working_directory=self.working_directory,
                    max_steps=self.max_steps,
                    simdt=self.simdt,
                    speeds=self.speeds,
                    LOS=self.LOS,
                    dGoal=self.dGoal,
                    intruderThreshold=self.intruderThreshold,
                    altChangePenalty=self.altChangePenalty,
                    weighting_factor_noise=self.weighting_factor_noise[run_idx],
                    weighting_factor_energy = self.weighting_factor_energy[run_idx],
                    weighting_factor_separation = self.weighting_factor_separation[run_idx],
                    max_alt = self.max_alt,
                    min_alt = self.min_alt,
                    stepPenalty=self.stepPenalty,
                    clearancePenalty=self.clearancePenalty,
                    gui=self.gui,
                    non_coop_tag=self.non_coop_tag,
                    traffic_manager_active=self.traffic_manager_active,
                    n_waypoints=self.n_waypoints,
                )
                for i in range(self.num_workers)
            }
            str_noise = str(self.weighting_factor_noise[run_idx]).replace(".", "")
            str_energy = str(self.weighting_factor_energy[run_idx]).replace(".", "")
            str_separation = str(self.weighting_factor_separation[run_idx]).replace(".", "")
            total_string = f'noise_{str_noise}_energy_{str_energy}_separation_{str_separation}'
            path_models_individual = f'{self.path_models}/noise_{str_noise}_energy_{str_energy}_separation_{str_separation}'
            path_results_individual = f'{self.path_results}/noise_{str_noise}_energy_{str_energy}_separation_{str_separation}'
            
            os.makedirs(path_models_individual, exist_ok=True)
            os.makedirs(path_results_individual, exist_ok=True)
            
            rewards = []
            episodic_rewards = []
            total_nmacs = []
            iteration_record = []
            total_nmac_time = []
            total_transitions = 0
            best_reward = -np.inf

            if self.agent.equipped:
                if self.weights_file is not None:
                    self.agent.model.load_weights(self.weights_file)

                weights = self.agent.model.get_weights()
            else:
                weights = []

            runner_sims = [
                workers[agent_id].run_one_iteration.remote(weights)
                for agent_id in workers.keys()
            ]
            convergence_threshold = 0.01  # Small change in rolling mean reward
            window_size = 20  # Same as the rolling mean calculation
            converged_iteration = None
            episodic_reward_list = {i: [] for i in range(self.num_workers)}
            episode_rewards = {i: 0 for i in range(self.num_workers)}  # Track episode rewards per worker
            
            for i in range(self.iterations):
                done_id, runner_sims = ray.wait(runner_sims, num_returns=self.num_workers)
                results = ray.get(done_id)

                transitions, workers_to_remove = self.agent.update_weights(results)

                if self.agent.equipped:
                    weights = self.agent.model.get_weights()

                total_reward = []
                
                mean_total_reward = None
                total_episode = []
                mean_total_episodic = None
                nmacs = []
                total_ac = []

                for r_index, result in enumerate(results):
                    data = ray.get(result)

                    try:
                        total_reward.append(float(np.sum(data[0]["raw_reward"])))
                        episode_rewards[r_index] += float(np.sum(data[0]["raw_reward"]))
                    except:
                        pass

                    if data[0]["environment_done"]:
                        # === MODIFIED: Adjusted Stored Metrics During Training ===
                        episodic_reward_list[r_index].append(episode_rewards[r_index])
                        total_episode.append(episode_rewards[r_index])
                        nmacs.append(data[0]["nmacs"])
                        total_nmac_time += [data[0]["nmac_time"]]
                        max_noise_increase = float(data[0]['max_noise_increase'])
                        num_alt_adjustments = data[0]['num_alt_adjustments']
                        avg_alt_adjustments = np.mean(list(num_alt_adjustments.values()))
                        testing_params[total_string] = (avg_alt_adjustments, max_noise_increase)
                        total_ac.append(data[0]["total_ac"])
                        # === END OF MODIFIED SEGMENT ===


                if total_reward:
                    mean_total_reward = np.mean(total_reward)

                if len(total_episode) != 0:
                    mean_total_episodic = np.mean(total_episode)
                    episodic_rewards.append(mean_total_episodic)
                    np.save("{}/episodic_reward.npy".format(path_results_individual), np.array(episodic_rewards))

                for j, nmac in enumerate(nmacs):
                    # === MODIFIED: Adjusted Printed Log ===
                    print(f"     Scenario Complete {self.weighting_factor_noise[run_idx]} {self.weighting_factor_energy[run_idx]}     ")
                    print("|------------------------------|")
                    print(f"| Total LOS:      {nmac}      |")
                    print(f"| Maximum Noise Increase: {max_noise_increase}  |")
                    print(f"| Average Number of Altitude Adjustments: {avg_alt_adjustments}  |")
                    print(f"| Total Aircraft:   {total_ac[j]}  |")
                    print("|------------------------------|")
                    print(" ")
                    total_nmacs.append(nmac)
                    iteration_record.append(i)
                    # === END OF MODIFIED SEGMENT ===


                if mean_total_reward is not None:
                    rewards.append(mean_total_reward)
                    np.save("{}/reward.npy".format(path_results_individual), np.array(rewards))                

                if len(nmacs) > 0:
                    np.save("{}/nmacs.npy".format(path_results_individual), np.array(total_nmacs))

                    np.save(
                        "{}/nmac_time.npy".format(path_results_individual),
                        np.array(total_nmac_time),
                    )

                    np.save(
                        "{}/iteration_record.npy".format(path_results_individual),
                        np.array(iteration_record),
                    )

                total_transitions += transitions

                if not mean_total_reward:
                    mean_total_reward = 0

                print(f"     Iteration {i} Complete     ")
                print(f"Name of Training Run: {self.run_name}")
                print("|------------------------------|")
                print(f"| Mean Total Reward:   {np.round(mean_total_reward,1)}  |")
                roll_mean = np.mean(rewards[-150:])
                print(f"| Rolling Mean Reward: {np.round(roll_mean,1)}  |")
                print("|------------------------------|")
                print(" ")

                if self.agent.equipped:
                    if len(rewards) > 150:
                        print(f"Saving Best Model {path_models_individual}")
                        if np.mean(rewards[-150:]) > best_reward:
                            best_reward = np.mean(rewards[-150:])
                            self.agent.model.save_weights(
                                "{}/best_model.h5".format(path_models_individual)
                            )
                    print(f"Saving Current Model {path_models_individual}")
                    self.agent.model.save_weights("{}/model.h5".format(path_models_individual))

                if len(episodic_rewards) > window_size:
                    roll_mean = np.mean(episodic_rewards[-window_size:])
                    prev_roll_mean = np.mean(episodic_rewards[-(window_size + 10):-10])  # Compare with an earlier window

                    if abs(roll_mean - prev_roll_mean) < convergence_threshold and converged_iteration is None:
                        converged_iteration = i
                        print(f"Model has converged at iteration {converged_iteration}")

                np.save("{}/convergence_iteration.npy".format(path_results_individual), np.array([converged_iteration]))
                runner_sims = [
                    workers[agent_id].run_one_iteration.remote(weights)
                    for agent_id in workers.keys()
                ]
        print("Final Output: ", testing_params)
        print("Run Time: ", time.time() - current_time)

    def evaluate(self):
        # Start simulations on actors
        for w_file in self.weights_file:
            tf.keras.backend.clear_session()
            # Force garbage collection
            gc.collect()
            self.agent = Agent()
            self.agent_template = deepcopy(self.agent)
            self.agent.initialize(
                tf, self.observation_dim, self.context_dim, self.action_dim
            )
            workers = {
                i: Runner.remote(
                    i,
                    self.agent_template,
                    scenario_file=self.scenario_file,
                    config_file=self.config_file,
                    working_directory=self.working_directory,
                    max_steps=self.max_steps,
                    simdt=self.simdt,
                    speeds=self.speeds,
                    LOS=self.LOS,
                    dGoal=self.dGoal,
                    intruderThreshold=self.intruderThreshold,
                    altChangePenalty=self.altChangePenalty,
                    max_alt = self.max_alt,
                    min_alt = self.min_alt,
                    stepPenalty=self.stepPenalty,
                    clearancePenalty=self.clearancePenalty,
                    gui=self.gui,
                    non_coop_tag=self.non_coop_tag,
                    traffic_manager_active=self.traffic_manager_active,
                    n_waypoints=self.n_waypoints,
                )
                for i in range(self.num_workers)
            }

            rewards = []
            total_nmacs = []
            cumulative_nmacs = 0
            total_nmac_time = []
            iteration_record = []
            total_transitions = 0
            best_reward = -np.inf
            scenario = 0
            metric_list =[]
            num_alt_adjustments = {}
            
            second_to_last = os.path.basename(os.path.dirname(w_file))
            second_to_last = second_to_last.replace("train", "test")
            print(second_to_last)
            
            
            if self.agent.equipped:
                self.agent.model.load_weights(w_file)
                weights = self.agent.model.get_weights()
            else:
                weights = []

            runner_sims = [
                workers[agent_id].run_one_iteration.remote(weights)
                for agent_id in workers.keys()
            ]

            for i in range(self.iterations):
                done_id, runner_sims = ray.wait(runner_sims, num_returns=self.num_workers)
                results = ray.get(done_id)

                total_reward = []

                nmacs = []
                total_ac = []

                for result in results:
                    data = ray.get(result)
                    total_reward.append(float(np.sum(data[0]["raw_reward"])))
                    if data[0]["environment_done"]:

                        nmacs.append(data[0]["nmacs"])
                        cumulative_nmacs += data[0]["nmacs"]
                        total_nmac_time += [data[0]["nmac_time"]]
                        total_ac.append(data[0]["total_ac"])
                        max_noise_increase = float(data[0]['max_noise_increase'])
                        avg_noise_increase = data[0]['avg_noise_increase']
                        congestion_distribution = data[0]['congestion_distribution']
                        num_alt_adjustments = data[0]['num_alt_adjustments']
                        avg_noise_dict = {}
                        for id_ in avg_noise_increase.keys():
                            avg_noise_dict[id_] = np.mean(avg_noise_increase[id_])
                        

                mean_total_reward = np.mean(total_reward)

                for j, nmac in enumerate(nmacs):
                    print(f"     Scenario Complete {self.run_name}    ")
                    print("|------------------------------|")
                    print(f"| Total LOS:      {nmac}      |")
                    print(f"| Maximum Noise Increase: {max_noise_increase}  |")
                    print(f"| Average Number of Altitude Adjustments: {np.mean(list(num_alt_adjustments.values()))}  |")
                    print(f"| Total Aircraft:   {total_ac[j]}  |")
                    print("|------------------------------|")
                    print(" ")

                    # === MODIFIED: Adjusted Stored Metrics During Testing ===
                    total_nmacs.append(nmac)
                    iteration_record.append(i)
                    metric_dict = {}
                    metric_dict['scenario_num'] = scenario
                    scenario += 1
                    metric_dict['los'] = int(cumulative_nmacs)
                    cumulative_nmacs = 0
                    metric_dict['max_noise'] = float(max_noise_increase)
                    metric_dict['avg_noise'] = avg_noise_dict
                    metric_dict['congestion_distribution'] = congestion_distribution
                    metric_dict['num_alt_adjustments'] = num_alt_adjustments
                    metric_list.append(metric_dict)
                    # === END OF MODIFIED SEGMENT ===


                rewards.append(mean_total_reward)

                runner_sims = [
                    workers[agent_id].run_one_iteration.remote(weights)
                    for agent_id in workers.keys()
                ]
            
                # === MODIFIED: Store Metrics to a Separate Log ===
                folder_path = 'log/test_models_full_results'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                with open('log/test_models_full_results/{}.json'.format(second_to_last), 'w') as file:
                    json.dump(metric_list, file, indent=4)
                # === END OF MODIFIED SEGMENT ===


### Main code execution
### IMPORTANT NOTE: You need to adjust the following config file whether you are training or testing
gin.parse_config_file("conf/config_test.gin")

if args.cluster:
    ## Initialize Ray
    ray.init(address=os.environ["ip_head"])
    print(ray.cluster_resources())
else:
    # check if running on Mac
    if platform.release() == "Darwin":
        ray.init(_node_ip_address="0.0.0.0", local_mode=args.debug)
    else:
        ray.init(local_mode=args.debug)
    print(ray.cluster_resources())


# Now initialize the trainer with 30 workers and to run for 100k episodes 3334 episodes * 30 workers = ~100k episodes
Trainer = Driver(cluster=args.cluster)
if Trainer.run_type == "train":
    Trainer.train()
else:
    Trainer.evaluate()
