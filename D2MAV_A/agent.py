import tensorflow as tf
import numpy as np
import numba as nb
import gin
import ray
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()


def proximal_policy_optimization_loss(
    advantage, old_prediction, entropy_beta, clip_loss
):
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        r = prob / (old_prob + 1e-10)
        return -tf.keras.backend.mean(
            tf.keras.backend.minimum(
                r * advantage,
                tf.keras.backend.clip(
                    r, min_value=1 - clip_loss, max_value=1 + clip_loss
                )
                * advantage,
            )
            + entropy_beta * -(prob * tf.keras.backend.log(prob + 1e-10))
        )

    return loss


@nb.njit()
def discount(r, discounted_r, v, done, gae):
    """Compute the gamma-discounted rewards over an episode."""
    for t in range(len(r) - 1, -1, -1):
        if done[t] or t == (len(r) - 1):
            delta = r[t] - v[t][0]
            gae[t] = delta

        else:
            delta = r[t] + 0.95 * v[t + 1][0] - v[t][0]
            gae[t] = delta + 0.95 * 0.95 * gae[t + 1]

        discounted_r[t] = gae[t] + v[t][0]

    return discounted_r


@gin.configurable
class Agent:
    def __init__(
        self,
        batch_size=512,
        epochs=8,
        max_agents=30,
        learning_rate=1e-4,
        entropy_beta=1e-4,
        clip_loss=0.2,
        action_predict=False,
        nodes=128,
        num_models=1,
        equipped=True,
        loss_weights=[1.0, 0.01],
    ):
        self.max_agents = max_agents
        self.learning_rate = learning_rate
        self.entropy_beta = entropy_beta
        self.clip_loss = clip_loss
        self.action_predict = action_predict
        self.nodes = nodes
        self.num_models = num_models
        self.batch_size = batch_size
        self.epochs = epochs
        self.equipped = equipped
        self.loss_weights = loss_weights
        self.is_initialized = False

    def initialize(self, tf_module, ownship_obs_dim, intruder_obs_dim, action_dim):
        self.ownship_obs_dim = ownship_obs_dim
        self.intruder_obs_dim = intruder_obs_dim
        self.action_dim = action_dim

        if self.equipped:
            if self.action_predict:
                self.action_predidctor = [
                    self.build_action_predictor_network(tf_module, i)
                    for i in range(self.num_models)
                ]

            if self.num_models == 1:
                self.model, self.inference_model = self.build_actor_critic_network(
                    tf_module, 0
                )
            else:
                self.models = [
                    self.build_actor_critic_network(tf_module, i)
                    for i in range(self.num_models)
                ]

    def build_actor_critic_network(self, tf_module, i):
        I = tf_module.keras.layers.Input(shape=(self.ownship_obs_dim,), name="states")

        context = tf_module.keras.layers.Input(
            shape=(None, self.intruder_obs_dim), name="context"
        )

        advantage = tf_module.keras.layers.Input(shape=(1,), name="A")
        old_prediction = tf_module.keras.layers.Input(
            shape=(self.action_dim,), name="old_pred"
        )

        transformI = tf_module.keras.layers.Dense(self.nodes, activation=None)(I)
        transformI = tf_module.keras.layers.LeakyReLU(0.2)(transformI)

        transformC = tf_module.keras.layers.Dense(self.nodes, activation=None)(context)
        transformC = tf_module.keras.layers.LeakyReLU(0.2)(transformC)

        if self.action_predict:
            #### action prediction module

            action_predict = self.action_predictor[i](context)
            transformC = tf_module.keras.layers.concatenate(
                [transformC, action_predict], axis=2
            )
            transformC = tf_module.keras.layers.Dense(self.nodes, activation=None)(
                transformC
            )
            transformC = tf_module.keras.layers.LeakyReLU(0.2)(transformC)

            #### ---> end action prediction module

        score_first_part = tf_module.keras.layers.Dense(self.nodes, use_bias=False)(
            transformC
        )
        score = tf_module.keras.layers.dot([score_first_part, transformI], [2, 1])
        attention_weights = tf_module.keras.layers.Activation("softmax")(score)
        context_vector = tf_module.keras.layers.dot(
            [transformC, attention_weights], [1, 1]
        )
        attention_vector = tf_module.keras.layers.Dense(
            self.nodes, use_bias=False, activation="tanh"
        )(context_vector)
        combined = tf_module.keras.layers.concatenate(
            [transformI, attention_vector], axis=1
        )

        H2 = tf_module.keras.layers.Dense(self.nodes * 2, activation=None)(combined)
        H2 = tf_module.keras.layers.LeakyReLU(0.2)(H2)

        H3 = tf_module.keras.layers.Dense(self.nodes * 2, activation=None)(H2)
        H3 = tf_module.keras.layers.LeakyReLU(0.2)(H3)

        output = tf_module.keras.layers.Dense(self.action_dim + 1, activation=None)(H3)

        # Split the output layer into policy and value
        policy = tf_module.keras.layers.Lambda(
            lambda x: x[:, : self.action_dim], output_shape=(self.action_dim,)
        )(output)
        value = tf_module.keras.layers.Lambda(
            lambda x: x[:, self.action_dim :], output_shape=(1,)
        )(output)

        # now I need to apply activation
        policy_out = tf_module.keras.layers.Activation("softmax", name="policy_out")(
            policy
        )
        value_out = tf_module.keras.layers.Activation("linear", name="value_out")(value)

        model = tf_module.keras.models.Model(
            inputs=[I, context, advantage, old_prediction],
            outputs=[policy_out, value_out],
        )

        opt = tf_module.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate)

        # TODO: Need to revisit action_predict to ensure this setup is correct
        if self.action_predict:
            model = tf_module.keras.models.Model(
                inputs=[I, context, advantage, old_prediction],
                outputs=[policy_out, value_out, action_predict],
            )

            model.compile(
                optimizer=opt,
                loss=[
                    proximal_policy_optimization_loss(
                        advantage=advantage,
                        old_prediction=old_prediction,
                        entropy_beta=self.entropy_beta,
                        clip_loss=self.clip_loss,
                    ),
                    tf_module.keras.losses.Huber(),
                    tf_module.losses.softmax_cross_entropy,
                ],
            )
        else:
            model = tf_module.keras.models.Model(
                inputs=[I, context, advantage, old_prediction],
                outputs=[policy_out, value_out],
            )

            inference_model = tf_module.keras.models.Model(
                inputs=[I, context], outputs=[policy_out, value_out]
            )

            model.compile(
                optimizer=opt,
                loss={
                    "policy_out": proximal_policy_optimization_loss(
                        advantage=advantage,
                        old_prediction=old_prediction,
                        entropy_beta=self.entropy_beta,
                        clip_loss=self.clip_loss,
                    ),
                    "value_out": tf_module.keras.losses.Huber(),
                },
                loss_weights={
                    "policy_out": self.loss_weights[0],
                    "value_out": self.loss_weights[1],
                },
            )

        return model, inference_model

    def build_action_predictor_network(self, tf_module, model_id):
        I = tf_module.keras.layers.Input(shape=(None, self.context_dim))
        transformC = tf_module.keras.layers.Dense(self.nodes, activation=None)(I)
        transformC = tf_module.keras.layers.LeakyReLU(0.2)(transformC)

        ap_H1 = tf_module.keras.layers.Dense(self.nodes, activation=None)(transformC)
        ap_H1 = tf_module.keras.layers.LeakyReLU(0.2)(ap_H1)

        ap_H2 = tf_module.keras.layers.Dense(self.nodes, activation=None)(ap_H1)
        ap_H2 = tf_module.keras.layers.LeakyReLU(0.2)(ap_H2)

        action_predict = tf_module.keras.layers.Dense(
            self.action_dim, activation="softmax"
        )(ap_H2)
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        model = tf_module.keras.models.Model(inputs=[I], outputs=[action_predict])
        model.compile(optimizer=opt, loss=[tf_module.losses.softmax_cross_entropy])

        return model

    def predict(self, obs, non_coop_tag, LControl_lst, LComm_lst):
        """
        2022/11/1 modified to support non-cooperative behaviors
        """
        ac_ids = np.array([x for x in obs.keys()])
        if self.equipped:
            ownship_obs = np.array([obs[x]["ownship_obs"] for x in ac_ids]).reshape(
                -1, self.ownship_obs_dim
            )
            intruder_obs = np.array([obs[x]["intruder_obs"] for x in ac_ids]).reshape(
                -1, self.max_agents, self.intruder_obs_dim
            )

            # TODO: Implement learned intent
            if self.action_predict:
                pass

            else:
                ## TODO: 0 is hardcoded, will need to revisit for multiple models

                # [1] is inference model
                policy, value = self.inference_model.predict(
                    [ownship_obs, intruder_obs], batch_size=max(ownship_obs.shape[0], 1)
                )

                actions = {
                    x: np.random.choice(self.action_dim, 1, p=policy[i].flatten())[0]
                    for (i, x) in enumerate(ac_ids)
                }

                policy = {x: policy[i] for (i, x) in enumerate(ac_ids)}

                value = {x: value[i].flatten()[0] for (i, x) in enumerate(ac_ids)}

                # Non-cooperative behaviors
                if non_coop_tag == 0:
                    pass
                elif non_coop_tag == 1:  # maintain same speed
                    lcontrol_idx = [i for i in obs.keys() if i in LControl_lst]
                    for ac_id in lcontrol_idx:
                        policy[ac_id] = np.array([0, 1, 0])
                        actions[ac_id] = 1

                elif non_coop_tag == 2:  # other aircraft cannot see it
                    pass
                    print("Loss of Communication not finished yet")
                else:
                    raise ValueError("Invalid Non_coop_tag!")

        else:
            actions = {x: -1 for (i, x) in enumerate(ac_ids)}

            policy = {x: np.random.rand(1, 3) for (i, x) in enumerate(ac_ids)}

            value = {x: np.random.rand(1) for (i, x) in enumerate(ac_ids)}

        return actions, policy, value

    def reset(self):
        self.memory = {}
        self.data = {}

    def store_step(self, ac_id, last_obs, actions, rewards, obs, dones, policy, value):
        if ac_id not in self.memory.keys():
            self.memory[ac_id] = {
                "ownship_obs": [last_obs[ac_id]["ownship_obs"]],
                "intruder_obs": [last_obs[ac_id]["intruder_obs"]],
                "actions": [actions[ac_id]],
                "rewards": [rewards[ac_id]],
                "dones": [dones[ac_id]],
                "policy": [policy[ac_id]],
                "value": [value[ac_id]],
            }

        else:
            self.memory[ac_id]["ownship_obs"].append(last_obs[ac_id]["ownship_obs"])
            self.memory[ac_id]["intruder_obs"].append(last_obs[ac_id]["intruder_obs"])
            self.memory[ac_id]["actions"].append(actions[ac_id])
            self.memory[ac_id]["rewards"].append(rewards[ac_id])
            self.memory[ac_id]["dones"].append(dones[ac_id])
            self.memory[ac_id]["policy"].append(policy[ac_id])
            self.memory[ac_id]["value"].append(value[ac_id])

        if dones[ac_id]:
            self.process_memory(ac_id)

    def process_memory(self, ac_id):
        ownship_obs = np.reshape(
            self.memory[ac_id]["ownship_obs"], (-1, self.ownship_obs_dim)
        )
        intruder_obs = np.reshape(
            self.memory[ac_id]["intruder_obs"],
            (-1, self.max_agents, self.intruder_obs_dim),
        )
        reward = np.reshape(self.memory[ac_id]["rewards"], (-1,))
        done = np.reshape(self.memory[ac_id]["dones"], (-1,))
        action = np.reshape(self.memory[ac_id]["actions"], (-1,))
        policy = np.reshape(self.memory[ac_id]["policy"], (-1, self.action_dim))
        value = np.reshape(self.memory[ac_id]["value"], (-1, 1))

        episode_length = ownship_obs.shape[0]
        discounted_r = np.zeros_like(reward)
        gae = np.zeros_like(reward)
        discounted_rewards = discount(reward, discounted_r, value, done, gae)

        advantages = np.zeros((episode_length, self.action_dim))
        index = np.arange(episode_length)
        advantages[index, action] = 1
        A = discounted_rewards - value[:, 0]

        if not "ownship_obs" in self.data.keys():
            self.data = {
                "ownship_obs": [ownship_obs.astype(np.float32)],
                "intruder_obs": [intruder_obs.astype(np.float32)],
                "action_one_hot": [advantages.astype(np.float32)],
                "A": [A.astype(np.float32)],
                "reward": [discounted_rewards.astype(np.float32)],
                "policy": [policy.astype(np.float32)],
                "raw_reward": [reward.astype(np.float32)],
            }

        else:
            self.data["ownship_obs"].append(ownship_obs.astype(np.float32))
            self.data["intruder_obs"].append(intruder_obs.astype(np.float32))
            self.data["action_one_hot"].append(advantages.astype(np.float32))
            self.data["A"].append(A.astype(np.float32))
            self.data["reward"].append(discounted_rewards.astype(np.float32))
            self.data["raw_reward"].append(reward.astype(np.float32))
            self.data["policy"].append(policy.astype(np.float32))

    def update_weights(self, total_data):
        agent_ids = []
        workers_to_remove = []
        (
            ownship_obs,
            intruder_obs,
            action_one_hot,
            policy,
            advantage,
            discounted_rewards,
        ) = ([], [], [], [], [], [])

        for worker_data_id in total_data:
            worker_data, agent_id = ray.get(worker_data_id)
            agent_ids.append(agent_id)

            if not "ownship_obs" in worker_data.keys():
                continue

            ownship_obs.append(worker_data["ownship_obs"])
            intruder_obs.append(worker_data["intruder_obs"])
            action_one_hot.append(worker_data["action_one_hot"])
            advantage.append(worker_data["A"])
            policy.append(worker_data["policy"])
            discounted_rewards.append(worker_data["reward"])

            if worker_data["environment_done"]:
                workers_to_remove.append(agent_id)

        if len(ownship_obs) == 0:
            return 0, workers_to_remove

        ownship_obs = np.concatenate(ownship_obs, axis=0)
        intruder_obs = np.concatenate(intruder_obs, axis=0)
        action_one_hot = np.concatenate(action_one_hot, axis=0)
        advantage = np.concatenate(advantage, axis=0)
        policy = np.concatenate(policy, axis=0)
        discounted_rewards = np.concatenate(discounted_rewards, axis=0)

        if self.equipped:
            self.model.fit(
                [ownship_obs, intruder_obs, advantage, policy],
                [action_one_hot, discounted_rewards],
                shuffle=True,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=0,
            )

            transitions = ownship_obs.shape[0]
        else:
            transitions = 0

        return transitions, workers_to_remove
