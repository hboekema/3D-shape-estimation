import random
import numpy as np

from tqdm import tqdm
import keras.backend as K
from keras.models import Model
from keras import regularizers
from keras.layers import Input, Dense, Flatten, Concatenate

from .critic import Critic
from .actor import Actor


class A2C:
    """ Multi-Agent Advantage Actor-Critic main algorithm """
    def __init__(self, act_dim, env_dim, actor_network, critic_network, k=1, batch_size=32, gamma=0.99, actor_lr=0.0001, critic_lr=0.002):
        """ Initialization """
        # Environment and A2C parameters
        self.batch_size = batch_size
        self.k = k
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        # Create actor and critic networks
        self.actor = Actor(env_dim, act_dim, actor_network, actor_lr)
        self.critic = Critic(env_dim, act_dim, critic_network, critic_lr)

        # Build optimizers
        self.a_opt = self.actor.optimizer()
        self.c_opt = self.critic.optimizer()

        # Results storage
        self.results = []

    def policy_action(self, state):
        """ Use the actor to predict the next action to take, using the policy """
        policy_mu, policy_scale = self.actor.predict(np.array(state))
        #print("avg. magnitude of mu of first example: " + str(np.mean(np.abs(policy_mu[0]))))
        #print("avg. magnitude of stddev of first example: " + str(np.mean(np.abs(np.diag(policy_scale[0])))))
        sample = self.actor.sample(policy_mu, policy_scale)

        return sample  # shape (batch_size, self.act_dim)

    def discount(self, r):
        """ Compute the gamma-discounted rewards over a time frame """
        discounted_r, cumul_r = np.zeros_like(r), np.zeros((r.shape[1],))
        for t in reversed(range(0, r.shape[0])):
            #print("t:" +str(t))
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

    def train_models(self, states, actions, discounted_rewards, new_state, done):
        """ Update actor and critic networks from experience """
        # Compute discounted rewards and Advantage (TD. Error)
        state_values = np.squeeze(self.critic.predict(np.array(states)))
        #print("state_values shape: " + str(state_values.shape))
        new_state_values = np.squeeze(self.critic.predict(np.array(new_state)))
        #print("new_state_values shape: " + str(new_state_values.shape))

        #print("discounted_rewards shape: " + str(discounted_rewards.shape))
        td_target = discounted_rewards + self.gamma * new_state_values
        #print("td_target shape: " + str(td_target.shape))
        advantages = td_target - state_values
        #print("advantages shape: " + str(advantages.shape))

        # Only show examples that were not done
        states = states[~done]
        actions = actions[~done]
        discounted_rewards = np.squeeze(discounted_rewards[~done])
        td_target = np.squeeze(td_target[~done])
        advantages = np.squeeze(advantages[~done])
        print("discounted_rewards: \n" + str(discounted_rewards[:1]))
        print("state_values: \n" + str(state_values[:1]))
        print("new_state_values: \n" + str(new_state_values[:1]))
        print("td_target: \n" + str(td_target[:1]))
        print("advantages: \n" + str(advantages[:1]))

        #print("states shape: " + str(states.shape))
        #print("actions shape: " + str(actions.shape))
        #print("discounted_rewards shape: " + str(discounted_rewards.shape))
        #print("advantages shape: " + str(advantages.shape))

        if states.size > 0:
            # Networks optimization
            K.set_learning_phase(1)
            a_loss, eligibility, entropy, penalty = self.a_opt([states, actions, advantages])
            print("actor loss: " + str(a_loss) + "     eligibility: " + str(np.mean(eligibility)) + "   entropy: " + str(np.mean(entropy)) + "    penalty: " + str(np.mean(penalty)))
            c_loss = self.c_opt([states, td_target])
            print("critic loss: " + str(c_loss[0]))
            K.set_learning_phase(0)

    def model_train_step(self, states, actions, rewards, new_states, done_history):
        """ Perform a single training step """
        # Get the appropriate time elements from each array
        actions_np = np.squeeze(np.array(actions)[0])
        rewards_np = np.array(rewards)
        discounted_rewards_np = np.squeeze(self.discount(rewards_np)[0])
        states_np = np.squeeze(np.array(states)[0])
        new_states_np = np.squeeze(np.array(new_states)[-1])
        done_history_np = np.squeeze(np.array(done_history)[0])

        # Train using discounted rewards over k steps i.e. compute updates and average over the number of examples in the batch
        self.train_models(states_np, actions_np, discounted_rewards_np, new_states_np, done_history_np)


    def train(self, env, num_episodes, save_period=None, save_dir="."):
        """ Main A2C training algorithm """
        self.results = []

        # Main loop
        tqdm_e = tqdm(range(num_episodes), desc='Mean cum. reward', leave=True, unit=" episodes")
        for e in tqdm_e:
            # Reset episode
            time, tau, cumul_reward, done = 0, 0, np.zeros((self.batch_size,)), np.array([False for i in range(self.batch_size)])
            old_state = env.reset()
            actions, states, rewards, new_states = [], [], [], []
            done_history = []

            print("\n")
            while len(done) > 0 and np.any(~done):
                # Actor picks an action (following the policy)
                a = self.policy_action(old_state)
                #print("avg. magnitude of action for first example: " + str(np.mean(np.abs(a[0]))))
                #print("magnitude of action for first example: " + str(a[0]))

                # Retrieve new state, reward, and whether the state is terminal
                new_state, r, done, _ = env.step(a)

                # Memorize (s, a, r) for k steps for training (only for examples that have not finished)
                actions.append(a)
                rewards.append(r)
                states.append(old_state)
                new_states.append(new_state)
                done_history.append(done)

                # Update current state
                old_state = new_state
                cumul_reward += r
                time += 1

                # Print statistics
                print("Step: {:02d}/{:02d}    No. done: {:02d}/{:02d}   Mean reward: {:04f}".format(time, env.step_limit, np.sum(done), self.batch_size, np.nan_to_num(np.nansum(r)/np.sum(~done))))

                tau = time - self.k
                if np.all(done) or tau >= 0:
                    # Perform learning step
                    self.model_train_step(states, actions, rewards, new_states, done_history)

                    # Pop an element off the front of the history vectors
                    actions.pop(0)
                    rewards.pop(0)
                    states.pop(0)
                    new_states.pop(0)
                    done_history.pop(0)

            # Display and store score
            self.results.append(np.mean(cumul_reward))
            tqdm_e.set_description("Mean cum. reward: {:03f}".format(np.mean(cumul_reward)))
            tqdm_e.refresh()


            if save_period is not None and e > 0 and e % save_period == 0:
                print("Saving model to directory:\n{}".format(save_dir))
                self.save_weights(path=save_dir, episode=e)

            print("\n")

        return self.results


    def validate(self, period, path):
        pass

    def save_weights(self, path, episode=0):
        mean_reward = np.mean(self.results)
        path += 'A2C-E{:05d}-R_{:02f}-'.format(episode, mean_reward)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)

