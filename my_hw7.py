import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from stoppedcar import StoppedCar
from fourierbasis import FourierBasis
from softmax import LinearSoftmax
import time

class LinearValueFunction(object):
    def __init__(self, basis: FourierBasis):
        self.basis = basis
        self.num_features = basis.getNumFeatures()
        self.w = np.zeros(self.num_features)

    def get_value(self, state):
        """
        This function compute the value function for the given state v_w(s).

        Parameters
        ----------
        state : np.ndarray
            array containing the state features (not basis function features)

        Returns
        -------
        v : float
            the value function v_w(s)

        """
        basis = self.basis.encode(state)
        v = np.dot(self.get_params(),basis)
        return v

    def get_value_grad(self, state : np.ndarray)->Tuple[float, np.ndarray]:
        """
        This function compute the value function for the given state v_w(s) and
        partial derivatives dv_w(s)/dw_j.

        Parameters
        ----------
        state : np.ndarray
            array containing the state features (not basis function features)

        Returns
        -------
        v : float
            the value function v_w(s)
        grad : np.ndarray
            the partial derivatives dv_w(s)/dw_j
        """
        grad = np.zeros_like(self.w)
        v = self.get_value(state)
        # finish this function

        grad = self.basis.encode(state)

        return v, grad

    def get_params(self)->np.ndarray:
        """
        This function returns the weights of the policy. This is just a helper
        function.
        Returns
        -------
        theta : np.ndarray
            The weights of the policy
        """
        return self.w

    def add_to_params(self, dw: np.ndarray):
        """
        This function adds the input array to the weights. You can use this
        function to update the policy weights.

        Parameters
        ----------
        dw : np.ndarray
            An array that is used to change the policy weights

        Returns
        -------
        None

        """
        assert self.w.shape == dw.shape, "dw and w have different shapes"
        self.w += dw

class ActorCritic(object):
    def __init__(self, policy:LinearSoftmax, vf:LinearValueFunction, alpha:float, beta:float, gamma:float):
        """

        Parameters
        ----------
        policy : LinearSoftmax
            policy to optimize
        vf : LinearValueFunction
            value function to optimize
        alpha : float
            policy step size
        beta : float
            critic step size
        gamma : float
            reward discount parameter
        """
        self.policy = policy
        self.vf = vf
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def update(self, state:np.ndarray, action:int, reward:float, next_state:np.ndarray, terminal:bool):
        """
        This function performs the Actor-Critic update for both the value function and policy.
        See Algorithm 16.2 in the course notes. If terminal is true, then v_w(S_{t+1})=0 is used.

        Parameters
        ----------
        state : np.ndarray
            The state features at time t
        action : int
            The action at time t
        reward : int
            The reward at time t
        next_state : np.ndarray
            The state features at time t+1
        terminal : bool
            A flag indicating if the episode has ended

        Returns
        -------

        """
        # TODO implement this function
        if terminal:
            vWPlus = 0
        else:
            vWPlus = self.vf.get_value(next_state)

        rho = reward + self.gamma*vWPlus-self.vf.get_value((state))
        #print(rho)
       #print(self.vf.get_value_grad(state))
        self.policy.add_to_params(self.alpha*rho*self.policy.gradient_prob(state, action))
        self.vf.add_to_params(self.beta*rho*self.vf.get_value_grad(state)[1])
        return None



def train_actorcritic(env, agent:ActorCritic, num_episodes: int):
    """
    This function trains the Actor-Critic agent every step for the number
    of episodes specified. It returns the list of sum of rewards for every
    episode.

    Parameters
    ----------
    env : StoppedCar
        The environment to interact with
    policy : ActorCritic
        The actor critic used to interact with the environment
    num_episodes : int
        The number of episodes to train for

    Returns
    -------
    Gs : List[float]
        Sum of rewards (no discounting)
    """
    Gs = []  # list of all sum of rewards
    for episode in range(num_episodes):
        G = 0.0 # variable to store the sum of rewards with no discounting (gamma=1.0)
        s = env.reset()  # sample the initial state from the environment
        done = False  # a flag to see if the episode is over

        while not done:
            a = agent.policy.get_action(s)  # sample action from policy
            snext,reward,done = env.step(a)  # get the next state, reward, and see if episode is over
            G += reward  # add the reward to the sum of rewards
            agent.update(s, a, reward, snext, done)  # update policy and value function
            s = snext  # update the state to be the next state

        Gs.append(G)  # save the sum of rewards to the list

    return Gs



def learning_curve(all_sums_of_rewards):
    """
    This function just makes the learning curve plot
    Parameters
    ----------
    all_sums_of_rewards : list
        a list of the sums_of_rewards from each lifetime

    Returns
    -------
    nothing
    """
    fig, axs = plt.subplots()
    for sums_of_rewards in all_sums_of_rewards:
        x = range(len(sums_of_rewards))
        axs.scatter(x, sums_of_rewards, color="dodgerblue", alpha=0.01, s=0.5)
    mn = np.mean(all_sums_of_rewards, axis=0)
    std = np.std(all_sums_of_rewards, axis=0)
    axs.plot(mn, linewidth=2, color="crimson")
    axs.fill_between(range(len(mn)), mn+std, mn-std, alpha=0.5, color="crimson")
    axs.hlines(7.5, 0, len(mn), color="black", ls="dashed")
    axs.set_xlabel("Episode")
    axs.set_ylabel("Sum of Rewards")
    axs.set_title("Performance of Policy")
    plt.savefig("learning_curve.png")
    plt.show()


def main():

    env = StoppedCar()
    order = 18  # order for fourier basis TODO tune this parameter
    basis = FourierBasis(env.obs_ranges, order)  # NOTE:  the fourier basis uses coupled terms not just the independent terms as in the previous homework


    alpha = 1.0 / basis.getNumFeatures() # TODO optimize the policy steps size
    beta  = 0.8 / basis.getNumFeatures()  # TODO optimize the critic step size
    gamma = 1.0  # TODO optimize the discount factor

    all_sums_of_rewards = []
    number_of_lifetimes = 100  # number of lifetimes (trials) to run the algorithm. Set the number of lifetimes to 1 to debug and tune your step size faster, but you must run 100 lifetimes to report your performance
    number_of_episodes = 400  # number of episodes per lifetime. This can be set small for debuging and tuning the step size, but must be 400 to report your results

    start = time.time()  # start a timer to see how long it takes to run the algorithm. On my old laptop I can run 100 lifetimes of 400 episodes in about 620 seconds. However, different hyperparmeters will have different run times.
    # run the algorithm for each lifetime
    for i in range(number_of_lifetimes):
        print(i)
        policy = LinearSoftmax(basis, env.num_actions)  # initialize the policy for each lifetime
        vf = LinearValueFunction(basis)  # initialize the value function for each lifetime
        agent = ActorCritic(policy, vf, alpha, beta, gamma)  # create the Actor-Critic agent
        sums_of_rewards = train_actorcritic(env, agent, num_episodes=number_of_episodes)  # train the agent
        all_sums_of_rewards.append(sums_of_rewards)  # log the performances from this lifetime

    end = time.time()  # take note of the stopping time
    print("Time to run all lifetimes: {0:.1f}(s)".format(end-start))

    learning_curve(all_sums_of_rewards)  # plot the performance of the agent for all lifetimes.


if __name__ == "__main__":
    main()