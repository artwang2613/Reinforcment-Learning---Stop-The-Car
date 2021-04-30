import numpy as np
from fourierbasis import FourierBasis

from typing import Tuple

class LinearSoftmax(object):
    def __init__(self, basis:FourierBasis, n_actions:int):
        """
        This creates a linear softmax policy using the specified basis function
        Parameters
        ----------
        basis: FourierBasis
            The basis function to use with the policy
        n_actions : int
            The number of possible actions
        """

        self.basis = basis
        self.n_actions = n_actions
        self.n_inputs =  basis.getNumFeatures()

        self.basis = basis

        # These are the policy weights. They are a 2D numpy array.
        # To get the vector of weights for the a^th action you can do self.theta[a]
        self.theta = np.zeros((self.n_actions, self.n_inputs))

        self.num_params = int(self.theta.size)


    def get_action(self, state:np.ndarray)->int:
        """
        This function samples an action for the provided state features.

        Parameters
        ----------
        state : np.ndarray
            The state features (no basis function applied yet)

        Returns
        -------
        a : int
            The sampled action
        """
        x = self.basis.encode(state)  # Computes the basis function representation of the state features
        p = self.get_action_probabilities(x)  # computes the probabilities of each action
        a = int(np.random.choice(range(p.shape[0]), p=p, size=1))  # samples the action from p

        return a

    def get_params(self)->np.ndarray:
        """
        This function returns the weights of the policy. This is just a helper
        function.
        Returns
        -------
        theta : np.ndarray
            The weights of the policy
        """
        return self.theta

    def add_to_params(self, x: np.ndarray):
        """
        This function adds the input array to the weights. You can use this
        function to update the policy weights.

        Parameters
        ----------
        x : np.ndarray
            An array that is used to change the policy weights

        Returns
        -------
        None

        """
        assert self.theta.shape == x.shape, "x and theta have different shapes"
        self.theta += x

    def get_action_probabilities(self, x: np.ndarray)->np.ndarray:
        """
        Compute the probabilities for each action for the features x. This
        function should compute the outputs values for each action unit then
        perform a softmax over all outputs. The return value should be a 1D
        numpy array containing the probabilities for each action.

        Parameters
        ----------
        x : np.ndarray
            The state features after the basis function is applied.

        Returns
        -------
        p : np.ndarray
            The probabilities for each action
        """
        theta = self.theta
        out = np.dot(theta,x)
        p = np.exp(out)
        p /= p.sum()

        return p

    def gradient_logprob(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
        """
        This function computes the partial derivative of ln pi(s,a) with
        respect to the policy weights. The functions returns two quantities:
        a numpy array of the partial derivatives and the log probability of
        the action specified, e.g., ln pi(s,a). Note we also mean the natural
        log here and not log with base 2 or 10.

        Parameters
        ----------
        state : np.ndarray
            The state features (no basis function is applied)
        action : int
            The action that was chosen

        Returns
        -------
        dtheta : np.ndarray
            A 2D numpy array containing the partial derivatives
        logp : float
            The log probability of the action for the state features
        """
        x = self.basis.encode(state)  # transform the state features using the basis function
        u = -self.get_action_probabilities(x)
        logp = np.log(-u[action])
        u[action] += 1

        dtheta = np.outer(u, x)

        return dtheta, logp

    def gradient_prob(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        This function computes the partial derivative of the pi(s,a) with
        respect to the policy parameters. The function should return a 2D numpy
        array wit the same shape as the weights of the policy.

        Parameters
        ----------
        state : np.ndarray
            The state features (basis function has not been applied)
        action : int
            The action that was chosen

        Returns
        -------
        dtheta : np.ndarray
            2D numpy array containing the partial derivatives for all weights
        """
        dtheta, logp = self.gradient_logprob(state, action)
        dtheta = dtheta * np.exp(logp)
        return dtheta
