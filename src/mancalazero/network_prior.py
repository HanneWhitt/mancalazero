from mancalazero.Game import GameState
from mancalazero.utils import fill
import torch


class NetworkPrior:

    """
    Just a wrapper for torch policy/value model; puts in evaluation mode, adapts torch.tensor/numpy
    """

    def __init__(
        self,
        p_v_network
    ):
        self.network = p_v_network


    def __call__(self, state: GameState):

        """
        Evaluate the policy/value network for a single gamestate

        TODO: introduce queueing/batch evaluation
        """

        observation = state.get_observation(state.current_player)

        observation = torch.from_numpy(observation).float()
        observation = observation[None, :]

        legal_actions = state.legal_actions
        mask = fill(state.total_actions(), legal_actions)
        mask = torch.from_numpy(mask).float()
        mask = mask[None, :]

        self.network.eval()
        p, v = self.network(observation, mask)

        return p.detach().numpy()[0][legal_actions], v.detach().numpy()[0][0]