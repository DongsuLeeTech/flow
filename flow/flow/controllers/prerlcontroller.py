"""Contains the RLController class."""

from flow.controllers.base_controller import BaseController

class PreRLController(BaseController):

    def __init__(self, veh_id, car_following_params):
        """Instantiate an RL Controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params)

    def PRE_ACTOR(self, ):
        weights

        for key, val in weights.items():
            weights_val = val

        wei_vlist = []
        for i in weights_val.items():
            wei_vlist.append(i)

        policy_wei = []
        policy_bias = []
        for i in range(6):
            if i % 2 == 0:
                policy_wei.append(wei_vlist[i])
            else:
                policy_bias.append(wei_vlist[i])

        w1 = torch.from_numpy(policy_wei[0][1]).float()
        w2 = torch.from_numpy(policy_wei[1][1]).float()
        w3 = torch.from_numpy(policy_wei[2][1]).float()
        b1 = torch.from_numpy(policy_bias[0][1]).float()
        b2 = torch.from_numpy(policy_bias[1][1]).float()
        b3 = torch.from_numpy(policy_bias[2][1]).float()
        # policy_bias = torch.Tensor([policy_bias[1][1], policy_bias[2][1], policy_bias[0][1]])
        return w1, w2, w3, b1, b2, b3

    def get_accel(self, env):
        """Pass, as this is never called; required to override abstractmethod."""

        pass