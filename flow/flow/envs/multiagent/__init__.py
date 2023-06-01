"""Empty init file to ensure documentation for multi-agent envs is created."""

from flow.envs.multiagent.base import MultiEnv
from flow.envs.multiagent.ring.wave_attenuation import \
    MultiWaveAttenuationPOEnv
from flow.envs.multiagent.ring.wave_attenuation import \
    MultiAgentWaveAttenuationPOEnv
from flow.envs.multiagent.ring.accel import AdversarialAccelEnv
from flow.envs.multiagent.ring.accel import MultiAgentAccelPOEnv
from flow.envs.multiagent.traffic_light_grid import MultiTrafficLightGridPOEnv
from flow.envs.multiagent.highway import MultiAgentHighwayPOEnv
from flow.envs.multiagent.merge import MultiAgentMergePOEnv
from flow.envs.multiagent.i210 import I210MultiEnv
from flow.envs.multiagent.ring.MIDLC_Ring import TD3MILCAccelEnv, TD3MILCAccelPOEnv
from flow.envs.multiagent.ring.MADLC_Des_Ring import TD3MADLC_DESAccelEnv, TD3MADLC_DESAccelPOEnv
from flow.envs.multiagent.ring.CeMIDLC_Ring import *
from flow.envs.multiagent.ring.MADLCwithNGSIM import *
from flow.envs.multiagent.ring.MADDPG_3Ring import *
from flow.envs.multiagent.ring.QMIX_3Ring import *
from flow.envs.multiagent.ring.MA_Foward_5LC_ring import *
from flow.envs.multiagent.ring.MA_Foward_5LC_ring_idm import *
from flow.envs.multiagent.ring.MA_Foward_5LC_ring_5vis import *

__all__ = [
    'MultiEnv',
    'AdversarialAccelEnv',
    'MultiWaveAttenuationPOEnv',
    'MultiTrafficLightGridPOEnv',
    'MultiAgentHighwayPOEnv',
    'MultiAgentAccelPOEnv',
    'MultiAgentWaveAttenuationPOEnv',
    'MultiAgentMergePOEnv',
    'I210MultiEnv',
    'TD3MILCAccelEnv',
    'TD3MILCAccelPOEnv',
    'TD3MADLC_DESAccelEnv',
    'TD3MADLC_DESAccelPOEnv',
    'TD3CeMILCAccelEnv',
    'TD3CeMILCAccelPOEnv',
    'TD3NGSIMAccelEnv',
    'TD3NGSIMAccelPOEnv',
    'MADDPGForwardLaneChange3RingEnv',
    'MADDPGForwardLaneChange3RingPOEnv',
    'QMIXForwardLaneChange3RingEnv',
    'QMIXForwardLaneChange3RingPOEnv',
    'MADRingLCPOEnv',
    'MADRingLCPOIDMEnv',
    'MADRingLCPO_5vis_Env',
    'MARing_5vis_Env'
]
