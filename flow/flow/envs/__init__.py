"""Contains all callable environments in Flow."""
from flow.envs.base import Env
from flow.envs.bay_bridge import BayBridgeEnv
from flow.envs.bottleneck import BottleneckAccelEnv, BottleneckEnv, \
    BottleneckDesiredVelocityEnv
from flow.envs.traffic_light_grid import TrafficLightGridEnv, \
    TrafficLightGridPOEnv, TrafficLightGridTestEnv, TrafficLightGridBenchmarkEnv
from flow.envs.ring.lane_change_accel import LaneChangeAccelEnv, \
    LaneChangeAccelPOEnv
from flow.envs.ring.accel import AccelEnv
from flow.envs.ring.wave_attenuation import WaveAttenuationEnv, \
    WaveAttenuationPOEnv
from flow.envs.merge import MergePOEnv
from flow.envs.test import TestEnv
from flow.envs.ring.LC_Inverse_ring import *
from flow.envs.ring.DLC_Inverse_ring import *
from flow.envs.ring.DLC_Inverse_ring_v2 import *
from flow.envs.ring.DLC_Foward_3ring import *
from flow.envs.ring.D3PG_Forward_3ring import *
from flow.envs.ring.Inverse_3Lane_ring import *
from flow.envs.ring.NGSIM_IRL import *
from flow.envs.ring.NGSIM_OFFRL import *
# deprecated classes whose names have changed
from flow.envs.bottleneck_env import BottleNeckAccelEnv
from flow.envs.bottleneck_env import DesiredVelocityEnv
from flow.envs.green_wave_env import PO_TrafficLightGridEnv
from flow.envs.green_wave_env import GreenWaveTestEnv
from flow.envs.multiagent.ring.MIDLC_Ring import *
from flow.envs.ring.DetInvLCStoChar import *
from flow.envs.ring.DetInvLCStoChar_Poisson import *
from flow.envs.ring.DetInvLCStoChar_Gamma import *
from flow.envs.multiagent.ring.MADDPG_3Ring import *
from flow.envs.multiagent.ring.UnifiedRing import *


__all__ = [
    'Env',
    'AccelEnv',
    'LCIAccelEnv',
    'LCIAccelPOEnv',
    'LaneChangeAccelEnv',
    'LaneChangeAccelPOEnv',
    'TrafficLightGridTestEnv',
    'MergePOEnv',
    'BottleneckEnv',
    'BottleneckAccelEnv',
    'WaveAttenuationEnv',
    'WaveAttenuationPOEnv',
    'TrafficLightGridEnv',
    'TrafficLightGridPOEnv',
    'TrafficLightGridBenchmarkEnv',
    'BottleneckDesiredVelocityEnv',
    'TestEnv',
    'BayBridgeEnv',
    # deprecated classes
    'BottleNeckAccelEnv',
    'DesiredVelocityEnv',
    'PO_TrafficLightGridEnv',
    'GreenWaveTestEnv',
    'DLCIAccelEnv',
    'DLCIAccelPOEnv',
    'DLCIAccelEnv2',
    'DLCIAccelPOEnv2',
    'DLCFAccelEnv',
    'DLCFAccelPOEnv',
    'D3PGAccelEnv',
    'D3PGAccelPOEnv',
    'TD3LCIAccelEnv',
    'TD3LCIAccelPOEnv',
    'TD3MILCAccelPOEnv',
    'TD3MILCAccelEnv',
    'TD3LCIStoCharAccelEnv',
    'TD3LCIStoCharAccelPOEnv',
    'NGSIMAccelEnv',
    'NGSIMAccelPOEnv',
    'NGSIMOnlyVelAccelPOEnv',
    'NGSIMOffRLAccelEnv',
    'NGSIMOffRLAccelPOEnv',
    'MADDPGForwardLaneChange3RingEnv',
    'MADDPGForwardLaneChange3RingPOEnv',
    'TD3LCIStoPoissonAccelEnv',
    'TD3LCIStoPoissonAccelPOEnv',
    'TD3LCIStoGammaAccelEnv',
    'TD3LCIStoGammaAccelPOEnv',
    'UniRingAccelEnv',
    'UniRingAccelPOEnv'
]
