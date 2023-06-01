"""Contains the ring road scenario class."""

from time import time
from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from numpy import pi, sin, cos, linspace
import numpy as np
import random

SCALEING = 1


ADDITIONAL_NET_PARAMS = {
    # length of the ring road
    "length": 300,
    # number of lanes
    "num_lanes": 1,
    # speed limit for all edges
    "speed_limit": 25,
    # resolution of the curves on the ring
    "resolution": 800,
}


class myNetwork(Network):

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Initialize a ring scenario."""
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        length = net_params.additional_params["length"]
        bottleneck_length = length//8
        r = length / (2 * pi)

        nodes = [
        # 도로 좁아지는 구간
        {
            "id": "bottleneck_1",
            "x": 0,
            "y": -r,
            # "type" : "zipper",
            # "radius": 5,
        },
        # ring bottom -> right
        {
            "id": "bottom",
            "x": bottleneck_length,
            "y": -r,
            "type" : "zipper",
            "radius": 10,
        },
        # ring right -> ring top
        {
            "id": "right",
            "x": bottleneck_length + r,
            "y": 0,
        },
        # 도로 넓어지는 구간
        {
            "id": "bottleneck_2",
            "x": bottleneck_length,
            "y": r,
            # "type" : "zipper",
            # "radius": 5,
        },
        # ring top -> ring left
        {
            "id": "top",
            "x": 0,
            "y": r,
            "type" : "zipper",
            "radius": 10,
        },
        {
            "id": "left",
            "x": -r,
            "y": 0
        }]

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        length = net_params.additional_params["length"]
        bottleneck_length = length//8
        resolution = net_params.additional_params["resolution"]
        r = length / (2 * pi)
        edgelen = length / 4.
        lanes = net_params.additional_params["num_lanes"]

        edges = [
        ## bottleneck1 -> ring_right
        {
            "id":
                "bottleneck_1",
            "type":
                "edgeType",
            "from":
                "bottleneck_1",
            "to":
                "bottom",
            "length":
                bottleneck_length,
            "numLanes": lanes//2,
            "spreadType": "center",
        },
        ## ring_bottome -> bottleneck1
        {
            "id":
                "bottom",
            "type":
                "edgeType",
            "from":
                "bottom",
            "to":
                "right",
            "length":
                edgelen,
            "shape":
                [
                    (bottleneck_length + r * cos(t), r * sin(t))
                    for t in linspace(-pi / 2, 0, resolution)
                ],
            "numLanes": lanes,
            "spreadType": "center",
        },
        ## ring_right -> ring_top
        {
            "id":
                "right",
            "type":
                "edgeType",
            "from":
                "right",
            "to":
                "bottleneck_2",
            "length":
                edgelen,
            "shape":
                [
                    (bottleneck_length+r * cos(t), r * sin(t))
                    for t in linspace(0, pi / 2, resolution)
                ],
            "numLanes": lanes,
            "spreadType": "center",
        },
        {
            "id":
                "bottleneck_2",
            "type":
                "edgeType",
            "from":
                "bottleneck_2",
            "to":
                "top",
            "length":
                bottleneck_length,
            "numLanes": lanes,
            "spreadType": "center",
        },
        ## ring_top -> bottleneck2
        {
            "id":
                "top",
            "type":
                "edgeType",
            "from":
                "top",
            "to":
                "left",
            "length":
                edgelen,
            "shape":
                [
                    (r * cos(t), r * sin(t))
                    for t in linspace(pi / 2, pi, resolution)
                ],
            "numLanes": lanes//2,
            "spreadType": "center",
        },

        {
            "id":
                "left",
            "type":
                "edgeType",
            "from":
                "left",
            "to":
                "bottleneck_1",
            "length":
                edgelen,
            "shape":
                [
                    (r * cos(t), r * sin(t))
                    for t in linspace(pi, 3 * pi / 2, resolution)
                ],
            "numLanes": lanes//2,
            "spreadType": "center",
        }]

        return edges

    def specify_types(self, net_params):
        """See parent class."""
        speed_limit = net_params.additional_params["speed_limit"]
        # print(speed_limit)
        types = [{
            "id": "edgeType",
            "speed": speed_limit,
            #"spreadType": "center",
        }]

        return types

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {
            "bottleneck_1" :["bottleneck_1", "bottom", "right", "bottleneck_2", "top", "left"],
            "bottom": ["bottom", "right", "bottleneck_2", "top", "left", "bottleneck_1"],
            "right": ["right", "bottleneck_2", "top", "left", "bottleneck_1", "bottom"],
            "bottleneck_2" :["bottleneck_2", "top", "left", "bottleneck_1", "bottom", "right"],
            "top": ["top", "left", "bottleneck_1", "bottom", "right", "bottleneck_2"],
            "left": ["left", "bottleneck_1", "bottom", "right", "bottleneck_2", "top"],
        }

        return rts

    # edge간의 connection을 지정 하는 함수
    # 해당 함수 지정 안할 시 도로간 연결성이 없어서 바깥쪽 차량들은 좁아지는 도로에 진입하지 못하고 정차하게 됨.
    def specify_connections(self, net_params):
        """See parent class."""
        scaling = net_params.additional_params.get("scaling", 1)
        conn_dic = {}
        conn = []

    # 4->2 lane으로 좁아질 때 connection 관계 설정
        for i in range(4 * scaling):
            conn += [{
            "from": "bottleneck_2",
            "to": "top",
            "fromLane": i,
            "toLane": int(np.floor(i / 2)),
            }]
        conn_dic["bottleneck_2"] = conn

    # 2 -> 4 lane으로 넓어질 때 connection 관계 설정 -> 중앙 차선으로 유지해서 가는 것을 기본으로 설정
        conn_dic["bottleneck_1"] = [
                        {'from': 'bottleneck_1', 'to': 'bottom', 'fromLane': 0, 'toLane': 1},
                        {'from': 'bottleneck_1', 'to': 'bottom', 'fromLane': 1, 'toLane': 2},
                        ]
        return conn_dic

    def specify_edge_starts(self):
        """See parent class."""
        ring_length = self.net_params.additional_params["length"]
        bottleneck_length = ring_length//8
        junction_length = 0.1  # length of inter-edge junctions

        a = 10*2

        edgestarts = [("bottleneck_1", 0),
                    ('bottom', bottleneck_length+a),
                    ("right", 0.25 * ring_length + junction_length + bottleneck_length+ a ),
                    ("bottleneck_2", 0.5 * ring_length + 2 * junction_length+ bottleneck_length+ a ),
                    ("top", 0.5 * ring_length + 2 * junction_length+ 2*bottleneck_length+ 2*a),
                    ("left", 0.75 * ring_length + 3 * junction_length+ 2*bottleneck_length+ 2*a )]

        return edgestarts

    def specify_internal_edge_starts(self):
        """See parent class."""
        ring_length = self.net_params.additional_params["length"]
        bottleneck_length = ring_length//8
        junction_length = 0.1  # length of inter-edge junctions

        a = 10*2

        edgestarts = [
                    (":bottleneck_1_0", ring_length + 3 * junction_length+ 2*bottleneck_length+ 2*a),
                    (':bottom_0', bottleneck_length),
                    (":right_0", 0.25 * ring_length + bottleneck_length+ a ),
                    (":bottleneck_2_0", 0.5 * ring_length + junction_length+ bottleneck_length+ a ),
                    (":top_0", 0.5 * ring_length + junction_length+ 2*bottleneck_length+ a ),
                    (":left_0", 0.75 * ring_length + 2 * junction_length+ 2*bottleneck_length+ 2*a )]


        return edgestarts


    @staticmethod
    def gen_custom_start_pos(cls, net_params, initial_config, num_vehicles):
        VEHICLE_LENGTH = 5
        ring_length = net_params.additional_params['length']
        bottle_length = ring_length // 8
        edge_length = ring_length // 4

        num_lane = net_params.additional_params['num_lanes']
        min_gap = initial_config.min_gap
        my_min_gap = min_gap + 10
        startpos, startlane = [], []
        # print(VEHICLE_LENGTH + my_min_gap)
        # [('bottom', 13.5)] [3]
        num_RL = 1
        edges = ['bottleneck_1','bottom', 'right', 'bottleneck_2', 'top', 'left']
        lane_list = np.array([2, 4, 4, 4, 2, 2])

        idx = random.randint(0,len(edges)-1)
        pos = round(random.uniform(5,35), 1)
        lane = random.randint(0, lane_list[idx]-1)
        startpos = [(edges[idx], pos)]
        startlane = [lane]

        # print(startpos, startlane)
        # import time
        # time.sleep(5)

        return startpos, startlane


    @staticmethod
    def gen_custom_start_pos2(cls, net_params, initial_config, num_vehicles):
        VEHICLE_LENGTH = 5
        ring_length = net_params.additional_params['length']
        bottle_length = ring_length // 8
        edge_length = ring_length // 4

        num_lane = net_params.additional_params['num_lanes']
        min_gap = initial_config.min_gap
        my_min_gap = min_gap + 15
        startpos, startlane = [], []
        # print(VEHICLE_LENGTH + my_min_gap)

        num_RL = 1
        edges = ['bottom', 'right', 'bottleneck_2']
        lane_list = np.array([4, 4, 4, 2, 2, 2])


        load_storage, test_pos = 0, 0
        while test_pos < ring_length:
            load_storage += lane_list[int(test_pos//edge_length)]
            test_pos += (VEHICLE_LENGTH + my_min_gap)

        if load_storage < num_vehicles:
            raise ValueError(f'''num of vehicles are too many.
            max_vehicles : {load_storage}, defined_vehicles : {num_vehicles}''')

        pos = 0
        pos_list = []
        RL_startpos, RL_startlane = [] , []
        not_deployed_vehicle = num_vehicles

        while not_deployed_vehicle != 0:

            #print(not_deployed_vehicle)
            now_edge_num = int(pos // edge_length)
            now_lane_num = lane_list[now_edge_num]
            now_veh_order = np.arange(now_lane_num)
            np.random.shuffle(now_veh_order)

            if not_deployed_vehicle >= now_lane_num:
                for lane in now_veh_order:
                    now_pos = pos+round(random.uniform(5,15), 1)
                    pos_list.append(now_pos)
                    startpos.append((edges[int(now_pos // edge_length)], now_pos % edge_length))
                    startlane.append(lane)

            else:
                # print(now_veh_order)
                now_veh_order = now_veh_order[:not_deployed_vehicle]
                for lane in now_veh_order:
                    now_pos = pos+round(random.uniform(5,15), 1)
                    pos_list.append(now_pos)
                    startpos.append((edges[int(now_pos // edge_length)], now_pos % edge_length))
                    startlane.append(lane)

            pos+= (my_min_gap + VEHICLE_LENGTH)

            not_deployed_vehicle -= len(now_veh_order)

        # print(startpos)
        # print(startlane)
        startpos = np.array(startpos)
        startlane = np.array(startlane)

        starting_order = np.arange(startpos.shape[0])
        np.random.shuffle(starting_order)

        startpos = startpos[starting_order]
        startlane = startlane[starting_order]

        return startpos, startlane

    @staticmethod
    def gen_custom_start_pos3(cls, net_params, initial_config, num_vehicles):
        VEHICLE_LENGTH = 5
        ring_length = net_params.additional_params['length']
        bottle_length = ring_length // 8
        edge_length = ring_length // 4

        num_lane = net_params.additional_params['num_lanes']
        min_gap = initial_config.min_gap
        my_min_gap = min_gap + 15
        startpos, startlane = [], []
        # print(VEHICLE_LENGTH + my_min_gap)

        num_RL = 1
        edges = ['bottom', 'right', 'bottleneck_2']
        lane_list = np.array([4, 4, 4, 2, 2, 2])


        load_storage, test_pos = 0, 0
        while test_pos < ring_length:
            load_storage += lane_list[int(test_pos//edge_length)]
            test_pos += (VEHICLE_LENGTH + my_min_gap)

        if load_storage < num_vehicles:
            raise ValueError(f'''num of vehicles are too many.
            max_vehicles : {load_storage}, defined_vehicles : {num_vehicles}''')

        pos = 0
        pos_list = []
        RL_startpos, RL_startlane = [] , []
        not_deployed_vehicle = num_vehicles

        while not_deployed_vehicle != 0:

            #print(not_deployed_vehicle)
            now_edge_num = int(pos // edge_length)
            now_lane_num = lane_list[now_edge_num]
            now_veh_order = np.arange(now_lane_num)
            np.random.shuffle(now_veh_order)

            if not_deployed_vehicle >= now_lane_num:
                for lane in now_veh_order:
                    now_pos = pos+round(random.uniform(5,15), 1)
                    pos_list.append(now_pos)
                    startpos.append((edges[int(now_pos // edge_length)], now_pos % edge_length))
                    startlane.append(lane)

            else:
                # print(now_veh_order)
                now_veh_order = now_veh_order[:not_deployed_vehicle]
                for lane in now_veh_order:
                    now_pos = pos+round(random.uniform(5,15), 1)
                    pos_list.append(now_pos)
                    startpos.append((edges[int(now_pos // edge_length)], now_pos % edge_length))
                    startlane.append(lane)

            pos+= (my_min_gap + VEHICLE_LENGTH)

            not_deployed_vehicle -= len(now_veh_order)

        # print(startpos)
        # print(startlane)

        # 현재 rl pos 저장
        tmp_pos = startpos[-1]
        tmp_lane = startlane[-1]

        rl_idx = random.randint(0, num_vehicles-1)

        # rl_pos 변경
        startpos[-1] = startpos[rl_idx]
        startlane[-1] = startlane[rl_idx]

        # 일반차량 pos 변경
        startpos[rl_idx] = tmp_pos
        startlane[rl_idx] = tmp_lane

        return startpos,