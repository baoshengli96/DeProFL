''' 
This demo is the main procedure of "Prototype-Based Decentralized Federated Learning for the Heterogeneous Time-Varying IoT Systems"
you access the https://ieeexplore.ieee.org/document/10246848

Ps: You can access the other auxiliary programs at https://github.com/TsingZ0/PFLlib
'''
from clientdepro import clientDePro
from serverbase import Server
from utils.data_utils import read_client_data
import networkx as nx
import copy

class DePro(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(args, clientDePro)
        self.average_protos = {}
        self.Budget = []
        self.num_clients = args.num_clients
        self.num_classes = args.num_classes
    def generate_graph(self):
        # time-varying topology
        while True:
            graph = nx.random_geometric_graph(self.num_clients,0.5)
            if nx.is_k_edge_connected(graph, 1)==True:
                break
        return graph
    
    def train(self):
        for i in range(self.global_rounds+1):
            self.selected_clients = self.select_clients()
            self.graph = self.generate_graph()
            self.local_protos = {}
            for client in self.selected_clients:
                client.train()
            self.receive_protos_from_neigh()
            self.send_protos()

    def send_protos(self):
        # send the protos to its neighbors
        assert (len(self.selected_clients) > 0)
        for client in self.selected_clients:
            client.set_protos(self.average_protos[client.id])

    def receive_protos_from_neigh(self):
        # receive the protos from its neighbors
        assert (len(self.selected_clients) > 0)
        for client in self.selected_clients:
            self.local_protos[client.id] = client.protos
        for client in self.selected_clients:    
            self.neigh_protos = {}
            self.agg_proto = {}
            self.neigh_id_list = list(self.graph.adj[client.id])
            for i in range(len(self.neigh_id_list)):
                self.neigh_protos[i] = self.local_protos[self.neigh_id_list[i]]
            self.agg_proto = copy.deepcopy(self.neigh_protos) # get neighbor protos
            self.agg_proto[i+1] = self.local_protos[client.id] # include its own protos
            self.average_protos[client.id] = proto_aggregation(self.agg_proto)


# FROM https://github.com/yuetan031/fedproto/blob/main/lib/utils.py
def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = [proto / len(proto_list)]
        else:
            agg_protos_label[label] = [proto_list[0].data]

    return agg_protos_label