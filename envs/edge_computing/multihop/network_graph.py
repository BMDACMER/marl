import argparse
import torch
from envs.edge_computing.multihop.edge_computing_config import add_edge_computing_env_args
from envs.edge_computing.multihop.links import Link
from collections import defaultdict
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import re

# Set random seed to ensure reproducible results
random.seed(2023)


class Graph:
    """
    Represents an undirected graph structure for edge computing networks
    
    Attributes:
        num_nodes: Number of nodes in the graph
        adj_list: Stores the adjacency list of the graph, format: {node: [(neighbor_node, bandwidth, failure_rate), ...]}
        args: Configuration parameters
    """
    def __init__(self, num_nodes, args):
        """
        Initialize graph structure
        
        Parameters:
            num_nodes: Number of nodes in the graph
            args: Configuration parameters
        """
        self.num_nodes = num_nodes
        self.adj_list = defaultdict(list)
        self.args = args

    def add_edge_bandwidth(self, u, v, bandwidth, failure_rate):
        """
        Add undirected edge with bandwidth and failure rate attributes
        
        Parameters:
            u, v: Two endpoints of the edge
            bandwidth: Edge bandwidth
            failure_rate: Edge transmission failure rate
        """
        # Check if edge already exists to avoid duplicate addition
        if not any(adj_node[0] == u for adj_node in self.adj_list[v]):
            self.adj_list[v].append((u, bandwidth, failure_rate))
        if not any(adj_node[0] == v for adj_node in self.adj_list[u]):
            self.adj_list[u].append((v, bandwidth, failure_rate))

    def get_adj_tuple(self, node_id):
        """
        Return triple information of adjacent nodes (node_id, bandwidth, transmission_failure_rate)
        
        Parameters:
            node_id: Target node ID
        Returns:
            List containing adjacent node information, each element is (node_id, bandwidth, failure_rate)
        """
        return self.adj_list[node_id]

    def get_adj_node(self, node_id):
        """
        Get all neighbor node IDs for the specified node
        
        Parameters:
            node_id: Target node ID
        Returns:
            List of neighbor node IDs
        """
        return [adj_node[0] for adj_node in self.adj_list[node_id]]

    def get_adj_edge(self, node_id):
        """
        Return all edges adjacent to the specified node
        
        Parameters:
            node_id: Target node ID
        Returns:
            List of edges, each edge represented as (node_id, neighbor_id)
        """
        if node_id not in self.adj_list:
            return []  # If node doesn't exist, return empty list

        return [(node_id, neighbor[0]) for neighbor in self.adj_list[node_id]]

    def get_all_adj_node(self, node_id):
        """
        Get all adjacent nodes and related information within maximum hop range for the specified node
        
        Sorted by 'shortest hop' and 'total bandwidth' in descending order
        
        Parameters:
            node_id: Target node ID
        Returns:
            (all_adj_nodes, all_adj_transmission, all_adj_failure, add_adj_info)
            - all_adj_nodes: List of all adjacent nodes
            - all_adj_transmission: Bandwidth of each link
            - all_adj_failure: Failure rate of each link
            - add_adj_info: Detailed information containing node, bandwidth, failure rate and hop count
        """
        max_hops = self.args.max_hops
        all_adj_nodes = []  # Store all adjacent nodes
        all_adj_transmission = []  # Store bandwidth on each link
        all_adj_failure = []  # Store failure rate on each link
        add_adj_info = []  # Store information of adjacent nodes, bandwidth, failure rate and hop count
        visited = set()  # Used to track visited nodes, avoid cycles

        def dfs(node, hop, path_transmission, path_failure):
            """
            Depth-first search to get all adjacent nodes within specified hop count
            """
            if hop > max_hops or node in visited:
                return
                
            visited.add(node)

            for neighbor, weight, failure in self.adj_list[node]:
                if neighbor != node_id and neighbor not in visited:  # Exclude current node itself and visited nodes
                    all_adj_nodes.append(neighbor)
                    new_transmission = path_transmission + [weight]
                    new_failure = path_failure + [failure]
                    
                    all_adj_transmission.extend(new_transmission)
                    all_adj_failure.extend(new_failure)
                    add_adj_info.append((neighbor, new_transmission, new_failure, hop))
                    
                    dfs(neighbor, hop + 1, new_transmission, new_failure)
            
            visited.remove(node)  # Remove node during backtracking, allow re-visiting through different paths

        # Start depth-first search
        dfs(node_id, 1, [], [])

        # Sort by hop count in ascending order and total bandwidth from current node to other nodes in descending order
        add_adj_info.sort(key=lambda x: (x[3], -sum(x[1])))

        return all_adj_nodes, all_adj_transmission, all_adj_failure, add_adj_info

    def get_all_edges(self):
        """
        Get all edges in the graph, return in PyTorch tensor format
        
        Returns:
            Edge list in torch.Tensor format [2, num_edges]
        """
        rows, cols = [], []
        for u, neighbors in self.adj_list.items():
            for v_info in neighbors:
                rows.append(u)
                cols.append(v_info[0])  # v_info[0] is the neighbor node ID

        return torch.tensor([rows, cols], dtype=torch.long)

    def get_all_edges_2(self):
        """
        Get all edges in the graph, return in tuple list format
        
        Returns:
            List of edges, each edge represented as (u, v)
        """
        all_edges = []
        for u, neighbors in self.adj_list.items():
            for v_info in neighbors:
                all_edges.append((u, v_info[0]))
                
        return all_edges

    def get_all_edges_3(self):
        """
        Get all edges in the graph (deduplicated), return in tuple list format
        
        Returns:
            List of unique edges, each edge represented as (u, v), where u < v
        """
        seen_edges = set()
        all_edges = []

        for u, neighbors in self.adj_list.items():
            for v_info in neighbors:
                v = v_info[0]
                # Ensure edges are not duplicated, sort edge order before adding
                edge = tuple(sorted([u, v]))
                
                if edge not in seen_edges:
                    all_edges.append(edge)
                    seen_edges.add(edge)

        return all_edges

    def get_edge_attr(self):
        """
        Get attributes of all edges (bandwidth and failure rate)
        
        Returns:
            Edge attributes in torch.Tensor format [num_edges, 2]
        """
        edge_index = self.get_all_edges()
        edge_bandwidth = []
        edge_failure_rate = []
        
        for u, v in edge_index.t().tolist():
            # Find corresponding edge attributes
            for adj_node, bandwidth, failure_rate in self.adj_list[u]:
                if adj_node == v:
                    edge_bandwidth.append(bandwidth)
                    edge_failure_rate.append(failure_rate)
                    break
                    
        return torch.tensor([edge_bandwidth, edge_failure_rate], dtype=torch.float).t()

    def print_graph(self):
        """Print the adjacency list of the graph"""
        for node in range(self.num_nodes):
            print(f"Node {node} -> {self.adj_list[node]}")


def generate_graph(args):
    """
    Generate an undirected connected graph (synthetic network topology)
    
    Parameters:
        args: Configuration parameters, including node count and other information
    Returns:
        (graph, G): Custom Graph object and corresponding NetworkX graph object
    """
    n = args.edge_node_num
    assert n >= 2, "Number of nodes must be greater than or equal to 2"
    
    # Create graph instance
    graph = Graph(n, args)

    # First create a ring to ensure the graph is connected
    edges = [(i, (i + 1) % n) for i in range(n - 1)] + [(n - 1, 0)]
    
    # Add additional edges to make the graph richer
    node_set = set(range(n))
    while len(node_set) > 1:
        node = random.choice(list(node_set))
        other_node = random.choice(list(node_set - {node}))
        
        # Avoid adding duplicate edges
        if (node, other_node) not in edges and (other_node, node) not in edges:
            edges.append((node, other_node))
            
        # Remove two nodes from unprocessed set to make constructed topology as sparse as possible
        node_set.discard(node)
        node_set.discard(other_node)
    
    # Generate transmission rates and failure rates for edges
    random_state = np.random.RandomState(args.link_seed)
    transmission_rates = [Link(args, random_state).transmission_rates for _ in range(len(edges))]
    transmission_failure_rates = [Link(args, random_state).transmission_failure_rates for _ in range(len(edges))]
    
    # Add edges to the graph
    for i, (u, v) in enumerate(edges):
        graph.add_edge_bandwidth(u, v, transmission_rates[i], transmission_failure_rates[i])

    # Create NetworkX graph object for visualization
    G = nx.Graph()
    G.add_edges_from(edges)
    
    return graph, G


def generate_graph2(args):
    """
    Generate graph using real-world network topology
    
    Parameters:
        args: Configuration parameters
    Returns:
        graph: Custom Graph object
    """
    import os
    
    # Get the directory where the current file is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Dynamically get topology file path, support different network topologies
    topology_file = getattr(args, 'topology_file', 'network_topology/abilene.graphml')
    if not os.path.isabs(topology_file):
        # If it's a relative path, then relative to project root directory
        file_path = os.path.join(script_dir, '../../../', topology_file)
    else:
        file_path = topology_file
    
    # Read real network topology
    G = nx.read_graphml(file_path)
    
    # Get all nodes and edges
    nodes = G.nodes()
    # Note: No longer modify args.edge_node_num here, as it should be set before calling this function
    edges = [(int(u), int(v)) for u, v in G.edges()]
    
    # Generate transmission rates and failure rates for edges
    random_state = np.random.RandomState(args.link_seed)
    transmission_rates = [Link(args, random_state).transmission_rates for _ in range(len(edges))]
    transmission_failure_rates = [Link(args, random_state).transmission_failure_rates for _ in range(len(edges))]

    # Create graph instance and add edges
    graph = Graph(len(nodes), args)
    for i, (u, v) in enumerate(edges):
        graph.add_edge_bandwidth(u, v, transmission_rates[i], transmission_failure_rates[i])

    return graph


if __name__ == '__main__':
    # Test code
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser = add_edge_computing_env_args(parser)
    args = parser.parse_args()

    graph_real, G = generate_graph(args)
    # print(len(graph_real.get_adj_edge(1)))
    print("All unique edges:")
    print(len(graph_real.get_all_edges_3()))

    # Visualize the generated graph
    import networkx as nx

    G = nx.Graph()
    for u, v in graph_real.get_all_edges_3():
        G.add_edge(u, v)
    nx.draw(G, with_labels=True)
    plt.show()