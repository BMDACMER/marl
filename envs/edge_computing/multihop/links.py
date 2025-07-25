class Link:
    def __init__(self, args, random_state):
        # Link Tuple
        self.args = args
        self.edge_node_random = random_state
        # Bandwidth of each edge in the network topology == weight of adjacency list
        self.transmission_rates = self.edge_node_random.uniform(args.transmission_rate_min, args.transmission_rate_max) * 1000000 * 8
        # Link reliability can be derived from failure rate
        self.transmission_failure_rates = self.edge_node_random.uniform(args.transmission_failure_rate_min, args.transmission_failure_rate_max)
