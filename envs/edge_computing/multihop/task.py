K = 1024
Byte = 8


class Task:
    def __init__(self, args, random_state):
        # Task Tuple
        self.args = args
        self.task_random = random_state
        self.id = self.task_random.random()
        self.task_size = self.task_random.uniform(args.task_size_min, args.task_size_max) * K * Byte  # Affects transmission time
        self.task_cpu_cycle = self.task_random.uniform(args.task_complexity_min, args.task_complexity_max) * self.task_size  # Affects computation time

        self.task_deadline = args.deadline  # Task deadline

        self.transmission_waiting_time = 0  # Transmission waiting time
        self.transmission_time = 0  # Required transmission time for the task
        self.current_transmission_time = 0  # Current transmission time spent

        self.execute_waiting_time = 0  # Execution waiting time
        self.execute_time = 0  # Required execution time for the task
        self.current_execute_time = 0  # Current execution time

        self.buffer_waiting_time = 0  # Task waiting time

        self.execution_failure_rate = 0
        self.execution_reliability = 0
        self.transmission_failure_rate = 0
        self.transmission_reliability = 0

        self.hop = 0  # Current node hop count is 0, neighbor nodes are 1, and so on

        self.is_success = False

    def __eq__(self, other):
        """Used to determine if two objects are equal after deep copy"""
        if not isinstance(other, Task):
            return False
        # Here we determine that two tasks are equal only when their ids are the same
        return self.id == other.id
