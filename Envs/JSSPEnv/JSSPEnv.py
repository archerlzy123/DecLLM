"""This script contains the JSSP environment for the JSSP problem. Here JSSPs are defined as: n jobs need to be processed on m machines. Each job consists of m operations, each operation needs to be processed on a specific machine. The processing time of each operation is given in the instance. The goal is to find a schedule that minimizes the makespan (the time when the last operation is completed)."""


import json
import random
import pickle
import numpy as np
from tqdm import tqdm
import os
import argparse
from multiprocessing import Pool
from ortools.sat.python import cp_model
import math


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def calculate_k_operations_with_lowest_processing_time(instance: np.array, k: int) -> list[list[tuple[int, int, float]]]:
    """
    Calculates the k operations with the lowest processing time for each machine.

    Parameters
    ----------
    instance : np.array
        A JSSP instance.
    k : int
        Number of operations to be calculated.

    Returns
    -------
    list[list[tuple[int, int, float]]]
        A list of k operations with the lowest processing time for each job. Each tuple contains operation index, machine index and processing time.
    """
    num_jobs, num_columns = instance.shape
    num_machines = num_columns // 2
    job_operations = {job: [] for job in range(num_jobs)}

    # Iterate through each job and operation
    for job_idx in range(num_jobs):
        for op_idx in range(num_machines):
            machine_idx, processing_time = instance[job_idx, op_idx * 2], instance[job_idx, op_idx * 2 + 1]
            job_operations[job_idx].append((op_idx, machine_idx, processing_time))

    # Sort operations for each job by processing time and select the k operations with the lowest processing time
    result = []
    for job in range(num_jobs):
        sorted_operations = sorted(job_operations[job], key=lambda x: x[2])  # Sort by processing time
        result.append(sorted_operations[:k])  # Take the k operations with the lowest processing time

    return result


def get_makespan(instance: np.array, schedule: list[list[int]]) -> float:
    """
    Decodes a machine-based job shop scheduling solution.
    
    Parameters:
        instance (np.array): 2D array with shape (n, 2*m) where each row represents a job.
                             For each job the operations are represented as consecutive pairs:
                             (machine, processing_time). There are m operations per job.
        schedule (list of lists): A list of m lists, each being a permutation of job indices.
                                        The k-th list gives the order in which machine k processes 
                                        its designated operations.
                                        
    Returns:
        The makespan (a numerical value) if a feasible schedule is constructed.
        
    Raises:
        ValueError: If the machine-based representation does not lead to a feasible schedule.
    """
    n, two_m = instance.shape
    m = two_m // 2  # each job has m operations
    
    # job_next[j] tracks the next operation index (0-indexed) to be scheduled for job j.
    job_next = [0] * n
    # job_finish[j] records the finish time of the last scheduled operation for job j.
    job_finish = [0] * n
    # machine_finish[k] records the finish time of the last operation scheduled on machine k.
    machine_finish = [0] * m
    # For each machine, machine_ptr[k] indicates how far we have advanced in its permutation.
    machine_ptr = [0] * m
    
    scheduled_ops = 0
    total_ops = n * m
    
    # Iterative scheduling: while there are operations not yet scheduled.
    while scheduled_ops < total_ops:
        available = []  # list to collect available operations as (start_time, machine, job)
        
        # For each machine, if there is still an operation assigned in its permutation ...
        for machine in range(m):
            if machine_ptr[machine] < n:
                job = schedule[machine][machine_ptr[machine]]
                op_idx = job_next[job]
                # Check that the next unscheduled operation for this job is indeed designated for this machine.
                if op_idx < m and instance[job, 2 * op_idx] == machine:
                    # The operation can start only after the job's previous operation and the machine are free.
                    start_time = max(job_finish[job], machine_finish[machine])
                    available.append((start_time, machine, job))
        
        # If no machine has a ready operation, then the representation is infeasible.
        if not available:
            raise ValueError("Infeasible schedule")
        
        # Choose the operation with the earliest possible start time.
        available.sort(key=lambda x: x[0])
        start_time, machine, job = available[0]
        op_idx = job_next[job]
        proc_time = instance[job, 2 * op_idx + 1]
        finish_time = start_time + proc_time
        
        # Update the finish times and pointers.
        job_finish[job] = finish_time
        machine_finish[machine] = finish_time
        job_next[job] += 1
        machine_ptr[machine] += 1
        scheduled_ops += 1

    # All operations scheduled successfully.
    makespan = max(job_finish)
    return makespan


class MultipleSolutionCollector(cp_model.CpSolverSolutionCallback):
    """
    收集 OR-Tools 搜索过程中的多个解。
    
    用于生成不同质量等级的轨迹数据集。
    """
    
    def __init__(self, tasks, instance, num_jobs, num_ops, max_solutions=10):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.tasks = tasks
        self.instance = instance
        self.num_jobs = num_jobs
        self.num_ops = num_ops
        self.max_solutions = max_solutions
        self.solutions = []  # [(makespan, schedule), ...]
    
    def on_solution_callback(self):
        """每找到一个解就调用此函数"""
        if len(self.solutions) >= self.max_solutions:
            return
        
        # 获取当前解的 makespan
        makespan = self.ObjectiveValue()
        
        # 构建调度表
        schedule = []
        unique_machines = set()
        for j in range(self.num_jobs):
            for op in range(self.num_ops):
                machine = int(self.instance[j, 2 * op])
                unique_machines.add(machine)
        
        sorted_machines = sorted(unique_machines)
        
        for m in sorted_machines:
            machine_ops = []
            for j in range(self.num_jobs):
                for op in range(self.num_ops):
                    if int(self.instance[j, 2 * op]) == m:
                        start_time = self.Value(self.tasks[(j, op)][0])
                        machine_ops.append((j, start_time))
            machine_ops.sort(key=lambda x: x[1])
            schedule.append([j for j, start in machine_ops])
        
        self.solutions.append((int(makespan), schedule))


def solve_jssp_multiple(instance: np.array, max_solutions=5, time_limit=60):
    """
    求解 JSSP 并收集多个解（从次优到最优）。
    
    这是生成不同质量轨迹的正确方法！
    
    Parameters
    ----------
    instance : np.array
        JSSP 实例
    max_solutions : int
        最多收集的解数量
    time_limit : float
        求解时间限制（秒）
        
    Returns
    -------
    list of (makespan, schedule) tuples, sorted by makespan (best first)
    """
    num_jobs = instance.shape[0]
    num_ops = instance.shape[1] // 2
    
    horizon = int(sum(instance[j, 2 * op + 1] for j in range(num_jobs) for op in range(num_ops)))
    
    model = cp_model.CpModel()
    
    tasks = {}
    machine_to_intervals = {}
    
    for j in range(num_jobs):
        for op in range(num_ops):
            machine = int(instance[j, 2 * op])
            processing_time = int(instance[j, 2 * op + 1])
            suffix = f'_j{j}_o{op}'
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, processing_time, end_var, 'interval' + suffix)
            tasks[(j, op)] = (start_var, end_var, processing_time, interval_var)
            
            if machine not in machine_to_intervals:
                machine_to_intervals[machine] = []
            machine_to_intervals[machine].append((j, op, interval_var))
    
    for j in range(num_jobs):
        for op in range(num_ops - 1):
            model.Add(tasks[(j, op)][1] <= tasks[(j, op + 1)][0])
    
    for machine, intervals in machine_to_intervals.items():
        model.AddNoOverlap([interval for (_, _, interval) in intervals])
    
    makespan = model.NewIntVar(0, horizon, 'makespan')
    last_ends = [tasks[(j, num_ops - 1)][1] for j in range(num_jobs)]
    model.AddMaxEquality(makespan, last_ends)
    model.Minimize(makespan)
    
    # 创建解收集器
    solution_collector = MultipleSolutionCollector(
        tasks, instance, num_jobs, num_ops, max_solutions
    )
    
    # 配置求解器以收集多个解
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = time_limit
    # 关键：设置为枚举所有解（在时间限制内）
    solver.parameters.enumerate_all_solutions = False  # 不枚举所有，只收集搜索过程中的解
    
    # 使用回调求解
    status = solver.SolveWithSolutionCallback(model, solution_collector)
    
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return []
    
    # 按 makespan 排序（最优的在前）
    solutions = sorted(solution_collector.solutions, key=lambda x: x[0])
    
    # 如果没有通过回调收集到足够的解，至少返回最终解
    if not solutions:
        # 构建最终解
        schedule = []
        unique_machines = set()
        for j in range(num_jobs):
            for op in range(num_ops):
                machine = int(instance[j, 2 * op])
                unique_machines.add(machine)
        
        sorted_machines = sorted(unique_machines)
        for m in sorted_machines:
            machine_ops = []
            if m in machine_to_intervals:
                for (j, op, interval) in machine_to_intervals[m]:
                    start_time = solver.Value(tasks[(j, op)][0])
                    machine_ops.append((j, start_time))
                machine_ops.sort(key=lambda x: x[1])
                schedule.append([j for j, start in machine_ops])
            else:
                schedule.append([])
        
        solutions = [(solver.Value(makespan), schedule)]
    
    return solutions


def solve_jssp(instance: np.array, time_limit=None):
    """
    Solve a JSSP instance using or-tools.
    
    Parameters
    ----------
    instance : np.array
        A JSSP instance.
    Returns
    -------
        A tuple (makespan, schedule) if a feasible solution is found, where:
          - makespan: the overall completion time (int)
          - schedule: a dictionary mapping each machine (int) to a list of tuples,
                      each tuple is (job, op, start_time, end_time) for that operation,
                      sorted by start time.
        If no feasible solution is found, returns "infeasible".
    """
    num_jobs = instance.shape[0]
    num_ops = instance.shape[1] // 2  # number of operations per job
    
    # Compute an upper bound on the makespan (sum of all processing times)
    horizon = int(sum(instance[j, 2 * op + 1] for j in range(num_jobs) for op in range(num_ops)))
    
    model = cp_model.CpModel()
    
    # Create task variables: tasks[(j, op)] = (start, end, processing_time, interval)
    tasks = {}
    machine_to_intervals = {}  # Map machine -> list of (job, op, interval)
    
    for j in range(num_jobs):
        for op in range(num_ops):
            machine = int(instance[j, 2 * op])
            processing_time = int(instance[j, 2 * op + 1])
            suffix = f'_j{j}_o{op}'
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, processing_time, end_var, 'interval' + suffix)
            tasks[(j, op)] = (start_var, end_var, processing_time, interval_var)
            
            if machine not in machine_to_intervals:
                machine_to_intervals[machine] = []
            machine_to_intervals[machine].append((j, op, interval_var))
    
    # Add precedence constraints for each job: operation op must finish before op+1 starts.
    for j in range(num_jobs):
        for op in range(num_ops - 1):
            model.Add(tasks[(j, op)][1] <= tasks[(j, op + 1)][0])
    
    # Add no-overlap constraints for operations assigned to the same machine.
    for machine, intervals in machine_to_intervals.items():
        model.AddNoOverlap([interval for (_, _, interval) in intervals])
    
    # Define makespan: maximum end time over all jobs.
    makespan = model.NewIntVar(0, horizon, 'makespan')
    last_ends = [tasks[(j, num_ops - 1)][1] for j in range(num_jobs)]
    model.AddMaxEquality(makespan, last_ends)
    model.Minimize(makespan)
    
    # Solve the model.
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    if time_limit is None:
        try:
            time_limit = float(os.environ.get("JSSP_ORTOOLS_TIME_LIMIT", 300))
        except Exception:
            time_limit = 300
    solver.parameters.max_time_in_seconds = float(time_limit)
    status = solver.Solve(model)
    
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return "infeasible"
    
    # Build the schedule: for each machine, list the operations (job, op, start, end)
    schedule = []
    
    # Get all unique machine IDs from the instance
    unique_machines = set()
    for j in range(num_jobs):
        for op in range(num_ops):
            machine = int(instance[j, 2 * op])
            unique_machines.add(machine)
    
    # Sort the machine IDs to ensure consistent ordering
    sorted_machines = sorted(unique_machines)
    
    # Create schedule for each machine
    for m in sorted_machines:
        machine_ops = []
        if m in machine_to_intervals:
            # Collect each operation's job and its start time on machine m.
            for (j, op, interval) in machine_to_intervals[m]:
                start_time = solver.Value(tasks[(j, op)][0])
                machine_ops.append((j, start_time))
            # Sort operations by their start time.
            machine_ops.sort(key=lambda x: x[1])
            # Extract only the job ids in processing order.
            schedule.append([j for j, start in machine_ops])
        else:
            schedule.append([])
    
    return solver.Value(makespan), schedule


def tag_prompt_and_transform_to_json(instance, k=2):
    """
    Combines tagging and JSON transformation for a JSSP instance.

    Parameters
    ----------
    instance : np.array
        A JSSP instance.
    k : int
        Number of tags to be used.
    
    Returns
    -------
    dict
        A dictionary with the tagged JSSP instance.
    """
    n_jobs, n_columns = instance.shape
    n_machines = n_columns // 2
    n_ops = n_columns // 2
    instruction = (
        f"Solve the Job Shop Scheduling Problem (JSSP) with {n_jobs} jobs and {n_machines} machines. "
        f"Each job consists of {n_ops} operations which need to be sequentially processed on specific machines. Each machine can process only one job at a time and each job can be processed by only one machine at a time. "
        "Identify the schedule that minimizes the maximum completion time (makespan). "
        "The input includes the information of operations for each job, including their specific machine and processing time, as well as the operations with lowest processing time and their respective machines and processing times. "
        "Provide the solution in the following format:\n\n"
        "1. Schedule: List the order that jobs are processed on each machine.\n"
        "2. Makespan: The makespan of the schedule.\n"        
    )

    instance_reformated = []
    for job_idx in range(n_jobs):
        job_info = instance[job_idx].tolist()
        job_info_reformated = []
        for op_idx in range(n_ops):
            job_info_reformated.append((job_info[2 * op_idx], job_info[2 * op_idx + 1]))
        instance_reformated.append(job_info_reformated)

    k_operations_with_lowest_processing_time = calculate_k_operations_with_lowest_processing_time(instance, k)
    # Remove instance_reformated as we'll generate structured descriptions directly
    # instance_reformated = []
    # for job_idx in range(n_jobs):
    #     job_info = instance[job_idx].tolist()
    #     job_info_reformated = []
    #     for op_idx in range(n_ops):
    #         job_info_reformated.append((job_info[2 * op_idx], job_info[2 * op_idx + 1]))
    #     instance_reformated.append(job_info_reformated)

    job_descriptions = []
    for job_idx in range(n_jobs):
        # Create structured description for each job's operations
        op_descriptions = []
        for op_idx in range(n_ops):
            machine = instance[job_idx, 2 * op_idx]
            processing_time = instance[job_idx, 2 * op_idx + 1]
            op_descriptions.append(f"Operation {op_idx} requires Machine {machine} (processing time {processing_time})")
        
        # Create job description
        job_desc = f"Job {job_idx} has {n_ops} operations: {'; '.join(op_descriptions)}."
        job_descriptions.append(job_desc)

    # Add information about operations with lowest processing time
    for job_idx, k_operations in enumerate(k_operations_with_lowest_processing_time):
        if k_operations:
            lowest_ops = []
            for op_info in k_operations:
                op_idx, machine, processing_time = op_info
                lowest_ops.append(f"Operation {op_idx} (Machine {machine}, processing time {processing_time})")
            
            if len(lowest_ops) == 1:
                job_descriptions[job_idx] += f" The operation with the lowest processing time is: {lowest_ops[0]}."
            else:
                job_descriptions[job_idx] += f" The operations with the lowest processing times are: {'; '.join(lowest_ops)}."

    # solve the JSSP
    result = solve_jssp(instance)
    if result == "infeasible":
        raise ValueError("OR-Tools timed out or found instance infeasible")
    makespan, schedule = result
    output = f"Schedule: {schedule}, Makespan: {makespan}"

    # create the JSON-ready dictionary
    jssp_json = {
        "n": str(n_jobs),
        "m": str(n_machines),
        "instruction": instruction,
        "output": output,
        "input": ' '.join(job_descriptions),
    }

    return jssp_json


class JSSPEnv:
    """
    A JSSP Environment class to generate JSSP instances (processing time matrices) using random integers and save them in JSON format.
    """
    def __init__(
            self, 
            n_job_range: list[int] = None, 
            n_mac_range: list[int] = None, 
            seed: int | None = None,
            external_instance: np.array = None,
            n_jobs: int = None,
            n_machines: int = None
    ) -> None:
        # 添加max_makespan初始化
        self.max_makespan = None
        """
        Initialize the JSSP environment.

        Parameters
        ----------
        n_job_range : list[int]
            [min_job, max_job] range for number of jobs.
        n_mac_range : list[int]
            [min_mac, max_mac] range of number of machines.
        seed : int | None
            Seed for random number generation.
        external_instance : np.array
            External JSSP instance provided by the user (job_machine_flattened format).
        n_jobs : int
            Number of jobs in the external instance.
        n_machines : int
            Number of machines in the external instance.
        """
        self.n_job_range = n_job_range if n_job_range is not None else [10, 10]
        self.n_mac_range = n_mac_range if n_mac_range is not None else [10, 10]
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)  # Only use numpy random for consistency
        
        # Initialize environment state variables
        self.current_instance = external_instance
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.state = None
        self.done = False
        self.step_count = 0
        self.max_steps = 100  # Default max steps
        
        # Dynamic state variables for scheduling progress
        self.job_status = None  # Tracks current operation index for each job
        self.machine_status = None  # Tracks remaining time on each machine
        self.schedule = None  # Tracks when operations start and finish
        self.current_time = 0
        self.completed_operations = 0
        self.total_operations = 0
        
        # Initialize action space as empty list - will be populated in reset
        self.action_space = []
        
        # Track if this is an externally provided instance
        self._is_external_instance = external_instance is not None
        

    
    def get_action_mask(self):
        """
        Return valid job indices as boolean vector.

        Mask semantics are strictly aligned with self.action_space:
        a job is valid iff it is currently schedulable at current_time.

        Returns:
            torch.Tensor: (n_jobs,) boolean mask where True indicates the job is in action_space
        """
        import torch
        self._refresh_action_space()
        mask = torch.zeros(self.n_jobs, dtype=torch.bool)
        for job_id in self.action_space:
            if 0 <= int(job_id) < self.n_jobs:
                mask[int(job_id)] = True
        return mask

    def reset(self):
        """
        Reset the environment to an initial state.

        Returns:
            dict: Initial dynamic state containing:
                - jobs: Job completion status
                - machines: Machine availability status
                - status: Overall scheduling progress
        """
        # Check if this is an externally provided instance
        is_external_instance = hasattr(self, '_is_external_instance') and self._is_external_instance

        if not is_external_instance:
            # Generate new instance dimensions (using +1 for high bound since np.random.randint is half-open interval)
            self.n_jobs = np.random.randint(self.n_job_range[0], self.n_job_range[1] + 1)
            self.n_machines = np.random.randint(self.n_mac_range[0], self.n_mac_range[1] + 1)
            self.total_operations = self.n_jobs * self.n_machines

            # 随机选择困难度以增加数据多样性
            difficulty = np.random.choice(['easy', 'medium', 'hard'], p=[0.2, 0.5, 0.3])

            # Generate new JSSP instance with difficulty
            self.current_instance = self._generate_single_instance(self.n_jobs, self.n_machines, difficulty=difficulty)
        else:
            # For external instances, decide whether to reuse or regenerate
            if hasattr(self, 'use_current_instance_once') and self.use_current_instance_once:
                # Use the externally provided instance once, then generate new instances
                self.total_operations = self.n_jobs * self.n_machines
                # Generate new instance dimensions and instance
                self.n_jobs = np.random.randint(self.n_job_range[0], self.n_job_range[1] + 1)
                self.n_machines = np.random.randint(self.n_mac_range[0], self.n_mac_range[1] + 1)
                self.total_operations = self.n_jobs * self.n_machines
                self.current_instance = self._generate_single_instance(self.n_jobs, self.n_machines)
                self._is_external_instance = False  # Mark as no longer external
            else:
                # Use the externally provided instance repeatedly
                self.total_operations = self.n_jobs * self.n_machines

        # Ensure current_instance is always set
        if self.current_instance is None:
            self.current_instance = self._generate_single_instance(self.n_jobs, self.n_machines)

        # ✅ 关键补丁
        assert self.current_instance is not None, "current_instance must be set"

        # Reset environment state
        self.flat_instance = self.current_instance.flatten()
        self.done = False
        self.step_count = 0

        # Initialize dynamic state variables
        self.current_time = 0
        self.completed_operations = 0
        self.job_status = np.zeros(self.n_jobs, dtype=int)  # Current operation index for each job
        self.machine_status = np.zeros(self.n_machines, dtype=int)  # Release time of each machine (when it becomes available)
        self.schedule = np.zeros((self.n_jobs, self.n_machines, 2), dtype=int)  # [start_time, end_time] for each operation

        # Calculate and set max_makespan based on the current instance
        self.max_makespan = self._calculate_upper_bound()

        # Update action space - now only job indices
        self.action_space = []
        for job in range(self.n_jobs):
            current_op = self.job_status[job]
            if current_op < self.n_machines:
                required_machine = int(self.current_instance[job, current_op * 2])
                if self.machine_status[required_machine] <= self.current_time:
                    self.action_space.append(job)

        return self._get_dynamic_state()

    def _calculate_upper_bound(self):
        """
        Calculate a reasonable upper bound for the makespan based on the current instance.

        Returns:
            float: Upper bound for the makespan
        """
        # 使用每台机器的总负载作为上界，这是一个更紧的估计
        n_jobs, two_m = self.current_instance.shape
        n_machines = two_m // 2
        machine_loads = np.zeros(n_machines)
        for j in range(n_jobs):
            for op in range(n_machines):
                m = int(self.current_instance[j, op*2])
                t = self.current_instance[j, op*2+1]
                machine_loads[m] += t
        return int(np.max(machine_loads)) * 2  # 取最大机器负载的两倍作为上界

    def _estimate_makespan(self):
        """
        Estimate the current makespan based on machine status.

        Returns:
            float: Estimated makespan
        """
        return np.max(self.machine_status)  # machine_status = 每台机器下次可用时间

    def _get_dynamic_state(self):
        """
        Get the current dynamic state of the environment with rich information.

        Returns:
            dict: Dynamic state with detailed job, machine, and scheduling information
        """
        max_jobs = self.n_jobs
        max_machines = self.n_machines

        # ----------------------------- Job-related Information -----------------------------
        # Job next operation indices
        job_next_op = np.zeros(max_jobs, dtype=float)
        for job in range(self.n_jobs):
            job_next_op[job] = self.job_status[job]

        # Job operation completion mask (1 if operation is completed, 0 otherwise)
        job_completion_mask = np.zeros((max_jobs, max_machines), dtype=float)
        for job in range(self.n_jobs):
            for op in range(self.n_machines):
                if op < self.job_status[job]:
                    job_completion_mask[job, op] = 1.0

        # Remaining processing times for each job's operations
        remaining_processing_times = np.zeros((max_jobs, max_machines), dtype=float)
        if self.current_instance is not None:
            for job in range(self.n_jobs):
                for op in range(self.n_machines):
                    if op >= self.job_status[job]:  # Only include remaining operations
                        # Get processing time from the instance
                        processing_time = self.current_instance[job, op * 2 + 1]
                        remaining_processing_times[job, op] = processing_time

        makespan_estimate = self._estimate_makespan()
        max_expected_time = getattr(self, 'max_makespan', makespan_estimate)
        time_scale = float(max(1.0, max_expected_time))

        normalized_machine_availability = np.zeros(max_machines, dtype=float)
        for machine in range(self.n_machines):
            normalized_machine_availability[machine] = float(self.machine_status[machine]) / time_scale

        # Machine busy mask (1 if machine is busy, 0 otherwise)
        machine_busy_mask = np.zeros(max_machines, dtype=float)
        for machine in range(self.n_machines):
            if self.machine_status[machine] > self.current_time:
                machine_busy_mask[machine] = 1.0

        overall_info = np.array([
            float(self.current_time) / time_scale,
            self.completed_operations / max(1, self.total_operations),
            self.step_count / max(1, self.max_steps),
        ], dtype=float)

        # Extract processing times and machine sequence from current instance
        processing_times = np.zeros((max_jobs, max_machines), dtype=float)
        machine_sequence = np.zeros((max_jobs, max_machines), dtype=int)
        if self.current_instance is not None:
            for job in range(self.n_jobs):
                for op in range(self.n_machines):
                    processing_times[job, op] = self.current_instance[job, op * 2 + 1]
                    machine_sequence[job, op] = int(self.current_instance[job, op * 2])

        # Directed machine-order edges for operations already fixed by the partial schedule.
        # If a machine processed O_a immediately before O_b, adj[O_a, O_b] = 1.
        total_nodes = max_jobs * max_machines
        scheduled_machine_adj = np.zeros((total_nodes, total_nodes), dtype=float)
        if self.schedule is not None:
            for machine in range(self.n_machines):
                machine_ops = []
                for job in range(self.n_jobs):
                    for op in range(self.n_machines):
                        end_time = self.schedule[job, op, 1]
                        if end_time <= 0:
                            continue
                        op_machine = int(self.current_instance[job, op * 2])
                        if op_machine == machine:
                            start_time = int(self.schedule[job, op, 0])
                            node_id = job * max_machines + op
                            machine_ops.append((start_time, end_time, node_id))
                machine_ops.sort(key=lambda x: (x[0], x[1]))
                for (_, _, src), (_, _, dst) in zip(machine_ops, machine_ops[1:]):
                    scheduled_machine_adj[src, dst] = 1.0

        # Create is_job_active array (1 if job is active, 0 otherwise)
        is_job_active = np.zeros(max_jobs, dtype=float)
        for job in range(self.n_jobs):
            if self.job_status[job] < self.n_machines:
                is_job_active[job] = 1.0

        legal_action_mask = np.zeros(max_jobs, dtype=float)
        for action in getattr(self, "action_space", []):
            try:
                action = int(action)
            except Exception:
                continue
            if 0 <= action < max_jobs:
                legal_action_mask[action] = 1.0

        return {
            'job_next_op': job_next_op,                      # Next operation index for each job
            'job_completion_mask': job_completion_mask,      # Which operations are completed for each job
            'remaining_processing_times': remaining_processing_times,  # Remaining processing times
            'normalized_machine_availability': normalized_machine_availability,  # Machine availability times
            'machine_busy_mask': machine_busy_mask,          # Which machines are busy
            'overall_info': overall_info,                    # Overall scheduling information
            'processing_times': processing_times,            # Processing times for each operation
            'machine_sequence': machine_sequence,            # Machine sequence for each operation
            'scheduled_machine_adj': scheduled_machine_adj,  # Directed edges for fixed machine order
            'legal_action_mask': legal_action_mask,          # Currently schedulable jobs
            'current_time': self.current_time,               # Current time
            'machine_status': self.machine_status,           # Machine status
            'completed_operations': int(np.sum(self.job_status)),  # Number of completed operations
            'makespan_estimate': makespan_estimate,          # Estimated makespan
            'max_expected_time': max_expected_time           # Maximum expected time
        }

    def _refresh_action_space(self):
        """
        Rebuild action_space at current_time based on precedence and machine availability.
        """
        self.action_space = []
        for job in range(self.n_jobs):
            current_op = self.job_status[job]
            if current_op < self.n_machines:
                if current_op > 0:
                    prev_operation_end_time = self.schedule[job, current_op - 1][1]
                    if prev_operation_end_time > self.current_time:
                        continue
                required_machine = int(self.current_instance[job, current_op * 2])
                if self.machine_status[required_machine] <= self.current_time:
                    self.action_space.append(job)

    def _advance_time(self):
        """
        Advance time to the earliest machine release time if all machines are busy.
        Updates current_time and action_space accordingly.

        Returns:
            bool: True if time was advanced, False otherwise
        """
        # Find the earliest machine release time that is greater than current time
        earliest_release_time = float('inf')
        for release_time in self.machine_status:
            if release_time > self.current_time and release_time < earliest_release_time:
                earliest_release_time = release_time

        if earliest_release_time < float('inf'):
            # Automatically advance time to the next available time
            self.current_time = earliest_release_time
            time_advanced = True
        else:
            time_advanced = False

        self._refresh_action_space()

        return time_advanced

    def _execute_operation(self, job_id, op_id, machine_id):
        """
        Execute an operation and update environment state.

        Args:
            job_id: Job index
            op_id: Operation index (the operation that is about to be executed)
            machine_id: Machine index

        Returns:
            tuple: (next_state, done, info)
        """
        # Get processing time
        processing_time = int(self.current_instance[job_id, op_id * 2 + 1])

        # Calculate start and end times
        # Start time is the maximum of current time and the machine's release time
        start_time = max(self.current_time, self.machine_status[machine_id])
        end_time = start_time + processing_time

        # Update schedule
        self.schedule[job_id, op_id] = [start_time, end_time]

        # Update machine status to reflect that this machine is now busy with the scheduled operation
        self.machine_status[machine_id] = end_time  # Set to release time (when machine becomes available again)

        # Update job status
        self.job_status[job_id] += 1

        # Keep the legacy attribute as completed operation count, not completed job count.
        self.completed_operations = int(np.sum(self.job_status))

        # Calculate current makespan estimate (maximum finish time across all completed operations)
        current_makespan_estimate = 0
        for j in range(self.n_jobs):
            for op in range(self.job_status[j]):
                current_makespan_estimate = max(current_makespan_estimate, self.schedule[j][op][1])

        # Check if episode is done (all operations completed or max steps reached)
        done = False
        if self.completed_operations == self.n_jobs * self.n_machines or self.step_count >= self.max_steps:
            done = True

        return self._get_dynamic_state(), done, {'job_completed': self.job_status[job_id] == self.n_machines,
                                                'current_makespan_estimate': current_makespan_estimate}

    def step(self, job=None, machine=None, action=None):
        """
        Step the environment with the given action.

        Args:
            job: Job index (integer) representing the job to schedule next operation (optional)
            machine: Machine index (integer) representing the machine to use for the operation (optional)
            action: Either:
                   - Job index (integer) representing the job to schedule next operation (legacy format)
                   - Tuple (job, machine) representing the job and machine to schedule next operation (legacy tuple format)

        Returns:
            tuple: (next_state, done, info)
        """
        # Calculate old makespan estimate before taking action
        # We should use a better estimate than just max(machine_status)
        # For now, use max(machine_status) as baseline
        old_makespan_est = self._estimate_makespan()

        # Fallback mechanism: if no valid actions, advance time automatically
        if len(self.action_space) == 0 and not self.done:
            time_advanced = self._advance_time()
            # Calculate new makespan estimate after advancing time
            new_makespan_est = self._estimate_makespan()
            # 归一化奖励
            time_scale = float(max(1, getattr(self, 'max_makespan', 1) or 1))
            reward = (old_makespan_est - new_makespan_est) / time_scale

            # Check if all jobs are completed
            all_jobs_completed = all(op >= self.n_machines for op in self.job_status)
            if all_jobs_completed:
                # All jobs are completed, episode is done
                self.done = True
                makespan = max(self.machine_status)
                return self._get_dynamic_state(), True, {'all_jobs_completed': True, 'makespan': makespan, 'action_space': self.action_space, 'reward': reward}
            # Don't force end if still no actions available - let the agent handle it
            # This prevents premature termination when we might need to advance time again later
            return self._get_dynamic_state(), self.done, {'time_advanced': time_advanced, 'action_space': self.action_space, 'reward': reward}

        # Determine input format
        if job is not None and machine is not None:
            # New explicit interface: (job, machine)
            is_tuple_input = True
        elif action is not None:
            # Legacy format: either tuple or single job index
            if isinstance(action, tuple):
                job, machine = action
                is_tuple_input = True
            else:
                job = action
                machine = None
                is_tuple_input = False
        else:
            # No action provided - advance time
            if self.done:
                return self._get_dynamic_state(), True, {}
            time_advanced = self._advance_time()
            # Calculate new makespan estimate after advancing time
            new_makespan_est = self._estimate_makespan()
            # 归一化奖励
            time_scale = float(max(1, getattr(self, 'max_makespan', 1) or 1))
            reward = (old_makespan_est - new_makespan_est) / time_scale
            return self._get_dynamic_state(), self.done, {'time_advanced': time_advanced, 'action_space': self.action_space, 'reward': reward}

        if self.done:
            # If episode is done, still validate action to provide meaningful feedback
            # Calculate reward as 0 since no change in makespan
            reward = 0.0

            if job < 0 or job >= self.n_jobs:
                return self._get_dynamic_state(), True, {'invalid': 'job_index_out_of_range', 'reward': reward}
            if is_tuple_input and (machine < 0 or machine >= self.n_machines):
                return self._get_dynamic_state(), True, {'invalid': 'machine_index_out_of_range', 'reward': reward}

            # Check if job is already completed (it should be if episode is done)
            current_op = self.job_status[job]
            if current_op >= self.n_machines:
                # Invalid action: job already completed
                return self._get_dynamic_state(), True, {'invalid': 'job_completed', 'reward': reward}

            # For now, if it's a valid job index and not completed, return normal done state
            return self._get_dynamic_state(), True, {'reward': reward}

        # Increment step counter
        self.step_count += 1

        # Validate job index
        if job < 0 or job >= self.n_jobs:
            return self._get_dynamic_state(), self.done, {'invalid': 'job_index_out_of_range', 'reward': 0.0}

        # Validate machine index if provided
        if is_tuple_input:
            if machine < 0 or machine >= self.n_machines:
                return self._get_dynamic_state(), self.done, {'invalid': 'machine_index_out_of_range', 'reward': 0.0}

        # Check if job is already completed
        current_op = self.job_status[job]
        if current_op >= self.n_machines:
            return self._get_dynamic_state(), self.done, {'invalid': 'job_completed', 'reward': 0.0}

        # Check if previous operation (if any) is completed
        if current_op > 0:
            prev_operation_end_time = self.schedule[job, current_op - 1][1]
            if prev_operation_end_time > self.current_time:
                return self._get_dynamic_state(), self.done, {'invalid': 'previous_operation_not_completed', 'reward': 0.0}

        # Get required machine for this operation from the instance
        required_machine = int(self.current_instance[job, current_op * 2])

        if is_tuple_input:
            # In new format, check if the specified machine matches the required machine
            if machine != required_machine:
                return self._get_dynamic_state(), self.done, {'invalid': 'machine_mismatch',
                                                             'required_machine': required_machine,
                                                             'specified_machine': machine,
                                                             'action_space': self.action_space,
                                                             'reward': 0.0}

            # Check if the specified machine is available
            if self.machine_status[machine] > self.current_time:
                return self._get_dynamic_state(), self.done, {'invalid': 'machine_busy',
                                                             'action_space': self.action_space,
                                                             'reward': 0.0}
        else:
            # Legacy format: check if required machine is available
            if self.machine_status[required_machine] > self.current_time:
                return self._get_dynamic_state(), self.done, {'invalid': 'machine_busy',
                                                             'action_space': self.action_space,
                                                             'reward': 0.0}

            # Check if job is in action space (legacy check)
            if job not in self.action_space:
                return self._get_dynamic_state(), self.done, {'invalid': 'not_in_action_space',
                                                             'action_space': self.action_space,
                                                             'reward': 0.0}

        # Determine which machine to use based on input format
        if is_tuple_input:
            # Use the specified machine from the tuple
            machine_to_use = machine
        else:
            # Use the required machine from the instance
            machine_to_use = required_machine

        # Execute the operation. The dynamic state returned inside
        # _execute_operation is intentionally discarded because action_space may
        # still need refreshing or automatic time advance before the next
        # decision state is valid.
        _, self.done, info = self._execute_operation(job, current_op, machine_to_use)

        # 计算reward：归一化的makespan改善量，让RTG量级稳定
        # 用 max_makespan 作为归一化基准，reward范围约[-1, 1]
        new_makespan_est = self._estimate_makespan()
        time_scale = float(max(1, getattr(self, 'max_makespan', 1) or 1))
        reward = (old_makespan_est - new_makespan_est) / time_scale
        info['reward'] = reward

        self._refresh_action_space()
        time_advanced = False
        if len(self.action_space) == 0 and not self.done:
            time_advanced = self._advance_time()
        info['action_space'] = self.action_space.copy()
        info['time_advanced'] = time_advanced

        next_state = self._get_dynamic_state()
        return next_state, self.done, info



    def get_makespan(self):
        """Return actual completion time of last finished operation."""
        if self.schedule is None or self.schedule.size == 0:
            return 0

        makespan = 0
        for job in range(self.n_jobs):
            for op in range(self.n_machines):
                end_time = self.schedule[job][op][1]
                if end_time > 0:
                    makespan = max(makespan, end_time)
        return makespan

    def get_valid_actions(self):
        """
        Get all valid actions at the current time step.

        Returns:
            list: List of (job_idx, machine_idx) tuples representing valid actions
        """
        valid_actions = []

        for job in range(self.n_jobs):
            current_op = self.job_status[job]
            if current_op < self.n_machines:
                # Check if previous operation (if any) is completed
                if current_op > 0:
                    prev_operation_end_time = self.schedule[job, current_op - 1][1]
                    if prev_operation_end_time > self.current_time:
                        continue
                # Check if required machine is available
                required_machine = int(self.current_instance[job, current_op * 2])
                if self.machine_status[required_machine] <= self.current_time:
                    valid_actions.append((job, required_machine))

        return valid_actions

    def get_available_actions(self):
        """
        Get a list of currently available actions (job indices).

        Returns:
            list: List of job indices that can be scheduled at the current time
        """
        return self.action_space

    def get_schedule(self):
        """
        Get the current schedule in a format compatible with the reward functions.

        Returns:
            list: Schedule of operations with job, machine, operation index, start time, and processing time
        """
        schedule_list = []
        for job in range(self.n_jobs):
            for op in range(self.n_machines):
                if self.schedule[job][op][1] > 0:  # Only include scheduled operations
                    machine = int(self.current_instance[job, op * 2])
                    schedule_list.append({
                        'job': job,
                        'machine': machine,
                        'operation': op,
                        'start_time': self.schedule[job][op][0],
                        'processing_time': int(self.current_instance[job, op * 2 + 1])
                    })
        return schedule_list

    def get_expert_demonstration(self, ortools_makespan=None):
        """
        Return an expert trajectory that can be used for BC training.

        Returns:
            dict: {
                'states': np.array([s0, s1, ..., sT]),      # shape (T+1, ...)
                'actions': np.array([a0, a1, ..., a_{T-1}]), # shape (T,), dtype=int (flat index)
                'action_masks': np.array([m0, m1, ..., m_{T-1}]), # shape (T, J, M), bool
                'current_times': np.array([t0, t1, ..., tT])     # shape (T+1,)
            }
        """
        # Import required modules
        import torch

        self.reset()
        states = [self._get_dynamic_state()]
        actions = []
        masks = []
        times = [self.current_time]

        # Get OR-Tools solution
        instance_tuple = (self.n_jobs, self.n_machines, self.current_instance)
        if ortools_makespan is None:
            ortools_result = solve_jssp(self.current_instance)
            if ortools_result == "infeasible":
                raise ValueError("OR-Tools timed out or found instance infeasible")
            ortools_makespan, ortools_schedule = ortools_result
        else:
            ortools_result = solve_jssp(self.current_instance)
            if ortools_result == "infeasible":
                raise ValueError("OR-Tools timed out or found instance infeasible")
            _, ortools_schedule = ortools_result

        # --- 关键修复：重建带时间戳的操作列表 ---
        # First, we need to reconstruct the schedule with start times
        # We'll use the provided `get_makespan` function's logic to decode the schedule
        n, two_m = self.current_instance.shape
        m = two_m // 2
        job_next = [0] * n
        job_finish = [0] * n
        machine_finish = [0] * m
        machine_ptr = [0] * m
        scheduled_ops = 0
        total_ops = n * m
        chronological_ops = []  # List of (start_time, job, machine)

        while scheduled_ops < total_ops:
            available = []
            for machine in range(m):
                if machine_ptr[machine] < n:
                    job = ortools_schedule[machine][machine_ptr[machine]]
                    op_idx = job_next[job]
                    if op_idx < m and int(self.current_instance[job, 2 * op_idx]) == machine:
                        start_time = max(job_finish[job], machine_finish[machine])
                        available.append((start_time, machine, job))

            if not available:
                raise ValueError("Infeasible schedule from OR-Tools")

            available.sort(key=lambda x: x[0])
            start_time, machine, job = available[0]
            op_idx = job_next[job]
            proc_time = self.current_instance[job, 2 * op_idx + 1]
            finish_time = start_time + proc_time

            # Record this operation in chronological order
            chronological_ops.append((start_time, job, machine))

            job_finish[job] = finish_time
            machine_finish[machine] = finish_time
            job_next[job] += 1
            machine_ptr[machine] += 1
            scheduled_ops += 1
        # --- 修复结束 ---

        # Now execute each operation in TRUE chronological order
        for start_time, job_id, machine_id in chronological_ops:
            # Record current state and mask before executing action
            action_mask = self.get_action_mask()  # shape: (n_jobs, n_machines), torch.Tensor
            masks.append(action_mask.numpy())  # Convert to numpy array and store

            # Convert to flat action index
            action_idx = job_id * self.n_machines + machine_id
            actions.append(action_idx)

            # Execute action
            next_state, done, info = self.step(job=job_id, machine=machine_id)

            # Record next state and time
            states.append(next_state)
            times.append(self.current_time)

            # Check if we're done
            if done:
                break

        # Convert lists to numpy arrays
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'action_masks': np.array(masks),
            'current_times': np.array(times)
        }

    def _generate_single_instance(self, n_jobs, n_machines, difficulty='medium'):
        """
        Generate a single JSSP instance with specified difficulty level.

        Args:
            n_jobs: Number of jobs
            n_machines: Number of machines
            difficulty: Difficulty level ('easy', 'medium', 'hard')

        Returns:
            np.ndarray: JSSP instance matrix where each row represents a job,
                        and for each operation, we have [machine, processing_time]
        """
        # 定义不同难度等级的参数
        difficulty_params = {
            'easy': {
                'processing_time_range': (1, 5),      # 处理时间短，容易调度
                'shuffle_probability': 0.2              # 机器顺序简单
            },
            'medium': {
                'processing_time_range': (3, 8),      # 处理时间中等
                'shuffle_probability': 0.5              # 机器顺序中等
            },
            'hard': {
                'processing_time_range': (6, 15),     # 处理时间长，难调度
                'shuffle_probability': 0.8              # 机器顺序复杂
            }
        }

        # 获取当前难度的参数
        params = difficulty_params.get(difficulty, difficulty_params['medium'])
        min_time, max_time = params['processing_time_range']
        shuffle_prob = params['shuffle_probability']

        # Create instance matrix: each row represents a job, with [machine, processing_time] for each operation
        instance = np.zeros((n_jobs, n_machines * 2), dtype=np.int32)

        # Generate random data for each job and operation
        for job_idx in range(n_jobs):
            # Generate random machine order for this job
            machine_order = np.arange(n_machines)

            # 根据难度决定是否打乱机器顺序
            if np.random.random() < shuffle_prob:
                # 打乱机器顺序，增加调度复杂度
                np.random.shuffle(machine_order)

            # Generate random processing times for this job based on difficulty
            processing_times = np.random.randint(min_time, max_time + 1, size=n_machines, dtype=np.int32)
            
            # Fill in the instance matrix
            for op_idx in range(n_machines):
                instance[job_idx, op_idx * 2] = machine_order[op_idx]
                instance[job_idx, op_idx * 2 + 1] = processing_times[op_idx]
        
        return instance
        
    def set_external_instance(self, processing_times, machine_order):
        """
        Set an external JSSP instance instead of generating a random one.
        
        Args:
            processing_times: 2D array of processing times [n_jobs, n_machines]
            machine_order: 2D array of machine orders [n_jobs, n_machines]
        """
        self.n_jobs = processing_times.shape[0]
        self.n_machines = processing_times.shape[1]
        self.total_operations = self.n_jobs * self.n_machines
        
        # Create the instance in the format expected by the environment
        # Each row represents a job, with [machine, processing_time] for each operation
        self.current_instance = np.zeros((self.n_jobs, self.n_machines * 2), dtype=int)
        for job in range(self.n_jobs):
            for op in range(self.n_machines):
                self.current_instance[job, op * 2] = machine_order[job, op]
                self.current_instance[job, op * 2 + 1] = processing_times[job, op]
        
        # Mark this as an externally provided instance
        self._is_external_instance = True


    def generate_jssp(self, num_jobs: int, num_machines: int, dataset_size: int, time_range: tuple = (1, 100)) -> list[np.array]:
        """
        Generate a list of JSSP instances.

        Parameters
        ----------
        num_jobs : int
            Number of jobs.
        num_machines : int
            Number of machines.
        dataset_size : int
            Number of instances to generate.
        time_range : tuple, optional
            Range of processing times (inclusive), by default (1, 100).

        Returns
        -------
        list[np.array]
            A list of JSSP instances as 2-d np-arrays.
        """
        instances = []
        for _ in range(dataset_size):
            processing_time = np.random.randint(time_range[0], time_range[1] + 1, size=(num_jobs, num_machines), dtype=np.int32)
            machine_assignment = np.zeros((num_jobs, num_machines))
            for job_idx in range(num_jobs):
                machines = list(range(num_machines))
                np.random.shuffle(machines)
                machine_assignment[job_idx] = machines
            instance = []
            for job_idx in range(num_jobs):
                job_info = []
                for op_idx in range(num_machines):
                    job_info += [machine_assignment[job_idx, op_idx], processing_time[job_idx, op_idx]]
                instance.append(job_info)
            instances.append(np.array(instance, dtype=np.int32))
        return instances

    
    def generate_instances(self, n_instance: int) -> list[np.array]:
        """
        Generate a list of JSSP instances.

        Parameters
        ----------
        n_instance : int
            Number of instances to generate.

        Returns
        -------
        list[np.array]
            A list of JSSP instances as 2-d np-arrays.
        """
        instances = []
        for _ in range(n_instance):
            num_jobs = np.random.randint(self.n_job_range[0], self.n_job_range[1] + 1)
            num_machines = np.random.randint(self.n_mac_range[0], self.n_mac_range[1] + 1)
            instances += self.generate_jssp(num_jobs, num_machines, 1)
        return instances
    

    def save_dataset(self, dataset, filename, disable_print=False):
        filedir = os.path.split(filename)[0]
        if not os.path.isdir(filedir):
            os.makedirs(filedir)
        with open(check_extension(filename), 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        if not disable_print:
            print(">> Save dataset to {}".format(filename))


    def generate_instances_and_save(self, n_instance: int, file_name: str, save_pkl: bool, k: int, pickle_path: str=None) -> None:
        """
        Generate JSSP instances and save them to a JSON file.

        Parameters
        ----------
        n_instance : int
            Number of JSSP instances to generate.
        file_name : str
            Path where the resulting JSON file should be saved.
        save_pkl : bool
            Whether to save the dataset as a pickle file.
        k : int
            Number of tags to be used.
        pickle_path : str
            Path where the resulting pickle file should be saved.
        """
        instances = self.generate_instances(n_instance)
        if save_pkl and pickle_path is not None:
            self.save_dataset(instances, pickle_path)
        jssp_data = []
        skipped_instances = 0
        for instance in tqdm(instances, desc="Generating JSSP instances"):
            try:
                jssp_item = tag_prompt_and_transform_to_json(instance, k)
                jssp_data.append(jssp_item)
            except ValueError:
                skipped_instances += 1
                continue
        
        if skipped_instances > 0:
            print(f"Warning: Skipped {skipped_instances} instances due to OR-Tools timeout")

        with open(file_name, 'w') as f:
            json.dump(jssp_data, f, indent=4)
        
       
def parse_args():
    """
    Parse command line arguments for the PFSP environment.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Job Shop Scheduling Problem (JSSP) Instance Generator')
    
    # Job and machine range parameters
    parser.add_argument('--min_jobs', type=int, default=5, help='Minimum number of jobs')
    parser.add_argument('--max_jobs', type=int, default=20, help='Maximum number of jobs')
    parser.add_argument('--min_machines', type=int, default=5, help='Minimum number of machines')
    parser.add_argument('--max_machines', type=int, default=20, help='Maximum number of machines')
    
    # Instance generation parameters
    parser.add_argument('--n_instances', type=int, help='Number of PFSP instances to generate')
    parser.add_argument('--output_file', type=str, default='./output/train_jssp.json', help='Path to save the JSON output file')
    parser.add_argument('--save_pkl', action='store_true', default=False, help='Whether to save the dataset as a pickle file')
    parser.add_argument('--pickle_path', type=str, default=None, help='Path to save the pickle file')
    # Random seed
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # JSON transformation parameters
    parser.add_argument('--k', type=int, default=2, help='Number of tags to include in the description')
    parser.add_argument('--n_processes', type=int, default=8, help='Number of processes to use for parallel generation')
    
    return parser.parse_args()


def generate_subset(process_index, seed, args, instances_per_process, n_processes):
    """
    Generate a subset of JSSP instances in a single process.
    
    Parameters
    ----------
    process_index : int
        Index of the current process.
    seed : int
        Random seed for this process.
    args : argparse.Namespace
        Command line arguments.
    instances_per_process : int
        Number of instances to generate per process.
    n_processes : int
        Total number of processes.
        
    Returns
    -------
    str
        Path to the file containing the generated instances.
    """
    # Create output file name with process index
    base_name, ext = os.path.splitext(args.output_file)
    process_file_name = f"{base_name}_part{process_index}{ext}"
    pickle_path = None
    if args.pickle_path is not None:
        pickle_path = f"{base_name}_part{process_index}.pkl"
    
    # Create JSSP environment with derived seed
    jssp_env = JSSPEnv(
        n_job_range=[args.min_jobs, args.max_jobs],
        n_mac_range=[args.min_machines, args.max_machines],
        seed=seed
    )

    # Calculate number of instances for this process
    if process_index == n_processes - 1:
        # Last process handles the remainder
        n_instances = args.n_instances - (process_index * instances_per_process)
    else:
        n_instances = instances_per_process

    # Generate instances and save to separate file
    print(f"Process {process_index}: Generating {n_instances} instances with seed {seed}")
    jssp_env.generate_instances_and_save(
        n_instance=n_instances,
        file_name=process_file_name,
        save_pkl=args.save_pkl,
        pickle_path=pickle_path,
        k=args.k
    )

    return process_file_name


def generate_instances_parallel(args, n_processes, combine_files=True):
    """
    Generate JSSP instances in parallel using multiple processes.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    n_processes : int
        Number of parallel processes to use.
    combine_files : bool
        Whether to combine individual JSON files into one.

    Returns
    -------
    list or list of str
        If combine_files is True, returns the combined data from all processes.
        If combine_files is False, returns the list of file paths to individual JSON files.
    """
    # Calculate instances per process
    instances_per_process = math.ceil(args.n_instances / n_processes)

    # Generate derived seeds from initial seed
    np.random.seed(args.seed)
    derived_seeds = np.random.randint(0, 100000, size=n_processes).tolist()

    # Create and start the process pool
    with Pool(processes=n_processes) as pool:
        result_files = pool.starmap(
            generate_subset, 
            [(i, derived_seeds[i], args, instances_per_process, n_processes) for i in range(n_processes)]
        )

    # Combine files if requested
    if not combine_files:
        print(f"Generated {args.n_instances} JSSP instances in {n_processes} separate files:")
        for file_name in result_files:
            print(f"  - {file_name}")
        return result_files
    
    # Combine the individual JSON files into one
    print("Combining results from all processes...")
    combined_data = []
    for file_name in result_files:
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                data = json.load(f)
                combined_data.extend(data)
            # Remove the temporary file
            os.remove(file_name)
    # Save the combined data
    with open(args.output_file, 'w') as f:
        json.dump(combined_data, f, indent=4)
    
    print(f"Successfully generated {len(combined_data)} JSSP instances and saved to {args.output_file}")
    return combined_data


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Generate instances in parallel
    generate_instances_parallel(args, n_processes=args.n_processes, combine_files=True)
    
    # generate 500,000 instances: 
    # python JSSPEnv.py --n_instances 500000 --output_file ./training_instance/train_jssp.json --k 2 --n_processes 32

    # generate 300 instances:
    # python JSSPEnv.py --n_instances 300 --output_file ./testing/test_jssp.json --k 2 --n_processes 1 --save_pkl --pickle_path ./testing/test_jssp.pkl
