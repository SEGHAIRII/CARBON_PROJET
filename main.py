# main_ibs.py
import time
import sys
import numpy as np

# Import necessary components from our other files
from utils import Node, calculate_makespan_of_complete_schedule, \
                  concatenate_schedule, calculate_bound_bi_directional, \
                  insert_forward, insert_backward, calculate_g_gap
from io_handler import parse_input

import random
from collections import deque

def compute_critical_path_positions(schedule, num_machines, proc_times_jm, completion_times):
    """
    Backtrack through the completion_times matrix to identify the indices
    in the schedule that lie on the critical path.
    Returns a list of job positions (indices in schedule) on the critical path.
    """
    n = len(schedule)
    i = n - 1
    k = num_machines - 1
    critical_positions = []

    # Backtrack from (i, k) to (0, 0)
    while i >= 0 and k >= 0:
        critical_positions.append(i)
        current_time = completion_times[i, k]
        job = schedule[i]
        p = proc_times_jm[job, k]

        # Move up or left depending on predecessor that matches timing
        if i == 0:
            k -= 1
        elif k == 0:
            i -= 1
        else:
            if np.isclose(completion_times[i-1, k] + p, current_time):
                i -= 1
            else:
                k -= 1

    return list(set(critical_positions))


def delta_makespan_swap(schedule, num_machines, proc_times_jm, completion_times, i, j):
    """
    Compute the makespan delta for swapping two jobs at positions i and j in the schedule.
    Returns new_makespan, delta, and the updated completion_times matrix.
    """
    n = len(schedule)
    # Copy original completion times for rollback
    new_ct = completion_times.copy()

    # Apply swap on a schedule copy
    s = schedule.copy()
    s[i], s[j] = s[j], s[i]

    # Determine start row for recomputation
    start = i
    # If swapping at position 0, recompute first job row
    if start == 0:
        job0 = s[0]
        new_ct[0, 0] = proc_times_jm[job0, 0]
        for m in range(1, num_machines):
            new_ct[0, m] = new_ct[0, m-1] + proc_times_jm[job0, m]
        start = 1

    # Recompute rows from start to j
    for idx in range(start, j+1):
        job_idx = s[idx]
        new_ct[idx, 0] = new_ct[idx-1, 0] + proc_times_jm[job_idx, 0]
        for m in range(1, num_machines):
            new_ct[idx, m] = max(new_ct[idx, m-1], new_ct[idx-1, m]) + proc_times_jm[job_idx, m]

    # Rows after j remain unchanged
    for idx in range(j+1, n):
        new_ct[idx] = completion_times[idx]

    old_mk = completion_times[n-1, num_machines-1]
    new_mk = new_ct[n-1, num_machines-1]
    delta = new_mk - old_mk
    return new_mk, delta, new_ct


def tabu_search(initial_schedule, num_machines, proc_times_jm,
                tabu_tenure=10, max_iterations=200, time_limit_seconds=None):
    """
    Tabu Search for the permutation flow-shop scheduling problem (PFSP).

    Args:
      initial_schedule: list of job indices (0-based).
      num_machines: int, number of machines.
      proc_times_jm: 2D numpy array [job, machine] of processing times.
      tabu_tenure: int, max size of the tabu list.
      max_iterations: int, max number of TS iterations.
      time_limit_seconds: float or None, overall time limit.

    Returns:
      best_schedule: list of job indices for best found solution.
      best_makespan: float, makespan of best_schedule.
    """

    # Initial solution and evaluation
    current_schedule = list(initial_schedule)
    # Compute initial completion times
    n = len(current_schedule)
    completion_times = np.zeros((n, num_machines))
    # fill matrix
    first_job = current_schedule[0]
    completion_times[0,0] = proc_times_jm[first_job,0]
    for m in range(1, num_machines):
        completion_times[0,m] = completion_times[0,m-1] + proc_times_jm[first_job,m]
    for i in range(1, n):
        job = current_schedule[i]
        completion_times[i,0] = completion_times[i-1,0] + proc_times_jm[job,0]
        for m in range(1, num_machines):
            completion_times[i,m] = max(completion_times[i,m-1], completion_times[i-1,m]) + proc_times_jm[job,m]
    current_makespan = float(completion_times[-1,-1])
    best_schedule = current_schedule.copy()
    best_makespan = current_makespan

    # Tabu list storing swapped index pairs
    tabu_list = deque()

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        if time_limit_seconds and (time.time() - start_time) > time_limit_seconds:
            break

        # Generate neighborhood: critical-path swaps
        crit_positions = compute_critical_path_positions(current_schedule, num_machines, proc_times_jm, completion_times)
        best_neighbor = None
        best_neighbor_mk = float('inf')
        best_move = None
        best_neighbor_ct = None

        # Evaluate all swaps on critical path
        for idx1 in range(len(crit_positions)):
            for idx2 in range(idx1+1, len(crit_positions)):
                i, j = crit_positions[idx1], crit_positions[idx2]
                move = (i, j)
                if move in tabu_list:
                    continue
                new_mk, delta, new_ct = delta_makespan_swap(
                    current_schedule, num_machines, proc_times_jm, completion_times, i, j)
                if new_mk < best_neighbor_mk:
                    best_neighbor_mk = new_mk
                    best_neighbor = (i, j)
                    best_neighbor_ct = new_ct

        # If no admissible neighbor, break
        if best_neighbor is None:
            break

        # Apply the best move
        i, j = best_neighbor
        current_schedule[i], current_schedule[j] = current_schedule[j], current_schedule[i]
        completion_times = best_neighbor_ct
        current_makespan = best_neighbor_mk

        # Update tabu list
        tabu_list.append(best_neighbor)
        if len(tabu_list) > tabu_tenure:
            tabu_list.popleft()

        # Aspiration: override tabu if new best
        if current_makespan < best_makespan:
            best_schedule = current_schedule.copy()
            best_makespan = current_makespan

    print(f'tabu search best result is {best_makespan}')
    return best_schedule, best_makespan


def compute_completion_times(schedule, proc_times_jm, num_machines):
    n = len(schedule)
    completion_times = np.zeros((n, num_machines))
    completion_times[0, 0] = proc_times_jm[schedule[0], 0]
    for m in range(1, num_machines):
        completion_times[0, m] = completion_times[0, m - 1] + proc_times_jm[schedule[0], m]
    for i in range(1, n):
        job = schedule[i]
        completion_times[i, 0] = completion_times[i - 1, 0] + proc_times_jm[job, 0]
        for m in range(1, num_machines):
            completion_times[i, m] = max(completion_times[i, m - 1], completion_times[i - 1, m]) + proc_times_jm[job, m]
    return completion_times

def hill_climbing(initial_schedule, num_machines, proc_times_jm,  max_iterations=100):
    current_schedule = list(initial_schedule)
    n = len(current_schedule)
    completion_times = np.zeros((n, num_machines))
    # fill matrix
    first_job = current_schedule[0]
    completion_times[0,0] = proc_times_jm[first_job,0]
    for m in range(1, num_machines):
        completion_times[0,m] = completion_times[0,m-1] + proc_times_jm[first_job,m]
    for i in range(1, n):
        job = current_schedule[i]
        completion_times[i,0] = completion_times[i-1,0] + proc_times_jm[job,0]
        for m in range(1, num_machines):
            completion_times[i,m] = max(completion_times[i,m-1], completion_times[i-1,m]) + proc_times_jm[job,m]
    current_makespan = completion_times[-1, -1]
    
    for it in range(max_iterations):
        improved = False
        best_delta = 0
        best_swap = None
        best_ct = None
        best_makespan = current_makespan

        # Try all neighbor swaps
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_mk, delta, new_ct = delta_makespan_swap(current_schedule, num_machines, proc_times_jm, completion_times, i, j)
                if new_mk < best_makespan:
                    improved = True
                    best_delta = delta
                    best_swap = (i, j)
                    best_ct = new_ct
                    best_makespan = new_mk

        if not improved:
            break  # No improvement found => local optimum

        # Apply the best move
        i, j = best_swap
        current_schedule[i], current_schedule[j] = current_schedule[j], current_schedule[i]
        completion_times = best_ct
        current_makespan = best_makespan

    return current_schedule, current_makespan


def create_root_node(num_machines, proc_times_jm, all_job_indices_list):
    # This function uses Node and calculate_bound_bi_directional from utils.py
    root = Node(num_machines=num_machines)
    root.bound = calculate_bound_bi_directional(root, all_job_indices_list, num_machines, proc_times_jm)
    return root

def generate_bi_directional_children(parent_node, current_UB,
                                     all_job_indices, num_jobs, num_machines, proc_times_jm,
                                     second_chance_prob=0.2):
    """
    Generate children by inserting unscheduled jobs forward/backward.
    If a child is pruned by bound, with probability p apply a random subsequence inversion
    on its partial schedule to give it a second chance.
    """
    f_children = []
    b_children = []

    scheduled_jobs_set = set(parent_node.starting) | set(parent_node.finishing)
    if len(scheduled_jobs_set) == num_jobs:
        return []

    unscheduled_job_indices = [j for j in all_job_indices if j not in scheduled_jobs_set]

    # Helper: attempt bound with second chance
    def try_bound_with_second_chance(child, current_UB):
        child.bound = calculate_bound_bi_directional(child, all_job_indices, num_machines, proc_times_jm)
        if child.bound < current_UB:
            return True  # passes normally
        # pruned: with probability p, invert a random subsequence and retry
        if random.random() < second_chance_prob:
            # choose subsequence in starting or finishing
            if child.starting and random.choice([True, False]):
                seq = child.starting
            else:
                seq = child.finishing
            if len(seq) >= 2:
                i, j = sorted(random.sample(range(len(seq)), 2))
                seq[i:j+1] = reversed(seq[i:j+1])
            # recompute bound
            child.bound = calculate_bound_bi_directional(child, all_job_indices, num_machines, proc_times_jm)
            return child.bound < current_UB
        return False

    # Generate forward insertions
    for job_idx in unscheduled_job_indices:
        child_f = insert_forward(parent_node, job_idx, num_machines, proc_times_jm)
        if try_bound_with_second_chance(child_f, current_UB + 1e-6):
            f_children.append(child_f)

    # Generate backward insertions
    for job_idx in unscheduled_job_indices:
        child_b = insert_backward(parent_node, job_idx, num_machines, proc_times_jm)
        if try_bound_with_second_chance(child_b, current_UB + 1e-6):
            b_children.append(child_b)

    # If one side empty, return the other
    if not f_children and not b_children:
        return []
    if not f_children:
        return b_children
    if not b_children:
        return f_children

    # Choose side with fewer children or better average bound
    sum_bounds_F = sum(c.bound for c in f_children)
    sum_bounds_B = sum(c.bound for c in b_children)
    if len(f_children) < len(b_children) or \
       (len(f_children) == len(b_children) and sum_bounds_F > sum_bounds_B):
        return f_children
    else:
        return b_children


def select_best_d_nodes(candidate_nodes, D_beam_width, current_UB, num_machines, proc_times_jm):
    # This function uses calculate_g_gap from utils.py and Node's __lt__ method
    if not candidate_nodes:
        return []

    for node_n in candidate_nodes:
        node_n.guide_value = calculate_g_gap(node_n, current_UB, node_n.bound, num_machines, proc_times_jm)
    
    candidate_nodes.sort() 
    return candidate_nodes[:D_beam_width]

def iterative_beam_search(num_jobs, num_machines, proc_times_jm,
                          initial_beam_width=1, beam_width_factor=2, max_ibs_iterations=10,
                          time_limit_seconds=None):
    
    start_overall_time = time.time()
    
    overall_best_makespan = float('inf')
    overall_best_schedule = []
    
    all_job_indices = list(range(num_jobs))

    D = initial_beam_width
    
    print(f"Starting Iterative Beam Search. Max IBS Iterations: {max_ibs_iterations}, Time Limit: {time_limit_seconds}s\n")

    for ibs_iter_count in range(max_ibs_iterations):
        print(f'===========================iter = {ibs_iter_count}')
        iter_process_start_time = time.time()
        
        root_node = create_root_node(num_machines, proc_times_jm, all_job_indices)
        current_level_nodes = [root_node]
        
        if overall_best_makespan == float('inf'):
            # For first iteration, use a loose upper bound based on sum of all processing times
            initial_estimate = 0
            for i in range(num_jobs):
                for j in range(num_machines):
                    initial_estimate += proc_times_jm[i, j]
            current_UB_for_pruning = initial_estimate  # Very loose bound for first iteration
        else:
            # Apply relaxation factor to existing best makespan
            relaxation_factor = 1.1  # 10% relaxation
            current_UB_for_pruning = overall_best_makespan * relaxation_factor
        
        iter_best_schedule= []
        iter_best_makespan = float('inf')
        for level in range(num_jobs):
            if not current_level_nodes:
                break
            
            next_level_candidates_partial = []
            
            for parent_c_node in current_level_nodes:
                # In early iterations with small beam width, use a more relaxed pruning
                iteration_adaptive_UB = current_UB_for_pruning
                if ibs_iter_count < 2 and D <= 5:  # For first two iterations with small beam width
                    iteration_adaptive_UB *= 1.2  # Extra 20% relaxation
                
                children_nodes = generate_bi_directional_children(parent_c_node, iteration_adaptive_UB,
                                                                  all_job_indices, num_jobs, num_machines, proc_times_jm)
                
                for child in children_nodes: 
                    if len(child.starting) + len(child.finishing) == num_jobs:
                        schedule_solution = concatenate_schedule(child.starting, child.finishing)
                        makespan_solution = calculate_makespan_of_complete_schedule(schedule_solution, num_machines, proc_times_jm)
                        
                        if makespan_solution < iter_best_makespan:
                            iter_best_makespan = makespan_solution
                            iter_best_schedule = schedule_solution
                    else: 
                        next_level_candidates_partial.append(child)

            if not next_level_candidates_partial:
                break 

            current_level_nodes = select_best_d_nodes(next_level_candidates_partial, D, current_UB_for_pruning, 
                                                      num_machines, proc_times_jm)
        

        print(f'=====result before hill climbing = {iter_best_makespan} ======')
        # Only apply hill climbing if we found a valid schedule
        if iter_best_makespan and len(iter_best_schedule) > 0:
            print("\n--- Starting Hill Climbing to further improve the solution ---")
            schedule_solution, makespan_solution = hill_climbing(iter_best_schedule, num_machines, proc_times_jm) 
            if makespan_solution < overall_best_makespan:
                overall_best_makespan = makespan_solution
                overall_best_schedule = schedule_solution
                current_UB_for_pruning = overall_best_makespan
        else:
            print("\n--- No valid schedule found for hill climbing in this iteration ---")
        
        current_total_execution_time = time.time() - start_overall_time
        iter_processing_duration = time.time() - iter_process_start_time
#================================current iter solution=======================================================================
        print(f"--- IBS Iteration {ibs_iter_count + 1} (Beam Width D={D}) Completed ---")
        print(f"    Time for this iteration's processing: {iter_processing_duration:.4f} seconds")
        print(f"Best Makespan Found So Far: {overall_best_makespan if overall_best_makespan != float('inf') else 'N/A'}")
        printable_schedule = [j + 1 for j in overall_best_schedule] if overall_best_schedule else 'N/A'
        print(f"Best Schedule Found So Far (1-indexed jobs): {printable_schedule}")
        print(f"Total Execution Time Up To This Point: {current_total_execution_time:.4f} seconds")
        print("-" * 60)

        if time_limit_seconds is not None and current_total_execution_time > time_limit_seconds:
            print("Time limit reached. Stopping IBS.")
            return overall_best_makespan, overall_best_schedule, time_limit_seconds

        D = D * beam_width_factor
        
        if D > (num_jobs**2) * 2 and num_jobs > 10 and D > 10000: 
             print(f"Beam width D={D} is becoming very large. Stopping IBS to conserve memory.")
             break
#=========================================final result==============================================
    final_execution_time = time.time() - start_overall_time
    print("\n--- Final Result After All Iterations ---")
    print(f"Best Makespan: {overall_best_makespan if overall_best_makespan != float('inf') else 'N/A'}")
    printable_schedule_final = [j + 1 for j in overall_best_schedule] if overall_best_schedule else 'N/A'
    print(f"Best Schedule (1-indexed jobs): {printable_schedule_final}")
    print(f"Total Execution Time: {final_execution_time:.4f} seconds")
    
    return overall_best_makespan, overall_best_schedule, final_execution_time

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python main_ibs.py <input_file_path> [max_ibs_iterations] [time_limit_seconds] [initial_beam_width] [beam_width_factor]")
        print("\nCreating a dummy test file 'dummy_pfsp.txt' for demonstration.")
        dummy_content = """3 2
10 20 30
15 5 25
"""
        with open("dummy_pfsp.txt", "w") as df:
            df.write(dummy_content)
        file_path = "dummy_pfsp.txt"
        print(f"Running with dummy file: {file_path}")
        max_iters_param = 5
        time_limit_param = 10 
        init_D_param = 1
        D_factor_param = 2
    else:
        file_path = sys.argv[1]
        max_iters_param = int(sys.argv[2]) if len(sys.argv) > 2 else 12
        time_limit_param = int(sys.argv[3]) if len(sys.argv) > 3 else 200000
        init_D_param = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        D_factor_param = int(sys.argv[5]) if len(sys.argv) > 5 else 2

    try:
        print(f"Parsing input file: {file_path}")
        num_jobs, num_machines, proc_times_jm = parse_input(file_path)
        print(f"Problem: {num_jobs} Jobs, {num_machines} Machines.")
        
        best_makespan, best_schedule, exec_time = iterative_beam_search(num_jobs, num_machines, proc_times_jm,
                              initial_beam_width=init_D_param, 
                              beam_width_factor=D_factor_param, 
                              max_ibs_iterations=max_iters_param,
                              time_limit_seconds=time_limit_param)
        
        if best_makespan == float('inf') or not best_schedule:
            print("\nNo feasible solution was found.")
        else:
            print(f"\nBest makespan found: {best_makespan}")
            print(f"Execution time: {exec_time:.4f} seconds")
                              

    except FileNotFoundError:
        print(f"Error: Input file '{file_path}' not found.")
    except ValueError as ve:
        print(f"Input or Value Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()