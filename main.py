# main_ibs.py
import time
import sys

# Import necessary components from our other files
from utils import Node, calculate_makespan_of_complete_schedule, \
                  concatenate_schedule, calculate_bound_bi_directional, \
                  insert_forward, insert_backward, calculate_g_gap
from io_handler import parse_input

def create_root_node(num_machines, proc_times_jm, all_job_indices_list):
    # This function uses Node and calculate_bound_bi_directional from utils.py
    root = Node(num_machines=num_machines)
    root.bound = calculate_bound_bi_directional(root, all_job_indices_list, num_machines, proc_times_jm)
    return root

def generate_bi_directional_children(parent_node, current_UB,
                                     all_job_indices, num_jobs, num_machines, proc_times_jm):
    # This function uses insert_forward, insert_backward, calculate_bound_bi_directional from utils.py
    f_children = []
    b_children = []

    scheduled_jobs_set = set(parent_node.starting) | set(parent_node.finishing)
    
    if len(scheduled_jobs_set) == num_jobs:
        return []

    unscheduled_job_indices = [j for j in all_job_indices if j not in scheduled_jobs_set]

    for job_idx in unscheduled_job_indices:
        child_f = insert_forward(parent_node, job_idx, num_machines, proc_times_jm)
        child_f.bound = calculate_bound_bi_directional(child_f, all_job_indices, num_machines, proc_times_jm)
        if child_f.bound < current_UB:
            f_children.append(child_f)

    for job_idx in unscheduled_job_indices:
        child_b = insert_backward(parent_node, job_idx, num_machines, proc_times_jm)
        child_b.bound = calculate_bound_bi_directional(child_b, all_job_indices, num_machines, proc_times_jm)
        if child_b.bound < current_UB:
            b_children.append(child_b)
    
    if not f_children and not b_children: return []
    if not f_children: return b_children
    if not b_children: return f_children

    sum_bounds_F = sum(child.bound for child in f_children)
    sum_bounds_B = sum(child.bound for child in b_children)

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
        iter_process_start_time = time.time()
        
        root_node = create_root_node(num_machines, proc_times_jm, all_job_indices)
        current_level_nodes = [root_node]
        
        current_UB_for_pruning = overall_best_makespan 

        for level in range(num_jobs):
            if not current_level_nodes:
                break
            
            next_level_candidates_partial = []
            
            for parent_c_node in current_level_nodes:
                children_nodes = generate_bi_directional_children(parent_c_node, current_UB_for_pruning,
                                                                  all_job_indices, num_jobs, num_machines, proc_times_jm)
                
                for child in children_nodes: 
                    if len(child.starting) + len(child.finishing) == num_jobs:
                        schedule_solution = concatenate_schedule(child.starting, child.finishing)
                        makespan_solution = calculate_makespan_of_complete_schedule(schedule_solution, num_machines, proc_times_jm)
                        
                        if makespan_solution < overall_best_makespan:
                            overall_best_makespan = makespan_solution
                            overall_best_schedule = schedule_solution
                            current_UB_for_pruning = overall_best_makespan
                    else: 
                        next_level_candidates_partial.append(child)

            if not next_level_candidates_partial:
                break 

            current_level_nodes = select_best_d_nodes(next_level_candidates_partial, D, current_UB_for_pruning, 
                                                      num_machines, proc_times_jm)
        
        current_total_execution_time = time.time() - start_overall_time
        iter_processing_duration = time.time() - iter_process_start_time

        print(f"--- IBS Iteration {ibs_iter_count + 1} (Beam Width D={D}) Completed ---")
        print(f"    Time for this iteration's processing: {iter_processing_duration:.4f} seconds")
        print(f"Best Makespan Found So Far: {overall_best_makespan if overall_best_makespan != float('inf') else 'N/A'}")
        printable_schedule = [j + 1 for j in overall_best_schedule] if overall_best_schedule else 'N/A'
        print(f"Best Schedule Found So Far (1-indexed jobs): {printable_schedule}")
        print(f"Total Execution Time Up To This Point: {current_total_execution_time:.4f} seconds")
        print("-" * 60)

        if time_limit_seconds is not None and current_total_execution_time > time_limit_seconds:
            print("Time limit reached. Stopping IBS.")
            break

        D = D * beam_width_factor
        
        if D > (num_jobs**2) * 2 and num_jobs > 10 and D > 10000: 
             print(f"Beam width D={D} is becoming very large. Stopping IBS to conserve memory.")
             break

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
        max_iters_param = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        time_limit_param = int(sys.argv[3]) if len(sys.argv) > 3 else 60
        init_D_param = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        D_factor_param = int(sys.argv[5]) if len(sys.argv) > 5 else 2

    try:
        print(f"Parsing input file: {file_path}")
        num_jobs, num_machines, proc_times_jm = parse_input(file_path)
        print(f"Problem: {num_jobs} Jobs, {num_machines} Machines.")
        
        iterative_beam_search(num_jobs, num_machines, proc_times_jm,
                              initial_beam_width=init_D_param, 
                              beam_width_factor=D_factor_param, 
                              max_ibs_iterations=max_iters_param,
                              time_limit_seconds=time_limit_param)

    except FileNotFoundError:
        print(f"Error: Input file '{file_path}' not found.")
    except ValueError as ve:
        print(f"Input or Value Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()