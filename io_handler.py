# io_handler.py
import numpy as np

def parse_input(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        
        if not lines:
            raise ValueError("Input file is empty.")

        first_line = lines[0].split()
        if len(first_line) != 2:
            raise ValueError("First line must contain number_jobs and number_machines.")
        
        try:
            num_jobs = int(first_line[0])
            num_machines = int(first_line[1])
        except ValueError:
            raise ValueError("Number_jobs and number_machines must be integers.")

        if num_jobs <= 0 or num_machines <= 0:
            raise ValueError("Number of jobs and machines must be positive.")

        expected_lines_for_matrix = num_machines + 1
        if len(lines) < expected_lines_for_matrix:
            raise ValueError(f"Input file error: Expected {expected_lines_for_matrix} lines for header and matrix, found {len(lines)}.")

        proc_times_machine_job = []
        # Lines from 1 to num_machines (inclusive, 0-indexed for lines list)
        for i in range(1, num_machines + 1):
            row_values_str = lines[i].split()
            if len(row_values_str) != num_jobs:
                raise ValueError(f"Input file error: Machine {i} (1-indexed) data has {len(row_values_str)} job times, expected {num_jobs}.")
            try:
                proc_times_machine_job.append(list(map(int, row_values_str)))
            except ValueError:
                raise ValueError(f"Processing times must be integers. Error in data for machine {i} (1-indexed).")
        
        proc_times_matrix_mj = np.array(proc_times_machine_job, dtype=int)
        # Transpose to get P[job_idx, machine_idx]
        proc_times_matrix_jm = proc_times_matrix_mj.T 
        
    return num_jobs, num_machines, proc_times_matrix_jm

# Example of how you might add output formatting here if desired:
# def print_iteration_results(iter_num, beam_width, iter_time, total_time, best_makespan, best_schedule):
#     print(f"--- IBS Iteration {iter_num} (Beam Width D={beam_width}) Completed ---")
#     print(f"    Time for this iteration's processing: {iter_time:.4f} seconds")
#     print(f"Best Makespan Found So Far: {best_makespan if best_makespan != float('inf') else 'N/A'}")
#     printable_schedule = [j + 1 for j in best_schedule] if best_schedule else 'N/A'
#     print(f"Best Schedule Found So Far (1-indexed jobs): {printable_schedule}")
#     print(f"Total Execution Time Up To This Point: {total_time:.4f} seconds")
#     print("-" * 60)

# def print_final_results(best_makespan, best_schedule, total_time):
#     print("\n--- Final Result After All Iterations ---")
#     print(f"Best Makespan: {best_makespan if best_makespan != float('inf') else 'N/A'}")
#     printable_schedule_final = [j + 1 for j in best_schedule] if best_schedule else 'N/A'
#     print(f"Best Schedule (1-indexed jobs): {printable_schedule_final}")
#     print(f"Total Execution Time: {total_time:.4f} seconds")