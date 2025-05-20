import subprocess
import re
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Define Taillard instances and their true upper bounds
TAILLARD_INSTANCES = {
    "20_5_1.txt": 1278,
    "20_5_2.txt": 1359,
    "20_5_3.txt": 1081,
    "20_5_4.txt": 1293,
    "20_5_5.txt": 1236,
    "20_5_6.txt": 1195,
    "20_5_7.txt": 1239,
    "20_5_8.txt": 1206,
    "20_5_9.txt": 1230,
    "20_5_10.txt": 1108,
    
    # ------------------
    "20_20_1.txt": 2297,
    "20_20_2.txt": 2100,
    "20_20_3.txt": 2326,
    "20_20_4.txt": 2223,
    "20_20_5.txt": 2291,
    "20_20_6.txt": 2226,
    "20_20_7.txt": 2273,
    "20_20_8.txt": 2200,
    "20_20_9.txt": 2237,
    "20_20_10.txt": 2178,
    #--------------------------------------------
    "50_10_1.txt": 3025,
    "50_10_2.txt": 2892,
    "50_10_3.txt": 2864,
    "50_10_4.txt": 3064,
    "50_10_5.txt": 2986,
    "50_10_6.txt": 3006,
    "50_10_7.txt": 3107,
    "50_10_8.txt": 3039,
    "50_10_9.txt": 2902,
    "50_10_10.txt": 3091,
    
    #-----------------------------------------
    "50_20_1.txt": 3875,
    "50_20_2.txt": 3715,
    "50_20_3.txt": 3668,
    "50_20_4.txt": 3752,
    "50_20_5.txt": 3635,
    "50_20_6.txt": 3698,
    "50_20_7.txt": 3716,
    "50_20_8.txt": 3709,
    "50_20_9.txt": 3765,
    "50_20_10.txt": 3777,
    
    #========================================
    # "100_5_1.txt": 5493,
    # "100_5_2.txt": 5268,
    # "100_5_3.txt": 5175,
    # "100_5_4.txt": 5014,
    # "100_5_5.txt": 5250,
    # "100_5_6.txt": 5135,
    # "100_5_7.txt": 5246,
    # "100_5_8.txt": 5106,
    # "100_5_9.txt": 5454,
    # "100_5_10.txt": 5328,
}

# Path to the main script and data folder, relative to this script's location
MAIN_SCRIPT_FILENAME = "main.py"
DATA_FOLDER_NAME = "data"
RESULTS_CSV_FILENAME = "experiment_results.csv"

# Configuration for running main.py (optional, main.py defaults will be used if None)
MAX_ITERS_ARG = None 
TIME_LIMIT_ARG = 3600 # e.g., "120" for 2 minutes per instance
INITIAL_BEAM_WIDTH_ARG = None
BEAM_WIDTH_FACTOR_ARG = None

# Maximum number of concurrent processes
# By default, use 75% of available cores to avoid overloading the system
MAX_WORKERS = max(1, int(multiprocessing.cpu_count() * 0.75))


def parse_output(output_str):
    """Parses the stdout of main.py to find makespan and execution time."""
    best_makespan = None
    exec_time = None

    # Regex to find "Best makespan found: <value>"
    makespan_match = re.search(r"Best makespan found: (\d+\.?\d*)", output_str)
    if makespan_match:
        try:
            best_makespan = float(makespan_match.group(1))
        except ValueError:
            print(f"Warning: Could not parse makespan value: {makespan_match.group(1)}")
            best_makespan = None

    # Regex to find "Execution time: <value> seconds" (from the main block of main.py)
    time_match = re.search(r"Execution time: (\d+\.?\d+) seconds", output_str)
    if time_match:
        try:
            exec_time = float(time_match.group(1))
        except ValueError:
            print(f"Warning: Could not parse execution time value: {time_match.group(1)}")
    else:
        # Fallback: Regex to find "Total Execution Time: <value> seconds" (from iterative_beam_search function)
        time_match_ibs = re.search(r"Total Execution Time: (\d+\.?\d+) seconds", output_str)
        if time_match_ibs:
            try:
                exec_time = float(time_match_ibs.group(1))
            except ValueError:
                print(f"Warning: Could not parse IBS total execution time value: {time_match_ibs.group(1)}")
            
    return best_makespan, exec_time


def process_instance(instance_data):
    """Process a single instance - can be executed in parallel."""
    instance_filename, instance_file_path, true_upper_bound, script_path, args = instance_data
    
    command = ["python3", script_path, instance_file_path]
    
    # Add optional arguments if they are defined
    for arg in args:
        if arg is not None:
            command.append(str(arg))
    
    result = {
        "instance": instance_filename,
        "true_upper_bound": true_upper_bound,
        "best_makespan": "N/A",
        "exec_time": "N/A",
        "rpd_percentage": "N/A",
        "status": "Success"
    }
    
    try:
        # Determine timeout
        process_timeout = 3600  # 10 minutes per instance by default
        if args[1]:  # TIME_LIMIT_ARG
            try:
                process_timeout = int(args[1]) + 60  # Add 60s buffer
            except ValueError:
                pass
        
        # Run the subprocess
        process = subprocess.run(command, capture_output=True, text=True, timeout=process_timeout)
        
        stdout = process.stdout
        stderr = process.stderr
        
        if process.returncode != 0:
            result["status"] = "Runtime Error"
            result["error_details"] = stderr[:500] if stderr else "Unknown error"
            return result
        
        best_makespan, exec_time = parse_output(stdout)
        
        if best_makespan is not None and best_makespan != float('inf'):
            result["best_makespan"] = f"{best_makespan:.0f}"  # Assuming makespan is integer
            rpd = ((best_makespan - true_upper_bound) / true_upper_bound) * 100
            result["rpd_percentage"] = f"{rpd:.2f}"
        elif "No feasible solution was found." in stdout:
            result["best_makespan"] = "No Solution"
        
        if exec_time is not None:
            result["exec_time"] = f"{exec_time:.4f}"
            
    except subprocess.TimeoutExpired:
        result["status"] = "Timeout"
    except Exception as e:
        result["status"] = "Script Error"
        result["error_details"] = str(e)
    
    return result


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_script_full_path = os.path.join(script_dir, MAIN_SCRIPT_FILENAME)
    data_folder_full_path = os.path.join(script_dir, DATA_FOLDER_NAME)
    results_csv_full_path = os.path.join(script_dir, RESULTS_CSV_FILENAME)

    if not os.path.exists(main_script_full_path):
        print(f"Error: Main script '{main_script_full_path}' not found.")
        sys.exit(1)

    if not os.path.isdir(data_folder_full_path):
        print(f"Error: Data folder '{data_folder_full_path}' not found.")
        sys.exit(1)

    # Sort instance filenames for consistent processing order
    instance_filenames = sorted(TAILLARD_INSTANCES.keys())
    
    # Prepare jobs for parallel execution
    jobs = []
    for instance_filename in instance_filenames:
        instance_file_path = os.path.join(data_folder_full_path, instance_filename)
        true_upper_bound = TAILLARD_INSTANCES[instance_filename]
        
        if not os.path.exists(instance_file_path):
            print(f"Warning: Instance file '{instance_file_path}' not found. Skipping.")
            continue
            
        # Pack job arguments
        args = [MAX_ITERS_ARG, TIME_LIMIT_ARG, INITIAL_BEAM_WIDTH_ARG, BEAM_WIDTH_FACTOR_ARG]
        job_data = (instance_filename, instance_file_path, true_upper_bound, main_script_full_path, args)
        jobs.append(job_data)
    
    print(f"Starting parallel experiments with {MAX_WORKERS} workers...")
    print(f"Processing {len(jobs)} instances...")
    
    start_time = time.time()
    results = []
    
    # Use a ProcessPoolExecutor to run jobs in parallel
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all jobs
        future_to_instance = {executor.submit(process_instance, job): job[0] for job in jobs}
        
        # Process results as they complete
        for future in as_completed(future_to_instance):
            instance_name = future_to_instance[future]
            try:
                result = future.result()
                results.append(result)
                
                # Print progress report
                status_marker = "✓" if result["status"] == "Success" else "✗"
                print(f"[{status_marker}] {instance_name}: {result['best_makespan']}, Time: {result['exec_time']}, RPD: {result['rpd_percentage']}%")
                
                if result["status"] != "Success":
                    print(f"  └─ Error: {result['status']}")
                    if "error_details" in result:
                        print(f"     └─ {result['error_details']}")
                
            except Exception as exc:
                print(f"[!] {instance_name} generated an exception: {exc}")
                results.append({
                    "instance": instance_name,
                    "true_upper_bound": TAILLARD_INSTANCES[instance_name],
                    "best_makespan": "Error",
                    "exec_time": "N/A",
                    "rpd_percentage": "N/A",
                    "status": "Exception",
                    "error_details": str(exc)
                })
    
    total_elapsed_time = time.time() - start_time
    print(f"\nAll experiments completed in {total_elapsed_time:.2f} seconds")
    
    # Sort results by instance name for consistent output
    results.sort(key=lambda x: x["instance"])
    
    # Write results to CSV
    csv_header = ["Instance", "BestMakespan", "ExecTime_s", "TrueUpperBound", "RPD_percentage", "Status"]
    
    with open(results_csv_full_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)
        
        for result in results:
            writer.writerow([
                result["instance"],
                result["best_makespan"],
                result["exec_time"],
                result["true_upper_bound"],
                result["rpd_percentage"],
                result["status"]
            ])
    
    print(f"Results saved to '{results_csv_full_path}'")
    
    # Calculate and display summary statistics
    success_count = sum(1 for r in results if r["status"] == "Success")
    timeout_count = sum(1 for r in results if r["status"] == "Timeout")
    error_count = len(results) - success_count - timeout_count
    
    print("\nSummary Statistics:")
    print(f"  Total instances: {len(results)}")
    print(f"  Successful runs: {success_count} ({success_count/len(results)*100:.1f}%)")
    print(f"  Timeouts: {timeout_count} ({timeout_count/len(results)*100:.1f}%)")
    print(f"  Errors: {error_count} ({error_count/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()