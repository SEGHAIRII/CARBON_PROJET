import subprocess
import re
import csv
import os
import sys

# Define Taillard instances and their true upper bounds
TAILLARD_INSTANCES = {
    "20_5_1.txt": 1278,
    "20_5_2.txt": 1359,
    "20_5_3.txt": 1081        ,
    "20_5_4.txt": 1293        ,
    "20_5_5.txt": 1236        ,
    "20_5_6.txt": 1195        ,
    "20_5_7.txt": 1239        ,
    "20_5_8.txt": 1206                ,
    "20_5_9.txt": 1230                ,
    "20_5_10.txt": 1108                ,
    
    # ------------------
    "20_20_1.txt": 2297                ,
    "20_20_2.txt": 2100,
    "20_20_3.txt": 2326        ,
    "20_20_4.txt": 2223        ,
    "20_20_5.txt": 2291,
    "20_20_6.txt": 2226,
    "20_20_7.txt": 2273        ,
    "20_20_8.txt": 2200        ,
    "20_20_9.txt": 2237        ,
    "20_20_10.txt": 2178        ,
#--------------------------------------------
    "50_10_1.txt": 3025,
    "50_10_2.txt": 2892,
    "50_10_3.txt": 2864        ,
    "50_10_4.txt": 3064        ,
    "50_10_5.txt": 2986        ,
    "50_10_6.txt": 3006        ,
    "50_10_7.txt": 3107        ,
    "50_10_8.txt": 3039        ,
    "50_10_9.txt": 2902        ,
    "50_10_10.txt": 3091        ,
    
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
}

# Path to the main script and data folder, relative to this script's location
MAIN_SCRIPT_FILENAME = "main.py"
DATA_FOLDER_NAME = "data"
RESULTS_CSV_FILENAME = "experiment_results.csv"

# Configuration for running main.py (optional, main.py defaults will be used if None)
# Example: MAX_ITERS_ARG = "5" # As a string
# Example: TIME_LIMIT_ARG = "30" # As a string
# Example: INITIAL_BEAM_WIDTH_ARG = "1"
# Example: BEAM_WIDTH_FACTOR_ARG = "2"
# Set to None to use defaults from main.py's __main__ block when only file_path is provided
MAX_ITERS_ARG = None 
TIME_LIMIT_ARG = None # e.g., "120" for 2 minutes per instance
INITIAL_BEAM_WIDTH_ARG = None
BEAM_WIDTH_FACTOR_ARG = None


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
            best_makespan = None # Or some indicator of parsing failure

    # Regex to find "Execution time: <value> seconds" (from the main block of main.py)
    time_match = re.search(r"Execution time: (\d+\.?\d+) seconds", output_str)
    if time_match:
        try:
            exec_time = float(time_match.group(1))
        except ValueError:
            print(f"Warning: Could not parse execution time value: {time_match.group(1)}")
            # exec_time remains None
    else:
        # Fallback: Regex to find "Total Execution Time: <value> seconds" (from iterative_beam_search function)
        time_match_ibs = re.search(r"Total Execution Time: (\d+\.?\d+) seconds", output_str)
        if time_match_ibs:
            try:
                exec_time = float(time_match_ibs.group(1))
            except ValueError:
                print(f"Warning: Could not parse IBS total execution time value: {time_match_ibs.group(1)}")
                # exec_time remains None
            
    if "No feasible solution was found." in output_str and best_makespan is None:
        # This confirms no solution was found, makespan should be treated as N/A
        pass

    return best_makespan, exec_time

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

    csv_header = ["Instance", "BestMakespan", "ExecTime_s", "TrueUpperBound", "RPD_percentage"]
    
    # Use sorted keys for consistent processing order (optional, but good practice)
    instance_filenames = sorted(TAILLARD_INSTANCES.keys())

    with open(results_csv_full_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)

        print(f"Running experiments. Results will be saved to '{results_csv_full_path}'")

        for instance_filename in instance_filenames:
            instance_file_path = os.path.join(data_folder_full_path, instance_filename)
            true_upper_bound = TAILLARD_INSTANCES[instance_filename]

            if not os.path.exists(instance_file_path):
                print(f"Warning: Instance file '{instance_file_path}' not found. Skipping.")
                writer.writerow([instance_filename, "File Not Found", "N/A", true_upper_bound, "N/A"])
                continue

            print(f"\nProcessing instance: {instance_filename}...")
            
            command = ["python3", main_script_full_path, instance_file_path]
            
            # Add optional arguments if they are defined
            if MAX_ITERS_ARG is not None:
                command.append(MAX_ITERS_ARG)
            if TIME_LIMIT_ARG is not None:
                command.append(TIME_LIMIT_ARG)
            if INITIAL_BEAM_WIDTH_ARG is not None:
                command.append(INITIAL_BEAM_WIDTH_ARG)
            if BEAM_WIDTH_FACTOR_ARG is not None:
                command.append(BEAM_WIDTH_FACTOR_ARG)

            try:
                # Increased timeout for potentially long-running instances
                process_timeout = 600000 # 10 minutes per instance
                if TIME_LIMIT_ARG:
                    try: # Add a small buffer to the specified time limit for overhead
                        process_timeout = int(TIME_LIMIT_ARG) + 60 
                    except ValueError:
                        pass

                process = subprocess.run(command, capture_output=True, text=True, timeout=process_timeout)
                
                stdout = process.stdout
                stderr = process.stderr

                # Uncomment to see full output for each run
                # print("--- STDOUT ---")
                # print(stdout)
                # if stderr:
                #     print("--- STDERR ---")
                #     print(stderr)
                # print("--------------")

                if process.returncode != 0:
                    print(f"Error running {instance_filename}. Return code: {process.returncode}")
                    print(f"Stderr: {stderr[:500]}...") # Print first 500 chars of stderr
                    writer.writerow([instance_filename, "Runtime Error", "N/A", true_upper_bound, "N/A"])
                    continue

                best_makespan, exec_time = parse_output(stdout)

                rpd_percentage_str = "N/A"
                best_makespan_str = "N/A"

                if best_makespan is not None and best_makespan != float('inf'):
                    best_makespan_str = f"{best_makespan:.0f}" # Assuming makespan is integer
                    rpd = ((best_makespan - true_upper_bound) / true_upper_bound) * 100
                    rpd_percentage_str = f"{rpd:.2f}"
                elif "No feasible solution was found." in stdout:
                     best_makespan_str = "No Solution"


                exec_time_str = f"{exec_time:.4f}" if exec_time is not None else "N/A"

                writer.writerow([instance_filename, best_makespan_str, exec_time_str, true_upper_bound, rpd_percentage_str])
                print(f"Logged: {instance_filename}, Makespan: {best_makespan_str}, Time: {exec_time_str}s, RPD: {rpd_percentage_str}%")

            except subprocess.TimeoutExpired:
                print(f"Timeout running {instance_filename} after {process_timeout} seconds.")
                writer.writerow([instance_filename, "Timeout", "N/A", true_upper_bound, "N/A"])
            except Exception as e:
                print(f"An unexpected error occurred while processing {instance_filename}: {e}")
                writer.writerow([instance_filename, "Script Error", "N/A", true_upper_bound, "N/A"])
        
        print(f"\nAll experiments completed. Results saved to '{results_csv_full_path}'")

if __name__ == "__main__":
    main()