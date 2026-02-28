import os
import subprocess
import pandas as pd
import time
import sys

# Use the current python executable (should be the venv one)
PYTHON_EXE = sys.executable
RENDERS = ["render1", "render2", "render3", "render4", "render5"]

def run_benchmark():
    results = []
    print("="*40)
    print("      RADAR SYSTEM BENCHMARK        ")
    print("="*40)
    
    for render in RENDERS:
        print(f"[TESTING] {render}...", end=" ", flush=True)
        start_time = time.time()
        # Headless mode for speed
        cmd = [PYTHON_EXE, "visualization_v2.py", "--render", render, "--thresh", "2", "--depth", "7", "--headless"]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        duration = time.time() - start_time
        
        rmse = 999.0
        found = False
        for line in stdout.splitlines():
            if "BENCHMARK_RESULT:" in line:
                try:
                    rmse = float(line.split("BENCHMARK_RESULT:")[-1].strip())
                    found = True
                except: pass
        
        results.append({
            "Render": render,
            "Duration (s)": round(duration, 2),
            "RMSE (m)": rmse,
            "Status": "PASS" if (rmse < 1.0 and found) else "FAIL"
        })
        print(f"Done. RMSE: {rmse:.4f}")
        
    df = pd.DataFrame(results)
    print("\n" + "="*40)
    print(df.to_string(index=False))
    print("="*40)
    df.to_csv("benchmark_results.csv", index=False)

if __name__ == "__main__":
    run_benchmark()
