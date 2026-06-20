import json
import subprocess
import os

repo_dir = "/Users/gunnarhostetler/Documents/GitHub/PlaudBlender_wt"
json_path = "/Users/gunnarhostetler/.gemini/antigravity-ide/brain/4e97d00f-bb33-4258-8864-f89291e20c1a/.system_generated/steps/12/output.txt"

with open(json_path) as f:
    prs = json.load(f)

results = []

for pr in prs:
    branch = pr['head']['ref']
    num = pr['number']
    
    # checkout branch
    print(f"\n--- Testing PR {num} ({branch}) ---")
    subprocess.run(["git", "reset", "--hard"], cwd=repo_dir, capture_output=True)
    res_co = subprocess.run(["git", "checkout", f"origin/{branch}"], cwd=repo_dir, capture_output=True)
    if res_co.returncode != 0:
        print(f"Failed to checkout {branch}")
        results.append({'pr': num, 'branch': branch, 'pytest': 'skip', 'ruff': 'skip', 'pyright': 'skip'})
        continue
    
    # Run tests
    # pytest
    res_pytest = subprocess.run(["pytest", "tests/"], cwd=repo_dir, capture_output=True, text=True)
    pytest_status = "PASS" if res_pytest.returncode == 0 else "FAIL"
    
    # ruff
    res_ruff = subprocess.run(["ruff", "check", "."], cwd=repo_dir, capture_output=True, text=True)
    ruff_status = "PASS" if res_ruff.returncode == 0 else "FAIL"
    
    # pyright
    res_pyright = subprocess.run(["pyright"], cwd=repo_dir, capture_output=True, text=True)
    pyright_status = "PASS" if res_pyright.returncode == 0 else "FAIL"
    
    print(f"pytest: {pytest_status}, ruff: {ruff_status}, pyright: {pyright_status}")
    results.append({
        'pr': num,
        'branch': branch,
        'pytest': pytest_status,
        'ruff': ruff_status,
        'pyright': pyright_status
    })

with open("pr_test_results.json", "w") as out:
    json.dump(results, out, indent=2)

print("\nFinished all PRs.")
