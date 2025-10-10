import subprocess

scripts = [
    "lead_ratios.py",
    "hackathon_train.py",
    "model_eval.py",
    "portfolio_eval.py"
]

for script in scripts:
    subprocess.run(["python", script], check=True)