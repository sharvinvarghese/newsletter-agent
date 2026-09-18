# 1) import + compile check done.
# 2) verify agent.graph and schemas import cleanly in this venv.
# 3) verify app.py imports with streamlit shimmed.
import subprocess

venv = r"C:\Users\sharv\.venvs\newsletter-agent\Scripts\python.exe"
root = r"c:\Users\sharv\OneDrive\Desktop\homework\newsletter-agent"
checks = [
    [venv, "-c", "import sys; sys.path.insert(0, r'"+root+"'); import agent.graph as g; print('graph import OK', g.run_newsletter_state.__name__)"],
    [venv, "-c", "import sys; sys.path.insert(0, r'"+root+"'); import schemas; print('schemas import OK', schemas.AgentResult.__name__)"],
    [venv, "-c", "import sys; sys.path.insert(0, r'"+root+"'); import streamlit; print('streamlit', streamlit.__version__)"],
]
for c in checks:
    print(c[2][:60], "...")
    r = subprocess.run(c, capture_output=True, text=True, check=False)
    print(" STDOUT:", r.stdout.strip()[:300])
    print(" STDERR:", r.stderr.strip()[:300])
    print(" rc:", r.returncode)
    print("---")