# How to run this project

This was tested on Windows with Python 3.12.10. If you're on Mac or
Linux, there are notes near the bottom.

## 1. What to hand over

You can send the whole project folder, but leave out a few things first.
They're either large or tied to this computer:

- `venv/`: a virtual environment won't work on another machine
- any `__pycache__/` folders
- `logs/` if you don't need the TensorBoard training curves (about 9 MB)
- `archive/` if you don't need the old V1–V6 experiments (about 7 MB)

Keep the four trained models in `models/`. The app needs them:

- `models/simple_dqn.zip`
- `models/simple_ppo.zip`
- `models/jobshop_dqn.zip`
- `models/jobshop_ppo.zip`

Without `venv/` the folder is about 54 MB. If you also skip `logs/` and
`archive/`, it's about 38 MB.

## 2. Install Python

Get Python 3.12 from <https://www.python.org/downloads/>.

On the first installer screen, tick **"Add python.exe to PATH"**. Then
check that it worked:

```powershell
python --version
```

If Windows can't find `python`, try the launcher:

```powershell
py -3.12 --version
```

## 3. Open a terminal in the project folder

Open the `Production-Optimization-RL` folder in File Explorer, click the
address bar, type `powershell`, and press Enter. Or open a terminal and
run:

```powershell
cd "C:\path\to\Production-Optimization-RL"
```

## 4. Create the virtual environment and install the packages

```powershell
python -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt
```

This downloads PyTorch and the other packages, so give it a few minutes.
On a slow connection, bump the timeout:

```powershell
venv\Scripts\python.exe -m pip install --default-timeout=1000 -r requirements.txt
```

If `python` wasn't on PATH in step 2, use `py -3.12` instead:

```powershell
py -3.12 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 5. Run the dashboard

```powershell
venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

The browser should open at <http://localhost:8501>. If it doesn't, copy
that address into your browser.

If the port is already taken, pick another one:

```powershell
venv\Scripts\python.exe -m streamlit run dashboard\app.py --server.port 8502
```

Press `Ctrl+C` in the terminal when you want to stop the app.

## 6. Use the app

- **📥 Your data** is the main page. Choose manual entry or an Excel
  upload, set the number of identical machines (1–5), pick the schedulers
  you want (FCFS, SJF, Priority, DQN, PPO), and the KPI cards, Gantt chart
  and comparison table appear below.
- **🏭 Job-shop demo** shows the original job-shop study and its models.

There's a sample upload at `data/example_jobs.csv`. You can also download
an Excel template from the sidebar.

Remember that priority `1` means the highest priority.

## Optional commands

Run the automatic checks:

```powershell
venv\Scripts\python.exe src\test_simple_tool.py
venv\Scripts\python.exe src\test_jobshop_environment.py
```

Rebuild the processed production data:

```powershell
venv\Scripts\python.exe src\build_processed_data.py
```

Re-run the 100-instance comparisons. These take a while:

```powershell
venv\Scripts\python.exe evaluation\evaluate_simple.py
venv\Scripts\python.exe evaluation\evaluate_jobshop.py
```

Retrain the simple-mode agents if you want to. The rule-based schedulers
work without them:

```powershell
venv\Scripts\python.exe agents\train_dqn_simple.py
venv\Scripts\python.exe agents\train_ppo_simple.py
```

## Mac and Linux

The steps are the same, but the commands look a little different:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## If something goes wrong

| Problem | What to try |
|---------|-------------|
| `python` is not recognized | Reinstall Python and tick "Add to PATH", or use `py -3.12` |
| `streamlit` is not recognized | Run it through Python: `venv\Scripts\python.exe -m streamlit run dashboard\app.py` |
| `Activate.ps1 cannot be loaded` | Skip activation and call `venv\Scripts\python.exe` directly |
| The port is already in use | Add `--server.port 8502` |
| Excel upload fails with an `openpyxl` error | Run `venv\Scripts\python.exe -m pip install openpyxl` |
| The dashboard says a model file is missing | Check that all four `.zip` files are in `models/` |
