# Deploying the scheduling dashboard (free)

The recommended route is **Streamlit Community Cloud**: it is free, needs no
Docker file, no server setup and no credit card. The only requirement is a
GitHub repository you control.

## What was prepared in this folder

| File | Why |
|------|-----|
| `requirements.txt` | Python packages, already pinned to the versions used locally |
| `runtime.txt` | Tells Community Cloud to use Python 3.12 (same as local) |
| `.streamlit/config.toml` | Server/theme settings and a 10 MB upload limit |
| `.gitignore` | Keeps `venv/`, `logs/`, `archive/` and caches out of the repo |
| `models/*.zip` | The four trained models the app loads — **must be uploaded** |

The app entrypoint is `dashboard/app.py`. All paths are resolved relative to
the project root, so it works unchanged on Community Cloud.

## Push the project to your GitHub repo

From this folder in PowerShell (replace the URL with yours):

```powershell
git init -b main
git add .
git commit -m "Deploy scheduling dashboard"
git remote add origin https://github.com/<YOUR-USER>/<YOUR-REPO>.git
git push -u origin main
```

If the repo already exists locally, just copy these files into it, commit and
push instead. Make sure `models/` is included and `venv/` is not.

## Deploy in the browser (about 3 clicks)

1. Go to <https://share.streamlit.io> and sign in with **GitHub**.
2. Click **Create app** → **Yup, I have an app**.
3. Fill in:
   - **Repository:** `<YOUR-USER>/<YOUR-REPO>`
   - **Branch:** `main`
   - **Main file path:** `dashboard/app.py`
4. Open **Advanced settings**, select Python **3.12**, leave Secrets empty,
   then click **Save**.
5. Click **Deploy**.

The first build installs PyTorch and the other packages, so it can take a few
minutes. Afterwards you get a public URL like
`https://<your-repo>.streamlit.app`. Every later `git push` redeploys it
automatically. Apps with no traffic for 12 hours go to sleep; the next visit
wakes them up.

## If the build fails

- **Model files missing:** confirm all four `.zip` files in `models/` are in
  the repo (GitHub shows them in the web view).
- **Package error:** check the build log in the Streamlit app dashboard; the
  versions in `requirements.txt` match this machine's Python 3.12 venv.
- **Private repo not visible:** authorize Streamlit for private repos in
  Community Cloud → Settings → Linked accounts, or make the repo public.
- **App wakes slowly:** normal on the free tier because PyTorch has to load.

## Alternative: run it on your own machine / LAN

```powershell
venv\Scripts\python.exe -m streamlit run dashboard\app.py --server.address 0.0.0.0
```

Open `http://localhost:8501` locally, or
`http://<your-LAN-IP>:8501` from another device on the same Wi-Fi (Windows
Firewall may ask for permission the first time).
