# OmegaX Trading Bot - Railway Deploy

## Prereqs
- Railway account + CLI (optional): https://railway.app
- Python 3.11 locally if you want to run tests before deploying

## Files
- requirements.txt
- Procfile
- runtime.txt
- .env.example
- app.py (your main file containing `if __name__ == "__main__": ...`)

## Deploy

1) Push your code to GitHub.

2) Create a new Railway project:
   - New Project -> Deploy from GitHub -> select your repo.

3) Configure service:
   - Railway will detect Python from requirements.txt and use Nixpacks.
   - It will honor Procfile:
     - web: python -u app.py
   - Expose port: the app reads PORT from env (Railway sets this automatically).

4) Environment variables:
   - Add your secrets in Railway (Settings -> Variables).
   - You can copy from `.env.example`. If WEB_UI_PASSWORD is unset, the app will auto-generate one (it prints to logs).

5) Deploy:
   - Railway will build and boot your service with Hypercorn via your script’s `main()`.

6) Access:
   - Use the Railway-generated domain.
   - Endpoints:
     - /metrics
     - /api/status
     - /performance