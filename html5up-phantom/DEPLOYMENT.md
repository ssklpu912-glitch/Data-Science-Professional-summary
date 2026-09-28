# Running LabLogbook

## Locally (Windows, no extra services)

```
pip install -r requirements.txt
pip install --no-deps -r requirements-nodeps.txt
python app.py
```

With no `DATABASE_URL` or `REDIS_URL` set, the app uses a single-machine setup:
SQLite at `instance/users.db`, sessions on disk in `instance/sessions/`, rate limits in
memory, and DFT runs inline in the request. PySCF has no Windows build, so DFT itself
only works on Linux/macOS; the page says so instead of erroring.

## Production

Two processes from the same code, plus Postgres and Redis:

| Process | Command | Notes |
|---|---|---|
| web | `python app.py` with `PRODUCTION=1` | waitress; applies pending migrations on start |
| compute worker | `rq worker compute --url $REDIS_URL` | runs DFT jobs; run N of them to allow N concurrent DFT runs |

### Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `PRODUCTION=1` | yes | waitress instead of the dev server; disables `DEV_AUTOLOGIN` |
| `FLASK_SECRET_KEY` | yes | long random string; keep it stable or everyone is signed out |
| `DATABASE_URL` | yes | e.g. `postgresql://user:pass@db:5432/lablogbook` (`postgres://` also accepted) |
| `REDIS_URL` | yes | e.g. `redis://redis:6379/0` — sessions, rate limits, compute queue |
| `BEHIND_PROXY=1` | behind Caddy/Nginx | trust one proxy hop's `X-Forwarded-*` (real client IPs, https links) |
| `SESSION_COOKIE_SECURE=1` | once HTTPS works | never before — sign-in silently breaks over plain HTTP |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `MAIL_FROM` | for password reset | transactional email provider |
| `PORT` | no | default 5000 |
| `WAITRESS_THREADS` | no | default 8 |

Uploads live in `static/uploads/` — mount persistent storage there.

## Changing the database schema

Never delete the database to pick up a model change. Instead:

1. Edit the model in `app.py`.
2. Generate a migration **against a fresh database**, so leftovers in an old local
   database don't leak into it:
   ```
   DATABASE_URL=sqlite:///C:/temp/fresh.db flask --app app db upgrade
   DATABASE_URL=sqlite:///C:/temp/fresh.db flask --app app db migrate -m "add X to Y"
   ```
3. Read the generated file in `migrations/versions/`, fix anything autogenerate got
   wrong (renames show up as drop + add, which loses data), and commit it.
4. It's applied automatically the next time the web process starts.

Databases created before migrations existed are detected on first start, patched,
and stamped at the baseline revision (`BASELINE_REVISION` in `app.py`).
