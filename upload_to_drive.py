"""
Reworld Haverhill — Google Drive Live Data Upload
==================================================
Uploads (or refreshes) all signal CSVs as Google Sheets inside the customer
data folder, plus the latest static HTML snapshot.

First run
---------
You need a Google OAuth credentials file. Get one in ~30 seconds:
  1. Go to https://console.cloud.google.com/apis/credentials
  2. Create an OAuth 2.0 Client ID → Desktop app
  3. Download the JSON → save as TOKEN_DIR/credentials.json  (path below)

Then run:
  python upload_to_drive.py

A browser URL will be printed. Open it, authorise, paste the code back.
The token is saved to TOKEN_DIR/token.json — future runs are fully automatic.

Scheduled / automated runs (no browser needed after first run)
--------------------------------------------------------------
  python upload_to_drive.py --no-browser

Or add to cron / use the /schedule skill in Claude Code to run on an interval.

What it does each run
---------------------
1. Loads/refreshes OAuth token
2. For each CSV: if a Sheet with that name already exists in the folder,
   clears and rewrites it; otherwise creates a new Sheet.
3. Generates a fresh static HTML snapshot and uploads it to the folder.
4. Prints shareable links for each file.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── config ─────────────────────────────────────────────────────────────────────
DASH_DIR     = Path('/home/shared/kumar/library/fastapi-jupyter-dashboard')
TOKEN_DIR    = DASH_DIR / '.gdrive_auth'        # where credentials.json lives
FOLDER_ID    = '1OVLgTTOuJgQg_ffOZKQTDgmF2p6FexP-'  # "Reworld Haverhill — Live Dashboard Data"

CSVS = {
    'Chute A — Signals':         DASH_DIR / 'data/achute_signals/signals.csv',
    'Chute B — Signals':         DASH_DIR / 'data/chuteb_signals/signals.csv',
    'West Pit — Signals':        DASH_DIR / 'data/eda_cache/west-pit_signals.csv',
    'Tipping Floor 1 — Signals': DASH_DIR / 'data/eda_cache/tipping1_signals.csv',
    'Tipping Floor 2 — Signals': DASH_DIR / 'data/eda_cache/tipping2_signals.csv',
    'Tipping Floor 3 — Signals': DASH_DIR / 'data/eda_cache/tipping3_signals.csv',
}

SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/spreadsheets',
]

# ── auth ───────────────────────────────────────────────────────────────────────
def get_credentials(no_browser: bool = False):
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request

    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    token_path = TOKEN_DIR / 'token.json'
    creds_path = TOKEN_DIR / 'credentials.json'

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not creds_path.exists():
                print(f'\nERROR: credentials.json not found at {creds_path}')
                print('\nTo get one:')
                print('  1. Go to https://console.cloud.google.com/apis/credentials')
                print('  2. Create OAuth 2.0 Client ID → Desktop app')
                print(f'  3. Download JSON → save as {creds_path}')
                sys.exit(1)
            if no_browser:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(creds_path), SCOPES,
                    redirect_uri='urn:ietf:wg:oauth:2.0:oob'
                )
                auth_url, _ = flow.authorization_url(prompt='consent')
                print(f'\nOpen this URL to authorise:\n\n  {auth_url}\n')
                code = input('Paste the authorisation code here: ').strip()
                flow.fetch_token(code=code)
                creds = flow.credentials
            else:
                flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
                creds = flow.run_local_server(port=0)

        token_path.write_text(creds.to_json())
        print(f'Token saved to {token_path}')

    return creds


# ── Drive helpers ──────────────────────────────────────────────────────────────
def _drive_service(creds):
    from googleapiclient.discovery import build
    return build('drive', 'v3', credentials=creds)


def _sheets_service(creds):
    from googleapiclient.discovery import build
    return build('sheets', 'v4', credentials=creds)


def _find_file_in_folder(drive, name: str, folder_id: str) -> str | None:
    """Return file ID if a file with this name exists in the folder."""
    q = (f"name = '{name}' and '{folder_id}' in parents "
         f"and trashed = false")
    r = drive.files().list(q=q, fields='files(id,name)').execute()
    files = r.get('files', [])
    return files[0]['id'] if files else None


def _set_public_read(drive, file_id: str):
    """Make a file viewable by anyone with the link."""
    drive.permissions().create(
        fileId=file_id,
        body={'type': 'anyone', 'role': 'reader'},
    ).execute()


# ── Sheet upload / refresh ─────────────────────────────────────────────────────
def upload_csv_as_sheet(drive, sheets, creds, name: str, csv_path: Path, folder_id: str):
    import csv as _csv
    from googleapiclient.http import MediaFileUpload

    existing_id = _find_file_in_folder(drive, name, folder_id)

    if existing_id:
        # Clear existing sheet and rewrite data
        print(f'  refreshing "{name}"…', end=' ', flush=True)
        sheets.spreadsheets().values().clear(
            spreadsheetId=existing_id, range='A1:ZZ100000'
        ).execute()
        with open(csv_path, newline='') as f:
            rows = list(_csv.reader(f))
        sheets.spreadsheets().values().update(
            spreadsheetId=existing_id,
            range='A1',
            valueInputOption='USER_ENTERED',
            body={'values': rows},
        ).execute()
        file_id = existing_id
        print('updated')
    else:
        # Create new Sheet by uploading the CSV (Drive auto-converts)
        print(f'  creating "{name}"…', end=' ', flush=True)
        meta = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.spreadsheet',
            'parents': [folder_id],
        }
        media = MediaFileUpload(str(csv_path), mimetype='text/csv')
        f = drive.files().create(body=meta, media_body=media, fields='id').execute()
        file_id = f['id']
        print('created')

    _set_public_read(drive, file_id)
    return file_id


# ── HTML snapshot upload ───────────────────────────────────────────────────────
def upload_html_snapshot(drive, folder_id: str) -> str:
    from googleapiclient.http import MediaFileUpload

    print('  generating HTML snapshot…', end=' ', flush=True)
    result = subprocess.run(
        [sys.executable, str(DASH_DIR / 'export_static.py')],
        capture_output=True, text=True, cwd=str(DASH_DIR),
    )
    if result.returncode != 0:
        print(f'FAILED:\n{result.stderr}')
        return ''

    # Find the newly created file
    snapshots = sorted(DASH_DIR.glob('reworld_haverhill_*.html'), reverse=True)
    if not snapshots:
        print('no snapshot file found')
        return ''
    snap = snapshots[0]
    print(f'done ({snap.name})')

    html_name = 'Reworld Haverhill — Dashboard Snapshot'
    existing_id = _find_file_in_folder(drive, html_name, folder_id)

    print(f'  uploading HTML snapshot…', end=' ', flush=True)
    meta = {
        'name': html_name,
        'parents': [folder_id],
        'mimeType': 'text/html',
    }
    media = MediaFileUpload(str(snap), mimetype='text/html', resumable=True)

    if existing_id:
        drive.files().update(
            fileId=existing_id, media_body=media
        ).execute()
        file_id = existing_id
    else:
        f = drive.files().create(
            body=meta, media_body=media, fields='id'
        ).execute()
        file_id = f['id']

    _set_public_read(drive, file_id)
    print('done')
    return file_id


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-browser', action='store_true',
                        help='Print auth URL instead of opening browser (for headless envs)')
    parser.add_argument('--skip-html', action='store_true',
                        help='Skip HTML snapshot generation (faster, CSVs only)')
    args = parser.parse_args()

    print('Authenticating with Google…')
    creds  = get_credentials(no_browser=args.no_browser)
    drive  = _drive_service(creds)
    sheets = _sheets_service(creds)
    print('Authenticated.\n')

    file_ids = {}

    print('Uploading signal data as Google Sheets:')
    for name, csv_path in CSVS.items():
        if not csv_path.exists():
            print(f'  SKIP "{name}" — file not found: {csv_path}')
            continue
        fid = upload_csv_as_sheet(drive, sheets, creds, name, csv_path, FOLDER_ID)
        file_ids[name] = fid

    if not args.skip_html:
        print('\nUploading HTML dashboard snapshot:')
        html_id = upload_html_snapshot(drive, FOLDER_ID)
        if html_id:
            file_ids['Dashboard Snapshot (HTML)'] = html_id

    # Print share links
    print('\n── Share links ──────────────────────────────────────────────────────')
    for name, fid in file_ids.items():
        print(f'  {name}')
        print(f'    https://drive.google.com/file/d/{fid}/view')
    print(f'\n  Folder: https://drive.google.com/drive/folders/{FOLDER_ID}')

    # Persist file IDs for future reference
    ids_path = DASH_DIR / '.gdrive_auth' / 'file_ids.json'
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(ids_path.read_text()) if ids_path.exists() else {}
    existing.update(file_ids)
    ids_path.write_text(json.dumps(existing, indent=2))

    print(f'\nDone at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
