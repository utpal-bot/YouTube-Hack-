#!/usr/bin/env python3
"""
Upload + schedule videos with the YouTube Data API v3.

  python upload.py --auth   # ONE-TIME, on your own PC: OAuth flow -> prints refresh token
  python upload.py          # uploads out/long.mp4 + out/short1-3.mp4 as private + publishAt
  DRY_RUN=1 python upload.py   # prints the plan, calls no API

Long video: custom thumbnail (out/thumb_long.jpg) is set automatically. Shorts thumbnails
(out/thumb_short*.jpg) are generated for cover-frame use but not uploaded via API.

Env: YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN
Optional: PUBLISH_SLOTS_UTC="12:00,13:00,16:30,23:30"  (long, short1, short2, short3 - UTC)
          YT_CATEGORY_ID (default 27 = Education)
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
TOKEN_URI = "https://oauth2.googleapis.com/token"
DEFAULT_SLOTS = "12:00,13:00,16:30,23:30"  # adjust to your audience's peak hours (in UTC)


def auth():
    """Run locally once. Needs client_secret.json (OAuth client of type 'Desktop app')."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
    print("\nSave these as GitHub Secrets:")
    print("YT_CLIENT_ID     =", creds.client_id)
    print("YT_CLIENT_SECRET =", creds.client_secret)
    print("YT_REFRESH_TOKEN =", creds.refresh_token)
    print("\nIMPORTANT: set the OAuth consent screen to 'In production', "
          "otherwise this refresh token expires after 7 days.")


def get_service():
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    creds = Credentials(
        token=None,
        refresh_token=os.environ["YT_REFRESH_TOKEN"],
        token_uri=TOKEN_URI,
        client_id=os.environ["YT_CLIENT_ID"],
        client_secret=os.environ["YT_CLIENT_SECRET"],
        scopes=SCOPES,
    )
    try:
        creds.refresh(Request())  # access token is minted fresh on every run
    except RefreshError as e:
        sys.exit(
            f"OAuth refresh failed: {e}\n"
            "Likely causes: consent screen still in 'Testing' (7-day tokens), token revoked, "
            "or wrong client id/secret. Re-run `python upload.py --auth`."
        )
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def next_slots(slots_csv: str, count: int = 4):
    """Next UTC datetimes for each slot; never earlier than 30 min from now, always non-decreasing."""
    now = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=30)
    slots = [tuple(int(x) for x in s.strip().split(":")) for s in slots_csv.split(",")]
    out, prev = [], now
    for i in range(count):
        h, m = slots[i % len(slots)]
        cand = now.replace(hour=h, minute=m, second=0, microsecond=0)
        while cand < prev:
            cand += dt.timedelta(days=1)
        out.append(cand)
        prev = cand
    return out


def clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def clean_tags(tags):
    out, total = [], 0
    for t in tags or []:
        t = str(t).replace("<", "").replace(">", "")[:30]
        if total + len(t) + 1 > 450:
            break
        out.append(t)
        total += len(t) + 1
    return out


def run_upload(request):
    from googleapiclient.errors import HttpError

    resp, retries = None, 0
    while resp is None:
        try:
            status, resp = request.next_chunk()
            if status:
                print(f"    {int(status.progress() * 100)}%", flush=True)
        except HttpError as e:
            if e.resp.status in (500, 502, 503, 504) and retries < 5:
                retries += 1
                time.sleep(2 ** retries)
                continue
            raise
    return resp


def upload_one(yt, path: Path, title, description, tags, publish_at: dt.datetime, category: str):
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    def body(with_synthetic: bool):
        status = {
            "privacyStatus": "private",  # private + publishAt = scheduled
            "publishAt": publish_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "selfDeclaredMadeForKids": False,
        }
        if with_synthetic:
            status["containsSyntheticMedia"] = True  # AI-content disclosure
        return {
            "snippet": {
                "title": clip(title, 100),
                "description": description[:4900],
                "tags": clean_tags(tags),
                "categoryId": category,
            },
            "status": status,
        }

    for with_synthetic in (True, False):
        media = MediaFileUpload(str(path), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
        req = yt.videos().insert(part="snippet,status", body=body(with_synthetic), media_body=media)
        try:
            return run_upload(req)
        except HttpError as e:
            if e.resp.status == 400 and with_synthetic:
                print("  400 with containsSyntheticMedia - retrying without it (set the flag manually in Studio).")
                continue
            raise


def set_thumbnail(yt, video_id: str, path: Path):
    """Custom thumbnail (long videos). Needs a phone-verified channel; failure is non-fatal."""
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    try:
        yt.thumbnails().set(
            videoId=video_id, media_body=MediaFileUpload(str(path), mimetype="image/jpeg")
        ).execute()
        print(f"  thumbnail set from {path.name}")
    except HttpError as e:
        print(f"  WARNING: thumbnail not set ({e.resp.status}). Verify the channel or set it in Studio.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auth", action="store_true")
    args = ap.parse_args()
    if args.auth:
        return auth()

    from googleapiclient.errors import HttpError

    script = json.loads(Path("build/script.json").read_text(encoding="utf-8"))
    verify = Path("build/verify.txt")
    if verify.exists() and verify.read_text(encoding="utf-8").strip():
        print("=== FACT-CHECK THESE CLAIMS BEFORE THE PUBLISH TIME ===")
        print(verify.read_text(encoding="utf-8"))
        print("=======================================================")

    dry = os.getenv("DRY_RUN") == "1"
    category = os.getenv("YT_CATEGORY_ID") or "27"
    slots = next_slots(os.getenv("PUBLISH_SLOTS_UTC") or DEFAULT_SLOTS)
    disclosure = "\n\nNarration and visuals in this video are AI-assisted."

    jobs = [{
        "name": "long", "path": Path("out/long.mp4"), "title": script["title"],
        "desc": script["description"] + disclosure,
    }]
    for k, s in enumerate(script["shorts"], 1):
        jobs.append({"name": f"short{k}", "path": Path(f"out/short{k}.mp4"),
                     "title": s["title"], "hook": s["hook"]})

    yt = None if dry else get_service()
    long_id, results = None, {}
    for job, when in zip(jobs, slots):
        if not job["path"].exists():
            print(f"SKIP {job['name']}: {job['path']} missing")
            if job["name"] == "long":
                sys.exit(1)
            continue
        if job["name"] != "long":
            link = f"Full video: https://youtu.be/{long_id}\n\n" if long_id else ""
            job["desc"] = f"{job['hook']}\n\n{link}#Shorts" + disclosure
        print(f"{job['name']}: {job['title']!r} -> {when:%Y-%m-%d %H:%M} UTC")
        if dry:
            continue
        try:
            resp = upload_one(yt, job["path"], job["title"], job["desc"], script.get("tags", []), when, category)
        except HttpError as e:
            if "quotaExceeded" in str(e):
                sys.exit("YouTube API daily quota exceeded (uploads cost ~1600 units each).")
            if e.resp.status == 403:
                print("403: if videos get locked as private, your API project needs the YouTube API compliance audit.")
            raise
        results[job["name"]] = {"id": resp["id"], "publishAt": when.isoformat()}
        if job["name"] == "long":
            long_id = resp["id"]
            thumb = Path("out/thumb_long.jpg")
            if thumb.exists():
                set_thumbnail(yt, resp["id"], thumb)
        print(f"  uploaded: https://youtu.be/{resp['id']}")

    Path("out").mkdir(exist_ok=True)
    Path("out/upload_result.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
