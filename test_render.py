import sys, requests
sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://localhost:8000"

srt = """1
00:00:00,000 --> 00:00:02,000
Bismillah

2
00:00:02,000 --> 00:00:03,000
Test caption
"""

print("Testing /render...")
with open("data/test_video.mp4", "rb") as f:
    r = requests.post(
        f"{BASE}/render",
        files={"file": ("test_video.mp4", f, "video/mp4")},
        data={"srt": srt, "width": "640", "height": "360", "font_size": "32"},
        timeout=120,
    )
print(f"Status: {r.status_code}")
print(f"Content-type: {r.headers.get('content-type')}")
print(f"Content-length: {r.headers.get('content-length')}")
if r.status_code == 200:
    with open("data/output_captioned.mp4", "wb") as f:
        f.write(r.content)
    print(f"Saved {len(r.content)} bytes to data/output_captioned.mp4")
else:
    print(r.text[:500])
