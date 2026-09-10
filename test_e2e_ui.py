import asyncio
import json
import sys
import threading

import requests
import websockets
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

APP_URL = "http://localhost:8501"
AUDIO = sys.argv[1] if len(sys.argv) > 1 else "data/test_ikhlas_16k.wav"
SKIP_RENDER = len(sys.argv) > 2 and sys.argv[2] == "norender"


class CORS(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, *a):
        pass


server = ThreadingHTTPServer(("127.0.0.1", 8899), CORS)
threading.Thread(target=server.serve_forever, daemon=True).start()


async def main():
    pages = requests.get("http://localhost:9222/json", timeout=10).json()
    target = next((p for p in pages if p["type"] == "page"), None)
    async with websockets.connect(target["webSocketDebuggerUrl"], max_size=2**28) as ws:
        msg_id = 0

        async def send(method, params=None, timeout=30):
            nonlocal msg_id
            msg_id += 1
            await ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
            while True:
                resp = json.loads(await asyncio.wait_for(ws.recv(), timeout))
                if resp.get("id") == msg_id:
                    return resp

        async def js(expr, await_promise=False):
            r = await send("Runtime.evaluate", {
                "expression": expr, "returnByValue": True, "awaitPromise": await_promise,
            })
            return r.get("result", {}).get("result", {}).get("value")

        async def wait_for(expr, timeout=600, interval=4):
            for _ in range(int(timeout / interval)):
                try:
                    v = await js(expr)
                    if v:
                        return v
                except Exception:
                    pass
                await asyncio.sleep(interval)
            return None

        await send("Page.enable")
        await send("Runtime.enable")
        await send("Page.navigate", {"url": APP_URL})
        await asyncio.sleep(12)
        print("title:", await js("document.title"))

        # ---- inject file via DataTransfer
        script = f"""
        (async () => {{
            const input = document.querySelector('input[type="file"]');
            const res = await fetch('http://127.0.0.1:8899/{AUDIO.replace(chr(92), '/')}');
            const blob = await res.blob();
            const name = '{AUDIO.split(chr(92))[-1].split("/")[-1]}';
            const f = new File([blob], name);
            const dt = new DataTransfer();
            dt.items.add(f);
            input.files = dt.files;
            input.dispatchEvent(new Event('change', {{bubbles: true}}));
            return 'injected ' + f.size;
        }})()
        """
        print("inject:", await js(script, await_promise=True))
        await asyncio.sleep(8)
        print("video shown:", await js("document.querySelectorAll('video').length > 0"))

        # ---- generate captions
        clicked = await js(
            """(() => { const b = [...document.querySelectorAll('button')]
                .find(x => x.innerText.includes('Generate Captions')); if (b) { b.click(); return true; } return false; })()"""
        )
        print("generate clicked:", clicked)

        # ---- wait for editor (align runs ~3-5 min)
        editor = await wait_for(
            """document.querySelector('[data-testid="stDataFrame"]') !== null""", 900
        )
        print("editor rendered:", editor)
        if not editor:
            print("body:", (await js("document.body.innerText"))[:400])
            return

        det = await js(
            """(() => { const t = document.body.innerText;
                const i = t.indexOf('Detected Surah');
                return i >= 0 ? t.slice(i, i + 220) : 'MISSING'; })()"""
        )
        print("detection:", (det or "").replace("\n", " | ")[:260])

        vid = await js(
            """(() => { const v = document.getElementById('vid');
                return v ? ('main-doc video, fsbtn=' + (document.getElementById('fsbtn') ? 'yes' : 'no')) : 'MISSING'; })()"""
        )
        print("player:", vid)

        # ---- playback position survives style reruns (no restart on font change)
        seeked = await js(
            """(() => { const v = document.getElementById('vid');
                if (!v) return false;
                const s = v.duration * 0.35;
                v.currentTime = s;
                v.paused = false;
                return true; })()"""
        )
        print("seeked to 35%:", seeked)
        await asyncio.sleep(3)
        await js(
            """(() => { const g = [...document.querySelectorAll('[data-testid^=stBaseButton-segmented_control]')];
                const el = g.find(b => b.innerText.trim() === 'top');
                if (el) { el.click(); return true; } return false; })()"""
        )
        await asyncio.sleep(6)
        t = await js(
            """(() => { const v = document.getElementById('vid');
                return v ? Math.round(v.currentTime * 100) / 100 + ' / dur ' + Math.round(v.duration * 100) / 100 : 'no video'; })()"""
        )
        print("time after rerun (expect ~35% of dur, not 0):", t)
        fs = await js(
            """(async () => { const b = document.getElementById('fsbtn');
                if (!b) return 'no btn';
                try { b.click(); await new Promise(r => setTimeout(r, 500));
                    return document.fullscreenElement ? (document.fullscreenElement.id || 'el') : 'not-fullscreen';
                } catch (e) { return 'ERR ' + e.message; } })()"""
        )
        print("fullscreen:", fs)

        rows = await js(
            """[...document.querySelectorAll('[data-testid="stDataFrame"] tbody tr')].length"""
        )
        print("editor rows:", rows)

        lang = await js(
            """(() => { const g = [...document.querySelectorAll('[data-testid^=stBaseButton-segmented_control]')];
                return g.length ? g.slice(0, 3).map(b => b.innerText).join('|') : 'none'; })()"""
        )
        print("language control:", lang)

        # ---- language switch to both + verify player text update
        await js(
            """(() => { const g = [...document.querySelectorAll('[data-testid^=stBaseButton-segmented_control]')];
                const el = g.find(b => b.innerText.includes('Arabic + English'));
                if (el) { el.click(); return true; } return false; })()"""
        )
        await asyncio.sleep(5)
        print("lang switched to both")

        # ---- downloads
        btns = await js(
            """[...document.querySelectorAll('button')].map(b => b.innerText).filter(t => t.includes('Download'))"""
        )
        print("download buttons:", btns)

        print("render button:", await js(
            """[...document.querySelectorAll('button')].some(b => b.innerText.includes('Render MP4'))"""))

        if SKIP_RENDER:
            return

        # ---- render
        await js(
            """(() => { const b = [...document.querySelectorAll('button')]
                .find(x => x.innerText.includes('Render MP4')); if (b) { b.click(); return true; } return false; })()"""
        )
        print("render clicked")
        done = await wait_for(
            """document.body.innerText.includes('Video rendered.')""", 1800, 5
        )
        print("render done:", done)
        if done:
            link = await js(
                """[...document.querySelectorAll('a')].map(a => a.textContent).filter(t => t.includes('Download MP4'))"""
            )
            print("download link:", link)
            print("st.video present:", await js("document.querySelectorAll('video').length > 0"))


asyncio.run(main())
