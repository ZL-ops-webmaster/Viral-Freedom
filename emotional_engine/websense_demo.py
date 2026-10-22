"""Demo: the second opt-in giving the engine read-only sight of the web.

Uses a canned fetcher so it runs offline and deterministically. Swap in
emotional_engine.websense.default_fetcher for live (allowlisted) fetching.

Run:
    python3 -m emotional_engine.websense_demo
"""

from __future__ import annotations

from .engine import EmotionalLearningEngine
from .websense import FetchResult, WebConsent, WebSense

PAGES = {
    "https://good.example/story":
        FetchResult(200, "https://good.example/story", "text/html",
                    "A beautiful day: love and hope and success, we win and thrive, safe and free."),
    "https://grim.example/story":
        FetchResult(200, "https://grim.example/story", "text/html",
                    "Breaking: urgent alert! war and danger and death, a terrible crisis, fear and attack now!"),
    "https://off.example/story":
        FetchResult(200, "https://off.example/story", "text/html", "nope"),
}


def fetcher(url, timeout_s, max_bytes):
    if url.endswith("/robots.txt"):
        return FetchResult(200, url, "text/plain", "User-agent: *\nAllow: /")
    return PAGES[url]


def main() -> None:
    engine = EmotionalLearningEngine(["reflect", "explore"], seed=1)
    # The operator opts in to web sense and allows exactly two domains.
    consent = WebConsent(
        enabled=True,
        allowlist=frozenset({"good.example", "grim.example"}),
        max_requests_per_min=30,
    )
    ws = WebSense(consent, fetcher=fetcher)

    print(f"start mood: {engine.mood()}\n")

    for url in ("https://good.example/story", "https://grim.example/story",
                "https://off.example/story"):
        obs = ws.observe(url)
        if obs.ok:
            ws.feed(engine, obs)
            print(f"read  {url}")
            print(f"      valence={obs.valence:+.2f} arousal={obs.arousal:.2f} "
                  f"nudges={ {k: round(v,2) for k,v in obs.emotion_nudges.items()} }")
            print(f"      -> mood now: {engine.mood()}\n")
        else:
            print(f"refused {url}  reason={obs.reason}\n")

    print("audit log (every attempt, allowed or not):")
    for o in ws.log:
        print(f"  {'OK ' if o.ok else 'NO '} {o.url}  {o.reason or o.status}")


if __name__ == "__main__":
    main()
