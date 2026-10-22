"""Web sense: a second, separate opt-in that lets the engine *read* the web.

This is deliberately a sensory organ, not a pair of hands. The engine can
look at pages the node operator allows, and what it reads becomes emotional
stimulus -- valence and arousal that nudge its affective state and that it can
learn from. It cannot post, submit forms, log in, or act as anyone. There is
no code path here that does anything but HTTP GET.

Consent is layered: a machine opts into *compute* and, entirely separately,
into *web sense*. Opting into one never implies the other. Web sense ships
disabled, with an empty allowlist, so it does nothing at all until the node's
human operator names the domains it may read.

Guardrails (all enforced before any byte leaves the machine):
  - GET only; no auth headers, no cookies, no form/body.
  - Operator allowlist: a domain the operator did not add is refused. The
    engine chooses *among* allowed domains; it cannot widen the set.
  - Redirects are blocked, so a 3xx cannot bounce a request off the allowlist.
  - Per-domain rate limit, so many nodes can't be aimed at one target.
  - robots.txt is respected when asked.
  - Response size and time are capped; every attempt is logged locally.

The HTTP fetcher is injectable so this is fully testable offline. A real
urllib-based fetcher (GET, no redirects, no cookies) is provided for live use.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple
from urllib.parse import urlsplit


class RedirectBlocked(Exception):
    """Raised by a fetcher when a response tried to redirect off-allowlist."""


@dataclass(frozen=True)
class FetchResult:
    status: int
    final_url: str
    content_type: str
    text: str


# fetcher(url, timeout_s, max_bytes) -> FetchResult
Fetcher = Callable[[str, float, int], FetchResult]


@dataclass
class WebConsent:
    """The second opt-in. Off and empty until the operator fills it in."""

    enabled: bool = False
    allowlist: frozenset = frozenset()          # bare domains, e.g. "en.wikipedia.org"
    max_requests_per_min: int = 10
    max_bytes: int = 200_000
    timeout_s: float = 5.0
    respect_robots: bool = True

    def allows(self, domain: str) -> bool:
        return self.enabled and domain in self.allowlist


@dataclass
class WebObservation:
    url: str
    ok: bool
    reason: str = ""                 # why it was refused, if ok is False
    status: Optional[int] = None
    text: str = ""
    valence: float = 0.0             # pleasant(+)/unpleasant(-), roughly [-1,1]
    arousal: float = 0.0             # activation, [0,1]
    emotion_nudges: Dict[str, float] = field(default_factory=dict)


# A tiny, transparent sentiment lexicon. This is a placeholder appraisal, not
# a claim about NLP quality; swap in a real model later. Kept visible on
# purpose so the affect attribution is auditable.
_POS = {
    "good", "great", "happy", "love", "win", "success", "safe", "hope",
    "beautiful", "calm", "gain", "thrive", "welcome", "free", "trust",
}
_NEG = {
    "bad", "terrible", "sad", "hate", "lose", "danger", "fear", "threat",
    "attack", "death", "pain", "fail", "crisis", "angry", "war",
}
_HIGH_AROUSAL = {
    "now", "breaking", "urgent", "alert", "shock", "explode", "attack",
    "emergency", "warning", "!",
}


def appraise_text(text: str) -> Tuple[float, float, Dict[str, float]]:
    """Turn page text into (valence, arousal, emotion nudges)."""
    words = [w.strip(".,!?;:\"'()[]").lower() for w in text.split()]
    if not words:
        return 0.0, 0.0, {}
    pos = sum(w in _POS for w in words)
    neg = sum(w in _NEG for w in words)
    arousal_hits = sum(w in _HIGH_AROUSAL for w in words) + text.count("!")

    denom = pos + neg or 1
    valence = (pos - neg) / denom
    valence = max(-1.0, min(1.0, valence))
    arousal = max(0.0, min(1.0, arousal_hits / max(10, len(words) / 20)))

    nudges: Dict[str, float] = {}
    if valence >= 0:
        nudges["joy"] = 0.20 * valence
        nudges["trust"] = 0.10 * valence
    else:
        m = -valence
        nudges["sadness"] = 0.18 * m
        nudges["fear"] = 0.14 * m
    nudges["surprise"] = 0.25 * arousal
    nudges["anticipation"] = 0.10 * arousal
    return valence, arousal, nudges


class WebSense:
    """Guarded, read-only web access for one node."""

    def __init__(
        self,
        consent: WebConsent,
        fetcher: Optional[Fetcher] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.consent = consent
        self._fetch = fetcher if fetcher is not None else default_fetcher
        self._clock = clock
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self.log: List[WebObservation] = []

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _domain(url: str) -> str:
        return (urlsplit(url).hostname or "").lower()

    def _rate_ok(self, domain: str) -> bool:
        now = self._clock()
        q = self._hits[domain]
        while q and now - q[0] > 60.0:
            q.popleft()
        if len(q) >= self.consent.max_requests_per_min:
            return False
        q.append(now)
        return True

    def _robots_ok(self, url: str, domain: str) -> bool:
        if not self.consent.respect_robots:
            return True
        try:
            from urllib.robotparser import RobotFileParser
            robots_url = f"{urlsplit(url).scheme}://{domain}/robots.txt"
            res = self._fetch(robots_url, self.consent.timeout_s, 50_000)
            rp = RobotFileParser()
            rp.parse(res.text.splitlines())
            return rp.can_fetch("*", url)
        except RedirectBlocked:
            return False
        except Exception:
            # No robots.txt (or it failed to load) -> not disallowed.
            return True

    # -- the one public action: read ---------------------------------------

    def observe(self, url: str) -> WebObservation:
        """Fetch `url` read-only, appraise it, log it. Refuses unless allowed."""
        domain = self._domain(url)
        scheme = urlsplit(url).scheme

        def refuse(reason: str) -> WebObservation:
            obs = WebObservation(url=url, ok=False, reason=reason)
            self.log.append(obs)
            return obs

        if scheme not in ("http", "https"):
            return refuse("scheme_not_allowed")
        if not self.consent.enabled:
            return refuse("web_sense_not_opted_in")
        if not self.consent.allows(domain):
            return refuse("domain_not_on_allowlist")
        if not self._rate_ok(domain):
            return refuse("rate_limited")
        if not self._robots_ok(url, domain):
            return refuse("robots_disallow")

        try:
            res = self._fetch(url, self.consent.timeout_s, self.consent.max_bytes)
        except RedirectBlocked:
            return refuse("redirect_blocked")
        except Exception as exc:  # network error, timeout, etc.
            return refuse(f"fetch_error:{type(exc).__name__}")

        # A redirect that landed off-allowlist is refused even if the fetcher
        # followed it (belt and braces; the default fetcher blocks redirects).
        if self._domain(res.final_url) != domain:
            return refuse("redirect_off_allowlist")

        valence, arousal, nudges = appraise_text(res.text)
        obs = WebObservation(
            url=url, ok=True, status=res.status, text=res.text[:2000],
            valence=valence, arousal=arousal, emotion_nudges=nudges,
        )
        self.log.append(obs)
        return obs

    def feed(self, engine, obs: WebObservation) -> None:
        """Apply an observation's emotional nudges to an engine."""
        if obs.ok and obs.emotion_nudges:
            engine.emotion.nudge(obs.emotion_nudges)


def default_fetcher(url: str, timeout_s: float, max_bytes: int) -> FetchResult:
    """Real GET-only fetcher: no redirects, no cookies, no auth, size-capped."""
    import urllib.request

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise RedirectBlocked(newurl)

    opener = urllib.request.build_opener(_NoRedirect)
    # Honest, identifiable agent string; no cookie processor is installed.
    req = urllib.request.Request(
        url, method="GET",
        headers={"User-Agent": "emotional-engine-websense/0.1 (read-only)"},
    )
    with opener.open(req, timeout=timeout_s) as resp:
        raw = resp.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raw = raw[:max_bytes]
        ctype = resp.headers.get("Content-Type", "")
        charset = resp.headers.get_content_charset() or "utf-8"
        text = raw.decode(charset, errors="replace")
        return FetchResult(
            status=resp.status, final_url=resp.geturl(),
            content_type=ctype, text=text,
        )
