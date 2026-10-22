import unittest

from emotional_engine.engine import EmotionalLearningEngine
from emotional_engine.websense import (
    FetchResult,
    RedirectBlocked,
    WebConsent,
    WebSense,
    appraise_text,
)


def make_fetcher(pages):
    """pages: {url: FetchResult or Exception}. Serves robots.txt as allow-all
    unless explicitly provided."""
    def fetch(url, timeout_s, max_bytes):
        if url.endswith("/robots.txt") and url not in pages:
            return FetchResult(200, url, "text/plain", "User-agent: *\nAllow: /")
        val = pages.get(url)
        if isinstance(val, Exception):
            raise val
        if val is None:
            raise ConnectionError("not found")
        return val
    return fetch


class TestAppraisal(unittest.TestCase):
    def test_positive_text_positive_valence(self):
        v, a, n = appraise_text("a great and happy success, we win and thrive")
        self.assertGreater(v, 0)
        self.assertIn("joy", n)

    def test_negative_text_negative_valence(self):
        v, a, n = appraise_text("a terrible crisis, danger and death and war")
        self.assertLess(v, 0)
        self.assertIn("fear", n)


class TestConsentGating(unittest.TestCase):
    def test_refuses_when_not_opted_in(self):
        ws = WebSense(WebConsent(enabled=False), fetcher=make_fetcher({}))
        obs = ws.observe("https://example.com/")
        self.assertFalse(obs.ok)
        self.assertEqual(obs.reason, "web_sense_not_opted_in")

    def test_refuses_domain_off_allowlist(self):
        ws = WebSense(
            WebConsent(enabled=True, allowlist=frozenset({"allowed.org"})),
            fetcher=make_fetcher({}),
        )
        obs = ws.observe("https://evil.example/")
        self.assertFalse(obs.ok)
        self.assertEqual(obs.reason, "domain_not_on_allowlist")

    def test_refuses_non_http_scheme(self):
        ws = WebSense(
            WebConsent(enabled=True, allowlist=frozenset({"x"})),
            fetcher=make_fetcher({}),
        )
        obs = ws.observe("file:///etc/passwd")
        self.assertFalse(obs.ok)
        self.assertEqual(obs.reason, "scheme_not_allowed")

    def test_allowed_domain_returns_observation(self):
        url = "https://news.org/a"
        ws = WebSense(
            WebConsent(enabled=True, allowlist=frozenset({"news.org"})),
            fetcher=make_fetcher(
                {url: FetchResult(200, url, "text/html", "good happy win safe hope")}
            ),
        )
        obs = ws.observe(url)
        self.assertTrue(obs.ok)
        self.assertGreater(obs.valence, 0)


class TestGuards(unittest.TestCase):
    def test_rate_limited(self):
        url = "https://news.org/a"
        page = {url: FetchResult(200, url, "text/html", "hello world")}
        ws = WebSense(
            WebConsent(enabled=True, allowlist=frozenset({"news.org"}),
                       max_requests_per_min=2),
            fetcher=make_fetcher(page),
            clock=lambda: 0.0,  # time frozen: window never advances
        )
        self.assertTrue(ws.observe(url).ok)
        self.assertTrue(ws.observe(url).ok)
        third = ws.observe(url)
        self.assertFalse(third.ok)
        self.assertEqual(third.reason, "rate_limited")

    def test_redirect_blocked(self):
        url = "https://news.org/a"
        ws = WebSense(
            WebConsent(enabled=True, allowlist=frozenset({"news.org"})),
            fetcher=make_fetcher({url: RedirectBlocked("https://elsewhere/")}),
        )
        obs = ws.observe(url)
        self.assertFalse(obs.ok)
        self.assertEqual(obs.reason, "redirect_blocked")

    def test_robots_disallow(self):
        url = "https://news.org/secret"
        pages = {
            "https://news.org/robots.txt":
                FetchResult(200, "https://news.org/robots.txt", "text/plain",
                            "User-agent: *\nDisallow: /secret"),
            url: FetchResult(200, url, "text/html", "whatever"),
        }
        ws = WebSense(
            WebConsent(enabled=True, allowlist=frozenset({"news.org"})),
            fetcher=make_fetcher(pages),
        )
        obs = ws.observe(url)
        self.assertFalse(obs.ok)
        self.assertEqual(obs.reason, "robots_disallow")

    def test_everything_is_logged(self):
        url = "https://news.org/a"
        ws = WebSense(
            WebConsent(enabled=True, allowlist=frozenset({"news.org"})),
            fetcher=make_fetcher({url: FetchResult(200, url, "text/html", "ok")}),
        )
        ws.observe(url)
        ws.observe("https://blocked.org/")
        self.assertEqual(len(ws.log), 2)


class TestFeedIntoEngine(unittest.TestCase):
    def test_observation_nudges_engine_emotion(self):
        url = "https://news.org/scary"
        ws = WebSense(
            WebConsent(enabled=True, allowlist=frozenset({"news.org"})),
            fetcher=make_fetcher(
                {url: FetchResult(200, url, "text/html",
                                  "danger war death crisis fear threat")}
            ),
        )
        eng = EmotionalLearningEngine(["a", "b"], seed=1)
        before = eng.emotion.intensities["fear"]
        ws.feed(eng, ws.observe(url))
        self.assertGreater(eng.emotion.intensities["fear"], before)


class TestSecondOptInOnNode(unittest.TestCase):
    def test_web_opt_in_is_independent_of_compute(self):
        from emotional_engine.swarm import Application, VolunteerNode
        app = Application("t", "s", ("a", "b"), lambda a, rng: 0.0)
        node = VolunteerNode("n", app)
        # No opt-ins at all.
        self.assertFalse(node.active)
        self.assertFalse(node.web_active)
        self.assertEqual(node.observe_web("https://x.org/").reason,
                         "web_sense_not_opted_in")
        # Web opt-in WITHOUT compute opt-in.
        url = "https://x.org/a"
        node.opt_in_web(
            WebConsent(enabled=True, allowlist=frozenset({"x.org"})),
            fetcher=make_fetcher({url: FetchResult(200, url, "text/html", "good win")}),
        )
        self.assertFalse(node.active)      # still not a compute volunteer
        self.assertTrue(node.web_active)   # but is a web volunteer
        self.assertTrue(node.observe_web(url).ok)


if __name__ == "__main__":
    unittest.main()
