from types import SimpleNamespace

import pytest
from google.genai import errors

from pagesnap_lib import gemini


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.models = self

    def generate_content(self, model, contents):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def api_error(code: int, message: str = "x") -> errors.APIError:
    return errors.APIError(code, {"error": {"message": message}})


def ok(text="解説"):
    return SimpleNamespace(text=text)


def test_retries_rate_limit_then_succeeds():
    client = FakeClient([api_error(429), api_error(503), ok()])
    sleeps = []
    assert gemini.explain_page(client, "m", b"img", None, sleep=sleeps.append) == "解説"
    assert sleeps == [2.0, 4.0]


def test_gives_up_after_max_attempts():
    client = FakeClient([api_error(429)] * 3)
    with pytest.raises(gemini.GeminiPageError):
        gemini.explain_page(client, "m", b"img", None, max_attempts=3, sleep=lambda s: None)
    assert client.calls == 3


@pytest.mark.parametrize("code,msg", [(404, "model not found"), (403, "denied"), (400, "API key not valid")])
def test_fatal_errors_stop_immediately(code, msg):
    client = FakeClient([api_error(code, msg)])
    with pytest.raises(gemini.GeminiFatalError):
        gemini.explain_page(client, "m", b"img", None, sleep=lambda s: None)
    assert client.calls == 1


def test_other_client_error_is_page_error():
    client = FakeClient([api_error(400, "image too large")])
    with pytest.raises(gemini.GeminiPageError):
        gemini.explain_page(client, "m", b"img", None, sleep=lambda s: None)


def test_empty_response_is_page_error():
    client = FakeClient([SimpleNamespace(text="", candidates=[SimpleNamespace(finish_reason="RECITATION")])])
    with pytest.raises(gemini.GeminiPageError, match="RECITATION"):
        gemini.explain_page(client, "m", b"img", None, sleep=lambda s: None)


def test_prompt_includes_subject():
    assert "数学B" in gemini.build_prompt("数学B")
    assert "科目" not in gemini.build_prompt(None).split("#")[0]
