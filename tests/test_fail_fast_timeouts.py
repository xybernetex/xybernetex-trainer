"""The experiment runners don't retry Workers AI's 408 (a model that deliberated
past the request limit), but still retry rate limits and server errors."""
import unittest

import httpx
from openai import AsyncOpenAI

from scenarios.sdk_agent import fail_fast_on_timeouts


class FailFastOnTimeouts(unittest.TestCase):
    def test_408_fails_fast_429_and_5xx_still_retry(self):
        fail_fast_on_timeouts()
        fail_fast_on_timeouts()   # idempotent: patched once
        client = AsyncOpenAI(api_key="test", base_url="http://localhost:9", max_retries=10)
        retry = lambda code: client._should_retry(httpx.Response(code))
        self.assertFalse(retry(408))
        self.assertTrue(retry(429))
        self.assertTrue(retry(500))
        self.assertFalse(retry(400))


if __name__ == "__main__":
    unittest.main()
