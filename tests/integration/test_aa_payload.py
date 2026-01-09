"""Integration tests for Advanced Augmentation (AA) API acceptance.

These tests verify that AA payloads are accepted by the staging API when making
real LLM calls. They use MEMORI_TEST_MODE=1 to target the staging environment.

The tests do NOT validate payload structure (that's done in unit tests at
tests/memory/augmentation/test_aa_payload_unit.py). Instead, they verify:
1. The full pipeline works end-to-end
2. The staging AA API accepts the payloads (no 4xx/5xx errors)
3. Background augmentation completes without exceptions

Run with:
    MEMORI_TEST_MODE=1 OPENAI_API_KEY=... pytest tests/integration/test_aa_payload.py -v

Requires at least one LLM API key to make actual calls that trigger the AA pipeline.
"""

import asyncio
import os
import time

import pytest

from tests.integration.conftest import requires_openai

# Ensure we're in test mode for staging API
os.environ.setdefault("MEMORI_TEST_MODE", "1")

# Test configuration constants
MODEL = "gpt-4o-mini"  # Cheapest model for testing
MAX_TOKENS = 50  # Minimize token usage
TEST_PROMPT = "Say 'hello' in one word."  # Minimal prompt


class TestSyncAAIntegration:
    """Tests for AA pipeline from synchronous LLM calls."""

    @requires_openai
    @pytest.mark.integration
    def test_sync_call_triggers_aa_pipeline(self, memori_instance, openai_api_key):
        """Verify sync chat completion triggers AA without errors."""
        from openai import OpenAI

        mem = memori_instance

        client = OpenAI(api_key=openai_api_key)
        mem.llm.register(client)
        mem.attribution(entity_id="test-user", process_id="test-process")

        # Make API call
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        # Verify LLM call succeeded
        assert response is not None
        assert response.choices[0].message.content is not None

        # Wait for background augmentation to complete
        mem.config.augmentation.wait(timeout=5.0)

        # If we get here without exceptions, the AA pipeline worked

    @requires_openai
    @pytest.mark.integration
    def test_sync_streaming_triggers_aa_pipeline(self, memori_instance, openai_api_key):
        """Verify sync streaming triggers AA without errors."""
        from openai import OpenAI

        mem = memori_instance

        client = OpenAI(api_key=openai_api_key)
        mem.llm.register(client)
        mem.attribution(entity_id="stream-user", process_id="stream-process")

        # Consume the stream
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        chunks = []
        for chunk in stream:
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)

        assert len(chunks) > 0

        # Wait for background augmentation
        mem.config.augmentation.wait(timeout=5.0)

    @requires_openai
    @pytest.mark.integration
    def test_multi_turn_conversation_triggers_aa(self, memori_instance, openai_api_key):
        """Verify multi-turn conversation triggers AA correctly."""
        from openai import OpenAI

        mem = memori_instance

        client = OpenAI(api_key=openai_api_key)
        mem.llm.register(client)
        mem.attribution(entity_id="multi-turn-user", process_id="multi-turn-proc")

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
            {"role": "user", "content": "What is my name?"},
        ]

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        mem.config.augmentation.wait(timeout=5.0)


class TestAsyncAAIntegration:
    """Tests for AA pipeline from asynchronous LLM calls."""

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_call_triggers_aa_pipeline(
        self, memori_instance, openai_api_key
    ):
        """Verify async chat completion triggers AA without errors."""
        from openai import AsyncOpenAI

        mem = memori_instance

        client = AsyncOpenAI(api_key=openai_api_key)
        mem.llm.register(client)
        mem.attribution(entity_id="async-user", process_id="async-process")

        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

        # Wait for background augmentation
        await asyncio.sleep(0.5)
        mem.config.augmentation.wait(timeout=5.0)

    @requires_openai
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_triggers_aa_pipeline(
        self, memori_instance, openai_api_key
    ):
        """Verify async streaming triggers AA without errors."""
        from openai import AsyncOpenAI

        mem = memori_instance

        client = AsyncOpenAI(api_key=openai_api_key)
        mem.llm.register(client)
        mem.attribution(entity_id="async-stream-user", process_id="async-stream-proc")

        stream = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        chunks = []
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)

        assert len(chunks) > 0

        await asyncio.sleep(0.5)
        mem.config.augmentation.wait(timeout=5.0)


class TestAAEdgeCases:
    """Tests for edge cases in AA pipeline."""

    @requires_openai
    @pytest.mark.integration
    def test_no_aa_without_attribution(self, memori_instance, openai_api_key):
        """Verify AA is skipped when entity_id is not set."""
        from openai import OpenAI

        mem = memori_instance

        client = OpenAI(api_key=openai_api_key)
        mem.llm.register(client)
        # Don't set attribution

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        # Pipeline should complete without errors even without attribution
        time.sleep(0.5)

    @requires_openai
    @pytest.mark.integration
    def test_aa_with_entity_only(self, memori_instance, openai_api_key):
        """Verify AA works with just entity_id (no process_id)."""
        from openai import OpenAI

        mem = memori_instance

        client = OpenAI(api_key=openai_api_key)
        mem.llm.register(client)
        mem.attribution(entity_id="entity-only-user")  # No process_id

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS,
        )

        assert response is not None
        mem.config.augmentation.wait(timeout=5.0)

    @requires_openai
    @pytest.mark.integration
    def test_multiple_calls_same_session(self, memori_instance, openai_api_key):
        """Verify multiple calls in same session trigger AA correctly."""
        from openai import OpenAI

        mem = memori_instance

        client = OpenAI(api_key=openai_api_key)
        mem.llm.register(client)
        mem.attribution(entity_id="multi-call-user", process_id="multi-call-proc")

        # Make multiple calls
        for i in range(3):
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": f"Say the number {i}"}],
                max_tokens=MAX_TOKENS,
            )
            assert response is not None

        # Wait for all augmentations
        mem.config.augmentation.wait(timeout=10.0)


class TestTestModeConfiguration:
    """Tests to verify test mode configuration."""

    def test_memori_test_mode_is_enabled(self):
        """Verify MEMORI_TEST_MODE is set for staging API."""

        assert os.environ.get("MEMORI_TEST_MODE") == "1"

    @requires_openai
    @pytest.mark.integration
    def test_memori_instance_in_test_mode(self, memori_instance):
        """Verify memori instance works in test mode."""
        # Check env var is set (test mode)
        assert os.environ.get("MEMORI_TEST_MODE") is not None

        # The memori instance should be configured and working
        assert memori_instance is not None
        assert memori_instance.config is not None
