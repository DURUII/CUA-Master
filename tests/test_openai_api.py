from types import SimpleNamespace

from cuamaster.openai_api import create_text_response, usage_token_counts


class _Recorder:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_responses_request_converts_multimodal_content() -> None:
    recorder = _Recorder(
        SimpleNamespace(
            output_text='{"ok":true}',
            usage=SimpleNamespace(input_tokens=12, output_tokens=4),
        )
    )
    client = SimpleNamespace(responses=recorder)

    result = create_text_response(
        client,
        wire_api="responses",
        model="GPT-5.6-sol",
        content=[
            {"type": "text", "text": "inspect"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ],
        max_output_tokens=2048,
        reasoning_effort="xhigh",
        store=False,
    )

    assert result.text == '{"ok":true}'
    assert recorder.kwargs == {
        "model": "GPT-5.6-sol",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "inspect"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,abc",
                    },
                ],
            }
        ],
        "max_output_tokens": 2048,
        "store": False,
        "reasoning": {"effort": "xhigh"},
    }
    assert usage_token_counts(result.response) == (12, 4)


def test_chat_completions_request_preserves_legacy_shape() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="done"))],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
    )
    recorder = _Recorder(response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=recorder))

    result = create_text_response(
        client,
        wire_api="chat_completions",
        model="kimi-k3",
        content="hello",
        system_prompt="system",
        max_output_tokens=128,
        chat_response_format={"type": "json_object"},
    )

    assert result.text == "done"
    assert recorder.kwargs == {
        "model": "kimi-k3",
        "stream": False,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
        ],
        "max_tokens": 128,
        "response_format": {"type": "json_object"},
    }
    assert usage_token_counts(result.response) == (5, 2)


def test_deepseek_chat_request_preserves_explicit_none_reasoning_effort() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
    )
    recorder = _Recorder(response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=recorder))

    create_text_response(
        client,
        wire_api="chat_completions",
        model="DeepSeek-V4.1-Flash",
        content="hello",
        max_output_tokens=128,
        reasoning_effort="none",
    )

    assert recorder.kwargs["reasoning_effort"] == "none"
