from scripts.generate_data import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    OPENROUTER_DEFAULT_BASE_URL,
    resolve_client_kwargs,
    resolve_model,
)


def test_empty_env_returns_no_credentials() -> None:
    assert resolve_client_kwargs({}) == ("", None)


def test_openrouter_key_uses_openrouter_base_url() -> None:
    env = {"OPENROUTER_API_KEY": "sk-or-test"}
    assert resolve_client_kwargs(env) == (
        "sk-or-test",
        OPENROUTER_DEFAULT_BASE_URL,
    )


def test_openrouter_custom_base_url_is_respected() -> None:
    env = {"OPENROUTER_API_KEY": "sk-or-test", "OPENROUTER_BASE_URL": "https://proxy.example"}
    assert resolve_client_kwargs(env) == ("sk-or-test", "https://proxy.example")


def test_openai_fallback_without_base_url() -> None:
    assert resolve_client_kwargs({"OPENAI_API_KEY": "sk-proj-test"}) == ("sk-proj-test", None)


def test_openai_fallback_with_custom_base_url() -> None:
    env = {"OPENAI_API_KEY": "sk-proj-test", "OPENAI_BASE_URL": "https://custom.example"}
    assert resolve_client_kwargs(env) == ("sk-proj-test", "https://custom.example")


def test_openrouter_key_preferred_over_openai_key() -> None:
    env = {
        "OPENAI_API_KEY": "sk-proj-test",
        "OPENROUTER_API_KEY": "sk-or-test",
        "OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
    }
    api_key, base_url = resolve_client_kwargs(env)
    assert api_key == "sk-or-test"
    assert base_url == OPENROUTER_DEFAULT_BASE_URL


def test_resolve_model_uses_openrouter_default() -> None:
    model, provider = resolve_model({"OPENROUTER_API_KEY": "sk-or-test"}, None)
    assert model == DEFAULT_OPENROUTER_MODEL
    assert provider == "openrouter"


def test_resolve_model_honors_requested_over_default() -> None:
    model, provider = resolve_model({"OPENROUTER_API_KEY": "sk-or-test"}, "gpt-4o-mini")
    assert model == "gpt-4o-mini"
    assert provider == "openrouter"


def test_resolve_model_openai_default() -> None:
    model, provider = resolve_model({}, None)
    assert model == DEFAULT_OPENAI_MODEL
    assert provider == "openai"


def test_resolve_model_honors_requested_for_openai() -> None:
    model, provider = resolve_model({}, "gpt-4o-mini")
    assert model == "gpt-4o-mini"
    assert provider == "openai"
