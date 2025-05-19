import pytest
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import Runnable
from langchain_openai import ChatOpenAI

from chatbot import Chatbot


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    """Set dummy API keys before each test."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")


def test_init_attributes():
    """Ensure Chatbot initializes all core attributes when API keys are present."""
    bot = Chatbot()
    assert hasattr(bot, "prompt_template")
    assert isinstance(bot.prompt_template, PromptTemplate)
    assert hasattr(bot, "llm")
    assert isinstance(bot.llm, ChatOpenAI)
    assert hasattr(bot, "search_tool")
    assert isinstance(bot.search_tool, TavilySearchResults)
    assert hasattr(bot, "parser")
    assert isinstance(bot.parser, StrOutputParser)
    assert hasattr(bot, "chain")
    assert isinstance(bot.chain, Runnable)


@pytest.mark.parametrize(
    "env_vars_to_remove, expected_missing, should_raise",
    [
        ([], [], False),
        (["OPENAI_API_KEY"], ["OPENAI_API_KEY"], True),
        (["TAVILY_API_KEY"], ["TAVILY_API_KEY"], True),
        (
            ["OPENAI_API_KEY", "TAVILY_API_KEY"],
            ["OPENAI_API_KEY", "TAVILY_API_KEY"],
            True,
        ),
    ],
)
def test_validate_api_keys(
    monkeypatch, env_vars_to_remove, expected_missing, should_raise
):
    """Test _validate_api_keys for all combinations of missing API keys, including none."""
    for var in env_vars_to_remove:
        monkeypatch.delenv(var, raising=False)
    for var in {"OPENAI_API_KEY", "TAVILY_API_KEY"} - set(env_vars_to_remove):
        monkeypatch.setenv(var, "dummy")

    if should_raise:
        with pytest.raises(RuntimeError) as exc:
            Chatbot._validate_api_keys()
        for key in expected_missing:
            assert key in str(exc.value)
    else:
        Chatbot._validate_api_keys()


@pytest.mark.parametrize(
    "response, expected_answer, expected_sources",
    [
        (
            'Here\'s the answer.\n\n{"Wikipedia": "https://wikipedia.org"}',
            "Here's the answer.",
            {"Wikipedia": "https://wikipedia.org"},
        ),
        (
            "Just a plain answer with no sources.",
            "Just a plain answer with no sources.",
            {},
        ),
    ],
)
def test_parse_response(response, expected_answer, expected_sources):
    """Test parsing of LLM response into answer and sources dictionary."""
    bot = Chatbot()
    answer, sources = bot._parse_response(response)
    assert answer == expected_answer
    assert sources == expected_sources
