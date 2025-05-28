"""
chatbot.py

Core logic for a web-enabled question-answering chatbot using LangChain.

This module defines the `Chatbot` class, which integrates OpenAI's language model with
Tavily web search to provide enriched responses. It handles prompt formatting,
invocation with retry logic, optional search context, and post-processing of LLM
responses to extract citations in structured form.

Features:
- Supports retry logic for LLM and search errors using exponential backoff.
- Dynamically loads API keys from the environment (or .env via dotenv).
- Provides source citation parsing from model output, enabling transparency.
"""

import os

import backoff
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.exceptions import LangChainException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import Runnable
from langchain_core.tools import ToolException
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["prompt", "context"],
    template=(
        "You are a highly knowledgeable assistant. Your goal is to answer the user's "
        "question clearly, concisely, and with factual accuracy. If supporting context "
        "is provided, incorporate it into the answer and cite it.\n\n"
        "Please follow these guidelines:\n"
        "1. Answer in a factual and neutral tone.\n"
        "2. Prefer concise sentences (2-3 lines max).\n"
        "3. Use bullet points or short paragraphs if multiple points are needed.\n"
        "4. Do not make up information if context is insufficient.\n"
        "5. If no context is provided, DO NOT include any information on sources in "
        "the response.\n"
        "6. If a context is provided, ALWAYS include citations.\n"
        "7. Format the response as JSON with this structure:\n"
        '{{\n  "answer": "<main response>",\n  "sources": {{ "Name": "URL" }}\n}}\n\n'
        "User question: {prompt}\n\n"
        "Context:\n{context}"
    ),
)


class ChatbotResponse(BaseModel):
    answer: str = Field(..., description="The assistant's main answer to the prompt.")
    sources: dict[str, str] = Field(
        default_factory=dict,
        description="A dictionary of cited sources: source name → URL.",
    )


class Chatbot:
    """
    A minimal chatbot that can optionally enrich responses with real-time web
    search context via the Tavily API, and robustly invoke an LLM chain with
    retry/backoff on transient errors.

    Attributes:
        - prompt_template (PromptTemplate): The template used to format the user's
          prompt and optional context into a prompt for the LLM.
        - llm (ChatOpenAI): The OpenAI chat model instance used to generate responses.
        - search_tool (TavilySearchResults): The Tavily search tool instance used to
          fetch real-time search results and context.
        - parser (StrOutputParser): Parses the raw LLM output into a clean string.
        - chain (Runnable): A composed runnable sequence: prompt_template | llm | "
        "parser.
    """

    max_retries: int = 5

    def __init__(
        self,
        llm_model: str = "gpt-3.5-turbo",
        temperature: float = 0.2,
    ):
        """
        Initialize the Chatbot with specified LLM settings and retry behavior.

        Parameters:
            - llm_model (str, optional): The model name to instantiate via ChatOpenAI
              (default "gpt-3.5-turbo").
            - temperature (float, optional): Sampling temperature for the LLM; lower
              values produce more deterministic outputs (default 0.2).
        """
        self._validate_api_keys()
        self.prompt_template = PROMPT_TEMPLATE
        self.llm = ChatOpenAI(name=llm_model, temperature=temperature)
        self.search_tool = TavilySearchResults(include_answer=True)
        self.parser = PydanticOutputParser(pydantic_object=ChatbotResponse)
        self.chain: Runnable = self.prompt_template | self.llm | self.parser

    @staticmethod
    def _validate_api_keys() -> None:
        """
        Check if required API keys are set in the environment and raise an error if any
        are missing.
        """
        missing = []
        if "OPENAI_API_KEY" not in os.environ:
            missing.append("OPENAI_API_KEY")
        if "TAVILY_API_KEY" not in os.environ:
            missing.append("TAVILY_API_KEY")
        if missing:
            raise RuntimeError(f"Missing required API key(s): {', '.join(missing)}")

    @backoff.on_exception(
        backoff.expo,
        ToolException,
        max_tries=max_retries,
    )
    def _search(self, prompt: str) -> str:
        """
        Perform a web search for the given prompt using Tavily, retrying on failure.

        This method wraps the TavilySearchResults tool, which sends the prompt to the
        Tavily API and returns an LLM-optimized summary of the top results. It retries
        up to `self.max_retries` times on network or API errors.

        Parameters:
            - prompt (str): The user's question or search phrase.

        Returns:
            - str: A human-readable summary of the top search results.

        Raises:
            - ToolException: If the Tavily tool encounters a network error, downtime, or
              invalid API key.
        """
        return self.search_tool.run(prompt)

    @backoff.on_exception(
        backoff.expo,
        LangChainException,
        max_tries=max_retries,
    )
    def _invoke(self, prompt: str, context: str) -> ChatbotResponse:
        """
        Invoke the LLM chain with the formatted prompt and context, retrying on failure.

        This method sends the combined prompt and context to the composed Runnable chain
        (prompt → llm → parser), strips whitespace, and returns the result. It retries
        on any LangChain-exposed exception, such as model timeouts or downstream tool
        errors.

        Parameters:
            - prompt (str): The original user input.
            - context (str): Optional context string, e.g. search results.

        Returns:
            - str: The processed and stripped LLM response.

        Raises:
            - LangChainException: If the chain invocation fails due to transient LLM or
              parsing errors.
        """
        return self.chain.invoke({"prompt": prompt, "context": context})

    def generate_response(
        self, prompt: str, enable_search: bool = False
    ) -> tuple[str, dict[str, str]]:
        """
        Generate a concise response to the given prompt, optionally enriched by web
        search.

        This method:
        1. Optionally performs a web search using the Tavily API to retrieve context.
        2. Sends the prompt and context to the LLM via a LangChain chain.
        3. Parses the raw response to extract both the answer and any cited sources.

        Parameters:
            - prompt (str): The user's question or prompt.
            - enable_search (bool, optional): Whether to include web search context.
              Defaults to False.

        Returns:
            - answer (str): The assistant's main textual response.
            - sources (dict[str, str]): A dictionary mapping source names to URLs, if
              present. Sources are only returned if `enable_search` is True, even if the
              LLM includes them in the response.
        """
        context = self._search(prompt) if enable_search else ""
        response = self._invoke(prompt, context)
        # Discard sources if web search was not enabled, even if LLM included them
        sources = response.sources if enable_search else {}
        return response.answer, sources
