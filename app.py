"""
app.py

Streamlit application for a web-enabled QA chatbot powered by a large language model
(OpenAI GPT) and optional web search via the Tavily API.

This module defines the web interface, manages chat history using Streamlit session
state, and handles user interactions. Users can ask questions through a chat input
field, and the chatbot responds with factual answers, optionally citing sources from
live search.

Main responsibilities:
- Render chat messages with an expandable "Sources" section.
- Manage persistent chat history across Streamlit reruns.
- Trigger backend logic in `chatbot.py` to generate answers.
"""

import os

import streamlit as st

from chatbot import Chatbot


def sync_secrets_to_env() -> None:
    """
    Load API keys from Streamlit secrets (if defined) and export them to
    environment variables.

    This function allows seamless operation in both local and deployed
    environments:
    - In production (Streamlit Cloud), secrets will be set via `secrets.toml`.
    - In local development, if the file is missing, it fails gracefully.

    It sets OPENAI_API_KEY and TAVILY_API_KEY into os.environ, enabling usage by
    LangChain and Tavily that rely on environment-based configuration.
    """
    if st.secrets.load_if_toml_exists():
        openai_key = st.secrets.get("OPENAI_API_KEY")
        tavily_key = st.secrets.get("TAVILY_API_KEY")

        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        if tavily_key:
            os.environ["TAVILY_API_KEY"] = tavily_key


def initialize_session_state() -> None:
    """
    Initialize chat history list in Streamlit session state if not already present.
    """
    if "history" not in st.session_state:
        st.session_state.history = []


def display_chat_history() -> None:
    """
    Display all messages stored in the session's chat history.
    """
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "sources" in message:
                render_sources_expander(message["sources"])


def generate_response(
    prompt: str, chatbot: Chatbot, enable_search: bool = False
) -> tuple[str, dict[str, str]]:
    """
    Generate a response from the chatbot, optionally using web search for additional
    context.

    This function wraps the chatbot's response generation logic and displays a visual
    spinner while processing. It handles both direct responses and responses enriched
    with citations when web search is enabled.

    Parameters:
        - prompt (str): The user's input question or statement.
        - chatbot (Chatbot): An instance of the Chatbot class.
        - enable_search (bool): If True, performs a web search to inform the response.

    Returns:
        - response (str): The assistant's main answer.
        - sources (dict[str, str]): A dictionary of source name → URL pairs extracted
          from the response.
    """
    message = "Searching..." if enable_search else "Generating response..."
    with st.spinner(message):
        return chatbot.generate_response(prompt, enable_search)


def handle_user_input(
    prompt: str, chatbot: Chatbot, enable_search: bool = False
) -> None:
    """
    Process the user input, update chat history, and display the chatbot's response.

    Parameters:
        - prompt (str): The user's input question or statement.
        - chatbot (Chatbot): An instance of the Chatbot class.
        - enable_search (bool): Flag to determine if web search should be used.

    Returns:
        - None
    """
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})

    # Create a placeholder to defer rendering of the assistant's message
    # until the chatbot response is fully generated (prevents grayed-out UI)
    placeholder = st.empty()

    answer, sources = generate_response(prompt, chatbot, enable_search)

    with placeholder.chat_message("assistant"):
        st.write(answer)
        if sources:
            render_sources_expander(sources)
    st.session_state.history.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


def render_sources_expander(sources: dict[str, str]) -> None:
    """
    Render a collapsible expander in the right column of a two-column layout to
    display a dictionary of source names and their corresponding URLs.

    Parameters:
        - sources (dict[str, str]): A dictionary mapping source names to their URLs.

    Returns:
        - None
    """
    cols = st.columns([0.75, 0.25])
    with cols[1]:
        with st.expander("Sources"):
            for name, url in sources.items():
                st.write(f"[[{name}]({url})]")


def main() -> None:
    """
    Run the Streamlit-based LangChain QA Chatbot application.

    This function sets up the Streamlit interface, initializes the chatbot, manages
    session state for chat history, and handles user interactions, including optional
    web search functionality.
    """
    st.set_page_config(page_title="LangChain QA Chatbot", page_icon="🤖")
    st.title("LangChain QA Chatbot")
    sync_secrets_to_env()
    enable_search = st.checkbox("Enable Web Search")
    chatbot = Chatbot()
    initialize_session_state()
    display_chat_history()
    prompt = st.chat_input("Ask anything")
    if prompt:
        handle_user_input(prompt, chatbot, enable_search)


if __name__ == "__main__":
    main()
