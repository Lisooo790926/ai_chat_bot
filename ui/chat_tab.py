import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import logging

logger = logging.getLogger(__name__)

def render_chat_tab(bot, provider, dataset_name):
    """Render the chat tab UI"""
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        initial_message = "您好，我是 AI 助手。有什麼可以幫您？" if st.session_state.get('language', 'en') == "zh-TW" else "Hello, I am a bot. How can I help you?"
        st.session_state.chat_history = [
            AIMessage(content=initial_message),
        ]

    # Display conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    # Get text for current language
    txt = st.session_state.get('langs', {}).get(st.session_state.get('language', 'en'), {})
    chat_placeholder = txt.get("chat_placeholder", "Type your message here...")
    
    # User input
    user_query = st.chat_input(chat_placeholder)
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            response = st.write_stream(
                bot.get_response_stream(
                    user_query=user_query,
                    chat_history=st.session_state.chat_history[-10:],
                    collection_name="bootcamp",
                    dataset_name=dataset_name,
                    provider=provider,
                )
            )

        st.session_state.chat_history.append(AIMessage(content=response)) 