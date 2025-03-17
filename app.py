import streamlit as st
import logging
from ragbot import RAGBot
from ui.chat_tab import render_chat_tab
from ui.upload_tab import render_upload_tab
from ui.youtube_tab import render_youtube_tab
from utils.language import get_language_options, get_localization_dict, get_system_prompt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main application entry point"""
    st.set_page_config(page_title="RAG bot", page_icon="🤖")
    
    # Initialize RAG bot
    bot = RAGBot()
    
    # Set up localization
    if 'langs' not in st.session_state:
        st.session_state.langs = get_localization_dict()
    
    # Add language selection in sidebar
    language = st.sidebar.selectbox(
        "Language / 語言",
        options=list(get_language_options().keys()),
        format_func=lambda x: get_language_options()[x],
        key="language"
    )
    
    # Get text for current language
    txt = st.session_state.langs[language]
    
    # Set page title
    st.title(txt["title"])
    
    # Update bot system prompt based on language
    bot.system_prompt = get_system_prompt(language)
    
    # Add provider selection in sidebar
    provider = st.sidebar.selectbox(
        txt["provider_select"],
        options=["azure", "gemini"],
        index=0
    )
    
    # Set dataset selection
    dataset_name = st.sidebar.selectbox(
        txt["dataset_select"], 
        options=bot.datasets
    )
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        txt["tab_chat"], 
        txt["tab_upload"], 
        txt["tab_youtube"]
    ])
    
    # Render Chat tab
    with tab1:
        render_chat_tab(bot, provider, dataset_name)
    
    # Render Upload Documents tab
    with tab2:
        render_upload_tab(dataset_name, provider)
    
    # Render YouTube tab
    with tab3:
        render_youtube_tab(bot, provider, dataset_name)

if __name__ == "__main__":
    main() 