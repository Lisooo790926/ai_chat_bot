import streamlit as st
import tempfile
import os
import re
import traceback
import logging
from components.youtube_to_md import YoutubeLoader, get_youtube_videos, get_playlist_videos, parse_video_id
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

def render_youtube_tab(bot, provider, dataset_name):
    """Render the YouTube tab UI"""
    # Get text for current language
    txt = st.session_state.get('langs', {}).get(st.session_state.get('language', 'en'), {})
    
    # Set default text if localization not found
    youtube_header = txt.get("youtube_header", "YouTube Video Summarizer")
    tab_single = txt.get("tab_single", "Single Video")
    tab_multi = txt.get("tab_multi", "Multiple Videos")
    tab_playlist = txt.get("tab_playlist", "Playlist")
    youtube_url = txt.get("youtube_url", "Enter YouTube Video URL")
    youtube_urls = txt.get("youtube_urls", "Enter YouTube Video URLs (one per line)")
    summary_style = txt.get("summary_style", "Summary Style:")
    save_md = txt.get("save_md", "Save as Markdown file")
    save_mds = txt.get("save_mds", "Save as Markdown files")
    add_rag = txt.get("add_rag", "Add to RAG database")
    output_folder = txt.get("output_folder", "Output folder for markdown files")
    process_video = txt.get("process_video", "Process Video")
    process_videos = txt.get("process_videos", "Process Videos")
    channel_url = txt.get("channel_url", "Enter YouTube Channel URL")
    max_videos = txt.get("max_videos", "Maximum videos to process")
    channel_info = txt.get("channel_info", "Channels will be processed using the 'Bullet Points' summary style for efficient processing.")
    process_channel = txt.get("process_channel", "Process Channel")
    error_no_url = txt.get("error_no_url", "Please enter a YouTube video URL")
    error_no_urls = txt.get("error_no_urls", "Please enter at least one YouTube URL")
    error_no_channel = txt.get("error_no_channel", "Please enter a YouTube channel URL")
    
    language = st.session_state.get('language', 'en')
    
    st.header(youtube_header)
    
    # Create tabs
    video_tab, multi_tab, playlist_tab = st.tabs([
        tab_single, tab_multi, tab_playlist
    ])
    
    # Render tabs
    with video_tab:
        _render_single_video_tab(bot, provider, dataset_name, language, txt)
    with multi_tab:
        _render_multiple_videos_tab(bot, provider, dataset_name, language, txt)
    with playlist_tab:
        _render_playlist_tab(bot, provider, dataset_name, language, txt)
    
    # Add helpful information
    with st.expander("About YouTube Processing"):
        if language == "zh-TW":
            st.markdown("""
            ### 如何使用 YouTube 摘要工具
            
            此工具可以通過三種方式處理 YouTube 影片：
            
            **單一影片模式：**
            - 輸入特定 YouTube 影片的網址
            - 選擇您喜歡的摘要風格（詳細、簡潔或重點列表）
            - 工具將提取字幕並生成摘要
            
            **多部影片模式：**
            - 輸入多個 YouTube 網址（每行一個）
            - 選擇您喜歡的摘要風格
            - 批量處理所有影片
            - 結果保存到指定資料夾
            - 失敗的影片會單獨跟踪
            
            **播放清單模式：**
            - 輸入 YouTube 播放清單的網址
            - 選擇您喜歡的摘要風格
            - 批量處理播放清單中的所有影片
            - 結果保存到指定資料夾
            - 可以追蹤處理失敗的影片
            
            **提示：**
            - 確保影片有字幕/註釋
            - 摘要對內容豐富的影片效果最佳
            - 您可以將結果保存為 markdown 文件和/或添加到 RAG 數據庫
            - 處理多個影片時請耐心等待 - 每個影片處理都需要時間
            """)
        else:
            st.markdown("""
            ### How to use YouTube Summarizer
            
            This tool can process YouTube videos in three ways:
            
            **Single Video Mode:**
            - Enter a URL to a specific YouTube video
            - Choose your preferred summary style (Detailed, Concise, or Bullet Points)
            - The tool will extract the transcript and generate a summary
            
            **Multiple Videos Mode:**
            - Enter multiple YouTube URLs (one per line)
            - Choose your preferred summary style
            - Process all videos in batch
            - Results are saved to the specified folder
            - Failed videos are tracked separately
            
            **Playlist Mode:**
            - Enter a URL to a YouTube playlist
            - Choose your preferred summary style
            - Process all videos in the playlist
            - Results are saved to the specified folder
            - Failed videos are tracked separately
            
            **Tips:**
            - Make sure the videos have subtitles/captions available
            - Summaries work best with content-rich videos
            - You can save results as markdown files and/or add them to your RAG database
            - When processing multiple videos, be patient - each video takes time to process
            """)

def process_single_video(url, content_type, save_to_md, embed_to_rag, bot, provider, dataset_name, language):
    """Process a single YouTube video"""
    try:
        # Load video transcript
        with st.spinner("Extracting video transcript..." if language == "en" else "正在提取影片字幕..."):
            yt_loader = YoutubeLoader()
            success, content, dataset_name = yt_loader.process_video(url, language)
            
            if not success:
                st.error("Could not extract transcript from this video." if language == "en" else "無法從此影片提取字幕。")
                return
                
        with st.spinner("Processing content..." if language == "en" else "處理內容中..."):
            # Get LLM model and process content
            generator_llm, _ = bot.get_llm_and_embeddings(provider)
            system_prompt = _get_summary_prompt(content_type, language)
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", f"Video transcript: {content}" if language == "en" else f"影片字幕: {content}"),
            ])
            
            chain = prompt | generator_llm
            response = chain.invoke({})
                        
            # Handle response
            markdown_content = response.content if hasattr(response, 'content') else response
            st.markdown(markdown_content)

            video_info = (
                {"url": url, "title": ""}
            )

            _process_single_video_content(video_info, yt_loader, chain, "single", save_to_md, embed_to_rag, dataset_name, provider, language)
            
    except Exception as e:
        _handle_error(e, "processing YouTube video", language)

def process_multiple_videos(urls, content_type, output_folder, save_to_md, embed_to_rag, bot, provider, dataset_name, language):
    """Process multiple YouTube videos"""
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        progress_bar = st.progress(0)
        chain = _create_processing_chain(bot, provider, content_type, language)
        
        return _process_video_batch(
            urls, chain, output_folder, save_to_md, embed_to_rag,
            dataset_name, provider, language, progress_bar
        )
        
    except Exception as e:
        _handle_error(e, "processing multiple videos", language)
        return [], []

def process_channel_videos(channel_url, max_videos, output_folder, save_to_md, embed_to_rag, bot, provider, dataset_name, language):
    """Process videos from a YouTube channel"""
    try:
        videos = get_youtube_videos(channel_url)[:max_videos]
        if not videos:
            st.error("No videos found in this channel." if language == "en" else "在此頻道中找不到影片。")
            return
            
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        progress_bar = st.progress(0)
        chain = _create_processing_chain(bot, provider, "Bullet Points", language)
        
        return _process_video_batch(
            videos, chain, output_folder, save_to_md, embed_to_rag,
            dataset_name, provider, language, progress_bar
        )
        
    except Exception as e:
        _handle_error(e, "processing YouTube channel", language)

def process_playlist_videos(playlist_url, content_type, output_folder, save_to_md, embed_to_rag, bot, provider, dataset_name, language):
    """Process videos from a YouTube playlist"""
    try:
        # Get videos from playlist
        with st.spinner("Fetching playlist videos..." if language == "en" else "正在獲取播放清單影片..."):
            video_urls = get_playlist_videos(playlist_url)
            
            if not video_urls:
                st.error(
                    "No videos found in this playlist." if language == "en" 
                    else "在此播放清單中找不到影片。"
                )
                return
            
            st.info(
                f"Found {len(video_urls)} videos in playlist" if language == "en"
                else f"在播放清單中找到 {len(video_urls)} 個影片"
            )
        
        # Create output folder if needed
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Process videos
        progress_bar = st.progress(0)
        chain = _create_processing_chain(bot, provider, content_type, language)
        
        return _process_video_batch(
            video_urls, chain, output_folder, save_to_md, embed_to_rag,
            dataset_name, provider, language, progress_bar
        )
        
    except Exception as e:
        _handle_error(e, "processing YouTube playlist", language)

def _embed_to_rag_database(file_path, dataset_name, provider, language):
    """Embed content into RAG database"""
    try:
        from components.embedding import embed_text
        # Shows a loading spinner while embedding content
        # Uses English or Chinese text based on language setting
        with st.spinner(f"Adding content to {dataset_name} dataset..." if language == "en" else f"將內容添加到 {dataset_name} 數據集..."):
            embed_text(
                collection="bootcamp",
                dataset=dataset_name,
                markdown_file=file_path,
                provider=provider,
                force_recreate=False
            )
        
        # Shows success message after embedding completes
        # Uses English or Chinese text based on language setting
        st.success(f"Content successfully added to the {dataset_name} dataset!" if language == "en" else f"內容已成功添加到 {dataset_name} 數據集！")
    except Exception as e:
        st.warning(f"Error adding to RAG database: {str(e)}" if language == "en" else f"添加到 RAG 數據庫時出錯: {str(e)}")

def _handle_error(error, context, language):
    """Handle and display errors"""
    
    error_detail = traceback.format_exc()
    logger.error(f"Error {context}: {str(error)}\n{error_detail}")
    st.error(f"Error {context}: {str(error)}" if language == "en" else f"處理時出錯: {str(error)}")
    
    with st.expander("Debug Information"):
        st.code(error_detail)
                            
def _render_single_video_tab(bot, provider, dataset_name, language, txt):
    """Render the single video tab"""
    youtube_url = st.text_input(txt.get("youtube_url", "Enter YouTube Video URL"), 
                                placeholder="https://www.youtube.com/watch?v=...")
    
    # Use common content type selection
    content_type = get_content_type_selection(language, "single")
    
    col1, col2 = st.columns(2)
    with col1:
        save_to_md = st.checkbox(txt.get("save_md", "Save as Markdown file"), value=True)
    with col2:
        embed_to_rag = st.checkbox(txt.get("add_rag", "Add to RAG database"), value=True)
    
    if st.button(txt.get("process_video", "Process Video"), key="single_video_process_btn"):
        if youtube_url:
            process_single_video(youtube_url, content_type, save_to_md, embed_to_rag, bot, provider, dataset_name, language)
        else:
            st.warning(txt.get("error_no_url", "Please enter a YouTube video URL"))

def _render_multiple_videos_tab(bot, provider, dataset_name, language, txt):
    """Render the multiple videos tab"""
    youtube_urls = st.text_area(
        txt.get("youtube_urls", "Enter YouTube Video URLs (one per line)"),
        placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/watch?v=...",
        height=100
    )
    
    # Use common content type selection
    content_type = get_content_type_selection(language, "multi")
    
    # Processing options
    col1, col2 = st.columns(2)
    with col1:
        save_to_md = st.checkbox(txt.get("save_mds", "Save as Markdown files"), value=True, key="multi_save_md")
    with col2:
        embed_to_rag = st.checkbox(txt.get("add_rag", "Add to RAG database"), value=True, key="multi_embed_rag")
    
    # Output folder
    output_folder = st.text_input(txt.get("output_folder", "Output folder for markdown files"), value="video_summaries", key="multi_output_folder")
    
    if st.button(txt.get("process_videos", "Process Videos"), key="multiple_videos_process_btn"):
        # Split the input by lines and clean
        urls = [url.strip() for url in youtube_urls.split('\n') if url.strip()]
        
        if urls:
            process_multiple_videos(urls, content_type, output_folder, save_to_md, embed_to_rag, bot, provider, dataset_name, language)
        else:
            st.warning(txt.get("error_no_urls", "Please enter at least one YouTube URL"))

def _render_playlist_tab(bot, provider, dataset_name, language, txt):
    """Render the playlist tab"""
    playlist_url = st.text_input(
        txt.get("playlist_url", "Enter YouTube Playlist URL"), 
        placeholder="https://www.youtube.com/playlist?list=..."
    )
    
    # Use common content type selection
    content_type = get_content_type_selection(language, "playlist")
    
    # Output folder
    output_folder = st.text_input(
        txt.get("output_folder", "Output folder for markdown files"), 
        value="playlist_summaries",
        key="playlist_output_folder"
    )
    
    # Processing options
    col1, col2 = st.columns(2)
    with col1:
        save_to_md = st.checkbox(
            txt.get("save_mds", "Save as Markdown files"), 
            value=True, 
            key="playlist_save_md"
        )
    with col2:
        embed_to_rag = st.checkbox(
            txt.get("add_rag", "Add to RAG database"), 
            value=True, 
            key="playlist_embed_rag"
        )
    
    if st.button(txt.get("process_playlist", "Process Playlist"), key="playlist_process_btn"):
        if playlist_url:
            process_playlist_videos(
                playlist_url, content_type, output_folder, 
                save_to_md, embed_to_rag, bot, provider, dataset_name, language
            )
        else:
            st.warning(txt.get("error_no_playlist", "Please enter a YouTube playlist URL"))

def _get_summary_prompt(content_type, language):
    """Get the appropriate summary prompt based on content type and language"""
    if content_type == "Bullet Points":
        if language == "zh-TW":
            return (
                "請將以下影片字幕整理成結構良好的 markdown 文件。"
                "包含以下部分：\n\n"
                "# 影片標題\n\n"
                "## 摘要\n"
                "1-2 段簡短介紹影片內容的摘要。\n\n"
                "## 重點\n"
                "- 重點 1\n"
                "- 重點 2\n"
                "- ...\n\n"
                "專注於建立清晰簡潔的要點，捕捉主要觀點。"
            )
        else:
            return (
                "Please summarize the following video transcript into a well-structured markdown document. "
                "Include these sections:\n\n"
                "# Video Title\n\n"
                "## Summary\n"
                "A brief overview of the video content in 1-2 paragraphs.\n\n"
                "## Key Points\n"
                "- Bullet point 1\n"
                "- Bullet point 2\n"
                "- ...\n\n"
                "Focus on creating clear, concise bullet points that capture the main ideas."
            )
    elif content_type == "Concise":
        if language == "zh-TW":
            return (
                "請為以下影片字幕創建一個簡潔的 markdown 格式摘要。"
                "保持簡短但信息豐富。包含以下部分：\n\n"
                "# 影片標題\n\n"
                "## 摘要\n"
                "一段能夠捕捉影片精髓的摘要。\n\n"
                "## 主要收穫\n"
                "影片中最重要的一點。"
            )
        else:
            return (
                "Please create a concise summary of the following video transcript in markdown format. "
                "Keep it brief but informative. Include these sections:\n\n"
                "# Video Title\n\n"
                "## Summary\n"
                "A single paragraph summary that captures the essence of the video.\n\n"
                "## Main Takeaway\n"
                "The single most important point from the video."
            )
    else:  # Detailed
        if language == "zh-TW":
            return (
                "請為以下影片字幕創建一個詳細的 markdown 格式摘要。"
                "包含以下部分：\n\n"
                "# 影片標題\n\n"
                "## 概述\n"
                "1-2 段關於影片內容的概述。\n\n"
                "## 詳細摘要\n"
                "3-5 段影片內容的綜合梳理。\n\n"
                "## 關鍵點\n"
                "- 點 1\n"
                "- 點 2\n"
                "...\n\n"
                "## 結論\n"
                "影片的最終思考和收穫。"
            )
        else:
            return (
                "Please create a detailed summary of the following video transcript in markdown format. "
                "Include these sections:\n\n"
                "# Video Title\n\n"
                "## Overview\n"
                "A 1-2 paragraph overview of what the video covers.\n\n"
                "## Detailed Summary\n"
                "A comprehensive breakdown of the video content in 3-5 paragraphs.\n\n"
                "## Key Points\n"
                "- Point 1\n"
                "- Point 2\n"
                "...\n\n"
                "## Conclusion\n"
                "Final thoughts and takeaways from the video."
            ) 

def _process_single_video_content(video_info, yt_loader, chain, output_folder, save_to_md, embed_to_rag, dataset_name, provider, language):
    """Process a single video and return its processed content or None if failed"""
    try:
        # Get video details
        url = video_info.get('url')
        title = video_info.get('title', '')
        
        # Extract transcript
        success, content, dataset_name = yt_loader.process_video(url, language)
        if not success:
            raise ValueError("No transcript available")
            
        # Process with AI
        input_text = f"Video transcript: {content}" if language == "en" else f"影片字幕: {content}"
        response = chain.invoke({"transcript": input_text})
        markdown_content = response.content if hasattr(response, 'content') else response
        
        # Generate filename
        if not title:
            video_id = parse_video_id(url)
            title_match = re.search(r'# (.+)', markdown_content)
            title = title_match.group(1) if title_match else f"youtube_{video_id}"
            
        safe_title = "".join(c if c.isalnum() or c in [' ', '-', '_'] else '_' for c in title)
        filename = f"downloads/{output_folder}/{safe_title[:50]}.md"
        
        # Save to file if requested
        if save_to_md:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            
        # Add to RAG if requested
        if embed_to_rag:
            _embed_to_rag_database(filename, dataset_name, provider, language)
        
        return {
            "url": url,
            "title": title,
            "file": filename,
            "content": markdown_content,
            "success": True
        }
        
    except Exception as e:
        logger.warning(f"Error processing video {title or url}: {str(e)}")
        return {
            "url": url,
            "title": title,
            "success": False,
            "reason": str(e)
        }

def _process_video_batch(video_urls, chain, output_folder, save_to_md, embed_to_rag, dataset_name, provider, language, progress_bar):
    """Process multiple videos and batch embed their content"""
    youtube_loader = YoutubeLoader()
    processed_files = []
    failed_urls = []
    
    # Process all videos
    with st.spinner("Processing videos..."):
        for url in video_urls:
            try:
                success, result, dataset_name = youtube_loader.process_video(url, language)
                if success:
                    processed_files.append(result)
                else:
                    failed_urls.append((url, result))
            except Exception as e:
                logger.error(f"Error processing video {url}: {str(e)}")
                failed_urls.append((url, str(e)))
    
    # Show processing results
    _display_processing_results(processed_files, failed_urls)
    
    # Batch embed if we have successful processes
    if processed_files:
        _embed_processed_content(youtube_loader)

def _display_processing_results(processed_files, failed_urls):
    """Display the results of video processing"""
    if processed_files:
        st.success(f"Successfully processed {len(processed_files)} videos")
    if failed_urls:
        st.error("Failed to process these videos:")
        for url, error in failed_urls:
            st.error(f"{url}: {error}")

def _embed_processed_content(youtube_loader):
    """Embed all processed content to RAG database"""
    with st.spinner("Embedding all content to RAG database..."):
        try:
            success, message = youtube_loader.batch_embed_content(provider="gemini")
            if success:
                st.success("Successfully embedded all content!")
            else:
                st.error(f"Error during batch embedding: {message}")
        except Exception as e:
            logger.error(f"Error during batch embedding: {str(e)}")
            st.error(f"Error during batch embedding: {str(e)}") 

def _create_processing_chain(bot, provider, content_type, language):
    """
    Create a processing chain including LLM setup and prompt creation
    
    Args:
        bot: RAGBot instance
        provider: The model provider ("azure" or "gemini")
        content_type: Type of summary to generate
        language: Current language setting
        
    Returns:
        Chain: A LangChain chain for processing transcripts
    """
    # Get LLM model
    generator_llm, _ = bot.get_llm_and_embeddings(provider)
    
    # Get appropriate system prompt
    system_prompt = _get_summary_prompt(content_type, language)
    
    # Create and return the chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{transcript}")
    ])
    
    return prompt | generator_llm 

def get_content_type_selection(language, key_suffix=""):
    """Common function to handle content type selection across all tabs
    
    Args:
        language: Current language setting
        key_suffix: Suffix for the radio button key to make it unique
        
    Returns:
        str: Selected content type in English
    """
    content_options = ["Bullet Points", "Concise", "Detailed"]
    content_options_zh = ["重點列表", "簡潔", "詳細"]
    
    content_type = st.radio(
        st.session_state.get('langs', {}).get(language, {}).get("summary_style", "Summary Style:"),
        options=content_options_zh if language == "zh-TW" else content_options,
        horizontal=True,
        key=f"content_type_{key_suffix}"
    )
    
    # Map Chinese options back to English for processing
    if language == "zh-TW":
        content_map = dict(zip(content_options_zh, content_options))
        content_type = content_map.get(content_type, content_type)
    
    return content_type 