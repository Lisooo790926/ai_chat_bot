import streamlit as st
import tempfile
import os
import re
import traceback
import logging
from components.youtube_to_md import YoutubeLoader, parse_video_id, get_youtube_videos
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
    tab_channel = txt.get("tab_channel", "Channel")
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
    
    # Add tabs for single video vs multiple videos vs channel
    video_tab, multi_tab, channel_tab = st.tabs([tab_single, tab_multi, tab_channel])
    
    # Single video tab
    with video_tab:
        _render_single_video_tab(bot, provider, dataset_name, language, txt)
    
    # Multiple videos tab
    with multi_tab:
        _render_multiple_videos_tab(bot, provider, dataset_name, language, txt)
    
    # Channel tab
    with channel_tab:
        _render_channel_tab(bot, provider, dataset_name, language, txt)
    
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
            
            **頻道模式：**
            - 輸入 YouTube 頻道的網址
            - 選擇要處理的影片數量（最多 10 個）
            - 工具將為每個影片創建重點列表摘要
            
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
            
            **Channel Mode:**
            - Enter a URL to a YouTube channel
            - Choose how many videos to process (up to 10)
            - The tool will create bullet-point summaries for each video
            
            **Tips:**
            - Make sure the videos have subtitles/captions available
            - Summaries work best with content-rich videos
            - You can save results as markdown files and/or add them to your RAG database
            - When processing multiple videos, be patient - each video takes time to process
            """)

def _render_single_video_tab(bot, provider, dataset_name, language, txt):
    """Render the single video tab"""
    # YouTube URL input
    youtube_url = st.text_input(
        txt.get("youtube_url", "Enter YouTube Video URL"), 
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    # Content type selection
    content_options = ["Detailed", "Concise", "Bullet Points"]
    content_options_zh = ["詳細", "簡潔", "重點列表"]
    content_type = st.radio(
        txt.get("summary_style", "Summary Style:"),
        options=content_options_zh if language == "zh-TW" else content_options,
        horizontal=True
    )
    
    # Map Chinese options back to English for processing
    if language == "zh-TW":
        content_map = dict(zip(content_options_zh, content_options))
        content_type = content_map.get(content_type, content_type)
    
    # Processing options
    col1, col2 = st.columns(2)
    with col1:
        save_to_md = st.checkbox(txt.get("save_md", "Save as Markdown file"), value=True)
    with col2:
        embed_to_rag = st.checkbox(txt.get("add_rag", "Add to RAG database"), value=True)
    
    if st.button(txt.get("process_video", "Process Video")):
        if youtube_url:
            try:
                # Load video transcript
                with st.spinner("Extracting video transcript..." if language == "en" else "正在提取影片字幕..."):
                    try:
                        yt_loader = YoutubeLoader()
                        content = yt_loader.load(youtube_url)
                        
                        if not content:
                            st.error("Could not extract transcript from this video." if language == "en" else "無法從此影片提取字幕。")
                            st.stop()
                    except Exception as transcript_error:
                        error_detail = traceback.format_exc()
                        logger.error(f"Error extracting transcript: {str(transcript_error)}\n{error_detail}")
                        st.error(f"Error extracting transcript: {str(transcript_error)}" if language == "en" else f"提取字幕時出錯: {str(transcript_error)}")
                        with st.expander("Debug Information"):
                            st.code(error_detail)
                        st.stop()
                
                # Get video metadata for filename
                video_id = parse_video_id(youtube_url)
                
                with st.spinner("Processing content..." if language == "en" else "處理內容中..."):
                    try:
                        # Choose prompt based on content type and language
                        system_prompt = _get_summary_prompt(content_type, language)
                        
                        # Get LLM model based on provider selection
                        generator_llm, _ = bot.get_llm_and_embeddings(provider)
                        
                        # Process with LangChain
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", system_prompt),
                            ("human", f"Video transcript: {content}" if language == "en" else f"影片字幕: {content}"),
                        ])
                        
                        chain = prompt | generator_llm
                        logger.info(f"Processing video: {youtube_url} with style: {content_type}")
                        response = chain.invoke({})
                        
                        # Handle both string responses and object responses with .content attribute
                        if hasattr(response, 'content'):
                            markdown_content = response.content
                        else:
                            markdown_content = response  # If response is already a string
                    except Exception as processing_error:
                        error_detail = traceback.format_exc()
                        logger.error(f"Error processing content: {str(processing_error)}\n{error_detail}")
                        st.error(f"Error processing content: {str(processing_error)}" if language == "en" else f"處理內容時出錯: {str(processing_error)}")
                        with st.expander("Debug Information"):
                            st.code(error_detail)
                        st.stop()
                    
                    # Display the generated content
                    st.markdown(markdown_content)
                    
                    # Extract title for filename
                    title_match = re.search(r'# (.+)', markdown_content)
                    if title_match:
                        title = title_match.group(1)
                        safe_title = "".join(c if c.isalnum() or c in [' ', '-', '_'] else '_' for c in title)
                        filename = f"{safe_title[:50]}.md"
                    else:
                        filename = f"youtube_{video_id}.md"
                    
                    # Save to file if requested
                    if save_to_md:
                        try:
                            # Create temp file
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp_file:
                                tmp_path = tmp_file.name
                                tmp_file.write(markdown_content.encode('utf-8'))
                            
                            # Provide download button
                            with open(tmp_path, "r", encoding="utf-8") as f:
                                st.download_button(
                                    label="Download as Markdown" if language == "en" else "下載為 Markdown",
                                    data=f.read(),
                                    file_name=filename,
                                    mime="text/markdown"
                                )
                            
                            # Add to RAG if requested
                            if embed_to_rag:
                                try:
                                    with st.spinner(f"Adding content to {dataset_name} dataset..." if language == "en" else f"將內容添加到 {dataset_name} 數據集..."):
                                        from components.embedding import embed_text
                                        result = embed_text(
                                            collection="bootcamp",
                                            dataset=dataset_name,
                                            markdown_file=tmp_path,
                                            provider=provider,
                                            force_recreate=False
                                        )
                                        st.success(f"Content successfully added to the {dataset_name} dataset!" if language == "en" else f"內容已成功添加到 {dataset_name} 數據集！")
                                except Exception as embed_error:
                                    error_detail = traceback.format_exc()
                                    logger.error(f"Error adding to RAG database: {str(embed_error)}\n{error_detail}")
                                    st.error(f"Error adding to RAG database: {str(embed_error)}" if language == "en" else f"添加到 RAG 數據庫時出錯: {str(embed_error)}")
                                    with st.expander("Debug Information"):
                                        st.code(error_detail)
                            
                            # Clean up temp file
                            os.unlink(tmp_path)
                        except Exception as file_error:
                            error_detail = traceback.format_exc()
                            logger.error(f"Error handling file operations: {str(file_error)}\n{error_detail}")
                            st.error(f"Error handling file operations: {str(file_error)}" if language == "en" else f"處理文件操作時出錯: {str(file_error)}")
                            with st.expander("Debug Information"):
                                st.code(error_detail)
            
            except Exception as e:
                error_detail = traceback.format_exc()
                logger.error(f"Error processing YouTube video: {str(e)}\n{error_detail}")
                st.error(f"Error processing YouTube video: {str(e)}" if language == "en" else f"處理 YouTube 影片時出錯: {str(e)}")
                with st.expander("Debug Information"):
                    st.code(error_detail)
        else:
            st.warning(txt.get("error_no_url", "Please enter a YouTube video URL"))

def _render_multiple_videos_tab(bot, provider, dataset_name, language, txt):
    """Render the multiple videos tab"""
    # Multiple YouTube URLs input
    youtube_urls = st.text_area(
        txt.get("youtube_urls", "Enter YouTube Video URLs (one per line)"),
        placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/watch?v=...",
        height=100
    )
    
    # Content type selection
    content_options = ["Bullet Points", "Concise", "Detailed"]
    content_options_zh = ["重點列表", "簡潔", "詳細"]
    content_type = st.radio(
        txt.get("summary_style", "Summary Style:"),
        options=content_options_zh if language == "zh-TW" else content_options,
        horizontal=True,
        key="multi_content_type"
    )
    
    # Map Chinese options back to English for processing
    if language == "zh-TW":
        content_map = dict(zip(content_options_zh, content_options))
        content_type = content_map.get(content_type, content_type)
    
    # Processing options
    col1, col2 = st.columns(2)
    with col1:
        save_to_md = st.checkbox(txt.get("save_mds", "Save as Markdown files"), value=True, key="multi_save_md")
    with col2:
        embed_to_rag = st.checkbox(txt.get("add_rag", "Add to RAG database"), value=True, key="multi_embed_rag")
    
    # Output folder
    output_folder = st.text_input(txt.get("output_folder", "Output folder for markdown files"), value="video_summaries", key="multi_output_folder")
    
    if st.button(txt.get("process_videos", "Process Videos")):
        # Split the input by lines and clean
        urls = [url.strip() for url in youtube_urls.split('\n') if url.strip()]
        
        if urls:
            try:
                # Create output folder if it doesn't exist
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                
                # Display progress bar
                progress_bar = st.progress(0)
                
                # Get LLM model based on provider selection
                try:
                    generator_llm, _ = bot.get_llm_and_embeddings(provider)
                    
                    # Choose prompt based on content type and language
                    system_prompt = _get_summary_prompt(content_type, language)
                    
                    # Create prompt template
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "Video transcript: {transcript}" if language == "en" else "影片字幕: {transcript}"),
                    ])
                    
                    # Create processing chain
                    chain = prompt | generator_llm
                except Exception as model_error:
                    error_detail = traceback.format_exc()
                    logger.error(f"Error initializing AI model: {str(model_error)}\n{error_detail}")
                    st.error(f"Error initializing AI model: {str(model_error)}" if language == "en" else f"初始化 AI 模型時出錯: {str(model_error)}")
                    with st.expander("Debug Information"):
                        st.code(error_detail)
                    st.stop()
                
                # Process each video
                yt_loader = YoutubeLoader()
                processed_videos = []
                failed_videos = []
                
                for i, url in enumerate(urls):
                    try:
                        # Display status
                        st.write(f"Processing video {i+1}/{len(urls)}: {url}" if language == "en" else f"處理影片 {i+1}/{len(urls)}: {url}")
                        logger.info(f"Processing video [{i+1}/{len(urls)}]: {url}")
                        
                        # Extract transcript
                        try:
                            content = yt_loader.load(url)
                            
                            if not content:
                                st.warning(f"Could not extract transcript from: {url}" if language == "en" else f"無法從此影片提取字幕: {url}")
                                failed_videos.append({"url": url, "reason": "No transcript available" if language == "en" else "無法獲取字幕"})
                                continue
                        except Exception as transcript_error:
                            error_detail = traceback.format_exc()
                            logger.warning(f"Error extracting transcript: {str(transcript_error)}\n{error_detail}")
                            st.warning(f"Error extracting transcript from: {url}" if language == "en" else f"提取字幕時出錯: {url}")
                            failed_videos.append({"url": url, "reason": f"Transcript error: {str(transcript_error)}" if language == "en" else f"字幕錯誤: {str(transcript_error)}"})
                            continue
                        
                        # Process with AI
                        try:
                            # Generate summary
                            response = chain.invoke({"transcript": content})
                            
                            # Handle both string responses and object responses with .content attribute
                            if hasattr(response, 'content'):
                                markdown_content = response.content
                            else:
                                markdown_content = response  # If response is already a string
                            
                            # Extract title for filename
                            video_id = parse_video_id(url)
                            title_match = re.search(r'# (.+)', markdown_content)
                            if title_match:
                                title = title_match.group(1)
                                safe_title = "".join(c if c.isalnum() or c in [' ', '-', '_'] else '_' for c in title)
                                filename = f"{safe_title[:50]}.md"
                            else:
                                filename = f"youtube_{video_id}.md"
                            
                            # Save to file
                            file_path = f"{output_folder}/{filename}"
                            with open(file_path, "w", encoding="utf-8") as f:
                                f.write(markdown_content)
                            
                            # Add to processed list
                            processed_videos.append({"url": url, "file": file_path, "content": markdown_content})
                            
                            # Add to RAG if requested
                            if embed_to_rag:
                                try:
                                    from components.embedding import embed_text
                                    embed_text(
                                        collection="bootcamp",
                                        dataset=dataset_name,
                                        markdown_file=file_path,
                                        provider=provider,
                                        force_recreate=False
                                    )
                                except Exception as embed_error:
                                    error_detail = traceback.format_exc()
                                    logger.warning(f"Error adding to RAG database: {str(embed_error)}\n{error_detail}")
                                    st.warning(f"Error adding {url} to RAG database: {str(embed_error)}" if language == "en" else f"將 {url} 添加到 RAG 數據庫時出錯: {str(embed_error)}")
                        except Exception as processing_error:
                            error_detail = traceback.format_exc()
                            logger.warning(f"Error processing content: {str(processing_error)}\n{error_detail}")
                            st.warning(f"Error processing video: {url}" if language == "en" else f"處理影片時出錯: {url}")
                            failed_videos.append({"url": url, "reason": f"Processing error: {str(processing_error)}" if language == "en" else f"處理錯誤: {str(processing_error)}"})
                            continue
                    
                    except Exception as video_error:
                        error_detail = traceback.format_exc()
                        logger.warning(f"Error processing video: {str(video_error)}\n{error_detail}")
                        st.warning(f"Error processing video: {url}" if language == "en" else f"處理影片時出錯: {url}")
                        failed_videos.append({"url": url, "reason": f"Unknown error: {str(video_error)}" if language == "en" else f"未知錯誤: {str(video_error)}"})
                    
                    # Update progress bar
                    progress_bar.progress((i + 1) / len(urls))
                
                # Show results
                if processed_videos:
                    st.success(f"Successfully processed {len(processed_videos)} out of {len(urls)} videos!" if language == "en" else f"成功處理 {len(processed_videos)} 個影片，共 {len(urls)} 個！")
                    
                    # Show preview of first video
                    if len(processed_videos) > 0:
                        with st.expander("Preview first processed video" if language == "en" else "預覽第一個處理的影片"):
                            st.markdown(processed_videos[0]["content"])
                    
                    # List all processed videos
                    with st.expander("All Processed Videos" if language == "en" else "所有處理的影片"):
                        for video in processed_videos:
                            st.write(f"- {video['url']} → {video['file']}")
                
                # Show failed videos if any
                if failed_videos:
                    with st.expander(f"Failed Videos ({len(failed_videos)})" if language == "en" else f"處理失敗的影片 ({len(failed_videos)})"):
                        for video in failed_videos:
                            st.write(f"- {video['url']}: {video['reason']}")
            
            except Exception as e:
                error_detail = traceback.format_exc()
                logger.error(f"Error processing multiple videos: {str(e)}\n{error_detail}")
                st.error(f"Error processing multiple videos: {str(e)}" if language == "en" else f"處理多個影片時出錯: {str(e)}")
                with st.expander("Debug Information"):
                    st.code(error_detail)
        else:
            st.warning(txt.get("error_no_urls", "Please enter at least one YouTube URL"))

def _render_channel_tab(bot, provider, dataset_name, language, txt):
    """Render the channel tab"""
    # Channel URL input
    channel_url = st.text_input(
        txt.get("channel_url", "Enter YouTube Channel URL"), 
        placeholder="https://www.youtube.com/@channelname"
    )
    
    # Max videos to process
    max_videos = st.slider(txt.get("max_videos", "Maximum videos to process"), 1, 10, 3)
    
    # Summary style info
    st.info(txt.get("channel_info", "Channels will be processed using the 'Bullet Points' summary style for efficient processing."))
    
    # Output folder
    output_folder = st.text_input(txt.get("output_folder", "Output folder for markdown files"), value="summaries")
    
    # Process button
    col1, col2 = st.columns(2)
    with col1:
        save_to_md = st.checkbox(txt.get("save_mds", "Save as Markdown files"), value=True, key="channel_save_md")
    with col2:
        embed_to_rag = st.checkbox(txt.get("add_rag", "Add to RAG database"), value=True, key="channel_embed_rag")
    
    if st.button(txt.get("process_channel", "Process Channel")):
        if channel_url:
            try:
                with st.spinner("Fetching channel videos..." if language == "en" else "正在獲取頻道影片..."):
                    try:
                        videos = get_youtube_videos(channel_url)
                        
                        if not videos:
                            st.error("No videos found in this channel." if language == "en" else "在此頻道中找不到影片。")
                            st.stop()
                    except Exception as channel_error:
                        error_detail = traceback.format_exc()
                        logger.error(f"Error fetching channel videos: {str(channel_error)}\n{error_detail}")
                        st.error(f"Error fetching channel videos: {str(channel_error)}" if language == "en" else f"獲取頻道影片時出錯: {str(channel_error)}")
                        with st.expander("Debug Information"):
                            st.code(error_detail)
                        st.stop()
                    
                    # Limit to max_videos
                    videos = videos[:max_videos]
                    
                    # Create output folder if it doesn't exist
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    
                    # Display progress bar
                    progress_bar = st.progress(0)
                    
                    try:
                        # Get LLM model
                        generator_llm, _ = bot.get_llm_and_embeddings(provider)
                        
                        # Create summary prompt
                        summary_prompt = ChatPromptTemplate.from_messages([
                            ("system", (
                                _get_summary_prompt("Bullet Points", language)
                            )),
                            ("human", "{transcript}" if language == "en" else "影片字幕: {transcript}")
                        ])
                        
                        # Create chain
                        summary_chain = summary_prompt | generator_llm
                    except Exception as model_error:
                        error_detail = traceback.format_exc()
                        logger.error(f"Error initializing AI model: {str(model_error)}\n{error_detail}")
                        st.error(f"Error initializing AI model: {str(model_error)}" if language == "en" else f"初始化 AI 模型時出錯: {str(model_error)}")
                        with st.expander("Debug Information"):
                            st.code(error_detail)
                        st.stop()
                    
                    # Process each video
                    yt_loader = YoutubeLoader()
                    processed_videos = []
                    
                    for i, video in enumerate(videos):
                        try:
                            # Update status
                            st.write(f"Processing: {video['title']}" if language == "en" else f"正在處理: {video['title']}")
                            logger.info(f"Processing video [{i+1}/{len(videos)}]: {video['title']}")
                            
                            # Get transcript
                            try:
                                content = yt_loader.load(video['url'])
                                
                                if not content:
                                    st.warning(f"Could not extract transcript for: {video['title']}" if language == "en" else f"無法提取此影片的字幕: {video['title']}")
                                    continue
                            except Exception as transcript_error:
                                error_detail = traceback.format_exc()
                                logger.warning(f"Error extracting transcript for video {video['title']}: {str(transcript_error)}\n{error_detail}")
                                st.warning(f"Error extracting transcript for: {video['title']} - {str(transcript_error)}" if language == "en" else f"提取字幕時出錯: {video['title']} - {str(transcript_error)}")
                                continue
                            
                            # Process with LangChain
                            try:
                                response = summary_chain.invoke({"transcript": f"Video transcript: {content}" if language == "en" else f"影片字幕: {content}"})
                                
                                # Handle both string responses and object responses with .content attribute
                                if hasattr(response, 'content'):
                                    markdown_summary = response.content
                                else:
                                    markdown_summary = response
                                
                                # Create safe filename
                                safe_title = "".join(c if c.isalnum() or c in [' ', '-', '_'] else '_' for c in video['title'])
                                filename = f"{output_folder}/{safe_title[:50]}.md"
                                
                                # Save to file
                                if save_to_md:
                                    with open(filename, "w", encoding="utf-8") as f:
                                        f.write(markdown_summary)
                                
                                # Add to processed list
                                processed_videos.append({"title": video['title'], "file": filename})
                                
                                # Optionally add to RAG
                                if embed_to_rag:
                                    try:
                                        from components.embedding import embed_text
                                        embed_text(
                                            collection="bootcamp",
                                            dataset=dataset_name,
                                            markdown_file=filename,
                                            provider=provider,
                                            force_recreate=False
                                        )
                                    except Exception as embed_error:
                                        error_detail = traceback.format_exc()
                                        logger.warning(f"Error adding video {video['title']} to RAG: {str(embed_error)}\n{error_detail}")
                                        st.warning(f"Error adding {video['title']} to RAG database: {str(embed_error)}" if language == "en" else f"將 {video['title']} 添加到 RAG 數據庫時出錯: {str(embed_error)}")
                            except Exception as processing_error:
                                error_detail = traceback.format_exc()
                                logger.warning(f"Error processing video {video['title']}: {str(processing_error)}\n{error_detail}")
                                st.warning(f"Error processing video: {video['title']} - {str(processing_error)}" if language == "en" else f"處理影片時出錯: {video['title']} - {str(processing_error)}")
                                continue
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(videos))
                        
                        except Exception as video_error:
                            error_detail = traceback.format_exc()
                            logger.warning(f"Error in video loop for {video['title']}: {str(video_error)}\n{error_detail}")
                            st.warning(f"Error processing {video['title']}: {str(video_error)}" if language == "en" else f"處理 {video['title']} 時出錯: {str(video_error)}")
                    
                    # Show completion message
                    if processed_videos:
                        st.success(f"Successfully processed {len(processed_videos)} videos!" if language == "en" else f"成功處理 {len(processed_videos)} 個影片！")
                        
                        # Display processed files
                        st.subheader("Processed Videos:" if language == "en" else "已處理的影片:")
                        for video in processed_videos:
                            st.write(f"- {video['title']} → {video['file']}")
                    else:
                        st.error("Could not process any videos from this channel." if language == "en" else "無法處理此頻道的任何影片。")
            
            except Exception as e:
                error_detail = traceback.format_exc()
                logger.error(f"Error processing YouTube channel: {str(e)}\n{error_detail}")
                st.error(f"Error processing YouTube channel: {str(e)}" if language == "en" else f"處理 YouTube 頻道時出錯: {str(e)}")
                with st.expander("Debug Information"):
                    st.code(error_detail)
        else:
            st.warning(txt.get("error_no_channel", "Please enter a YouTube channel URL"))

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