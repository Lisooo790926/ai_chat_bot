"""Language utilities for multilingual support"""

def get_language_options():
    """Get supported language options"""
    return {
        "en": "English",
        "zh-TW": "繁體中文"
    }

def get_localization_dict():
    """Return dictionary with all translatable strings"""
    return {
        "en": {
            "title": "RAG bot",
            "provider_select": "Select AI Provider",
            "dataset_select": "Select dataset to query",
            "tab_chat": "Chat",
            "tab_upload": "Upload Documents",
            "tab_youtube": "YouTube Video Summarizer",
            "chat_placeholder": "Type your message here...",
            "upload_header": "Upload Documents to Knowledge Base",
            "upload_info": "Documents will be embedded using the {provider} provider selected in the sidebar.",
            "upload_label": "Upload a markdown (.md) or text (.txt) file",
            "force_recreate": "Force recreate collection (use if you encounter dimension errors)",
            "process_embed": "Process and Embed Document",
            "please_upload": "Please upload a file first.",
            "embedding_spinner": "Embedding document into {dataset} dataset using {provider} provider...",
            "embed_success": "Document successfully embedded into the {dataset} dataset!",
            "youtube_header": "YouTube Video Summarizer",
            "tab_single": "Single Video",
            "tab_multi": "Multiple Videos",
            "tab_channel": "Channel",
            "youtube_url": "Enter YouTube Video URL",
            "youtube_urls": "Enter YouTube Video URLs (one per line)",
            "summary_style": "Summary Style:",
            "save_md": "Save as Markdown file",
            "save_mds": "Save as Markdown files",
            "add_rag": "Add to RAG database",
            "output_folder": "Output folder for markdown files",
            "process_video": "Process Video",
            "process_videos": "Process Videos",
            "channel_url": "Enter YouTube Channel URL",
            "max_videos": "Maximum videos to process",
            "channel_info": "Channels will be processed using the 'Bullet Points' summary style for efficient processing.",
            "process_channel": "Process Channel",
            "error_no_url": "Please enter a YouTube video URL",
            "error_no_urls": "Please enter at least one YouTube URL",
            "error_no_channel": "Please enter a YouTube channel URL"
        },
        "zh-TW": {
            "title": "RAG 機器人",
            "provider_select": "選擇 AI 提供者",
            "dataset_select": "請選擇要查詢的資料集名稱",
            "tab_chat": "對話",
            "tab_upload": "上傳文件",
            "tab_youtube": "YouTube 影片摘要",
            "chat_placeholder": "在此輸入您的訊息...",
            "upload_header": "上傳文件至知識庫",
            "upload_info": "文件將使用側邊欄中選擇的 {provider} 提供者進行嵌入。",
            "upload_label": "上傳 markdown (.md) 或文本 (.txt) 文件",
            "force_recreate": "強制重新創建集合（如果遇到維度錯誤請使用）",
            "process_embed": "處理並嵌入文件",
            "please_upload": "請先上傳文件。",
            "embedding_spinner": "使用 {provider} 提供者將文件嵌入到 {dataset} 數據集中...",
            "embed_success": "文件已成功嵌入到 {dataset} 數據集中！",
            "youtube_header": "YouTube 影片摘要",
            "tab_single": "單一影片",
            "tab_multi": "多部影片",
            "tab_channel": "頻道",
            "youtube_url": "輸入 YouTube 影片網址",
            "youtube_urls": "輸入 YouTube 影片網址（每行一個）",
            "summary_style": "摘要風格：",
            "save_md": "儲存為 Markdown 文件",
            "save_mds": "儲存為 Markdown 文件",
            "add_rag": "添加到 RAG 數據庫",
            "output_folder": "Markdown 文件輸出資料夾",
            "process_video": "處理影片",
            "process_videos": "處理影片",
            "channel_url": "輸入 YouTube 頻道網址",
            "max_videos": "最多處理的影片數量",
            "channel_info": "頻道處理將使用「重點列表」摘要風格以提高效率。",
            "process_channel": "處理頻道",
            "error_no_url": "請輸入 YouTube 影片網址",
            "error_no_urls": "請至少輸入一個 YouTube 網址",
            "error_no_channel": "請輸入 YouTube 頻道網址"
        }
    }

def get_system_prompt(language="en"):
    """Get system prompt based on language"""
    if language == "zh-TW":
        return (
            "你是一位專門根據文件回答問題的 AI 助手。如果你無法從文件得到答案，請說你不知道。"
            "請根據以下參考資料回答問題："
            "歷史紀錄：{chat_history}"
            "參考資料：{context}"
        )
    else:
        return (
            "You are an AI assistant that answers questions based on provided documents. If you don't know the answer based on the documents, say you don't know."
            "Please answer the question based on the following reference materials:"
            "Chat history: {chat_history}"
            "References: {context}"
        ) 