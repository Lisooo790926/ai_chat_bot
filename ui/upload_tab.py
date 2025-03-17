import streamlit as st
import tempfile
import os
import traceback
import logging

logger = logging.getLogger(__name__)

def render_upload_tab(dataset_name, provider):
    """Render the document upload tab UI"""
    # Get text for current language
    txt = st.session_state.get('langs', {}).get(st.session_state.get('language', 'en'), {})
    
    # Set default text if localization not found
    upload_header = txt.get("upload_header", "Upload Documents to Knowledge Base")
    upload_info = txt.get("upload_info", "Documents will be embedded using the {provider} provider selected in the sidebar.")
    upload_label = txt.get("upload_label", "Upload a markdown (.md) or text (.txt) file")
    force_recreate = txt.get("force_recreate", "Force recreate collection (use if you encounter dimension errors)")
    process_embed = txt.get("process_embed", "Process and Embed Document")
    please_upload = txt.get("please_upload", "Please upload a file first.")
    embedding_spinner = txt.get("embedding_spinner", "Embedding document into {dataset} dataset using {provider} provider...")
    embed_success = txt.get("embed_success", "Document successfully embedded into the {dataset} dataset!")
    
    st.header(upload_header)
    
    # Import here to avoid circular imports
    from components.embedding import embed_text
    
    # Display the selected provider (informational only)
    st.info(upload_info.format(provider=provider.upper()))
    
    # File uploader
    uploaded_file = st.file_uploader(
        upload_label, 
        type=["md", "txt"]
    )
    
    # Force recreate option
    force_recreate_option = st.checkbox(
        force_recreate,
        value=False
    )
    
    if st.button(process_embed):
        if uploaded_file is not None:
            # Create a temporary file
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    # Write the uploaded file content
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    with st.spinner(embedding_spinner.format(dataset=dataset_name, provider=provider)):
                        # Call the embed_text function
                        logger.info(f"Embedding document: {uploaded_file.name} into dataset: {dataset_name} with provider: {provider}")
                        result = embed_text(
                            collection="bootcamp",
                            dataset=dataset_name,
                            markdown_file=tmp_path,
                            provider=provider,
                            force_recreate=force_recreate_option
                        )
                        st.success(embed_success.format(dataset=dataset_name))
                except Exception as e:
                    error_detail = traceback.format_exc()
                    logger.error(f"Error embedding document: {str(e)}\n{error_detail}")
                    st.error(f"Error embedding document: {str(e)}")
                    
                    # Show detailed error information in an expander for debugging
                    with st.expander("Debug Information"):
                        st.code(error_detail)
                finally:
                    # Clean up the temporary file
                    os.unlink(tmp_path)
            except Exception as e:
                error_detail = traceback.format_exc()
                logger.error(f"Error handling file upload: {str(e)}\n{error_detail}")
                st.error(f"Error handling file upload: {str(e)}")
                
                # Show detailed error information in an expander for debugging
                with st.expander("Debug Information"):
                    st.code(error_detail)
        else:
            st.warning(please_upload)
    
    # Add some helpful instructions
    with st.expander("How to use document upload"):
        if st.session_state.get('language', 'en') == "zh-TW":
            st.markdown("""
            ### 上傳文件的技巧：
            
            1. **文件格式**：上傳 markdown (.md) 或文本 (.txt) 文件
            2. **內容結構**：具有清晰標題的文件（在 markdown 中使用 # 語法）會有更好的分塊效果
            3. **提供者選擇**：使用側邊欄選擇用於聊天和嵌入的 AI 提供者
            4. **強制重新創建**：如果遇到維度不匹配問題，請使用此選項
            
            上傳後，您的文件將被分塊並嵌入到向量數據庫中，使機器人能夠在回答中引用它。
            """)
        else:
            st.markdown("""
            ### Tips for uploading documents:
            
            1. **File Format**: Upload markdown (.md) or text (.txt) files
            2. **Content Structure**: Documents with clear headings (using # syntax in markdown) will be chunked better
            3. **Provider Selection**: Use the sidebar to select the AI provider for both chat and embedding
            4. **Force Recreate**: Use this option if you encounter dimension mismatches
            
            After uploading, your document will be chunked and embedded in the vector database, allowing the bot to reference it in answers.
            """) 