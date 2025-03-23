import streamlit as st
from components.vector_store import VectorStoreManager

def handle_collection_cleaning(collections: list[str], dataset_name: str, vector_manager: VectorStoreManager):
    """Handle the cleaning process for selected collections"""
    results = []
    with st.spinner("Cleaning collections..."):
        for collection in collections:
            deleted_count, status = vector_manager.clean_collection_by_dataset(
                collection, 
                dataset_name
            )
            results.append({
                "collection": collection,
                "deleted": deleted_count,
                "status": status
            })
    
    display_cleaning_results(results)

def display_cleaning_results(results: list[dict]):
    """Display the results of the cleaning operation"""
    st.subheader("Cleaning Results", divider="rainbow")
    for result in results:
        if "Error" in result["status"]:
            st.error(f"Failed to clean {result['collection']}: {result['status']}")
        else:
            st.success(f"Cleaned {result['deleted']} entries from {result['collection']}")

def display_collection_info(collections: list[str], vector_manager: VectorStoreManager):
    """Display information for selected collections"""
    cols = st.columns(len(collections))
    for idx, collection in enumerate(collections):
        with cols[idx]:
            with st.expander(f"Collection: {collection}"):
                collection_info = vector_manager.get_collection_info(collection)
                st.json(collection_info)

def render_vector_store_tab(dataset_name: str):
    """Render the vector store management tab"""
    st.header("Vector Store Management")

    vector_manager = VectorStoreManager()
    collections = vector_manager.list_collections()
    
    if not collections:
        st.warning("No collections found in the vector store.")
        return

    st.info(f"Managing collections for dataset: **{dataset_name}**", icon="ℹ️")

    selected_collections = st.multiselect(
        "Select Collections to Clean",
        options=collections,
        help="Select one or more collections to clean data from"
    )

    if not selected_collections:
        return

    # Display collection information
    display_collection_info(selected_collections, vector_manager)

    # Clean button
    col1, col2 = st.columns([1, 3])
    with col1:
        clean_clicked = st.button(
            "Clean Collections", 
            key="clean_collections_btn", 
            type="primary"
        )
    
    if clean_clicked:
        with col2:
            confirm_clicked = st.button(
                "⚠️ Confirm Clean", 
                key="confirm_clean_btn", 
                type="secondary",
                help=f"This will clean dataset '{dataset_name}' from selected collections"
            )
            if confirm_clicked:
                handle_collection_cleaning(selected_collections, dataset_name, vector_manager) 