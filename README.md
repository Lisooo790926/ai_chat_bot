## Slides

https://kuiming.github.io/rag_path/output/#/

## Code

### Packages

```bash
pip install requirements.txt
```

### Get recipe from youtube for RAG

- `youtube_recipe.py`: get recipe from youtube
- `rag.py`: tranform recipe into embedding vector
- `bot.py`: create chatbot with `streamlit`

### Crawler with structured output
- `espn_openai.py`: get commentaries from ESPN

## Estimated Monthly Cost

The following table provides a ballpark estimate of the monthly costs to run this project on Azure. Please note that actual costs may vary based on usage, deployment configurations, and scaling factors.

| **Service Component**       | **Description**                                                                                             | **Estimated Monthly Cost** |
| --------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------- |
| Azure OpenAI Service        | Chat completions for GPT-4 usage (prompt and completion tokens)                                             | \$20 - \$50                |
| Azure OpenAI Embeddings     | Embedding operations for document processing (lower cost per token; typically included in usage costs)      | Included in usage          |
| Compute & Hosting           | Hosting of the Streamlit UI (bot.py, espn_openai.py) on Azure App Service, Container, or VM                   | \$10 - \$50                |
| Qdrant Vector Store         | Deployment and hosting of the Qdrant vector store for document retrieval and embedding management           | \$10 - \$50                |
| **Total Estimate**          | Combined estimated cost for Azure services and compute resources                                            | \$30 - \$100               |

*Note:* These figures are estimates. Actual costs depend on your specific usage, chosen service tiers, and optimization strategies.

## Acknowledgements
This project was inspired by [iamlazy](https://github.com/narumiruna/iamlazy).
