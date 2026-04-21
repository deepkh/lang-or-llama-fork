## Naive RAG

One simple round of QA (i.e. no conversation memory); no external tools

* Data: single pdf file
* Basic indexing: customized size + overlap
* Simple top-k embedding retrieval
* Qdrant vector DB
* Customized embedding (using HuggingFace)
* Customized LLM (OpenAI API for LangChain; Ollama for LlamaIndex)
* Customized prompt

This comparison focuses on their styles and the mindset/philosophy.
It does NOT push the limit of two frameworks on
* How good they can parse a PDF
* How good the index data structure is

`constants.py` contains common things that the separate ETL and inference scripts
can leverage.

### Run

Start Qdrant before running ETL or retrieval.
The separate ETL and inference scripts import modules from the repo root, so run
them with `PYTHONPATH=$(pwd)`. The combined LlamaIndex script is self-contained
and can run directly.

```bash
### ETL (once)
# LangChain
$ PYTHONPATH=$(pwd) python src/naive_rag/la_etl.py 
# LlamaIndex
$ PYTHONPATH=$(pwd) python src/naive_rag/ll_etl.py

### Inference
# LangChain
$ PYTHONPATH=$(pwd) python src/naive_rag/la_rag.py 
# LlamaIndex
$ PYTHONPATH=$(pwd) python src/naive_rag/ll_rag.py

### Combined ETL + inference
# LlamaIndex with Ollama qwen3:4b
# Uses collection naive_rag_ll_etl_rag, then deletes it before exit.
$ python src/naive_rag/ll_etl_rag.py
```
