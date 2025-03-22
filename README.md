# Taller LLM Clase

Este repositorio contiene dos archivos Jupyter Notebook (`TALLER_LLMCLASE.ipynb` y `TALLER_LLMCLASE2.ipynb`) que demuestran el uso de modelos de lenguaje y procesamiento de documentos utilizando la biblioteca `langchain`.

## Archivos

### `TALLER_LLMCLASE.ipynb`

Este archivo muestra cómo inicializar un modelo de chat, enviar mensajes y recibir respuestas. También incluye ejemplos de cómo usar plantillas de prompts para traducir texto.

#### Contenido

1. **Inicialización del modelo de chat**:
    ```python
    from langchain.chat_models import init_chat_model

    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    ```

2. **Enviar y recibir mensajes**:
    ```python
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage("Translate the following from English into Italian"),
        HumanMessage("hi!"),
    ]

    response = model.invoke(messages)
    print(response.content)  # Esto debería imprimir "Ciao!"
    ```

3. **Uso de plantillas de prompts**:
    ```python
    from langchain_core.prompts import ChatPromptTemplate

    system_template = "Translate the following from English into {language}"
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )

    prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
    messages = prompt.to_messages()
    response = model.invoke(prompt)
    print(response.content)  # Esto debería imprimir "Ciao!"
    ```

### `TALLER_LLMCLASE2.ipynb`

Este archivo muestra cómo inicializar un modelo de embeddings, cargar y dividir contenido web, y construir una aplicación de preguntas y respuestas utilizando un grafo de estado.

#### Contenido

1. **Inicialización del modelo de embeddings y almacenamiento vectorial**:
    ```python
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.vectorstores import InMemoryVectorStore

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = InMemoryVectorStore(embeddings)
    ```

2. **Cargar y dividir contenido web**:
    ```python
    import bs4
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = WebBaseLoader(
        web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=["post-content", "post-title", "post-header"]
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    ```

3. **Indexar los fragmentos y definir el prompt**:
    ```python
    _ = vector_store.add_documents(documents=all_splits)

    from langchain import hub
    prompt = hub.pull("rlm/rag-prompt")
    ```

4. **Inicializar el modelo de chat y definir el estado**:
    ```python
    from langchain.chat_models import init_chat_model
    from typing_extensions import List, TypedDict
    from langchain_core.documents import Document

    llm = init_chat_model("gpt-4o-mini", model_provider="openai")

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str
    ```

5. **Definir los pasos de la aplicación y compilar el grafo**:
    ```python
    from langgraph.graph import START, StateGraph

    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    response = graph.invoke({"question": "What is Task Decomposition?"})
    print(response["answer"])
    ```

## Requisitos

- Python 3.8 o superior
- Bibliotecas necesarias: `langchain`, `langchain_openai`, `langchain_core`, `langchain_community`, `langgraph`, `bs4`, `pkg_resources`

## Instalación

Para instalar las bibliotecas necesarias, puedes usar los siguientes comandos:

```sh
pip install langchain langchain_openai langchain_core langchain_community langgraph beautifulsoup4