from typing import Iterator, Protocol, TypedDict

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langgraph.graph import START, StateGraph
from llama_cpp import CreateChatCompletionStreamResponse, Llama

from .db import get_vector_store


class Llm(Protocol):
    def invoke(self, messages: list[str]) -> str: ...


class LocalLlm:
    def __init__(
        self,
        repo_id: str = "TheBloke/CodeLlama-7B-Instruct-GGUF",
        filename: str = "codellama-7b-instruct.Q4_K_M.gguf",
    ):
        self.model = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=16384,
            n_batch=512,
            n_gpu_layers=32,
            stream=True,
        )

        self.system_prompt = (
            "You are a helpful assistant. "
            "You will answer questions based on the provided context. "
            "If the context does not contain enough information, "
            "you will say 'I don't know'."
        )
        self.inference_params = {
            "temperature": 0.1,
            "top_p": 0.95,
            "repeat_penalty": 1.1,
        }

    def _collect_response_content(
        self, response: Iterator[CreateChatCompletionStreamResponse]
    ) -> str:
        """Collects the content from a streamed response."""
        content = []
        for chunk in response:
            if "choices" in chunk and chunk["choices"]:
                content.append(chunk["choices"][0].get("delta", {}).get("content", ""))
                print(content[-1], end="", flush=True)
        return "".join(content)

    def invoke(self, messages: list[str]) -> str:
        system_message = {"role": "system", "content": self.system_prompt}
        user_messages = [{"role": "user", "content": msg} for msg in messages]
        response = self.model.create_chat_completion(
            messages=[system_message] + user_messages,  # type: ignore
            **self.inference_params,  # type: ignore
        )
        return response["choices"][0]["message"]["content"]  # type: ignore


class State(TypedDict, total=False):
    question: str
    context: list[Document]
    answer: str


class Prompt:
    """Dummy class for now"""

    def invoke(self, state: State) -> str:
        sep = "\n\n" + 10 * "-" + "\n\n"
        return f"""
        Q: {state.get("question", "")}
        
        Context: {sep.join(doc.page_content for doc in state.get("context", []))}

        Generate a concise answer based on the question and context."""


class FakeLlm:
    def invoke(self, messages: list[str]) -> str:
        prompt_text = messages[0] if messages else "No prompt provided"
        return f"Generated answer for prompt: {prompt_text}"


def _retrieve(state: State, vector_store: VectorStore) -> State:
    retrieved_docs = vector_store.similarity_search(state.get("question", ""), k=5)
    return {"context": retrieved_docs}


def _generate(state: State, prompt: Prompt, llm: Llm) -> State:
    prompt_text = prompt.invoke(state)
    answer = llm.invoke([prompt_text])
    return {"answer": answer}


def ask_question(question: str) -> None:
    llm = LocalLlm()
    prompt = Prompt()
    vector_store = get_vector_store()

    def retrieve(state: State) -> State:
        return _retrieve(state, vector_store)

    def generate(state: State) -> State:
        return _generate(state, prompt, llm)

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    question = "What type of database is used in PKB?"
    response = graph.invoke({"question": question})
    print(response.get("answer", "No answer generated"))
