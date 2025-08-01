from typing import List, Dict, Union, Generator, Optional

from ..base import ChatHistory, FunctionCalling, BaseModel
from ...backend import BackendType, CoreRuntime


# Set model id
model_id = "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"
context_length = 131072  # Set context length to 131072 tokens (max)


# Prompt setting
system_prompt = "" \
                + "Your name is `Llama 3.1` developed by Meta AI. " \
                + "You are a helpful, smart, kind, and efficient AI Assistant. " \
                + "Your knowledge is based on data available up to December 2023. " \
                + "You always fulfill the user's requests to the best of your ability. " \
                + "You are not aware of any events or developments that occurred after that date unless explicitly provided by the user. " \
                + "Do not fabricate information about events beyond your knowledge cutoff. If uncertain, state that clearly. " \
                + "Detect the user's input language and always respond in the same language. " \
                + "Do not switch languages unless explicitly instructed. " \
                + "Maintain a consistent, polite, and contextually appropriate tone for each language."
print("INFO:     Use default system prompt -", system_prompt)


class Llama3Model(BaseModel):
    """
    Llama 3.1 8B 4bitQ Instruct model implementation.
    This class extends BaseModel and provides methods for chatting and token streaming.
    """
    model_id = model_id
    context_length = context_length
    supported_backends = tuple([BackendType.GGUF])
    supported_tools: FunctionCalling = FunctionCalling.DISABLED

    def _get_runtime(self, backend: BackendType | None = None):
        if backend is None:  # Default to GGUF backend
            backend = self.supported_backends[0]
        super()._get_runtime(backend)

        return CoreRuntime(
            model_id=self.model_id,
            context_length=self.context_length,
            filename="*Q4_K_M.gguf",  # 4bit quantized model
            verbose=False,
            backend=backend.value
        )

    def chat(
        self,
        chat_history: ChatHistory,
        user_prompt: str,
        system_prompt: str = system_prompt,
        tools: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        stream: bool = True,
        max_new_tokens: int = 0,
        repeat_penalty: float = 1.0,
        print_output: bool = False,
        **kwargs
    ) -> Union[Generator[str, None, None], str]:
        return super().chat(
            chat_history=chat_history,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            tools=tools,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            stream=stream,
            max_new_tokens=max_new_tokens,
            repeat_penalty=repeat_penalty,
            print_output=print_output,
            **kwargs
        )
