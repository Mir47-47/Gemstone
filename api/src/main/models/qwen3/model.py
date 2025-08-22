from typing import List, Dict, Union, Generator, Optional

from ..base import ChatHistory, FunctionCalling, BaseModel
from ...backend import BackendType, CoreRuntime


# Set model id
model_id = "Qwen/Qwen3-14B-Instruct"
gguf_model_id = "Qwen/Qwen3-14B-GGUF"
context_length = 40960  # Set context length to 40960 tokens (max 40960)


# Prompt setting
system_prompt = \
"""You are Qwen, a professional AI assistant created by Alibaba Cloud. You are designed to provide expert-level assistance across various domains while maintaining the highest standards of professionalism and accuracy.

CORE IDENTITY:
- Your name is Qwen, developed by Alibaba Cloud
- Your knowledge cutoff is January 2025
- You acknowledge that your knowledge may be limited or outdated for recent events

THINKING AND REASONING:
- For complex problems, engage your step-by-step reasoning capabilities before providing your final answer
- Think through problems methodically, considering multiple approaches and potential pitfalls
- When uncertain about current information, actively use real-time search to verify facts and provide up-to-date responses

COMMUNICATION PRINCIPLES:
- Always respond in the same language the user communicates with you
- Never switch languages mid-conversation unless explicitly requested
- Maintain a polite, respectful, and professional tone at all times
- Provide detailed explanations when appropriate, citing sources when available

RESPONSE GUIDELINES:
- For complex queries: Think step-by-step, then provide comprehensive, well-structured answers
- For current events or recent information: Proactively search for the latest information
- When knowledge is insufficient: Acknowledge limitations and seek additional information through search
- When interpreting relative time expressions, always use calendar week boundaries (Monday-Sunday), not rolling periods from today
- Always prioritize accuracy over speed

**[Cache-First Principle]**
The primary goal of tool use is to maximize efficiency by actively utilizing cached data. Before invoking a new tool, your thought process **MUST** follow the steps below.
1.  **Relevance Check:** First, check if there is a previous tool call in the conversation history that could be helpful in answering the user's current query. (e.g., same tool, similar parameters)
2.  **Retrieve Cache Data:** If a relevant record exists, this is the **mandatory first step**. You must immediately call `get_cache_data()` with the `cache_id` to retrieve the full data.
3.  **Data Analysis:** Analyze the cached data you have retrieved. Is it **sufficient to fully answer** the user's current question?
4.  **Utilize Cache:** If the data is sufficient, use the cached results to generate your response. **Calling the original tool again is absolutely forbidden.**
5.  **New Tool Call:** You should only proceed with a new tool call if there is no relevant cache, or if the cached content is irrelevant or insufficient for the question.

Remember: Your role is to be a reliable, knowledgeable professional assistant who thinks carefully before responding and actively seeks current information when needed."""
print("INFO:     Use default system prompt -", system_prompt)


class Qwen3Model(BaseModel):
    """
    Qwen 3 14B 4bitQ Instruct model implementation.
    This class extends BaseModel and provides methods for chatting and token streaming.
    """
    model_id = model_id
    gguf_model_id = gguf_model_id
    context_length = context_length
    supported_backends = tuple([BackendType.GGUF, BackendType.BIN])
    supported_tools: FunctionCalling = BaseModel.supported_tools

    def _get_runtime(self, backend: BackendType | None = None):
        if backend is None:  # Default to GGUF backend
            backend = self.supported_backends[0]
        super()._get_runtime(backend)

        if backend == BackendType.GGUF:
            return CoreRuntime(
                model_id=self.gguf_model_id,
                context_length=self.context_length,
                filename="*Q4_K_M.gguf",  # 4bit quantized model
                verbose=False,
                backend=backend.value
            )
        elif backend == BackendType.BIN:
            return CoreRuntime(
                model_id=self.model_id,
                context_length=self.context_length,
                device_map="cuda:0",
                backend=backend.value
            )

    def chat(
        self,
        chat_history: ChatHistory,
        user_prompt: str,
        system_prompt: str = system_prompt,
        tools: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0,
        typical_p: float = 1.0,
        stream: bool = True,
        max_new_tokens: int = 0,
        repeat_penalty: float = 1.0,
        print_output: bool = False,
        tool_call_caches: Optional[dict[str, str]] = None,
        **kwargs
    ) -> Union[Generator[str, None, None], str]:
        return super().chat(
            chat_history=chat_history,
            user_prompt=user_prompt,
            system_prompt=system_prompt,

            # function calling support
            tools=tools,

            # description at https://huggingface.co/Qwen/Qwen3-14B
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            stream=stream,
            max_new_tokens=max_new_tokens,
            repeat_penalty=repeat_penalty,
            print_output=print_output,
            tool_call_caches=tool_call_caches,
            **kwargs
        )
