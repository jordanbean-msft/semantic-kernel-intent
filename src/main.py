import asyncio

import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.connectors.ai.open_ai.request_settings.azure_chat_request_settings import (
    AzureAISearchDataSources,
    AzureChatRequestSettings,
    AzureDataSources,
    ExtraBody,
)

sk_prompt = """
ChatBot can have a conversation with you about any topic.
It can give explicit instructions or say 'I don't know'
when it doesn't know the answer.

{{$chat_history}}
User:> {{$user_input}}
ChatBot:>
"""

kernel = sk.Kernel()

deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env() # type: ignore

#req_settings = AzureChatRequestSettings()#extra_body=extra)
prompt_config = sk.PromptTemplateConfig()#execution_settings=req_settings)

chat_service = sk_oai.AzureChatCompletion(
    deployment_name=deployment,
    api_key=api_key,
    endpoint=endpoint,
    api_version="2023-12-01-preview",
    use_extensions=False,
)

kernel.add_chat_service("chat-gpt", chat_service)

prompt_template = sk.PromptTemplate(sk_prompt, kernel.prompt_template_engine, prompt_config)

function_config = sk.SemanticFunctionConfig(prompt_config, prompt_template)
chat_function = kernel.register_semantic_function("ChatBot", "Chat", function_config)

context = kernel.create_new_context()

async def chat(context_vars: sk.ContextVariables) -> bool:
    try:
        user_input = input("User:> ")
        context_vars["user_input"] = user_input
    except KeyboardInterrupt:
        print("\n\nExiting chat...")
        return False
    except EOFError:
        print("\n\nExiting chat...")
        return False

    if user_input == "exit":
        print("\n\nExiting chat...")
        return False
    
    answer = await kernel.run(chat_function, input_vars=context_vars)
    context_vars["chat_history"] += f"\nUser:> {user_input}\nChatBot:> {answer}\n"

    print(f"ChatBot:> {answer}")
 
    return True

async def main() -> None:
    context = sk.ContextVariables()
    context["chat_history"] = ""
    chatting = True
    while chatting:
        chatting = await chat(context)

if __name__ == "__main__":
    asyncio.run(main())