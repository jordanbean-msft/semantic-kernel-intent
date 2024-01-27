import asyncio
import os

import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.connectors.ai.open_ai.request_settings.azure_chat_request_settings import (
    AzureAISearchDataSources,
    AzureChatRequestSettings,
    AzureDataSources,
    ExtraBody,
)
from semantic_kernel.planning import ActionPlanner
from semantic_kernel.connectors.ai.open_ai.utils import (
    chat_completion_with_function_call,
    get_function_calling_object,
)
from semantic_kernel.connectors.ai.open_ai.semantic_functions.open_ai_chat_prompt_template import (
    OpenAIChatPromptTemplate,
)

plugins_directory = os.path.join(__file__, "../plugins")
system_message = """
ChatBot can have a conversation with you about any topic.
It can give explicit instructions or say 'I don't know'
when it doesn't know the answer.

{{$chat_history}}
"""

kernel = sk.Kernel()

deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env() # type: ignore

tools=[
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "hvac_questions",
    #         "description": "HVAC Questions",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "user_input": {
    #                     "type": "string",
    #                     "description": "User Input",
    #                 }
    #             },
    #             "required": ["user_input"]
    #         }
    #     }
    # }
]

req_settings = AzureChatRequestSettings(
    service_id="chat-gpt",
    ai_model_id=deployment,
    max_tokens=2000,
    temperature=0.7,
    top_p=0.8,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    number_of_responses=1,
    tool_choice="intentdetectionplugin-assistantintent",
    #tool_choice="auto",
    tools=get_function_calling_object(kernel, { "exclude_plugin": ["ChatBot"] })
)
prompt_config = sk.PromptTemplateConfig(execution_settings=req_settings)

chat_service = sk_oai.AzureChatCompletion(
    deployment_name=deployment,
    api_key=api_key,
    endpoint=endpoint,
    api_version="2023-12-01-preview",
    use_extensions=False,
)

kernel.add_chat_service("chat-gpt", chat_service)

prompt_template = OpenAIChatPromptTemplate(
    template="{{$user_input}}", 
    template_engine=kernel.prompt_template_engine, 
    prompt_config=prompt_config)

prompt_template.add_system_message(system_message)

function_config = sk.SemanticFunctionConfig(prompt_config, prompt_template)
chat_function = kernel.register_semantic_function("ChatBot", "Chat", function_config)

intent_plugin = kernel.import_semantic_plugin_from_directory(plugins_directory, "IntentDetectionPlugin")
#hvac_plugin = kernel.import_semantic_plugin_from_directory(plugins_directory, "HVACPlugin")
#fire_plugin = kernel.import_semantic_plugin_from_directory(plugins_directory, "FirePlugin")

filter = {"exclude_plugin": ["ChatBot"]}
functions = get_function_calling_object(kernel, filter)

async def chat(context: sk.KernelContext) -> bool:
    try:
        user_input = input("User:> ")
        context.variables["user_input"] = user_input
    except KeyboardInterrupt:
        print("\n\nExiting chat...")
        return False
    except EOFError:
        print("\n\nExiting chat...")
        return False

    if user_input == "exit":
        print("\n\nExiting chat...")
        return False
    
    context = await kernel.run(intent_plugin["AssistantIntent"], input_context=context)
    #context = await chat_function.invoke_async(context=context)
    #context = await chat_completion_with_function_call(
    #    kernel, 
    #    chat_plugin_name="ChatBot",
    #    chat_function_name="Chat",
    #    context=context,
    #    functions=functions)
    #context.variables["chat_history"] += f"\nUser:> {user_input}\nChatBot:> {context.result}\n"

    print(f"ChatBot:> {context.result}")
 
    return True

async def main() -> None:
    context_vars = sk.ContextVariables()
    context_vars["chat_history"] = ""
    context = kernel.create_new_context(context_vars)
    chatting = True
    while chatting:
        chatting = await chat(context)

if __name__ == "__main__":
    asyncio.run(main())