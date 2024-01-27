import asyncio
import os
from openai import NotFoundError

import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.connectors.ai.open_ai.request_settings.azure_chat_request_settings import (
    AzureAISearchDataSources,
    AzureChatRequestSettings,
    AzureDataSources,
    ExtraBody,
)
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
"""

kernel = sk.Kernel()

deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env() # type: ignore

req_settings = AzureChatRequestSettings(
    service_id="chat-gpt",
    ai_model_id=deployment,
    max_tokens=2000,
    temperature=0.7,
    top_p=0.8,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    number_of_responses=1
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

intent_plugin = kernel.import_semantic_plugin_from_directory(plugins_directory + "/semantic", "IntentDetectionPlugin")

def SetupKernel(context: sk.KernelContext, kernel: sk.Kernel) -> sk.KernelFunctionBase:
    if(context.result == "hvac"):    
        plugin = kernel.import_semantic_plugin_from_directory(plugins_directory + "/semantic", "HVACPlugin")
        
        search_api_key, search_url = sk.azure_aisearch_settings_from_dot_env(include_index_name=False) # type: ignore
        azure_ai_search_settings = {
            "key": search_api_key, 
            "endpoint": search_url, 
            "indexName": "hvac",
            "fieldsMapping": {
                "titleField": "title",
                "contentFields": ["content"],
                "vectorFields": ["contentVector"]
            }
        }
        
        az_source = AzureAISearchDataSources(**azure_ai_search_settings) # type: ignore
        az_data = AzureDataSources(type="AzureCognitiveSearch", parameters=az_source)
        extra = ExtraBody(dataSources=[az_data]) # type: ignore
        req_settings = AzureChatRequestSettings(extra_body=extra) # type: ignore
        prompt_config = sk.PromptTemplateConfig(execution_settings=req_settings)
        chat_service = sk_oai.AzureChatCompletion(
            deployment_name=deployment,
            api_key=api_key,
            endpoint=endpoint,
            api_version="2023-12-01-preview",
            use_extensions=True,
        )
        kernel.remove_chat_service("chat-gpt")
        kernel.add_chat_service("chat-gpt", chat_service)

        prompt_template = OpenAIChatPromptTemplate(
            template="{{$user_input}}", 
            template_engine=kernel.prompt_template_engine, 
            prompt_config=prompt_config)
        
        function_config = sk.SemanticFunctionConfig(prompt_config, prompt_template)        
        chat_function = kernel.register_semantic_function("ChatBot", "Chat", function_config)

        #return plugin["AssistantHVAC"]
        return chat_function
    
    raise Exception(f"No function found: {context.result}")

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

    if(context.result != "not_found"):
        answer_function = SetupKernel(context, kernel)

        context = await kernel.run(answer_function, input_context=context)

    context.variables["chat_history"] += f"\nUser:> {user_input}\nChatBot:> {context.result}\n"

    print(f"ChatBot:> {context.result}")

    print(f"Tool:> {context.objects.get('tool_message')}")
 
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