import asyncio
import os

import semantic_kernel as sk
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig
from semantic_kernel.prompt_template.input_variable import InputVariable
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureAISearchDataSources,
    AzureChatPromptExecutionSettings,
    AzureDataSources,
    ExtraBody,
)
from semantic_kernel.functions.function_result import FunctionResult

plugins_directory = os.path.join(__file__, "../plugins")
kernel = sk.Kernel()
deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env() # type: ignore
service_id = "chat-gpt"

def generate_initial_chat_function(kernel) -> sk.KernelFunction:
    chat_service = AzureChatCompletion(
        service_id = service_id,
        deployment_name=deployment,
        api_key=api_key,
        endpoint=endpoint,
        api_version="2023-12-01-preview",
        use_extensions=False,
    )

    try:
        kernel.remove_service(service_id)
    except:
        pass #it's okay if the service doesn't exist if this is the first run

    try:
        kernel.plugins.remove_by_name("chat_bot")
    except:
        pass #it's okay if the plugin doesn't exist if this is the first run

    kernel.add_service(chat_service)

    req_settings = kernel.get_service(service_id).instantiate_prompt_execution_settings(
        service_id=service_id, 
        max_tokens=2000,
        temperature=0.7,
        top_p=0.8,
    )

    prompt_template_config = PromptTemplateConfig(
        template="{{$chat_history}}{{$user_input}}",
        name="chat",
        input_variables=[
            InputVariable(name="input", description="User input", is_required=True),
            InputVariable(name="chat_history", description="The history of the conversation", is_required=True)
        ],
        execution_settings=req_settings
    )

    chat_function = kernel.create_function_from_prompt(
        plugin_name="chat_bot",
        function_name="chat",
        prompt_template_config=prompt_template_config
    )
    return chat_function

def setup_kernel_for_specific_index(context: FunctionResult, kernel: sk.Kernel) -> sk.KernelFunction:
    if(context == "movies"):
        print(f"ChatBot:> The context of your question has been determined to be {context}. Determining a response...")
        chat_function = generate_new_chat_function_using_index(kernel=kernel, index_name="movies")
        return chat_function
    
    if(context == "songs"):
        print(f"ChatBot:> The context of your question has been determined to be {context}. Determining a response...")
        chat_function = generate_new_chat_function_using_index(kernel=kernel, index_name="songs")
        return chat_function
    

    raise Exception(f"No function found: {context}")

def generate_new_chat_function_using_index(kernel: sk.Kernel, index_name: str) -> sk.KernelFunction:
    search_api_key, search_url = sk.azure_aisearch_settings_from_dot_env(include_index_name=False) # type: ignore
    azure_ai_search_settings = {
            "key": search_api_key, 
            "endpoint": search_url, 
            "indexName": index_name, # <-- this is the name of the specific Azure AI Search index you want to use
            "fieldsMapping": {
                "titleField": "title",
                "vectorFields": ["vector"]
            }
        }
        
    az_source = AzureAISearchDataSources(**azure_ai_search_settings) # type: ignore
    az_data = AzureDataSources(type="AzureCognitiveSearch", parameters=az_source)
    extra = ExtraBody(dataSources=[az_data]) # type: ignore
    req_settings = AzureChatPromptExecutionSettings(extra_body=extra) # type: ignore
    req_settings.service_id = service_id

    chat_service = AzureChatCompletion(
            service_id=service_id,
            deployment_name=deployment,
            api_key=api_key,
            endpoint=endpoint,
            api_version="2023-12-01-preview",
            use_extensions=True,
        )
    
    #replace the existing chat service (that wasn't pointed at a specific Azure AI Search index) with the new chat service
    kernel.remove_service(service_id)
    kernel.plugins.remove_by_name("chat_bot")
    
    kernel.add_service(chat_service)

    prompt_template_config = PromptTemplateConfig(
        template="{{$chat_history}}{{$input}}",
        name="chat",
        input_variables=[
            InputVariable(name="input", description="User input", is_required=True),
            InputVariable(name="chat_history", description="The history of the conversation", is_required=True)
        ],
        execution_settings=req_settings
    )

    chat_function = kernel.create_function_from_prompt(
        plugin_name="chat_bot",
        function_name="chat",
        prompt_template_config=prompt_template_config
    )
    return chat_function

async def chat(chat_history: ChatHistory) -> bool:
    generate_initial_chat_function(kernel)

    try:
        kernel.plugins.remove_by_name("IntentDetectionPlugin")
    except:
        pass #it's okay if the plugin doesn't exist if this is the first run

    intent_plugin = kernel.import_plugin_from_prompt_directory(plugins_directory + "/semantic", "IntentDetectionPlugin")

    try:
        user_input = input("User:> ")
        chat_history.add_user_message(user_input)
    except KeyboardInterrupt:
        print("\n\nExiting chat...")
        return False
    except EOFError:
        print("\n\nExiting chat...")
        return False

    if user_input == "exit":
        print("\n\nExiting chat...")
        return False
    
    #use the IntentDetectionPlugin-AssistantIntent plugin to detect the intent of the user input
    response = await kernel.invoke(
        intent_plugin["AssistantIntent"], 
        input=user_input, 
        chat_history=chat_history)

    context = response.value[-1].content

    #if the intent is found, then use the intent to setup the kernel for a specific index
    if(context != "not_found"):
        # get a new chat function that is pointed at a specific Azure AI Search index        
        chat_function_for_index = setup_kernel_for_specific_index(context, kernel)

        #answer the user's question using the specific index set up in the previous step
        response = await kernel.invoke(chat_function_for_index, input=user_input, chat_history=chat_history)

        context = response.value[-1].content

        chat_history.add_assistant_message(context)

        print(f"ChatBot:> {context}")
    else:
        message = f"ChatBox:> I was unable to determine the context of your question, please try again."
        chat_history.add_assistant_message(message)
        print(message)

    return True

async def main() -> None:
    system_message = """
    ChatBot can have a conversation with you about any topic.
    It can give explicit instructions or say 'I don't know'
    when it doesn't know the answer.
    """

    chat_history = ChatHistory(system_message=system_message)

    chatting = True
    while chatting:
        chatting = await chat(chat_history)

if __name__ == "__main__":
    asyncio.run(main())