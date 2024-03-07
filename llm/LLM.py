from autogen import oai
import openai

from .utils import *

def conversation(prompts, args):
    openai.api_key = args.openai_token
    model = 'gpt-3.5-turbo-1106'

    # create a chat completion request
    message_list = get_message_list(args, prompts)

    response = oai.ChatCompletion.create(
        # config_list=config_list_gpt4,
        model=model,
        messages=message_list,
        # request_timeout=300,  # may be necessary for larger models
    )
    print('Response {}:\n'.format(1))
    print(response.choices[0].message.content, '\n\n')
    rerank_text = response.choices[0].message.content

    return rerank_text