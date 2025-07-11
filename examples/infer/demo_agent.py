# Copyright (c) Alibaba, Inc. and its affiliates.
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def infer(engine: 'InferEngine', infer_request: 'InferRequest'):
    stop = [engine.default_template.agent_template.keyword.observation]  # compat react_en
    request_config = RequestConfig(max_tokens=512, temperature=0, stop=stop)
    resp_list = engine.infer([infer_request], request_config)
    query = infer_request.messages[0]['content']
    response = resp_list[0].choices[0].message.content
    print(f'query: {query}')
    print(f'response: {response}')

    tool = '{"temperature": 32, "condition": "Sunny", "humidity": 50}'
    print(f'tool: {tool}')
    infer_request.messages += [{'role': 'assistant', 'content': response}, {'role': 'tool', 'content': tool}]
    resp_list = engine.infer([infer_request], request_config)
    response2 = resp_list[0].choices[0].message.content
    print(f'response2: {response2}')


def infer_stream(engine: 'InferEngine', infer_request: 'InferRequest'):
    stop = [engine.default_template.agent_template.keyword.observation]
    request_config = RequestConfig(max_tokens=512, temperature=0, stream=True, stop=stop)
    gen_list = engine.infer([infer_request], request_config)
    query = infer_request.messages[0]['content']
    response = ''
    print(f'query: {query}\nresponse: ', end='')
    for resp in gen_list[0]:
        if resp is None:
            continue
        delta = resp.choices[0].delta.content
        response += delta
        print(delta, end='', flush=True)
    print()

    tool = '{"temperature": 32, "condition": "Sunny", "humidity": 50}'
    print(f'tool: {tool}\nresponse2: ', end='')
    infer_request.messages += [{'role': 'assistant', 'content': response}, {'role': 'tool', 'content': tool}]
    gen_list = engine.infer([infer_request], request_config)
    for resp in gen_list[0]:
        if resp is None:
            continue
        print(resp.choices[0].delta.content, end='', flush=True)
    print()


def get_infer_request():
    return InferRequest(
        messages=[{
            'role': 'user',
            'content': "How's the weather in Beijing today?"
        }],
        tools=[{
            'name': 'get_current_weather',
            'description': 'Get the current weather in a given location',
            'parameters': {
                'type': 'object',
                'properties': {
                    'location': {
                        'type': 'string',
                        'description': 'The city and state, e.g. San Francisco, CA'
                    },
                    'unit': {
                        'type': 'string',
                        'enum': ['celsius', 'fahrenheit']
                    }
                },
                'required': ['location']
            }
        }])


if __name__ == '__main__':
    from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig
    model = 'Qwen/Qwen2.5-1.5B-Instruct'
    infer_backend = 'pt'

    if infer_backend == 'pt':
        engine = PtEngine(model, max_batch_size=64)
    elif infer_backend == 'vllm':
        from swift.llm import VllmEngine
        engine = VllmEngine(model, max_model_len=8192)
    elif infer_backend == 'lmdeploy':
        from swift.llm import LmdeployEngine
        engine = LmdeployEngine(model)

    infer(engine, get_infer_request())
    infer_stream(engine, get_infer_request())
