

import os
import dashscope
from dotenv import load_dotenv
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

load_dotenv()

messages = [
    {
        "role": "user",
        "content": [
            {"image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"},
            {"text": "图中描绘的是什么景象?"}]
    }]

response = dashscope.MultiModalConversation.call(
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    model='qwen3.8-max',
    messages=messages
)

print(response.output.choices[0].message.content[0]["text"])
