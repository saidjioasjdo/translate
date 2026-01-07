import os
import re
import uvicorn
from fastapi import FastAPI, HTTPException
import dotenv
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

# 加载模型
dotenv.load_dotenv(dotenv_path=".env")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model_name="Qwen/Qwen2.5-7B-Instruct", )
prompt = PromptTemplate.from_template(
    template="""
    你是一位专业的中英双向口笔译专家。
请将以下句子翻译（中文→英文 或 英文→中文），要求：
1. 翻译准确、自然、符合母语者表达习惯
2. 保留原文语气和语感
3. 在翻译后单独列出翻译过程中最关键的 3-6 个词汇/短语，只需展示关键的词汇即可，关键词另起一行进行输出
格式要求：{{"translation": "翻译结果", "keywords": ["关键词1", "关键词2"]}}
原句：{sentence}

"""
)
chain = LLMChain(llm=model, verbose=False, prompt=prompt)

app = FastAPI(title="translate_server")
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/translate")
async def translate(sentence: str):
    try:
        response = chain.invoke(input={"sentence": sentence})
        return response["text"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"错误为:{e}")


if __name__ == '__main__':
    uvicorn.run(
        app=app,
        host="127.0.0.1",
        port=8000
    )
