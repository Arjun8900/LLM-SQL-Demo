# LLM
This Demo demonstrate to use a model offline. Since, its downloaded there is no need for any Key.
1. Install all packages using `!pip install openai langchain huggingface_hub transformers sentence_transformers datasets`
2. Download the model
3. Use the downloaded model to 

- [Using a Small model](#Using-a-Small-model)
  - [DEMO](#demo)
  - [Download the model in local](#download-the-model-in-local)
  - [Create pipeline](#create-pipeline)
  - [Create prompt and write the text](#create-prompt-and-write-the-text)
# Using a Small model
## DEMO
![Text2SQL](https://github.com/Arjun8900/LLM-SQL-Demo/assets/30146648/df3f86ae-1fd7-48ca-8cb9-76013b02372b)

![Text2SQL_Demo](https://github.com/Arjun8900/LLM-SQL-Demo/assets/30146648/47b03e7a-fcb2-4407-8a14-d1bf106f2237)
## Download the model in local
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate

model = "mrm8488/t5-base-finetuned-wikiSQL"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForSeq2SeqLM.from_pretrained(model)
```

## Create pipeline
```python
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=100)
local_llm = HuggingFacePipeline(pipeline=pipe)
```

## Create prompt and write the text
```python
prompt = PromptTemplate(
    input_variables=["question"],
    template="Translate English to SQL: {question}"
)
hub_chain = LLMChain(prompt=prompt, llm=local_llm, verbose=True)
print(hub_chain.run("What is the average age of respondents using mobile device"))
```
# Using a Large model