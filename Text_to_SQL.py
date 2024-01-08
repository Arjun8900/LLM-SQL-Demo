import warnings
warnings.filterwarnings("ignore")

# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate

model_name = "mrm8488/t5-base-finetuned-wikiSQL"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=100)
local_llm = HuggingFacePipeline(pipeline=pipe)


prompt = PromptTemplate(
    input_variables=["question"],
    template="Translate English to SQL: {question}"
)
hub_chain = LLMChain(prompt=prompt, llm=local_llm, verbose=True)

while True:

    print("Please enter the text:", end =" ")
    user_input = str(input())
    if (user_input == ''): 
        break

    print(hub_chain.run(user_input))
    print()