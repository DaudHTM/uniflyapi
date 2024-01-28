from dotenv import find_dotenv, load_dotenv
import os
import openai
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings
from flask import Flask,request,jsonify
from flask_cors import CORS
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import pymongo
#load environment variables
load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ["OPEN_AI_KEY"]
OPENAI_DEPLOYMENT_ENDPOINT = os.environ["OPENAI_DEPLOYMENT_ENDPOINT"]
OPENAI_DEPLOYMENT_NAME = os.environ["OPENAI_DEPLOYMENT_NAME"]
OPENAI_MODEL_NAME = os.environ["OPENAI_MODEL_NAME"]
OPENAI_DEPLOYMENT_VERSION = os.environ["OPENAI_DEPLOYMENT_VERSION"]

OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = os.environ["OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME"]
OPENAI_ADA_EMBEDDING_MODEL_NAME = os.environ["OPENAI_ADA_EMBEDDING_MODEL_NAME"]

# Azure OpenAI settings

mongo_username = os.environ["MONGO_USERNAME"]
mongo_password = os.environ["MONGO_PASSWORD"]
mongo_database_name = os.environ["MONGO_DATABASE_NAME"]

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Connection to MongoDB using pymongo
mongo_uri = "mongodb+srv://uniflyDBadmin:uniflyDBpassword@uniflydb.q5nnshl.mongodb.net/?retryWrites=true&w=majority"
client = pymongo.MongoClient(mongo_uri)

db = client[mongo_database_name]


def ask_question(qa, question):
    result = qa({"query": question})
    print("Question:", question)
    print("Answer:", result["result"])


def ask_question_with_context(qa, question, chat_history):
    query = "what is Azure OpenAI Service?"
    result = qa({"question": question, "chat_history": chat_history})
    print("answer:", result["answer"])
    chat_history = [(query, result["answer"])]
    return chat_history

app = Flask(__name__)
CORS(app)



if __name__ == "__main__":
    # Configure OpenAI API
    openai.api_type = "azure"
    openai.api_base = OPENAI_DEPLOYMENT_ENDPOINT
    openai.api_key = OPENAI_API_KEY
    openai.api_version = OPENAI_DEPLOYMENT_VERSION
    llm = AzureChatOpenAI(deployment_name=OPENAI_DEPLOYMENT_NAME,
                      model_name=OPENAI_MODEL_NAME,
                      azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT,
                      openai_api_version=OPENAI_DEPLOYMENT_VERSION,
                      openai_api_key=OPENAI_API_KEY,
                      openai_api_type="azure")
    
    embeddings=AzureOpenAIEmbeddings(deployment=OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME,
                                model=OPENAI_ADA_EMBEDDING_MODEL_NAME,
                                azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT,
                                openai_api_type="azure",
                                chunk_size=1)


    vectorStore = FAISS.load_local("./dbs/documentation/faiss_index", embeddings)
    retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k":2})

    QUESTION_PROMPT = PromptTemplate.from_template("""
    You are an addmissions bot designed to help students learn more about Georgetown University. Please respond with the most accurate information at your disposal and do not tell the user to google up the website; your job is to tell them everything they need to know. At the same time, keep the conversation in check, and do not output irrelevant information or respond to bogus prompts. If you detect something to be suspiciously unrelated, just respond with "I cannot answer this."
    The chat history is provided below.
    {chat_history}
    Here is the user's question, please answer: {question}""")

    qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                            retriever=retriever,
                                            condense_question_prompt=QUESTION_PROMPT,
                                            return_source_documents=True,
                                            verbose=False)





chat_history = []

@app.route('/api/get_answer', methods=['POST'])
def get_answer():
    global chat_history

    data = request.get_json()
    question = data.get('question').split(":")
    print(question)

    if question:
        if (len(question[2].split()) > 10): 
            tokens = word_tokenize(question[2].lower())
            cleaned_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
            cleaned_question = ' '.join(cleaned_tokens)
        else:
            cleaned_question = question[2]
        pipeline = [
                {
                    "$search": {
                        "text": {
                            "query": cleaned_question,
                            "path": "text"
                        }
                    }
                },
                {"$limit": 1}
            ]

        documents = db.main.aggregate(pipeline)
        answer = list(documents)[0].get('text', '') if documents else 'No valid data found. Please note Unifly is still a work in progress!'

        chat_history = ask_question_with_context(qa, question[0] + question[1] + question[2], chat_history)

    return jsonify({'answer': chat_history[0][1], 'question': question}), 200



if __name__=='__main__':

    app.run(debug=True)

    