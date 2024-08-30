import os
import pickle
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
import streamlit as st
from langchain_community.utilities import SQLDatabase
from src.utils import add_fallback_url, remove_unwanted_phrases 
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import GoogleGenerativeAI
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
import time
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv
#import google.generativeai as genai
#from langchain_community.llms import Ollama

load_dotenv(dotenv_path='C:/Users/Moussa/Desktop/projet/.env')

def load_or_download(url):
    cache_file = "url_content_cache.pkl"
    
    # Texte statique à ajouter si l'URL correspond
    static_texts = ( 
        "Les membres publics du CREA de l'atelier norme et certification sont : Région Marrakech-Safi Centre Régional d'Investissement Marrakech-Safi ONSSA Ministère de la Santé et de la Sécurité Sociale\n"
        "Les membres privés du CREA de l'atelier norme et certification sont : IRCOS laboratoires Fondation création d'entreprise (BCP) Natus Marrakech HUILAROME Marrakech Agro Food Industrie\n"
        "Les membres publics du CREA de l'atelier financement sont : Région Marrakech-Safi Centre Régional d'Investissement Marrakech-Safi TAMWILCOM Bank al Maghrib\n"
        "Les membres privés du CREA de l'atelier financement sont : AMWAL Fondation création d'entreprise (BCP) Confederation général des entreprises du Maroc(CGEM)\n"
        "Les membres publics du CREA de l'atelier formation sont : Région Marrakech-Safi Centre Régional d'Investissement Marrakech-Safi Ministère du Commerce et de l'Industrie Ministère de la Transition Énergétique et du Développement Durable\n"
        "Les membres privés du CREA de l'atelier formation sont : les entreprises du Maroc(CGEM) ONSSA\n"
        "Les membres publics du CREA de l'atelier foncier sont : Région Marrakech-Safi Centre Régional d'Investissement Marrakech-Safi Ministère du Commerce et de l'Industrie Ministère de la Transition Énergétique et du Développement Durable\n"
        "Les membres privés du CREA de l'atelier foncier est : Confederation général des entreprises du Maroc(CGEM)\n"
        "Les réalisations que le CREA a mis en place sont les differents guides :\n"
        "financement : http://crea.marrakechinvest.ma/realisation/guide_de_financement\n"
        "formation : http://crea.marrakechinvest.ma/realisation/guide_de_formation\n"
        "foncier : http://crea.marrakechinvest.ma/realisation/guide_foncier\n"
        "Le CREA a plusieurs actualités qui sont répertoriées sur ce lien :\n"
        "Actualité : http://crea.marrakechinvest.ma/activites/actualites\n"
        "Le compte rendu des ateliers organisés par le CREA sont répertoriés sur ces liens :\n"
        "Actualité : http://crea.marrakechinvest.ma/activites/actualites\n"
        "Document : http://crea.marrakechinvest.ma/mediatheque/documents\n"
        "Pour devenir membre du CREA, veuillez cliquer sur ce lien :\n"
        "Membre : http://crea.marrakechinvest.ma/participation\n"
        "Les formations au niveau du CREA sont toutes répertoriées sur ce guide :\n"
        "Formation : http://crea.marrakechinvest.ma/realisation/guide_de_formation\n"
        "Les financements au niveau du CREA sont toutes répertoriées sur ce guide :\n"
        "Financement : http://crea.marrakechinvest.ma/realisation/guide_de_financement\n"
        "Pour répondre précisément à votre question, les membres du CREA ne sont pas rémunérés pour leur participation. Il s'agit d'un engagement volontaire pour contribuer à l'amélioration de l'environnement des affaires dans la région.\n"
        "le directeur du CRI-MS s'appelle Farid Chourak \n"
    )
    
    # Charger le cache si le fichier existe
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            try:
                url_cache = pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                url_cache = {}
    else:
        url_cache = {}
    
    special_url = "http://crea.marrakechinvest.ma" 
    
    # Retourner le contenu mis en cache si disponible
    if url in url_cache:
        page_content = url_cache[url]
    else:
        # Télécharger le contenu de la page web
        response = requests.get(url)
        response.raise_for_status()  # Vérifier que la requête a réussi
        content = response.content

        # Analyser le contenu avec BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extraire tout le contenu texte et les attributs d'intérêt
        texts = set()
        for element in soup.find_all(True):
            if element.name in ['script', 'style', 'a']:
                continue
            text = element.get_text(strip=True)
            if text:
                texts.add(text)
            # Ajouter les attributs des éléments (par exemple, les alt des images)
            #if element.name == 'img' and element.has_attr('src'):
                #texts.add(element['src'])
        
        page_content ="\n".join(sorted(texts))
        
        if url == special_url:
            page_content = static_texts + "\n" + page_content

        # Mettre en cache le contenu de la page
        url_cache[url] = page_content
        with open(cache_file, "wb") as f:
            pickle.dump(url_cache, f)

    # Créer un objet Document avec le contenu de la page
    document = Document(page_content=page_content)
    return document


#@st.cache_resource
def get_vectorstore_from_urls(url_list):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    all_documents = []
    for url in url_list:
        document = load_or_download(url)
        all_documents.append(document)
    
    embeddings = model.encode([doc.page_content for doc in all_documents])
    
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings, dtype=np.float32))
    
    document_mapping = {i: all_documents[i] for i in range(len(all_documents))}
    
    return index, document_mapping


def retrieve_documents(user_query, index, document_mapping, model, top_k=5):
    query_embedding = model.encode([user_query])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    results = [document_mapping[idx] for idx in indices[0] if idx in document_mapping]
    return results

def get_context_retriever_chain(index, document_mapping, model):
    
    llm = GoogleGenerativeAI(model="models/gemini-1.5-pro-latest", google_api_key="AIzaSyCSi6WtM85eaF2LIG72gIL1UeVgu3sBG3k", temperature=0.2)
    #llm = genai.configure(api_key=os.getenv("google_api_key"), model="models/gemini-1.5-pro-latest",temperature=0.2 )
    #llm = GoogleGenerativeAI(model="models/gemini-1.0-pro", google_api_key=os.getenv("google_api_key"), temperature=0.2)
    def retriever(query):
        return retrieve_documents(query, index, document_mapping, model)
    
    retriever_runnable = RunnableLambda(lambda x: retriever(x))
    
    prompt = ChatPromptTemplate.from_messages([
        #MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "En tenant compte de la réquete, générez une requête de recherche afin d'obtenir des informations pertinentes pour la conversation.")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever_runnable, prompt)
    
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    
    llm = GoogleGenerativeAI(model="models/gemini-1.5-pro-latest", google_api_key="AIzaSyCSi6WtM85eaF2LIG72gIL1UeVgu3sBG3k", temperature=0.2)
    #llm = GoogleGenerativeAI(model="models/gemini-1.0-pro", google_api_key=os.getenv("google_api_key"), temperature=0.2)
    #llm = genai.configure(api_key=os.getenv("google_api_key"), model="models/gemini-1.5-pro-latest",temperature=0.2 )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Répondez aux questions de l'utilisateur en fonction du contexte ci-dessous:\n\n{context}"),
        #MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_query):
    try:
        retriever_chain = get_context_retriever_chain(st.session_state.vector_store, st.session_state.document_mapping, st.session_state.embedding_model)
        conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
        response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
      })
        response['answer'] = remove_unwanted_phrases(response['answer'])
    
       #response['answer'] = "Je n'ai pas trouvé de réponse précise à votre question :"
       #response['answer'] = add_fallback_url(response['answer'], user_query)
        return response['answer']
    
    except ResourceExhausted:
        time.sleep(60)  # Attendre 60 secondes avant de réessayer
        return "Le temps limite a été depassé. Veuillez recommen