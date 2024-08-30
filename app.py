import streamlit as st
from src.retriever import get_vectorstore_from_urls
from src.retriever import get_response
from sentence_transformers import SentenceTransformer

# Charger les variables d'environnement
urls = [
    "http://crea.marrakechinvest.ma",
    "http://crea.marrakechinvest.ma/activites/actualites",
    "http://crea.marrakechinvest.ma/mediatheque/documents",
    "http://crea.marrakechinvest.ma/realisation/plan-action/formation",
    "http://crea.marrakechinvest.ma/realisation/plan-action/foncier",
    "http://crea.marrakechinvest.ma/realisation/plan-action/financement",
    "http://crea.marrakechinvest.ma/realisation/plan-action/certification",
    "http://crea.marrakechinvest.ma/mediatheque/videotheque",
   
]



# Configuration de Streamlit
st.set_page_config(page_title="Chat with My CREA", page_icon="🤖")
st.title("Chat with My CREA")

# CSS personnalisé
st.markdown(
    """
    <style>
    /* Background color for the entire app */
    .main {
        background-color: #f0f2f6;
    }

    h1 {
        color: #ffc107; /* Jaune pour le titre de la page */
    }

    /* Chat bubble styling */
    .st-chat-message-human {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 5px;
    }

    .st-chat-message-ai {
        background-color: #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 5px;
    }

    /* Chat input box styling */
    input {
        border-radius: 5px;
        padding: 8px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Initialisation de l'historique du chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Bonjour, je suis My CREA. Comment puis-je vous aider?"}
    ]

# Initialisation du modèle d'embedding et du vector store
if "vector_store" not in st.session_state:
    st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    st.session_state.vector_store, st.session_state.document_mapping = get_vectorstore_from_urls(urls)

# Interface de chat
for message in st.session_state.chat_history:
    if message["role"] == "assistant":
        st.chat_message("AI").markdown(
            f"<div class='st-chat-message-ai'>{message['content']}</div>", 
            unsafe_allow_html=True
        )
    else:
        st.chat_message("Human").markdown(
            f"<div class='st-chat-message-human'>{message['content']}</div>", 
            unsafe_allow_html=True
        )

# Saisie de l'utilisateur
user_query = st.chat_input("Type your message here...")
if user_query:
    st.session_state.chat_history.append({"role": "human", "content": user_query})
    st.chat_message("Human").markdown(
        f"<div class='st-chat-message-human'>{user_query}</div>", 
        unsafe_allow_html=True
    )
    response = get_response(user_query)
    st.chat_message("AI").markdown(
        f"<div class='st-chat-message-ai'>{response}</div>", 
        unsafe_allow_html=True
    )
    st.session_state.chat_history.append({"role": "assistant", "content": response})
