import os
from fuzzywuzzy import fuzz

def contains_term_with_typo(term, query, threshold=80):
    """
    Vérifie si le terme donné se trouve dans la requête, en tenant compte des fautes d'orthographe.
    Le seuil (threshold) détermine la tolérance pour les erreurs.
    """
    query_words = query.split()
    for word in query_words:
        if fuzz.ratio(term.lower(), word.lower()) >= threshold:
            return True
    return False

def add_fallback_url(response, user_query):
    fallback_urls = {
        "financement": "http://crea.marrakechinvest.ma/realisation/guide_de_financement",
        "formation": "http://crea.marrakechinvest.ma/realisation/guide_de_formation",
        "foncier": "http://crea.marrakechinvest.ma/realisation/guide_foncier",
        "document": "http://crea.marrakechinvest.ma/mediatheque/phototheque",
        "membre": "https://crea.marrakechinvest.ma/participation",
        "videotheque": "https://crea.marrakechinvest.ma/mediatheque/videotheque",
        "actualité": "https://crea.marrakechinvest.ma/activites/actualites",
    }

    keywords = {
        'financement': ['financement', 'subvention'],
        'formation': ['formation', 'cours', 'apprentissage', 'éducation'],
        'foncier': ['foncier', 'terrain', 'propriété', 'immobilier'],
        'document': ['document', 'charte', 'projet', 'investissement'],
        'membre': ['devenir membre', 'participer', 'être membre', 'accéder', 'adhérer'],
        'videotheque': ['roadshow', 'video', 'vidéothèque'],
        'actualité': ['compte rendus', 'actualite', 'actualité', 'Atelier']
    }

    fallback_message = ""
    user_query_lower = user_query.lower()

    # Vérifier si le mot "réalisation" est dans la requête utilisateu
    
    for category, kw_list in keywords.items():
        if any(kw in user_query_lower for kw in kw_list):
            fallback_message = f"\nVeuillez cliquer sur ce lien pour plus de détails : {fallback_urls[category]}"
            break

    return response + fallback_message

def remove_unwanted_phrases(response):
    unwanted_phrases = [
        "je suis désolé",
        "je n'ai pas trouvé",
        "Je suis un agent conversationnel",
        "je ne suis pas qualifié",
        "Malheureusement",
        "Le document que vous avez fourni",
        "pour vous repondre au mieux",
        "Le site web",
        "dans la base de donnée",
        "Le texte fourni ne mentionne",
        "D'après le contexte fourni",
        "D'après le texte fourni"
    ]
    for phrase in unwanted_phrases:
        response = response.replace(phrase,"D'après les informations que j'ai")
    return response
