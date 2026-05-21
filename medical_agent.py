"""
MedAssist — Agent Médical Intelligent v2
Combines: RAG + Chatbot + AI Agents with Tools + Flask Web API
Fixed: tool schemas now use individual typed parameters (Groq-compatible)
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

KNOWLEDGE_BASE_PATH = Path("medical_knowledge_base.txt")
MODEL_NAME = "llama-3.3-70b-versatile"
CONVERSATION_HISTORY = []

SYSTEM_PROMPT = """Tu es MedAssist, un assistant médical intelligent et bienveillant.
Tu combines des outils médicaux pour aider les utilisateurs à :
- Calculer des indicateurs de santé (IMC, fréquence cardiaque cible)
- Rechercher des informations médicales fiables dans une base RAG locale
- Obtenir des guides de premiers secours
- Générer des plans de santé personnalisés
- Analyser des symptômes courants

RÈGLES STRICTES :
- Tu n'es PAS un médecin. Rappelle toujours de consulter un professionnel de santé.
- Réponds dans la langue de l'utilisateur (français ou anglais).
- Utilise TOUJOURS un outil quand la demande le justifie.
- Base ta réponse finale sur les résultats des outils.
- Sois clair, rassurant, et structuré dans tes réponses.
"""

# ─────────────────────────────────────────────
# TOOL 1 — IMC (individual typed params, Groq-compatible)
# ─────────────────────────────────────────────
@tool
def calculer_imc(poids_kg: float, taille_cm: float) -> str:
    """Calcule l'IMC (Indice de Masse Corporelle), la catégorie et le poids idéal.
    Args:
        poids_kg: Poids en kilogrammes (ex: 75.0)
        taille_cm: Taille en centimètres (ex: 180.0)
    """
    try:
        if poids_kg <= 0 or taille_cm <= 0:
            return "Erreur : poids et taille doivent être positifs."
        taille_m = taille_cm / 100
        imc = poids_kg / (taille_m ** 2)

        if imc < 18.5:
            categorie, conseil = "Insuffisance pondérale", "Consultez un nutritionniste pour atteindre un poids sain."
        elif imc < 25:
            categorie, conseil = "Poids normal ✅", "Maintenez vos habitudes alimentaires et d'activité physique."
        elif imc < 30:
            categorie, conseil = "Surpoids", "Une alimentation équilibrée et l'exercice régulier sont recommandés."
        elif imc < 35:
            categorie, conseil = "Obésité modérée (classe I)", "Consultez votre médecin pour un suivi adapté."
        else:
            categorie, conseil = "Obésité sévère (classe II+)", "Une consultation médicale urgente est fortement recommandée."

        poids_min = round(18.5 * (taille_m ** 2), 1)
        poids_max = round(24.9 * (taille_m ** 2), 1)

        return (
            f"📊 IMC : {imc:.1f} kg/m²\n"
            f"Catégorie : {categorie}\n"
            f"Poids idéal pour {taille_cm:.0f} cm : {poids_min} – {poids_max} kg\n"
            f"Conseil : {conseil}\n"
            f"⚠️ Consultez un médecin pour une évaluation complète."
        )
    except Exception as e:
        return f"Erreur de calcul : {e}"


# ─────────────────────────────────────────────
# TOOL 2 — RAG : recherche dans base de connaissances
# ─────────────────────────────────────────────
@tool
def rechercher_informations_medicales(mot_cle: str) -> str:
    """Recherche des informations médicales fiables dans la base de connaissances locale (RAG).
    Args:
        mot_cle: Mot-clé médical à rechercher (ex: diabète, tension, grippe, cholestérol)
    """
    kb_path = Path("medical_knowledge_base.txt")
    if not kb_path.exists():
        return "Erreur : base de connaissances introuvable (medical_knowledge_base.txt)."

    lines = [l.strip() for l in kb_path.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    mot = mot_cle.strip().lower()
    if not mot:
        return "Erreur : fournissez un mot-clé non vide."

    matches = [l for l in lines if mot in l.lower()]
    if not matches:
        return (f"Aucune information trouvée pour '{mot_cle}' dans la base locale.\n"
                f"Consultez who.int ou ameli.fr pour des informations fiables.")

    return (
        f"📚 {len(matches[:6])} résultat(s) pour '{mot_cle}' :\n"
        + "\n".join(f"• {r}" for r in matches[:6])
        + "\n\n⚠️ Ces informations sont éducatives. Consultez un professionnel de santé."
    )


# ─────────────────────────────────────────────
# TOOL 3 — Fréquence cardiaque cible
# ─────────────────────────────────────────────
@tool
def calculer_frequence_cardiaque(age: int) -> str:
    """Calcule la fréquence cardiaque maximale et les zones d'entraînement selon l'âge.
    Args:
        age: Âge en années entières (ex: 30)
    """
    if age < 5 or age > 120:
        return "Erreur : âge doit être entre 5 et 120 ans."

    fc_max = 220 - age
    zones = [
        ("Zone 1 – Récupération (50–60%)",  int(fc_max*0.50), int(fc_max*0.60)),
        ("Zone 2 – Endurance (60–70%)",     int(fc_max*0.60), int(fc_max*0.70)),
        ("Zone 3 – Aérobie (70–80%)",       int(fc_max*0.70), int(fc_max*0.80)),
        ("Zone 4 – Anaérobie (80–90%)",     int(fc_max*0.80), int(fc_max*0.90)),
        ("Zone 5 – Effort max (90–100%)",   int(fc_max*0.90), fc_max),
    ]
    lignes = "\n".join(f"  {n}: {lo}–{hi} bpm" for n, lo, hi in zones)
    return (
        f"❤️ FC maximale estimée ({age} ans) : {fc_max} bpm\n\n"
        f"Zones d'entraînement :\n{lignes}\n\n"
        f"⚠️ Consultez un médecin avant tout programme intensif."
    )


# ─────────────────────────────────────────────
# TOOL 4 — Guide premiers secours
# ─────────────────────────────────────────────
@tool
def guide_premiers_secours(situation: str) -> str:
    """Fournit un guide de premiers secours pour une urgence courante.
    Args:
        situation: Type d'urgence parmi : coupure, brulure, etouffement, malaise, fracture, saignement, choc_anaphylactique, avc, crise_cardiaque
    """
    guides = {
        "coupure": "🩹 Coupure\n1. Rincez à l'eau claire 5 min.\n2. Comprimez avec compresse propre.\n3. Désinfectez.\n4. Couvrez avec pansement stérile.\n5. Consultez si : saignement abondant, plaie profonde ou signe d'infection.",
        "brulure": "🔥 Brûlure\n1. Eau froide (15–20°C) pendant 15 min.\n2. Ne PAS appliquer beurre, crème ou glace.\n3. Ne pas percer les cloques.\n4. Couvrir avec film alimentaire propre.\n5. Appeler le 15 si : brûlure étendue, visage, mains, parties génitales.",
        "etouffement": "🚨 Étouffement\n1. Demandez à tousser fortement.\n2. 5 tapes vigoureuses dans le dos (penché en avant).\n3. Si inefficace : 5 compressions abdominales (Heimlich).\n4. Alternez tapes et compressions.\n5. Si inconscient : appelez le 15, commencez RCP.",
        "malaise": "😵 Malaise\n1. Allongez jambes surélevées.\n2. Desserrez les vêtements.\n3. Parlez pour maintenir l'attention.\n4. Si inconscient et respire : PLS.\n5. Appelez le 15 si : perte de conscience, douleur thoracique.",
        "fracture": "🦴 Fracture suspectée\n1. Ne PAS repositionner l'os.\n2. Immobilisez dans la position actuelle.\n3. Attelle de fortune (planche + bandage).\n4. Glace enveloppée dans un tissu.\n5. Appelez le 15 ou rendez-vous aux urgences.",
        "saignement": "🩸 Saignement abondant\n1. Pression directe continue avec compresse.\n2. N'enlever pas la compresse imbibée, ajoutez-en.\n3. Surélevez le membre.\n4. Maintenez 10 min minimum.\n5. Appelez le 15 si saignement ne s'arrête pas.",
        "choc_anaphylactique": "💉 Choc anaphylactique\n1. APPELEZ LE 15 IMMÉDIATEMENT.\n2. Utilisez l'auto-injecteur d'adrénaline (EpiPen) si disponible.\n3. Allongez jambes surélevées.\n4. Si difficultés respiratoires : asseyez la personne.\n5. Si inconsciente : PLS + préparez RCP.",
        "avc": "🧠 AVC — FAST\nF – Face : visage tombant d'un côté ?\nA – Arms : bras impossible à lever ?\nS – Speech : parole difficile/incompréhensible ?\nT – Time : APPELEZ LE 15 IMMÉDIATEMENT.\n→ Chaque minute compte. Ne donnez rien à manger ni à boire.",
        "crise_cardiaque": "💔 Crise cardiaque\n1. APPELEZ LE 15 IMMÉDIATEMENT.\n2. Faites asseoir la personne (pas allongée).\n3. Desserrez les vêtements.\n4. Si aspirine disponible et pas d'allergie : 250–500 mg à mâcher.\n5. Si inconsciente et n'est plus en arrêt : commencez RCP.",
    }

    s = re.sub(r"[éèê]","e", re.sub(r"[àâ]","a", situation.strip().lower()))
    for key, guide in guides.items():
        if key in s or s in key:
            return guide

    return (
        f"Situation '{situation}' non reconnue.\n"
        f"Disponibles : {', '.join(guides.keys())}\n"
        f"🚨 Urgence médicale : appelez le 15 (SAMU) ou 112."
    )


# ─────────────────────────────────────────────
# TOOL 5 — Plan de santé personnalisé
# ─────────────────────────────────────────────
@tool
def generer_plan_sante(age: int, sexe: str, objectif: str) -> str:
    """Génère un plan de santé personnalisé selon l'âge, le sexe et l'objectif.
    Args:
        age: Âge en années (ex: 35)
        sexe: homme ou femme
        objectif: perte_de_poids, prise_de_masse, bien_etre, endurance, ou prevention
    """
    objectifs_map = {
        "perte_de_poids": [
            "Déficit calorique modéré de 300–500 kcal/jour.",
            "150 min d'activité cardio par semaine minimum.",
            "Privilégiez protéines et fibres à chaque repas.",
            "Éliminez les aliments ultra-transformés et sucrés.",
            "Pesez-vous 1x/semaine, le matin à jeun.",
        ],
        "prise_de_masse": [
            "Surplus calorique de 300–500 kcal/jour.",
            "1,6–2,2 g de protéines par kg de poids corporel.",
            "Musculation 3–4 fois par semaine.",
            "Repos 48h entre les séances pour les mêmes muscles.",
            "Creatine monohydrate (5g/jour) peut être envisagée.",
        ],
        "bien_etre": [
            "Méditation ou yoga 10–20 min/jour.",
            "30 min de marche quotidienne.",
            "Alimentation méditerranéenne (légumes, poissons, huile d'olive).",
            "Limiter le temps d'écran avant le coucher.",
            "Maintenir un cercle social actif.",
        ],
        "endurance": [
            "3 séances de course par semaine pour commencer.",
            "Règle des 10% : n'augmentez pas plus de 10%/semaine.",
            "1 séance longue et lente par semaine.",
            "Hydratation avant, pendant et après l'effort.",
            "Intégrez du renforcement musculaire 1x/semaine.",
        ],
        "prevention": [
            "Dépistages recommandés pour votre âge (bilan sanguin annuel).",
            "Maintenir un IMC entre 18,5 et 25.",
            "Contrôle tension et glycémie régulièrement.",
            "Vaccination selon le calendrier vaccinal.",
            "Pas de tabac, alcool limité à 2 verres/jour max.",
        ],
    }

    obj_key = objectif.lower().replace(" ", "_").replace("-", "_")
    # fuzzy match
    for key in objectifs_map:
        if key in obj_key or obj_key in key:
            obj_key = key
            break
    else:
        obj_key = "bien_etre"

    conseils = objectifs_map[obj_key]

    generaux = [
        "Dormez 7 à 9 heures par nuit.",
        "Buvez 1,5 à 2 L d'eau par jour.",
        "Évitez le tabac et l'alcool excessif.",
        "Bilan médical annuel.",
    ]

    if age >= 60:
        conseils.append("Travaillez l'équilibre et la souplesse pour prévenir les chutes.")
    if age <= 25:
        conseils.append("Profitez de votre jeunesse pour établir de bonnes habitudes durables.")
    if sexe.lower() == "femme" and age >= 40:
        conseils.append("Dépistage mammographique recommandé tous les 2 ans après 50 ans.")

    return (
        f"🏥 Plan de santé — {age} ans, {sexe}, objectif : {obj_key.replace('_',' ')}\n\n"
        f"Conseils spécifiques :\n"
        + "\n".join(f"  {i+1}. {c}" for i, c in enumerate(conseils))
        + "\n\nConseils généraux :\n"
        + "\n".join(f"  • {c}" for c in generaux)
        + "\n\n⚠️ Consultez un médecin ou nutritionniste pour un suivi personnalisé."
    )


# ─────────────────────────────────────────────
# TOOL 6 — Analyseur de symptômes
# ─────────────────────────────────────────────
@tool
def analyser_symptomes(symptomes: str) -> str:
    """Analyse une liste de symptômes et suggère des pistes (non diagnostiques).
    Args:
        symptomes: Liste de symptômes séparés par des virgules (ex: fièvre, toux, fatigue)
    """
    base = {
        "fievre": ["grippe", "infection virale", "COVID-19", "angine"],
        "toux": ["grippe", "bronchite", "COVID-19", "asthme", "allergie"],
        "fatigue": ["anémie", "dépression", "diabète", "hypothyroïdie", "surmenage"],
        "maux_de_tete": ["migraine", "hypertension", "sinusite", "stress", "déshydratation"],
        "douleur_thoracique": ["problème cardiaque ⚠️ URGENT", "pleurésie", "reflux gastrique", "anxiété"],
        "essoufflement": ["asthme", "insuffisance cardiaque ⚠️", "anémie", "BPCO"],
        "nausees": ["gastro-entérite", "grossesse", "intoxication alimentaire", "migraine"],
        "douleur_abdominale": ["gastrite", "appendicite ⚠️", "syndrome côlon irritable", "calculs rénaux"],
        "vertiges": ["hypotension", "problème d'oreille interne", "anémie", "déshydratation"],
        "insomnie": ["anxiété", "dépression", "apnée du sommeil", "stress"],
    }

    symptomes_list = [re.sub(r"[éèê]","e", re.sub(r"[àâ]","a", s.strip().lower().replace(" ","_")))
                      for s in symptomes.split(",")]

    resultats = {}
    urgent = False
    for s in symptomes_list:
        for key, causes in base.items():
            if key in s or s in key:
                resultats[key.replace("_"," ")] = causes
                if any("⚠️" in c for c in causes):
                    urgent = True

    if not resultats:
        return (f"Symptômes '{symptomes}' non reconnus dans la base.\n"
                f"Consultez un médecin pour tout symptôme persistant.")

    lignes = []
    for symptome, causes in resultats.items():
        lignes.append(f"• {symptome} → possiblement : {', '.join(causes)}")

    warning = "\n🚨 SYMPTÔME URGENT DÉTECTÉ — Consultez un médecin immédiatement ou appelez le 15." if urgent else ""

    return (
        f"🔍 Analyse des symptômes : {symptomes}\n\n"
        + "\n".join(lignes)
        + warning
        + "\n\n⚠️ IMPORTANT : Ceci n'est PAS un diagnostic médical. Consultez un professionnel."
    )


# ─────────────────────────────────────────────
# Agent Core
# ─────────────────────────────────────────────
TOOLS = [
    calculer_imc,
    rechercher_informations_medicales,
    calculer_frequence_cardiaque,
    guide_premiers_secours,
    generer_plan_sante,
    analyser_symptomes,
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def run_agent(question: str, use_history: bool = True) -> str:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY manquante dans le fichier .env")

    llm = ChatGroq(model=MODEL_NAME, temperature=0.2, api_key=api_key)
    llm_with_tools = llm.bind_tools(TOOLS)
    # llm without tools for the final synthesis step — avoids empty content bug
    llm_plain = ChatGroq(model=MODEL_NAME, temperature=0.2, api_key=api_key)

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if use_history:
        messages.extend(CONVERSATION_HISTORY[-10:])
    messages.append(HumanMessage(content=question))

    first_response = llm_with_tools.invoke(messages)
    messages.append(first_response)

    tool_outputs = []

    if first_response.tool_calls:
        for tool_call in first_response.tool_calls:
            selected_tool = TOOLS_BY_NAME.get(tool_call["name"])
            tool_output = (
                selected_tool.invoke(tool_call["args"])
                if selected_tool
                else f"Outil '{tool_call['name']}' introuvable."
            )
            tool_outputs.append(tool_output)
            messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))

        # Use plain LLM (no tools) for final synthesis to prevent empty content
        final = llm_plain.invoke(messages)
        answer = final.content

        # Safety fallback: if LLM returned empty, return raw tool output directly
        if not answer or not answer.strip():
            answer = "\n\n---\n\n".join(tool_outputs)

    else:
        answer = first_response.content

    # Final guard: should never be empty
    if not answer or not answer.strip():
        answer = "Je n'ai pas pu générer une réponse. Veuillez reformuler votre question."

    if use_history:
        CONVERSATION_HISTORY.append(HumanMessage(content=question))
        CONVERSATION_HISTORY.append(AIMessage(content=answer))

    return answer


def reset_conversation():
    global CONVERSATION_HISTORY
    CONVERSATION_HISTORY = []


# ─────────────────────────────────────────────
# Flask Web API (for the UI)
# ─────────────────────────────────────────────
def start_server():
    try:
        from flask import Flask, request, jsonify, send_from_directory
        from flask_cors import CORS
    except ImportError:
        print("Flask not installed. Run: pip install flask flask-cors")
        return

    app = Flask(__name__, static_folder=".")
    CORS(app)

    @app.route("/")
    def index():
        return send_from_directory(".", "interface.html")

    @app.route("/chat", methods=["POST"])
    def chat():
        data = request.get_json()
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "Question vide"}), 400
        try:
            answer = run_agent(question)
            return jsonify({"answer": answer})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/reset", methods=["POST"])
    def reset():
        reset_conversation()
        return jsonify({"status": "ok"})

    print("\n🌐 Interface web : http://localhost:5000")
    print("   Appuyez sur Ctrl+C pour arrêter\n")
    app.run(debug=False, port=5000)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    import sys
    if "--web" in sys.argv:
        start_server()
        return

    print("\n" + "="*60)
    print("  🏥 MedAssist — Agent Médical Intelligent v2")
    print("  Propulsé par Groq + LangChain + RAG local")
    print("="*60)
    print("\nExemples :")
    print("  • Mon IMC pour 90 kg et 1m82 ?")
    print("  • Que faire en cas de brûlure ?")
    print("  • FC cible à 35 ans ?")
    print("  • Infos sur le diabète.")
    print("  • Plan santé pour perdre du poids, 30 ans, homme.")
    print("  • J'ai fièvre, toux et fatigue.")
    print("\nCommandes : 'reset' | 'quit'")
    print("Interface web : python medical_agent.py --web\n")

    while True:
        try:
            question = input("🩺 Vous > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Au revoir !")
            break

        if not question:
            continue
        if question.lower() in {"quit","exit","q"}:
            print("👋 Prenez soin de vous !")
            break
        if question.lower() == "reset":
            reset_conversation()
            print("💬 Historique réinitialisé.\n")
            continue

        try:
            print("\n🤔 MedAssist réfléchit...")
            answer = run_agent(question)
            print(f"\n🏥 MedAssist :\n{answer}\n")
            print("-"*60)
        except Exception as e:
            print(f"\n❌ Erreur : {e}\n")


if __name__ == "__main__":
    main()
