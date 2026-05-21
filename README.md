# 🏥 MedAssist — Agent Médical Intelligent

**Projet IA Générative — Module Agents IA**
Propulsé par **Groq LLM** + **LangChain** + **RAG local**

---

## 📋 Description

MedAssist est un agent médical intelligent qui combine :
- **RAG** (Retrieval-Augmented Generation) — base de connaissances médicales locale
- **Chatbot avec mémoire** — historique de conversation sur 5 échanges
- **5 outils spécialisés** — calculs, recherche, urgences, plans personnalisés

---

## 🛠️ Outils disponibles

| Outil | Description | Exemple |
|-------|-------------|---------|
| `calculer_imc` | Calcule l'IMC et interprétation | "Mon IMC pour 75 kg, 1m80 ?" |
| `rechercher_informations_medicales` | RAG sur base locale | "Infos sur le diabète" |
| `calculer_frequence_cardiaque_cible` | Zones cardio par âge | "FC cible pour 35 ans ?" |
| `guide_premiers_secours` | Guide urgences | "Que faire en cas de brûlure ?" |
| `generer_plan_sante` | Plan personnalisé | "Plan pour perdre du poids" |

---

## 🚀 Installation

### 1. Prérequis
- Python 3.10 ou 3.11
- Clé API Groq → https://console.groq.com

### 2. Environnement virtuel

**Windows :**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / macOS :**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configuration
```bash
cp .env.example .env
# Éditez .env et ajoutez votre clé GROQ_API_KEY
```

### 4. Lancement
```bash
python medical_agent.py
```

---

## 💬 Exemples d'utilisation

```
🩺 Vous > Calcule mon IMC, je pèse 82 kg pour 1m75

🏥 MedAssist : 
📊 Résultats IMC
IMC : 26.8 kg/m²
Catégorie : Surpoids
Plage de poids idéal : 56.7 – 76.3 kg
Conseil : Une alimentation équilibrée et l'exercice régulier sont recommandés.
```

```
🩺 Vous > Que faire en cas d'étouffement ?

🏥 MedAssist :
🚨 Premiers secours – Étouffement
1. Demandez à la personne de tousser fortement.
2. Penchez-la en avant, donnez 5 tapes vigoureuses dans le dos.
...
```

---

## 🏗️ Architecture

```
Question utilisateur
        ↓
 SystemMessage (prompt médical)
        ↓
 Historique de conversation (chatbot)
        ↓
 Groq LLM (llama-3.3-70b-versatile)
        ↓
 Tool calling (si nécessaire)
        ↓
 Exécution outil Python
        ↓
 Réponse finale du modèle
```

---

## 📁 Fichiers

```
medical_assistant/
├── medical_agent.py           # Agent principal
├── medical_knowledge_base.txt # Base RAG locale
├── requirements.txt           # Dépendances
├── .env.example               # Template variables d'environnement
└── README.md                  # Ce fichier
```

---

## ⚠️ Avertissement

> **MedAssist est un outil éducatif.** Il ne remplace pas un médecin.
> En cas d'urgence médicale, appelez le **15 (SAMU)** ou le **112**.

---
