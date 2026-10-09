# 🏛️ Dynamic Multi-Asset Portfolio Optimization & Regime-Switching Engine with Machine Learning & Options Hedging

Un moteur quantitatif complet d'ingénierie financière conçu pour l'allocation dynamique de portefeuille, la détection de régimes par Machine Learning Walk-Forward et la couverture systématique par dérivés (Black-Scholes).

Ce projet est calibré pour répondre aux critères stricts des entretiens quantitatifs, risk managers et data analysts en finance institutionnelle.

---

## 📌 Architecture & Modules du Projet

```text
quant_regime_engine/
│
├── app.py                      # Application Streamlit principale (Thème Dark Institutionnel)
├── requirements.txt            # Dépendances Python nécessaires
├── Dockerfile                  # Conteneurisation de production
├── docker-compose.yml          # Orchestration multi-plateforme en 1 commande
├── README.md                   # Documentation exhaustive et guide méthodologique
│
├── .streamlit/
│   └── config.toml             # Configuration du thème dark épuré (#0B0E14)
│
└── modules/
    ├── __init__.py
    ├── translations.py         # Sélecteur bilingue FR / EN
    ├── data_loader.py          # Ingestion yfinance + fallback synthétique stochastique
    ├── feature_engineering.py  # Momenta, EMA stretches, volatilité réalisée, RSI, labels
    ├── ml_regime_model.py      # Walk-forward cross validation avec fenêtre d'embargo (purging)
    ├── portfolio_optimizer.py  # Frontière de Markowitz & Covariance Shrinkage (Ledoit-Wolf)
    ├── options_hedging.py      # Moteur Black-Scholes, Grecques (Δ, Γ, Θ, ν) & Protective Put
    └── backtester.py           # Simulation institutionnelle avec frais & slippage
```

---

## 🚀 Installation & Démarrage Rapide

### Méthode 1 : En local avec Python

**Prérequis :** Python 3.10 ou supérieur installé.

```bash
# 1. Décompresser l'archive et entrer dans le dossier
unzip quant_regime_engine.zip
cd quant_regime_engine

# 2. Créer un environnement virtuel isolé
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer le dashboard
streamlit run app.py
```
L'interface s'ouvre automatiquement sur `http://localhost:8501`.

---

### Méthode 2 : Avec Docker & Docker Compose (Zero-Config)

Si Docker est installé sur votre machine :

```bash
# Lancement direct en conteneur
docker compose up --build
```
Accédez au dashboard sur `http://localhost:8501`.

---

### Méthode 3 : Déploiement sur Streamlit Cloud

1. Pousser le dossier extrait sur un dépôt GitHub public ou privé.
2. Rendez-vous sur [share.streamlit.io](https://share.streamlit.io/).
3. Connectez votre dépôt et spécifiez `app.py` comme fichier principal.
4. L'application est en ligne avec HTTPS automatique.

---

## 🎯 Comment Interpréter les Résultats

### 1. Détection de Régimes (Machine Learning Walk-Forward)
* **État 0 (Vert) : Bullish / Tendance Saine** $\rightarrow$ Volatilité contenue, momentum haussier. Allocation à 100% sur les actifs de croissance.
* **État 1 (Orange) : Neutre / Range** $\rightarrow$ Volatilité intermédiaire, indécision. Allocation pondérée (70% actions, 30% actifs défensifs).
* **État 2 (Rouge) : Risque Élevé / Sell-off** $\rightarrow$ Détection de pic de volatilité ou de drawdown imminent. Déclenchement de la réduction d'exposition (15% actions, 85% obligations/or) ou activation de l'overlay de couverture options.

### 2. Gestion de la Couverture Options (Black-Scholes & Grecques)
* **Delta (Δ) :** Mesure la sensibilité du portefeuille aux variations de l'indice sous-jacent. L'achat de puts génère un delta négatif qui vient amortir les pertes des positions actions.
* **Gamma (Γ) :** Convexité. Indique à quelle vitesse le delta de protection augmente lorsque le marché s'effondre (effet coussin amplifié en cas de krach).
* **Theta (Θ) :** Coût d'érosion temporelle quotidien du contrat de put (*time decay*).

---

## 💼 Justifications Méthodologiques pour les Entretiens

| Point Méthodologique | Justification Institutionnelle / Réponse pour Recruteur |
| :--- | :--- |
| **Pourquoi prédire des régimes et non le prix de demain ?** | Le ratio signal/bruit des rendements quotidiens est quasiment nul. Tenter de prédire le cours exact mène invariablement à un surapprentissage non stationnaire. Prédire les régimes de volatilité permet d'adapter l'espérance mathématique de la taille de position. |
| **Pourquoi avoir appliqué une fenêtre d'embargo (purging) ?** | Comme la cible est construite sur un horizon glissant de 20 jours futurs, les observations proches du découpage train/test partagent une information commune. L'embargo de 25 jours garantit une séparation stricte sans *look-ahead bias*. |
| **Comment sont valorisées les options dans le backtest ?** | L'historique tick-by-tick des options n'étant pas disponible publiquement sans terminaux payants, les contrats sont simulés analytiquement via **Black-Scholes (1973)** en utilisant le **VIX** comme proxy de volatilité implicite. |
| **Pourquoi le Shrinkage de Ledoit-Wolf pour Markowitz ?** | La matrice de covariance empirique surévalue les corrélations extrêmes lorsqu'elle est inversée. Le rétrécissement linéaire vers une cible diagonale stabilise les poids optimaux. |
| **Intégration du Slippage et Coûts de Transaction** | Chaque rééquilibrage de portefeuille intègre 10 bps de frais d'exécution et 5 bps de slippage pour refléter les conditions réelles d'exécution de marché. |

---

## ⚖️ Disclaimer Légal
*Ce logiciel est un outil de simulation et d'ingénierie financière développé à des fins strictement éducatives et de recherche quantitative. Il ne constitue en aucun cas une recommandation financière, un conseil en investissement ou un mandat de gestion.*
