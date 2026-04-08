# 📡 Classification de la Qualité Réseau Radio via CNN

> Pipeline CRISP-DM complet — Classification de signaux RadioML 2016.10a avec CNN PyTorch

---

## 🎯 Objectif

Ce projet vise à **estimer automatiquement la qualité d'un réseau radio** à partir de signaux I/Q bruts, en utilisant un réseau de neurones convolutionnel (CNN 1D).

Les signaux sont labellisés selon le niveau de bruit (SNR) :

| SNR | Classe |
|-----|--------|
| ≥ 10 dB | **Good** |
| 0 à 9 dB | **Average** |
| < 0 dB | **Poor** |

---

## 📦 Dataset

**RadioML 2016.10a** — dataset synthétique généré avec GNU Radio (DeepSig)

| Propriété | Valeur |
|-----------|--------|
| Format | `.pkl` (dictionnaire Python) |
| Clés | Tuples `(modulation, SNR)` |
| Modulations | 11 types (BPSK, QPSK, 8PSK, QAM16, QAM64, CPFSK, GFSK, PAM4, WBFM, AM-DSB, AM-SSB) |
| SNR couverts | −20 dB à +18 dB (pas de 2 dB) |
| Shape signal | `(2, 128)` — 2 canaux I/Q × 128 échantillons |
| Signaux / clé | 1 000 |
| Total signaux | 220 000 |

> Télécharger le dataset : [Kaggle — RadioML 2016.10a](https://www.kaggle.com/datasets/pinxau1000/radioml2016)

Placer le fichier dans :
```
Data/RML2016.10a_dict.pkl
```

---

## 🗂️ Structure du projet

```
projet-cnn-radioml/
│
├── Data/
│   └── RML2016.10a_dict.pkl       ← dataset (à télécharger)
│
├── CNN_FINALE.ipynb               ← notebook principal
├── best_model.pth                 ← meilleur modèle sauvegardé (généré à l'exécution)
├── requirements.txt               ← dépendances Python
└── README.md                      ← ce fichier
```

---

## 🏗️ Pipeline CRISP-DM

```
1. Business Understanding   →  Estimer la qualité réseau radio (QoS)
2. Data Understanding       →  Exploration RadioML, structure I/Q, distribution SNR
3. Data Preparation         →  Labels SNR, vérif. NaN/inf, under-sampling, split 70/15/15
4. Modeling                 →  CNN 1D PyTorch (3 blocs Conv + BatchNorm + Dropout)
5. Evaluation               →  Matrice de confusion, F1-score, Accuracy vs SNR
6. Deployment               →  Sauvegarde best_model.pth, pistes Streamlit
```

---

## 🧠 Architecture CNN 1D

```
Input (2, 128)
    │
    ├── Conv1d(2→64) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.2)
    ├── Conv1d(64→128) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.2)
    ├── Conv1d(128→256) + BatchNorm + ReLU + AdaptiveAvgPool(8)
    │
    ├── Flatten → Linear(2048→128) → ReLU → Dropout(0.5)
    └── Linear(128→3)  →  [Good, Average, Poor]
```

**Optimiseur** : Adam (`lr=0.001`)  
**Loss** : CrossEntropyLoss  
**Scheduler** : ReduceLROnPlateau (`patience=3`, `factor=0.5`)  
**Early Stopping** : patience = 3 epochs  

---

## ▶️ Exécution

### 1. Cloner le projet

```bash
git clone https://github.com/votre-username/projet-cnn-radioml.git
cd projet-cnn-radioml
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Lancer le notebook

```bash
jupyter notebook CNN_FINALE.ipynb
```

Exécuter les cellules dans l'ordre de haut en bas.

---

## 📊 Résultats attendus

| Métrique | Valeur approximative |
|----------|---------------------|
| Test Accuracy | ~85–95 % |
| Meilleure classe | Good (SNR élevé, signal propre) |
| Classe la plus difficile | Poor/Average (SNR proche de 0 dB) |

> Les performances varient selon la machine (CPU/GPU) et la graine aléatoire.

---

## 🔧 Visualisations incluses

- Distribution des classes (avant/après équilibrage)
- Distribution des SNR
- Signaux I/Q bruts
- Spectrogramme (canal I)
- Boxplot amplitudes I vs Q
- Corrélation I/Q
- Diagramme de constellation IQ par classe
- Courbes Loss & Accuracy (entraînement)
- Matrice de confusion
- Accuracy par niveau de SNR

---

## 🚀 Pistes d'amélioration

- Tester **LSTM** ou **Transformer** pour les dépendances temporelles longues
- Comparer avec des CNN pré-entraînés (**ResNet**, **MobileNet**) sur spectrogrammes
- Recherche d'hyperparamètres automatique via **Optuna** (alternative au Grid Search)
- Déploiement en démo interactive via **Streamlit**

---

## 👥 Auteurs

Projet réalisé dans le cadre du cours de Deep Learning — Pipeline CRISP-DM sur RadioML 2016.10a.
