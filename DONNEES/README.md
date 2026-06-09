# Analyse de la prime verte immobilière en Île-de-France

> Étude de l'impact de la performance énergétique (DPE/GES) sur le prix au m² - 1 839 annonces IAD France

**Projet réalisé par Emmanuel KOURAOGO **

<p align="center">
  <img src="DEMO/IMAGES/Carte de chaleur des corrélations Spearman.png" width="47%"/>
  <img src="DEMO/IMAGES/Distribution des prix au m² par GES et localité.png" width="47%"/>
</p>
<p align="center">
  <img src="DEMO/IMAGES/Nuage de Mots pour l'Analyse Textuelle.png" width="47%"/>
  <img src="DEMO/IMAGES/Relation Prix vs Surface, segmentée par performance énergétique.png" width="47%"/>
</p>

---

## Résultats clés

| Hypothèse | Résultat | Détail |
|-----------|----------|--------|
| H1 - Prime verte uniforme | ❌ Rejetée | DPE/GES non significatifs à l'échelle régionale (p > 0.46) |
| H2 - Effet localisé à Paris | ✅ Validée | Décote GES à Paris : **−739 €/m² par classe** (p = 0.002) |
| H3 - Prix multifactoriel | ✅ Validée | R² ajusté ≈ 0.59–0.60, ascenseur +1175 €/m², charges β = +0.56 |

---

## Présentation du projet

Ce projet analyse empiriquement si les biens immobiliers bien notés sur le plan énergétique bénéficient d'une **"Prime Verte"** sur le marché francilien.

**Périmètre :** 9 villes d'Île-de-France - Paris, Boulogne-Billancourt, Nanterre, Créteil, Saint-Denis, Villejuif, Palaiseau, Versailles, Melun.

**Conclusion principale :** La Prime Verte n'est pas uniforme. Elle se manifeste uniquement à Paris sous forme d'une **décote GES significative (~700 €/m² par classe)**, tandis qu'en banlieue, ce sont les attributs de standing (ascenseur, charges) et le marketing textuel qui gouvernent le prix.

---

## Structure du projet

```
analyse-immobilier-dpe/
│
├── main.py                    # Point d'entrée — pipeline complet
├── requirements.txt           # Dépendances Python
│
├── CODE/
│   ├── scraping.py            # Scraping Selenium + BeautifulSoup (IAD France)
│   ├── Processing.py          # Nettoyage, winsorisation, feature engineering
│   └── Analysis.py            # Modélisation OLS, NLP, visualisations
│
├── DONNEES/
│   └── Processing_data.csv    # Données traitées (1 839 annonces)
│
└── DEMO/
    ├── IMAGES/                # Visualisations générées (PNG + HTML interactifs)
    └── PROJET PDF/
        └── Rapport.pdf        # Rapport complet
```

---

## Méthodologie

### 1. Collecte des données
- **Web scraping** : Selenium + BeautifulSoup sur [IAD France](https://www.iadfrance.fr/annonces/vente)
- **1 839 annonces** de vente immobilière en Île-de-France
- Variables collectées : prix, surface, DPE, GES, localisation, équipements, description textuelle

### 2. Traitement des données
- **Winsorisation** des prix au m² et surfaces (percentiles 1%–99%)
- **Normalisation ordinale** des classes DPE/GES (1=Bon, 2=Moyen, 3=Mauvais)
- **Feature engineering** : segmentation géographique (Paris / banlieue proche / banlieue éloignée)
- **NLP** : extraction de scores de mots-clés positifs (`kw_pos`) et négatifs (`kw_neg`) depuis les descriptions

### 3. Modélisation
- **Régression OLS** avec erreurs standards robustes HC3 (hétéroscédasticité)
- **R² ajusté ≈ 0.59–0.60**
- Termes d'interaction DPE/GES × localisation pour tester l'effet différencié Paris/banlieue
- Corrélations de Spearman pour le diagnostic multifactoriel

### 4. Visualisations
- Cartes interactives (Folium) : prix moyen et DPE moyen par zone
- Distribution des prix par classe GES et localité
- Analyse textuelle : nuage de mots, impact des mots-clés sur le prix
- Matrice de corrélations Spearman

---

## Résultats détaillés

### La Prime Verte : un effet sélectif (H1 rejetée, H2 validée)
- Les coefficients directs DPE et GES sont **non significatifs** dans les modèles multivariés (p > 0.46)
- L'interaction `GES × Paris` est **fortement significative** : β ≈ **−739 €/m²** (p = 0.002)
- Le marché parisien, plus mature, est le seul à intégrer le risque énergétique dans le prix

### Déterminants majeurs du prix (H3 validée)
| Variable | Effet | Significativité |
|----------|-------|-----------------|
| Ascenseur | +1 175 €/m² | p < 0.001 |
| Charges de copropriété | β = +0.56 | p < 0.001 |
| Mots-clés positifs (kw_pos) | +577 €/m² | p = 0.008 |
| Mots-clés négatifs (kw_neg) | −397 €/m² | p = 0.006 |
| GES × Paris | −739 €/m²/classe | p = 0.002 |

### Limites
- Taux de données DPE/GES manquantes : ~40% (biais de sélection potentiel)
- Endogénéité : forte corrélation entre âge du bâti, localisation et DPE
- Perspective : modèles à effets fixes par code postal pour affiner l'hétérogénéité locale

---

## Installation & utilisation

```bash
# Cloner le dépôt
git clone https://github.com/EKOURAOGO/analyse-immobilier-dpe.git
cd analyse-immobilier-dpe

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Lancer le pipeline complet
python main.py
```

Les visualisations sont générées dans `DEMO/IMAGES/`.

---

## Stack technique

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Selenium](https://img.shields.io/badge/Selenium-43B02A?style=flat-square&logo=selenium&logoColor=white)
![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup4-orange?style=flat-square)
![Statsmodels](https://img.shields.io/badge/Statsmodels-OLS-blue?style=flat-square)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![Folium](https://img.shields.io/badge/Folium-maps-77B829?style=flat-square)
![NLTK](https://img.shields.io/badge/NLP-NLTK-yellow?style=flat-square)

---

## Auteur

| Nom | GitHub |
|-----|--------|
| Emmanuel KOURAOGO | [@EKOURAOGO](https://github.com/EKOURAOGO) |
