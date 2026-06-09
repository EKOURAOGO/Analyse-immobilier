# Analyse de la Prime verte immobilière en Île-de-France

> Étude empirique de l'impact de la performance énergétique (DPE/GES) sur le prix au m²
> 1 839 annonces IAD France - Scraping · NLP · Modélisation OLS HC3

**Auteur :** Emmanuel KOURAOGO

---

## Aperçu

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
| H3 - Prix multifactoriel | ✅ Validée | R² ajusté ≈ 0.59–0.60 |

### Déterminants majeurs du prix

| Variable | Effet | Significativité |
|----------|-------|-----------------|
| Ascenseur | +1 175 €/m² | p < 0.001 |
| Charges de copropriété | β = +0.56 | p < 0.001 |
| Mots-clés positifs (NLP) | +577 €/m² | p = 0.008 |
| Mots-clés négatifs (NLP) | −397 €/m² | p = 0.006 |
| GES × Paris | −739 €/m²/classe | p = 0.002 |

---

## Structure du projet

```
Analyse-immobilier/
│
├── main.py                    # Pipeline complet
├── requirements.txt
│
├── CODE/
│   ├── scraping.py            # Selenium + BeautifulSoup (IAD France)
│   ├── Processing.py          # Nettoyage, winsorisation, feature engineering
│   └── Analysis.py            # OLS HC3, NLP, visualisations
│
├── DONNEES/
│   └── Processing_data.csv    # 1 839 annonces traitées
│
└── DEMO/
    ├── IMAGES/                # Visualisations (PNG + cartes HTML interactives)
    └── PROJET PDF/
        └── Rapport.pdf
```

---

## Méthodologie

### Collecte des données
- **Web scraping** : Selenium + BeautifulSoup sur [IAD France](https://www.iadfrance.fr/annonces/vente)
- **1 839 annonces** de vente en Île-de-France
- 9 villes : Paris, Boulogne-Billancourt, Nanterre, Créteil, Saint-Denis, Villejuif, Palaiseau, Versailles, Melun

### Traitement des données
- **Winsorisation** des prix au m² et surfaces (percentiles 1%–99%)
- **Normalisation ordinale** DPE/GES (1=Bon, 2=Moyen, 3=Mauvais)
- **Segmentation géographique** : Paris / banlieue proche / banlieue éloignée
- **NLP** : scores de mots-clés positifs (`kw_pos`) et négatifs (`kw_neg`) depuis les descriptions

### Modélisation
- **Régression OLS** avec erreurs standards robustes **HC3** (hétéroscédasticité)
- **R² ajusté ≈ 0.59–0.60**
- Termes d'interaction GES × localisation pour tester l'effet différencié Paris/banlieue
- Corrélations de Spearman pour le diagnostic multifactoriel

### Visualisations
- Cartes interactives Folium : prix moyen et DPE moyen par zone
- Nuage de mots des annonces
- Matrice de corrélations Spearman
- Distributions prix par classe GES et localité

---

## Installation & lancement

```bash
# Cloner le dépôt
git clone https://github.com/EKOURAOGO/Analyse-immobilier.git
cd Analyse-immobilier

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Lancer le pipeline complet
python main.py
```

---

## Stack technique

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![Selenium](https://img.shields.io/badge/Selenium-43B02A?style=flat-square&logo=selenium&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-OLS%20HC3-blue?style=flat-square)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![Folium](https://img.shields.io/badge/Folium-maps-77B829?style=flat-square)
![NLTK](https://img.shields.io/badge/NLP-NLTK-yellow?style=flat-square)

---

## Auteurs

| Nom | Profil |
|-----|--------|
| Emmanuel KOURAOGO | [GitHub](https://github.com/EKOURAOGO) · [Email](mailto:ekouraogo73@gmail.com) |

