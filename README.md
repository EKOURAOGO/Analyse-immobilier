Projet Python — Analyse de l'impact de la performance énergétique sur le prix au m² en Île-de-France.
=============================================================================

1. Contexte et objectifs du projet
----------------------------------

Le projet “Prime Verte” vise à étudier l’impact des performances énergétiques des logements (DPE et GES)
sur leur prix de vente en France, à partir de données collectées sur des plateformes d’annonces immobilières.

L’hypothèse principale est que les logements les plus économes en énergie (DPE A ou B) bénéficient d’une
prime de valorisation par rapport aux logements énergivores (DPE F ou G).

Objectifs :

1. Constituer une base de données immobilières à partir de sources web réelles ;
2. Nettoyer et structurer ces données de manière exploitable ;
3. Analyser statistiquement les liens entre prix, caractéristiques et performances énergétiques ;
4. Visualiser les résultats sous forme de graphiques et de cartes interactives ;
5. Fournir une interface utilisateur via Streamlit.

2. Architecture du projet
-------------------------

PROJET_PYTHON/
│
├── CODE/
│   ├── scraping.py                # Collecte de données (web scraping)
│   ├── Processing.py              # Nettoyage et transformation des données
│   ├── Analysis.py                # Analyses statistiques et économétriques
│   └── scraping.ipynb             # Version Notebook pour tests et explorations
│
├── DEMO/
│   ├── IMAGES/                    # Exemples de visualisations
│   ├── VIDEOS/                    # Captures ou démonstrations vidéo
│   └── PROJET PDF/
│       └── Rapport.pdf            # Rapport final du projet
│
├── DONNEES/
│   ├── Donnees_brutes_annonces_immobilieres.csv
│   └── Processing_data.csv
│
├── main.py                        # Lancement de l’application Streamlit
├── README.md                      # Documentation du projet
└── requirements.txt                # Dépendances Python

3. Description des modules
--------------------------

- scraping.py : automatisation de la collecte des annonces immobilières avec BeautifulSoup et Selenium.
  Extraction des informations essentielles : prix, surface, pièces, DPE, GES, localisation.

- Processing.py : nettoyage des données (conversion, normalisation, suppression des doublons).
  Création de variables dérivées comme le prix au m² et catégorisation des performances énergétiques.

- Analysis.py : analyse statistique et économétrique (tests ANOVA, Kruskal, régressions linéaires).
  Génération de graphiques et de cartes Folium.

- Streamlit.py / app.py : interface utilisateur pour visualiser les analyses via un tableau de bord interactif.
  L’utilisateur peut importer ses données et visualiser les statistiques, les cartes et les graphiques.

4. Méthodologie
---------------

1. Collecte : scraping automatisé depuis IAD France ;
2. Nettoyage : standardisation et enrichissement des données ;
3. Analyse : statistiques descriptives et modélisation ;
4. Visualisation : cartes et tableaux de bord Streamlit.

5. Résultats attendus
---------------------

- Corrélation négative entre le DPE et le prix au m² ;
- Prime verte plus marquée dans les zones urbaines ;
- Résultats cohérents avec la transition énergétique et les politiques publiques.

6. Technologies utilisées
-------------------------

- Python 3.10+
- pandas, numpy, matplotlib, seaborn
- scipy, statsmodels
- beautifulsoup4, selenium, requests
- folium, streamlit, wordcloud, geopy

7. Limites et perspectives
--------------------------
Limites :

- Données limitées à IAD France ;
- Qualité variable des annonces ;
- Biais géographiques possibles.

Perspectives :

- Étendre le scraping à d’autres plateformes ;
- Ajouter des données socio-économiques externes ;
- Utiliser des modèles de Machine Learning ;
- Développer une API Flask/FastAPI pour la diffusion des analyses.

8. Auteur
---------
Projet réalisé par Diallo Amadou et Emmanuel.

=============================================================================


Introduction 

Le présent projet d'étude a pour objectif l'analyse empirique de l'impact de la performance énergétique, mesurée par les indices DPE (Diagnostic de Performance Énergétique) et GES (Gaz à Effet de Serre), sur le prix au mètre carré des biens immobiliers en Île-de-France. Dans un contexte de transition environnementale et de réglementation croissante des "passoires thermiques", cette recherche vise à quantifier l'existence et la nature de la « Prime Verte ».
La base de données primaire a été constituée par web scraping ciblé sur le portail immobilier IAD France (https://www.iadfrance.fr/annonces/vente), permettant de rassembler un corpus d'annonces représentatif des dynamiques de marché franciliennes. Cette approche méthodologique garantit l'actualité et la granularité des informations analysées.
L'étude s'articule autour de trois hypothèses fondamentales à vérifier par la modélisation économétrique. Nous postulons (H1) l'existence d'une corrélation positive uniforme entre performance et prix ; (H2) la dépendance significative de cet effet à la localisation (Paris vs Banlieue) ; et (H3) la détermination multifactorielle du prix, largement influencée par les attributs structurels, de confort et de qualité.




I.	Méthodologie et traitement des données

1.	Chaîne d'acquisition et ingénierie des variables 

L'échantillon de 1 839 annonces immobilières a été collecté via web scraping (Selenium / BeautifulSoup) sur le portail IAD France (https://www.ia dfrance.fr/annonces/vente), ciblant des villes clés de l'Île-de-France (Paris, Boulogne-Billancourt, Nanterre, Créteil, Saint-Denis, Villejuif, Palaiseau, Versailles, Melun). Le traitement des données a été essentiel : le prix au m² et la surface ont été winsorisés  (1%-99%) afin de neutraliser l'influence des valeurs extrêmes. Les diagnostics DPE et GES ont été normalisés en classes ordinales (1=Bon, 2=Moyen, 3=Mauvais). L’échantillon a également été enrichi par la création de variables de localisation (Paris, banlieue proche, banlieue éloignée) dérivées de la variable « Ville », ainsi que par des indicateurs de marketing textuel (kw_pos et kw_neg) extraits de la « Description », afin de permettre une analyse multifactorielle plus fine.

2.	Diagnostic et modélisation économétrique

L'analyse s'est appuyée sur une régression linéaire OLS avec erreurs standards robustes (HC3) pour tenir compte de l'hétéroscédasticité, atteignant un R² ajusté ≈ 0,59 - 0,60. Le Diagnostic de corrélation (Spearman) (Voire annexe 1) a confirmé la nature multifactorielle du prix, révélant une forte corrélation du prix au m² avec le confort (Ascenseur : ρ =0,29) et le standing (Charges : ρ =0,34). Ces diagnostics ont permis de valider la pertinence des variables retenues pour la modélisation des effets énergétiques.

II.	Résultats et Interprétation des facteurs de prix

1.	 La Prime verte : Un effet sélectif et localisé (Rejet H1, Validation H2)

L'Hypothèse H1 est rejetée pour un effet général : les coefficients directs DPE et GES sont non significatifs dans les modèles OLS multivariés (p-value> 0,46). L'Hypothèse H2 est validée par l'interaction significative : le terme GES_num : C(ville_classer)[T. Paris] est fortement négatif et significatif (β ≈ -739 €/m², p-value =0,002), indiquant qu'une performance GES plus mauvaise se traduit par une décote tangible d'environ 700€/m² par classe, uniquement à Paris. Ce résultat, confirmé par l'analyse du Prix moyen par GES et localité (Voire annexe 2), prouve que le marché parisien, plus contraint et mature, est le seul à intégrer le risque énergétique de manière significative.

2.	Le prix multifactoriel et la prime au standing (Validation H3)

L'Hypothèse H3 est validée par le pouvoir explicatif élevé du modèle et la puissance des coefficients structurels. Les facteurs de confort et de standing sont primordiaux : l'ascenseur (+1175€/ m², p-value <0,001) et les charges de copropriété (β =+0,56, p-value <0,001) sont des déterminants majeurs du prix. Le discours commercial exerce une influence directe sur la valorisation : les mots-clés positifs (kw_pos) sont associés à une prime de +577€/ m² (p-value =0,008), tandis que les mots-clés négatifs (kw_neg) entraînent une décote de -397€/ m² (p-value =0,006). Enfin, l'intensité marketing (Nombre de photos) est également significative, et les coefficients négatifs des pièces et du parking s'expliquent par l'effet de dégressivité du prix unitaire en fonction de la taille.

Conclusion, Limites et recommandations

	Conclusion 

Le projet conclut que la Prime Verte n'est pas un attribut uniforme du marché immobilier francilien, mais se manifeste comme une décote GES localisée et significative à Paris (β ≈ -739€/m², p-value =0,002), validant l'Hypothèse H2. Le prix au mètre carré est avant tout régi par le système multifactoriel (H3) des attributs structurels : la localisation, le standing (charges, ascenseur) et l'efficacité du discours marketing dominent l'effet énergétique. L'absence de signification des coefficients directs DPE/GES à l'échelle régionale confirme que le risque énergétique est intégré de manière inégale selon la maturité du marché.

	Limites de l'analyse 

Malgré la robustesse des modèles OLS, l'étude est confrontée à des limites méthodologiques. La principale réside dans le Biais des Données Manquantes, le taux de données DPE/GES manquantes atteignant jusqu'à 40%, ce qui risque d'introduire un biais de sélection. De plus, la forte corrélation entre l'âge du bâti, la localisation et le DPE complexifie l'isolement de l'effet purement énergétique (Endogénéité), limitant la portée des inférences causales.

	Recommandations et perspectives 

Afin d'optimiser la valorisation, il est recommandé d'adopter une stratégie de tarification asymétrique : cibler la performance GES pour éviter la décote à Paris, et prioriser le confort et les attributs de standing dans les zones périphériques. Les professionnels doivent également exploiter l'analyse sémantique pour maximiser l'utilisation des mots-clés positifs dont l'effet sur la valorisation est mesuré. Enfin, d'un point de vue analytique, l'exploration de modèles à Effets Fixes par code postal est recommandée pour affiner le contrôle de l'hétérogénéité locale non observée dans les travaux futurs.

=============================================================================

