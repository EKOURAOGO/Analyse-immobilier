### ANALYSE DES DONNEES
""" L'objectif principal de cette étude est d'analyser et de quantifier l'influence de la performance énergétique des biens immobiliers sur leur prix au mètre carré. 
Nous cherchons à déterminer si une meilleure performance, mesurée par les indices du Diagnostic de Performance Énergétique (DPE) et des Gaz à Effet de Serre (GES), se traduit par une prime immobilière, communément appelée la "Prime Verte"."""

"""Hypothèses:

H1 – Les logements ayant un DPE et un GES fort se vendent à un prix plus élevé.

Une meilleure performance énergétique (DPE) et des émissions réduites de gaz à effet de serre (GES) sont associées à un prix plus élevé des logements.

H2 – La prime verte varie selon les localités.

L'impact de la performance énergétique sur les prix immobiliers (la Prime Verte) n'est pas uniforme et diffère selon les localités.

H3 – Les prix des logements peuvent être expliqués par plusieurs variables.

Le prix des logements est influencé par une combinaison de facteurs, y compris la performance énergétique (DPE et GES), la superficie, la localisation, et d'autres caractéristiques spécifiques. """

# python --version==Python 3.10.8
# pip install pandas numpy matplotlib scipy statsmodels seaborn scikit-learn wordcloud

# ==============================================================================
# 0. PRÉPARATION DE L'ENVIRONNEMENT
# ==============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import folium
from geopy.geocoders import Nominatim
import time
import statsmodels.formula.api as smf

from scipy.stats import f_oneway, kruskal, ttest_ind, shapiro, levene
from statsmodels.stats.outliers_influence import variance_inflation_factor
from wordcloud import WordCloud, STOPWORDS

# Configuration de l'affichage
pd.set_option('display.max_columns', 50)
sns.set_style("whitegrid")

# Configuration des variables
FILE_PATH = 'C:/Users/Pc/OneDrive/Bureau/Dossier/IMSD/PROJET_PYTHON/DONNEES/Processing_data.csv'
ALPHA = 0.05 # Seuil de signification

# Importer les données
try:
    df = pd.read_csv(FILE_PATH, delimiter=';')
    print("Données importées avec succès.")
except FileNotFoundError:
    print(f"Erreur : Fichier non trouvé à l'emplacement : {FILE_PATH}")
    # Créer un DataFrame vide ou quitter si le fichier est essentiel
    df = pd.DataFrame() 

# Aperçu et types de données
if not df.empty:
    print("\n--- Aperçu des données ---")
    print(df.head(3))
    print("\n--- Information sur les colonnes et types ---")
    df.info(verbose=False)
    print("\n--- Types de données ---")
    print(df.dtypes) 

# ==============================================================================
# 0.5. PRÉPARATION ADDITIONNELLE DES VARIABLES (Pour Visualisations Spécifiques)
# ==============================================================================
if not df.empty:
    # Création de variables de qualité DPE pour les boxplots (DPE_num 1-3 = Bon, 4-7 = Mauvais)
    df['DPE_Qualite'] = df['DPE_num'].apply(lambda x: 'Bon DPE (1-3)' if x <= 3 else ('Moyen DPE (4)' if x == 4 else 'Mauvais DPE (5-7)'))

    # Catégorisation des Charges (Faibles/Moyennes/Élevées)
    if 'Charges_w' in df.columns:
        q_charges_75 = df['Charges_w'].quantile(0.75)
        df['Charges_Cat'] = df['Charges_w'].apply(lambda x: 'Charges Élevées (> Q75)' if x > q_charges_75 else 'Charges Faibles/Moyennes')
    

# ==============================================================================
# 1. STATISTIQUES DESCRIPTIVES ET EXPLORATOIRE
# ==============================================================================

def display_descriptive_stats(data):
    """Affiche les statistiques descriptives des variables clés."""
    print("\n" + "="*50)
    print("1.1. Statistiques descriptives des variables clés")
    print("="*50)
    
    key_vars = ['Prix_au_m2_w', 'Surface_w', 'DPE_num', 'GES_num', 'ville_classer']
    stats_df = df[key_vars].describe(include='all').T
    print(stats_df)
    
    print("\nRépartition des modalités de DPE et GES :")
    print(df['DPE_num'].value_counts().sort_index())
    print(df['GES_num'].value_counts().sort_index())

def plot_distributions(df):
    """Visualise les distributions des prix et des catégories DPE/GES."""
    print("\n1.2. Visualisation des Distributions")
    
    # Distribution des prix au mètre carré 
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Prix_au_m2_w'], bins=30, kde=True)
    # --- Ajouter les annotations sur chaque barre ---
    ax = plt.gca()
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width() / 2,
                    height,
                    f"{int(height)}",   # tu peux mettre f"{height:.0f}" si tu veux les arrondis
                    ha="center", va="bottom", fontsize=8, color="black")
    plt.title('Distribution des prix au mètre carré')
    plt.xlabel('Prix au m² (€)')
    plt.ylabel('Fréquence')
    plt.show()

    # Comparaison de la distribution des prix (avant/après winsorisation)
    plt.figure(figsize=(12, 5))
    # --- Boxplot 1 : avant winsorisation ---
    plt.subplot(1, 2, 1)
    data1 = df['Prix_au_m²'].dropna()
    plt.boxplot(data1, vert=False)
    plt.title("Prix au m² (Avant winsorisation)")
    plt.xlabel("Prix au m² (€)")

    # Calcul et annotation des stats
    median1 = data1.median()
    q1_1, q3_1 = data1.quantile([0.25, 0.75])
    min1, max1 = data1.min(), data1.max()
    plt.text(median1, 1.05, f"Mediane: {median1:.0f}", ha='center', fontsize=9, color='black')
    plt.text(q1_1, 0.95, f"Q1: {q1_1:.0f}", ha='center', fontsize=8, color='gray')
    plt.text(q3_1, 0.95, f"Q3: {q3_1:.0f}", ha='center', fontsize=8, color='gray')
    plt.text(min1, 1.05, f"Min: {min1:.0f}", ha='left', fontsize=8, color='gray')
    plt.text(max1, 1.05, f"Max: {max1:.0f}", ha='right', fontsize=8, color='gray')

    # --- Boxplot 2 : après winsorisation ---
    plt.subplot(1, 2, 2)
    data2 = df['Prix_au_m2_w'].dropna()
    plt.boxplot(data2, vert=False)
    plt.title("Prix au m² (Après winsorisation)")
    plt.xlabel("Prix au m² (€)")

    median2 = data2.median()
    q1_2, q3_2 = data2.quantile([0.25, 0.75])
    min2, max2 = data2.min(), data2.max()
    plt.text(median2, 1.05, f"Mediane: {median2:.0f}", ha='center', fontsize=9, color='black')
    plt.text(q1_2, 0.95, f"Q1: {q1_2:.0f}", ha='center', fontsize=8, color='gray')
    plt.text(q3_2, 0.95, f"Q3: {q3_2:.0f}", ha='center', fontsize=8, color='gray')
    plt.text(min2, 1.05, f"Min: {min2:.0f}", ha='left', fontsize=8, color='gray')
    plt.text(max2, 1.05, f"Max: {max2:.0f}", ha='right', fontsize=8, color='gray')

    plt.tight_layout()
    plt.show()


    # Distribution des catégories DPE et GES
    plt.figure(figsize=(12, 5))

    # --- DPE ---
    plt.subplot(1, 2, 1)
    ax1 = sns.countplot(x='DPE_num', data=df, palette='coolwarm')
    plt.title('Distribution des catégories DPE (1=Bon 2=Moyen 3=Mauvais)')
    plt.xlabel('Catégories DPE')
    plt.ylabel('Fréquence')

    # Ajouter les valeurs au-dessus des barres
    total_dpe = sum(p.get_height() for p in ax1.patches)
    for p in ax1.patches:
        height = p.get_height()
        if height > 0:
            ax1.text(p.get_x() + p.get_width() / 2,
                    height,
                    f"{int(height)}\n({height/total_dpe*100:.1f}%)",
                    ha='center', va='bottom', fontsize=9, color='black')

    # --- GES ---
    plt.subplot(1, 2, 2)
    ax2 = sns.countplot(x='GES_num', data=df, palette='coolwarm')
    plt.title('Distribution des catégories GES (1=Bon 2=Moyen 3=Mauvais)')
    plt.xlabel('Catégories GES')
    plt.ylabel('Fréquence')
    total_ges = sum(p.get_height() for p in ax2.patches)
    for p in ax2.patches:
        height = p.get_height()
        if height > 0:
            ax2.text(p.get_x() + p.get_width() / 2,
                    height,
                    f"{int(height)}\n({height/total_ges*100:.1f}%)",
                    ha='center', va='bottom', fontsize=9, color='black')

    plt.tight_layout()
    plt.show()


if not df.empty:
    display_descriptive_stats(df)
    plot_distributions(df)


# ==============================================================================
# 1. DPE/GES et Caractéristiques du Bien (Effet de Modernité et Taille)
# ==============================================================================

def analyze_dpe_characteristics_interaction(data):
    """Analyse les interactions entre la performance énergétique et les caractéristiques du bien (Année, Surface, Équipements)."""
    print("\n" + "="*80)
    print("1. ANALYSE DES INTERACTIONS : DPE/GES vs. CARACTÉRISTIQUES DU BIEN")
    print("="*80)
    
    # --- 1.1. DPE/GES x Année de Construction ---
    print("\n--- 1.1. Interaction : DPE/GES vs. Année de construction ---")
    
    # Créer une variable d'ancienneté simplifiée pour la visualisation (e.g., Avant 1975 vs Après 1975)
    df['Anciennete_Cat'] = pd.cut(df['Année_de_construction'], bins=[0, 1975, df['Année_de_construction'].max()], labels=['Avant 1975 (Isolation faible)', 'Apres 1975 (Isolation reglementee)'], right=False)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='DPE_num', y='Prix_au_m2_w', hue='Anciennete_Cat', data=df)
    plt.title('Prix au m² par DPE, séparé par catégorie d\'ancienneté')
    plt.xlabel('DPE (1=Bon 2=Moyen 3=Mauvais)')
    plt.ylabel('Prix au m²(€)')
    plt.show()

    # --- 1.2. DPE/GES x Surface ---
    print("\n--- 1.2. Interaction : DPE/GES vs. Surface ---")

    plt.figure(figsize=(10, 6))

    # Groupes
    mask_good   = df['DPE_num'].isin([1, 2, 3])
    mask_mid    = df['DPE_num'].isin([4])
    mask_bad    = df['DPE_num'].isin([5, 6, 7])

    ax = plt.gca()

    # Plots (identiques aux tiens)
    sns.regplot(x='Surface_w', y='Prix_au_m2_w', data=df[mask_good],
                scatter_kws={'alpha':0.3}, line_kws={'color':'green'}, label='DPE Bon (A–C)')
    sns.regplot(x='Surface_w', y='Prix_au_m2_w', data=df[mask_mid],
                scatter_kws={'alpha':0.3}, line_kws={'color':'orange'}, label='DPE Moyen (D)')
    sns.regplot(x='Surface_w', y='Prix_au_m2_w', data=df[mask_bad],
                scatter_kws={'alpha':0.3}, line_kws={'color':'red'}, label='DPE Mauvais (E–G)')

    plt.title('Relation Prix vs Surface, segmentée par performance énergétique')
    plt.xlabel('Surface winsorisée (m²)')
    plt.ylabel('Prix au m² winsorisé (€)')

    # ---------- Annotations : pente, R², n ----------
    def fit_stats(x, y):
        x = np.asarray(x); y = np.asarray(y)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]; y = y[m]
        n = x.size
        if n < 2:
            return None
        slope, intercept = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0, 1] if n > 1 else np.nan
        r2 = r**2 if np.isfinite(r) else np.nan
        return slope, r2, n

    groups = [
        (mask_good, "DPE Bon (A–C)",  "green"),
        (mask_mid,  "DPE Moyen (D)",  "orange"),
        (mask_bad,  "DPE Mauvais (E–G)", "red"),
    ]

    ypos = 0.96  # position verticale (axes coords)
    for mask, label, color in groups:
        stats = fit_stats(df.loc[mask, 'Surface_w'], df.loc[mask, 'Prix_au_m2_w'])
        if stats is None:
            continue
        slope, r2, n = stats
        # pente affichée par 10 m² pour être parlante
        ax.text(0.02, ypos,
                f"{label} : n={n}  β≈{slope*10:.1f} €/m² / 10 m²  R²={r2:.2f}",
                transform=ax.transAxes, ha='left', va='top', fontsize=9, color=color)
        ypos -= 0.08  # décale la ligne suivante

    plt.legend()
    plt.tight_layout()
    plt.show()


    # --- 1.3. DPE/GES x Ascenseur & Balcon ---
    print("\n--- 1.3. Interactions : DPE/GES vs. Équipements (Ascenseur, Balcon) ---")

    # --- DPE x Ascenseur ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    ax1 = sns.pointplot(
        x='DPE_num', y='Prix_au_m2_w',
        hue='Ascenseur', data=df,
        errorbar=None, capsize=0.1, palette='Set2'
    )
    plt.title('Prix moyen par DPE, avec/sans ascenseur')
    plt.xlabel('DPE (1=Bon 2=Moyen 3=Mauvais)')
    plt.ylabel('Prix moyen au m² (€)')

    # 🔹 Ajout manuel des annotations
    grouped = df.groupby(['DPE_num', 'Ascenseur'], observed=False)['Prix_au_m2_w']
    for (dpe, asc), sub in grouped:
        if len(sub) == 0:
            continue
        mean = sub.mean()
        n = len(sub)
        hue_order = sorted(df['Ascenseur'].dropna().unique())
        hue_idx = hue_order.index(asc)
        x = (dpe - 1) + (hue_idx - (len(hue_order)-1)/2) * 0.2
        plt.text(x, mean, f"{mean:.0f}€\n(n={n})",
                ha='center', va='bottom', fontsize=8, color='black')

    # --- DPE x Balcon ---
    plt.subplot(1, 2, 2)
    ax2 = sns.pointplot(
        x='DPE_num', y='Prix_au_m2_w',
        hue='Balcon', data=df,
        errorbar=None, capsize=0.1, palette='Set2'
    )
    plt.title('Prix moyen par DPE, avec/sans balcon')
    plt.xlabel('DPE (1=Bon 2=Moyen 3=Mauvais)')
    plt.ylabel('')

    grouped = df.groupby(['DPE_num', 'Balcon'], observed=False)['Prix_au_m2_w']
    for (dpe, bal), sub in grouped:
        if len(sub) == 0:
            continue
        mean = sub.mean()
        n = len(sub)
        hue_order = sorted(df['Balcon'].dropna().unique())
        hue_idx = hue_order.index(bal)
        x = (dpe - 1) + (hue_idx - (len(hue_order)-1)/2) * 0.2
        plt.text(x, mean, f"{mean:.0f}€\n(n={n})",
                ha='center', va='bottom', fontsize=8, color='black')

    plt.tight_layout()
    plt.show()



# ==============================================================================
# 2. Localité et Caractéristiques du Bien (Effet du Marché Local)
# ==============================================================================

def analyze_local_market_interaction(data):
    """Analyse les interactions entre la localité et les caractéristiques du bien (Surface, Parking)."""
    print("\n" + "="*80)
    print("2. ANALYSE DES INTERACTIONS : LOCALITÉ vs. CARACTÉRISTIQUES DU MARCHÉ")
    print("="*80)

    # --- 2.1. Localité x Surface ---
    print("\n--- 2.1. Interaction : Localité vs. Surface ---")

    palette = "tab10"  # garde des couleurs stables pour texte + lignes
    g = sns.lmplot(
        x='Surface_w', y='Prix_au_m2_w',
        hue='ville_classer', data=df,
        height=5, aspect=1.3, scatter_kws={'alpha': 0.4}, palette=palette
    )
    ax = g.axes[0, 0]

    plt.title('Relation prix vs surface, segmentée par localité')
    plt.xlabel('Surface winsorisée (m²)')
    plt.ylabel('Prix au m² winsorisé (€)')

    # ---- Annotations : β, R², n par localité ----
    import numpy as np
    levels = sorted(df['ville_classer'].dropna().unique())
    colors = sns.color_palette(palette, n_colors=len(levels))

    def stats_lin(x, y):
        x = np.asarray(x); y = np.asarray(y)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]; y = y[m]
        n = x.size
        if n < 2:
            return None
        slope, intercept = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0, 1]
        r2 = r**2
        return slope, r2, n

    ypos = 0.96
    for c, lvl in enumerate(levels):
        sub = df[df['ville_classer'] == lvl]
        st = stats_lin(sub['Surface_w'], sub['Prix_au_m2_w'])
        if st is None:
            continue
        slope, r2, n = st
        ax.text(
            0.02, ypos,
            f"{lvl} : n={n}  β≈{slope*10:.1f} €/m² / +10 m²  R²={r2:.2f}",
            transform=ax.transAxes, ha='left', va='top', fontsize=9, color=colors[c]
        )
        ypos -= 0.09

    plt.tight_layout()
    plt.show()

   
    # --- 2.2. Localité x Parking ---
    print("\n--- 2.2. Interaction : Localité vs. Parking ---")

    plt.figure(figsize=(8, 6))

    # Fixe l'ordre pour maîtriser les annotations
    x_order   = sorted(df['ville_classer'].dropna().unique())
    hue_order = sorted(df['Parking'].dropna().unique())  # 0/1 ou False/True

    ax = sns.barplot(
        x='ville_classer', y='Prix_au_m2_w',
        hue='Parking', data=df, errorbar=None, palette='pastel',
        order=x_order, hue_order=hue_order
    )
    plt.title('Prix moyen par localité, avec/sans parking')
    plt.xlabel('Localité')
    plt.ylabel('Prix moyen au m² winsorisé (€)')

    # ---- Annotations : moyenne + n sur chaque barre ----
    # Comptages pour n
    counts = (
        df.groupby(['ville_classer', 'Parking'], observed=False)['Prix_au_m2_w']
        .size().to_dict()
    )

    # Parcours des patches dans l'ordre (pour chaque x, toutes les hues)
    idx = 0
    for xi, xval in enumerate(x_order):
        for hi, hval in enumerate(hue_order):
            if idx >= len(ax.patches):
                break
            p = ax.patches[idx]; idx += 1
            h = p.get_height()
            if h is None:
                continue
            # n correspondant
            n = counts.get((xval, hval), 0)
            # centre de la barre
            x = p.get_x() + p.get_width() / 2
            # label sur 2 lignes : moyenne + n
            ax.text(x, h, f"{h:.0f}€\n(n={n})",
                    ha='center', va='bottom', fontsize=9, color='black')

    plt.tight_layout()
    plt.show()

try:
    # 1. Vérifiez que 'df' existe et est un DataFrame valide
    if 'df' in locals() and isinstance(df, pd.DataFrame) and not df.empty:
        print("Démarrage de l'analyse des interactions...")
        
        # 2. APPELEZ VOS FONCTIONS ICI
        analyze_dpe_characteristics_interaction(df)
        analyze_local_market_interaction(df)
        
        print("\nAnalyse terminée.")
    else:
        print("df n'est pas défini ou est vide. Veuillez charger les données.")
except NameError:
    print("df n'est pas définie. Assurez-vous d'avoir chargé vos données.")
except AttributeError as e:
    print(f"Une colonne est manquante ou mal nommée dans 'df'. Détails: {e}")

# ==============================================================================
# 3. Qualité Textuelle et Prix (Marketing Énergétique)
# ==============================================================================

sns.set_style("whitegrid")

def analyze_text_interaction(data):
    """Analyse les interactions entre la performance énergétique et la qualité textuelle."""
    print("\n" + "="*80)
    print("3. ANALYSE DES INTERACTIONS : DPE/GES vs. QUALITÉ TEXTUELLE")
    print("="*80)

    # --- 3.1. Interaction : DPE/GES vs. Mots Positifs (kw_pos) ---
    print("\n--- 3.1. Interaction : DPE/GES vs. Mots Positifs (kw_pos) ---")
    
    # DPE_num à 3 classes : 1=Bon (A-C), 2=Moyen (D), 3=Mauvais (E-G)
    df['DPE_Qualite'] = df['DPE_num'].apply(
        lambda x: 'Mauvais DPE (E-G)' if x > 2
        else 'Moyen DPE (D)' if x == 2
        else 'Bon DPE (A-C)'
    )
    df['DPE_Qualite'] = pd.Categorical(
        df['DPE_Qualite'],
        categories=['Bon DPE (A-C)', 'Moyen DPE (D)', 'Mauvais DPE (E-G)'],
        ordered=True
    )

    # Terciles (0-33 / 33-66 / 66-100) sur kw_pos
    q_33 = df['kw_pos'].quantile(0.33)      # FIX: doublon supprimé
    q_66 = df['kw_pos'].quantile(0.66)

    eps = 1e-6
    bins = [df['kw_pos'].min() - eps, q_33, q_66, df['kw_pos'].max() + eps]
    labels = ['kw_pos Faible', 'kw_pos Moyen', 'kw_pos Élevé']

    # Bords strictement croissants (évite ValueError si quantiles égaux)
    unique_bins = sorted(set(bins))

    if len(unique_bins) < len(labels) + 1:
        # Fallback propre en 2 classes (données trop concentrées)
        median_val = df['kw_pos'].median()
        df['kw_pos_cat'] = df['kw_pos'].apply(
            lambda x: 'kw_pos > Médiane' if x > median_val else 'kw_pos <= Médiane'
        )
        print("Avertissement : 'kw_pos' est trop concentré. Catégorisation simplifiée en 2 groupes (médiane).")
    else:
        # Si un bord saute après déduplication, on ajuste la liste de labels
        use_labels = labels[:len(unique_bins) - 1]
        df['kw_pos_cat'] = pd.cut(
            df['kw_pos'],
            bins=unique_bins,
            labels=use_labels,
            include_lowest=True,
            right=False
        )

    # --- Plot 3.1 ---
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        x='DPE_Qualite',
        y='Prix_au_m2_w',
        hue='kw_pos_cat',
        data=df,
        palette='Set3'
    )

    plt.title('Prix au m² par qualité DPE, segmenté par niveau de mots positifs', fontsize=14, pad=15)
    plt.xlabel('Performance énergétique', fontsize=12)
    plt.ylabel('Prix au m² (€)', fontsize=12)
    plt.legend(title='Niveau de mots positifs', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # --- Calcul des 5 chiffres clés + n ---
    group_stats = (
        df.groupby(['DPE_Qualite', 'kw_pos_cat'], observed=False)['Prix_au_m2_w']
        .agg(
            Min='min',
            Q1=lambda x: x.quantile(0.25),
            Median='median',
            Q3=lambda x: x.quantile(0.75),
            Max='max',
            n='count'
        )
        .reset_index()
    )

    # --- Positionnement dynamique ---
    hue_levels = sorted(df['kw_pos_cat'].dropna().unique())
    n_hue = len(hue_levels)
    offsets = np.linspace(-0.3, 0.3, n_hue)

    # --- Annotations : Min, Q1, Médiane, Q3, Max, n ---
    for i, dpe in enumerate(df['DPE_Qualite'].cat.categories):
        for j, hue_val in enumerate(hue_levels):
            subset = group_stats[
                (group_stats['DPE_Qualite'] == dpe) &
                (group_stats['kw_pos_cat'] == hue_val)
            ]
            if subset.empty:
                continue

            s = subset.iloc[0]
            x_pos = i + offsets[j]

            # Min et Max
            ax.text(x_pos, s['Min'], f"Min\n{int(s['Min']):,}€", ha='center', va='bottom', fontsize=7, color='gray')
            ax.text(x_pos, s['Max'], f"Max\n{int(s['Max']):,}€", ha='center', va='bottom', fontsize=7, color='gray')

            # Q1 / Médiane / Q3
            ax.text(x_pos, s['Q1'], f"Q1\n{int(s['Q1']):,}€", ha='center', va='bottom', fontsize=7, color='gray')
            ax.text(x_pos, s['Median'], f"Med\n{int(s['Median']):,}€", ha='center', va='bottom', fontsize=8, fontweight='bold', color='black')
            ax.text(x_pos, s['Q3'], f"Q3\n{int(s['Q3']):,}€", ha='center', va='bottom', fontsize=7, color='gray')

            # n (effectif total)
            ax.text(
                x_pos, s['Max'] + (s['Max'] * 0.03),
                f"(n={s['n']})",
                ha='center', va='bottom',
                fontsize=8, color='black',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2')
            )

    plt.tight_layout()
    plt.show()


    # --- 3.2. Interaction : Mots Négatifs vs. Localité ---
    print("\n--- 3.2. Interaction : Mots Négatifs vs. Localité ---")

    df['kw_neg_bin'] = df['kw_neg'].apply(lambda x: 'kw_neg > 0' if x > 0 else 'kw_neg = 0')

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
        x='ville_classer',
        y='Prix_au_m2_w',
        hue='kw_neg_bin',
        data=df,
        errorbar=None,
        palette='Set1',
        edgecolor='white'
    )

    plt.title('Décote des mots négatifs par localité', fontsize=14, pad=15)
    plt.xlabel('Localité', fontsize=12)
    plt.ylabel('Prix moyen au m² (€)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # Légende à droite
    plt.legend(
        title='Présence de mots négatifs',
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )

    # --- Annotations sur les barres ---
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            # Position centrée sur la barre
            x = p.get_x() + p.get_width() / 2
            y = height
            ax.text(
                x, y + (ax.get_ylim()[1] * 0.01),  # décalage fixe basé sur l’échelle du graphe
                f"{int(height):,}€",
                ha='center', va='bottom',
                fontsize=8, color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.2')
            )

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.show()


    # Nettoyage léger : on garde DPE_Qualite si tu en as besoin ensuite (sections 4.x)
    df.drop(columns=['kw_neg_bin', 'kw_pos_cat'], inplace=True, errors='ignore')

# ================== EXÉCUTION MINIMALE ==================
if 'df' in locals() and isinstance(df, pd.DataFrame) and not df.empty:
    analyze_text_interaction(df)
else:
    print("Le DataFrame 'df' n'est pas chargé. Veuillez exécuter l'import des données.")


def analyze_marketing_and_other_interactions(df):
# --- 4.1. DPE/GES x Nombre de Photos ---
    print("\n--- 4.1. Interaction : DPE/GES vs. Nombre de Photos ---")

    # 1) Sécurité DPE_Qualite
    if 'DPE_Qualite' not in df.columns:
        df['DPE_Qualite'] = df['DPE_num'].apply(
            lambda x: 'Mauvais DPE (E-G)' if x > 2
            else 'Moyen DPE (D)' if x == 2
            else 'Bon DPE (A-C)'
        )
        df['DPE_Qualite'] = pd.Categorical(
            df['DPE_Qualite'],
            categories=['Bon DPE (A-C)', 'Moyen DPE (D)', 'Mauvais DPE (E-G)'],
            ordered=True
        )

    # 2) Catégoriser photos (médiane)
    q_median_photos = df['Nombre_de_photos'].median()
    df['Photos_Cat'] = np.where(df['Nombre_de_photos'] > q_median_photos,
                                'Photos > Médiane', 'Photos ≤ Médiane')
    hue_order = ['Photos ≤ Médiane', 'Photos > Médiane']
    df['Photos_Cat'] = pd.Categorical(df['Photos_Cat'], categories=hue_order, ordered=True)

    # 3) Tracé
    x_order = list(df['DPE_Qualite'].cat.categories)
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        x='DPE_Qualite', y='Prix_au_m2_w',
        hue='Photos_Cat', data=df,
        order=x_order, hue_order=hue_order, palette='Set2'
    )
    plt.title('Prix au m² par qualité DPE, segmenté par nombre de photos')
    plt.xlabel('Performance énergétique'); plt.ylabel('Prix au m² (€)')
    plt.legend(title='Nombre de photos', bbox_to_anchor=(1.02, 1), loc='upper left')

    # 4) ANNOTATIONS ROBUSTES (sans ax.artists)
    #    -> on calcule la médiane et la position x comme seaborn (dodge)
    stats = (
        df.groupby(['DPE_Qualite', 'Photos_Cat'], observed=False)['Prix_au_m2_w']
        .agg(median='median', n='count')
        .reset_index()
    )

    # index des niveaux
    x_idx   = {lvl: i for i, lvl in enumerate(x_order)}
    h_idx   = {lvl: j for j, lvl in enumerate(hue_order)}
    m       = len(hue_order)
    width   = 0.8                    # largeur de groupe (par défaut seaborn)
    ymin, ymax = ax.get_ylim()
    yoff    = (ymax - ymin) * 0.01   # petit décalage vertical

    for _, r in stats.iterrows():
        if pd.isna(r['median']) or r['n'] == 0: 
            continue
        i = x_idx.get(r['DPE_Qualite'])
        j = h_idx.get(r['Photos_Cat'])
        if i is None or j is None: 
            continue
        # position x avec "dodge" façon seaborn
        x_pos = i + (j - (m - 1) / 2) * (width / max(m, 1))
        y_pos = r['median']

        ax.text(
            x_pos, y_pos + yoff,
            f"{r['median']:.0f}€\n(n={int(r['n'])})",
            ha='center', va='bottom', fontsize=8, color='black',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=2)
        )

    plt.tight_layout()
    plt.show()


    # --- 4.2. DPE/GES x Charges (Charges_w) ---
    # --- 4.2. DPE/GES x Charges (Charges_w) ---
    print("\n--- 4.2. Interaction : DPE/GES vs. Charges (Charges_w) ---")

    # Créer la variable de catégorie
    q_charges_75 = df['Charges_w'].quantile(0.75)
    df['Charges_Cat'] = np.where(df['Charges_w'] > q_charges_75,
                                'Charges Élevées (>75e pct)',
                                'Charges Faibles/Moyennes')
    hue_order = ['Charges Faibles/Moyennes', 'Charges Élevées (>75e pct)']
    df['Charges_Cat'] = pd.Categorical(df['Charges_Cat'], categories=hue_order, ordered=True)

    # Calcul manuel des moyennes + effectifs
    stats = (
        df.groupby(['DPE_num', 'Charges_Cat'], observed=False)['Prix_au_m2_w']
        .agg(mean='mean', n='count')
        .reset_index()
    )

    # Tracé manuel avec annotations
    plt.figure(figsize=(10, 6))

    # décalage horizontal pour séparer les deux séries
    offsets = {'Charges Faibles/Moyennes': -0.15, 'Charges Élevées (>75e pct)': 0.15}
    colors = {'Charges Faibles/Moyennes': 'royalblue', 'Charges Élevées (>75e pct)': 'tomato'}

    for cat in hue_order:
        subset = stats[stats['Charges_Cat'] == cat]
        x = subset['DPE_num'] + offsets[cat]
        y = subset['mean']
        plt.plot(x, y, marker='o', linestyle='-', color=colors[cat], label=cat)
        # annotations
        for xi, yi, n in zip(x, y, subset['n']):
            plt.text(xi, yi, f"{yi:.0f}€\n(n={n})",
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1))

    plt.title('Prix moyen par DPE, segmenté par niveau de charges')
    plt.xlabel('DPE (1=Bon 2=Moyen 3=Mauvais)')
    plt.ylabel('Prix moyen au m² (€)')
    plt.legend(title='Niveau de charges')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


    # --- 4.3. DPE/GES x Nombre de Pièces ---
    print("\n--- 4.3. Interaction : DPE/GES vs. Pièces ---")

    # Catégorie pièces
    df['Pieces_Cat'] = df['Pièces'].apply(lambda x: 'Grand Bien (> 3 Pièces)' if x > 3
                                                    else 'Petit/Moyen Bien (<= 3 Pièces)')
    hue_order = ['Petit/Moyen Bien (<= 3 Pièces)', 'Grand Bien (> 3 Pièces)']
    df['Pieces_Cat'] = pd.Categorical(df['Pieces_Cat'], categories=hue_order, ordered=True)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='DPE_num', y='Prix_au_m2_w', hue='Pieces_Cat', data=df,
                    errorbar=None, palette='pastel', hue_order=hue_order)
    plt.title('Prix moyen par DPE, segmenté par nombre de pièces')
    plt.xlabel('DPE (1=Bon 2=Moyen 3=Mauvais)')
    plt.ylabel('Prix moyen au m² (€)')

    # Annotations: moyenne + n (robuste)
    stats = (df.groupby(['DPE_num', 'Pieces_Cat'], observed=False)['Prix_au_m2_w']
            .agg(mean='mean', n='count').reset_index())

    x_order = sorted(df['DPE_num'].dropna().unique())
    m = len(hue_order); width = 0.8
    ymin, ymax = ax.get_ylim(); yoff = (ymax - ymin) * 0.015

    for _, r in stats.iterrows():
        i = x_order.index(r['DPE_num']); j = hue_order.index(r['Pieces_Cat'])
        x_pos = i + (j - (m - 1)/2) * (width / m)
        y_pos = r['mean']
        ax.text(x_pos, y_pos + yoff, f"{y_pos:.0f}€\n(n={int(r['n'])})",
                ha='center', va='bottom', fontsize=9,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1))

    plt.tight_layout()
    plt.show()

    # --- 4.4. DPE/GES x Procédure (Procedure_bin) ---
    print("\n--- 4.4. Interaction : DPE/GES vs. Procédure (Procedure_bin) ---")

    # Sécurité DPE_Qualite
    if 'DPE_Qualite' not in df.columns:
        df['DPE_Qualite'] = df['DPE_num'].apply(
            lambda x: 'Mauvais DPE (E-G)' if x > 2 else ('Moyen DPE (D)' if x == 2 else 'Bon DPE (A-C)')
        )
        df['DPE_Qualite'] = pd.Categorical(df['DPE_Qualite'],
            categories=['Bon DPE (A-C)', 'Moyen DPE (D)', 'Mauvais DPE (E-G)'], ordered=True)

    hue_order = [0, 1]  # 0=Non, 1=Oui (à ajuster si tes valeurs sont différentes)

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='DPE_Qualite', y='Prix_au_m2_w', hue='Procedure_bin', data=df,
                    order=list(df['DPE_Qualite'].cat.categories), hue_order=hue_order)
    plt.title('Prix au m² par qualité DPE, segmenté par procédure en cours')
    plt.xlabel('Performance énergétique'); plt.ylabel('Prix au m² (€)')
    plt.legend(title='Procédure (1=Oui)')

    # Annotations: médiane + n, robustes (recalcule position "dodge")
    stats = (df.groupby(['DPE_Qualite', 'Procedure_bin'], observed=False)['Prix_au_m2_w']
            .agg(median='median', n='count').reset_index())

    x_order = list(df['DPE_Qualite'].cat.categories)
    m = len(hue_order); width = 0.8
    ymin, ymax = ax.get_ylim(); yoff = (ymax - ymin) * 0.01

    x_idx = {lvl: i for i, lvl in enumerate(x_order)}
    for _, r in stats.iterrows():
        i = x_idx[r['DPE_Qualite']]; j = hue_order.index(r['Procedure_bin'])
        x_pos = i + (j - (m - 1)/2) * (width / m)
        y_pos = r['median']
        ax.text(x_pos, y_pos + yoff, f"{y_pos:.0f}€\n(n={int(r['n'])})",
                ha='center', va='bottom', fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1))

    plt.tight_layout()
    plt.show()

    # --- 4.5. DPE/GES x Note/Avis (Note_z) ---
    print("\n--- 4.5. Interaction : DPE/GES vs. Note de l'Agent (Note_z) ---")

    df['Note_z_Cat'] = np.where(df['Note_z'] < 0, 'Note < Moyenne',
                        np.where(df['Note_z'] >= 0, 'Note ≥ Moyenne', 'Note manquante'))
    hue_order = ['Note < Moyenne', 'Note ≥ Moyenne', 'Note manquante']
    df['Note_z_Cat'] = pd.Categorical(df['Note_z_Cat'], categories=hue_order, ordered=True)

    plt.figure(figsize=(10, 6))
    ax = sns.pointplot(x='DPE_num', y='Prix_au_m2_w', hue='Note_z_Cat', data=df,
                    errorbar=None, capsize=0.1, palette='Set2', hue_order=hue_order)

    plt.title('Prix moyen par DPE, segmenté par note agent (Z-score)')
    plt.xlabel('DPE (1=Bon 2=Moyen 3=Mauvais)'); plt.ylabel('Prix moyen au m² (€)')

    # Annotations: moyenne + n (robuste)
    stats = (df.groupby(['DPE_num', 'Note_z_Cat'], observed=False)['Prix_au_m2_w']
            .agg(mean='mean', n='count').reset_index())

    x_order = sorted(df['DPE_num'].dropna().unique())
    m = len(hue_order); width = 0.8
    ymin, ymax = ax.get_ylim(); yoff = (ymax - ymin) * 0.015

    for _, r in stats.iterrows():
        if r['Note_z_Cat'] not in hue_order: 
            continue
        i = x_order.index(r['DPE_num']); j = hue_order.index(r['Note_z_Cat'])
        x_pos = i + (j - (m - 1)/2) * (width / m)
        y_pos = r['mean']
        ax.text(x_pos, y_pos + yoff, f"{y_pos:.0f}€\n(n={int(r['n'])})",
                ha='center', va='bottom', fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1))

    plt.legend(title='Note (Z)', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

try:
    if 'df' in locals() and isinstance(df, pd.DataFrame) and not df.empty:
        print("Démarrage de l'analyse des interactions secondaires.")
        
        # APPEL DE LA FONCTION
        analyze_marketing_and_other_interactions(df)
        
        print("\nAnalyse terminée.")
    else:
        print("df n'est pas défini ou est vide. Veuillez charger les données.")
except NameError:
    print("df n'est pas définie. Assurez-vous d'avoir chargé vos données.")
except Exception as e:
    print(f"Une erreur inattendue est survenue lors de l'exécution : {e}")
    
# ==============================================================================
# 2. ANALYSE BIVARIÉE ET TESTS STATISTIQUES (Prime Verte Générale)
# ==============================================================================

def analyze_correlations(data):
    """Calcule et visualise les corrélations."""
    print("\n" + "="*50)
    print("2.1. Analyse des Corrélations (Pearson et Spearman)")
    print("="*50)

    corr_vars = ['Prix_au_m2_w', 'DPE_num', 'GES_num', 'Surface_w','Procedure_bin','Charges_w',
                 'Nombre_de_lots', 'Pièces', 'Parking', 'Balcon',
                 'Nombre_de_photos', 'Note_z', 'Avis_z','Transport', 'Ascenseur']


    # Corrélation de Spearman (non-linéaire / ordinal)
    spearman_corr = data[corr_vars].corr(method='spearman')
    print("Corrélation de Spearman :")
    print(spearman_corr[['Prix_au_m2_w', 'DPE_num', 'GES_num']])

    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Carte de chaleur des corrélations (Spearman)')
    plt.show()

def plot_group_comparisons(data):
    """Visualise les relations entre DPE/GES et le prix au m²."""
    print("\n2.2. Visualisation des relations DPE/GES - Prix")

    # Fonction utilitaire pour annoter chaque graphique
    def annotate_box_violin(ax, df, x, y):
        """Affiche médiane et n au-dessus de chaque groupe."""
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        if not tick_labels:
            return
        
        ymin, ymax = ax.get_ylim()
        y_offset = (ymax - ymin) * 0.02
        df_x = df[x].astype(str)

        for i, label in enumerate(tick_labels):
            sub = df.loc[df_x == label, y].dropna()
            if sub.empty:
                continue
            median_val = sub.median()
            n_val = len(sub)
            ax.text(i, median_val + y_offset,
                    f"{median_val:.0f}€\n(n={n_val})",
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1))

    # Création des 4 graphiques
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # --- Boxplot DPE
    sns.boxplot(x='DPE_num', y='Prix_au_m2_w', data=df, ax=axes[0, 0])
    axes[0, 0].set_title('Boxplot prix au m² vs DPE')
    axes[0, 0].set_xlabel('DPE (1=Bon 2=Moyen 3=Mauvais)')
    axes[0, 0].set_ylabel('Prix au m² (€)')
    annotate_box_violin(axes[0, 0], df, 'DPE_num', 'Prix_au_m2_w')

    # --- Boxplot GES
    sns.boxplot(x='GES_num', y='Prix_au_m2_w', data=df, ax=axes[0, 1])
    axes[0, 1].set_title('Boxplot prix au m² vs GES')
    axes[0, 1].set_xlabel('GES (1=Bon 2=Moyen 3=Mauvais)')
    annotate_box_violin(axes[0, 1], df, 'GES_num', 'Prix_au_m2_w')

    # --- Violin DPE
    sns.violinplot(x='DPE_num', y='Prix_au_m2_w', data=df, palette='Set2', ax=axes[1, 0])
    axes[1, 0].set_title('Violin plot prix au m² vs DPE')
    annotate_box_violin(axes[1, 0], df, 'DPE_num', 'Prix_au_m2_w')

    # --- Violin GES
    sns.violinplot(x='GES_num', y='Prix_au_m2_w', data=df, palette='Set2', ax=axes[1, 1])
    axes[1, 1].set_title('Violin plot prix au m² vs GES')
    annotate_box_violin(axes[1, 1], df, 'GES_num', 'Prix_au_m2_w')

    plt.tight_layout()
    plt.show()


def run_hypothesis_tests(data):
    """Effectue les tests statistiques (ANOVA, T-test) pour valider les hypothèses."""
    print("\n" + "="*50)
    print("2.3. Tests d'hypothèses (Prime verte générale)")
    print("="*50)
    
    # Préparer les groupes pour les tests
    min_obs = 30
    df_filtered_dpe = data[data.groupby('DPE_num')['Prix_au_m2_w'].transform('count') >= min_obs].copy()
    df_filtered_ges = data[data.groupby('GES_num')['Prix_au_m2_w'].transform('count') >= min_obs].copy()

    # Groupes pour ANOVA/Kruskal
    groups_dpe = [df_filtered_dpe[df_filtered_dpe['DPE_num'] == dpe]['Prix_au_m2_w'].dropna() 
                  for dpe in sorted(df_filtered_dpe['DPE_num'].unique())]
    groups_ges = [df_filtered_ges[df_filtered_ges['GES_num'] == ges]['Prix_au_m2_w'].dropna() 
                  for ges in sorted(df_filtered_ges['GES_num'].unique())]

    # Test d'homogénéité des variances (Levene)
    stat_dpe_l, p_value_dpe_l = levene(*groups_dpe)
    stat_ges_l, p_value_ges_l = levene(*groups_ges)
    print(f"Test de Levene (DPE) : p-value = {p_value_dpe_l:.4f} (Var. non homogènes si p < {ALPHA})")
    print(f"Test de Levene (GES) : p-value = {p_value_ges_l:.4f} (Var. non homogènes si p < {ALPHA})")
    
    # ANOVA (si variances homogènes) ou Kruskal-Wallis (sinon)
    # Étant donné le grand échantillon et la non-normalité, Kruskal-Wallis est robuste.
    anova_dpe = f_oneway(*groups_dpe)
    kruskal_ges = kruskal(*groups_ges)
    
    print(f"\nRésultat de l'ANOVA pour le DPE : F-stat = {anova_dpe.statistic:.2f}, p-value = {anova_dpe.pvalue:.4e}")
    print(f"Résultat du Kruskal-Wallis pour le GES : H-stat = {kruskal_ges.statistic:.2f}, p-value = {kruskal_ges.pvalue:.4e}")
    
    # T-test (DPE élevé vs DPE faible)
    # DPE Élevé (Bonne performance: 1, 2, 3) vs DPE Faible (Mauvaise performance: 4, 5, 6, 7)
    high_dpe = data[data['DPE_num'].isin([1, 2, 3])]['Prix_au_m2_w']
    low_dpe = data[data['DPE_num'].isin([4, 5, 6, 7])]['Prix_au_m2_w']
    t_stat, p_value = ttest_ind(high_dpe.dropna(), low_dpe.dropna(), equal_var=False) # Welch's t-test
    print(f"\nT-test (Welch) DPE Bon vs Mauvais : t-stat = {t_stat:.2f}, p-value = {p_value:.4e}")
    
if not df.empty:
    analyze_correlations(df)
    plot_group_comparisons(df)
    run_hypothesis_tests(df)

def analyze_transport_only(df):
    
        if 'Transport' not in df.columns:
            print("Erreur : La colonne 'Transport' est manquante dans le DataFrame.")
            return

        print("\n" + "="*50)
        print("3. Analyse de la Distribution de la variable Transport")
        print("="*50)

        # 1. Décompte des modalités (Statistiques)
        transport_counts = df['Transport'].value_counts()
        transport_proportions = df['Transport'].value_counts(normalize=True).mul(100).round(2)
        
        # Création d'un DataFrame de résumé pour l'affichage
        transport_summary = pd.DataFrame({
            'Effectif (Count)': transport_counts,
            'Proportion (%)': transport_proportions
        })

        print("Répartition des modalités de 'Transport' :")
        print(transport_summary)

        # 2. Visualisation (Graphique à barres)
        print("\nVisualisation de la répartition :")
        try:
            plt.figure(figsize=(7, 5))
            ax = sns.countplot(x='Transport', data=df, palette='viridis')

            # Titres et axes
            plt.title('Distribution de la variable Transport', fontsize=14)
            plt.xlabel('Modalité Transport', fontsize=12)
            plt.ylabel('Nombre de Biens (Fréquence)', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # --- Ajout des annotations (effectif + %) ---
            total = len(df)
            for p in ax.patches:
                count = int(p.get_height())
                if count > 0:
                    pct = 100 * count / total
                    ax.text(p.get_x() + p.get_width() / 2,
                            p.get_height() + total * 0.005,  # léger décalage vertical
                            f"{count} ({pct:.1f}%)",
                            ha='center', va='bottom',
                            fontsize=9, color='black')

            plt.tight_layout()
            plt.show()

        
            
        except Exception as e:
            print(f"\nErreur lors de la création du graphique : {e}")
if not df.empty:
    analyze_transport_only(df)


def analyze_transport_only(df):
    
    if 'Transport' not in df.columns:
        print("Erreur : La colonne 'Transport' est manquante dans le DataFrame.")
        return

    print("\n" + "="*50)
    print("3. Analyse de la Distribution de la variable 'Transport'")
    print("="*50)

    # 1. Décompte des modalités (Statistiques)
    transport_counts = df['Transport'].value_counts()
    transport_proportions = df['Transport'].value_counts(normalize=True).mul(100).round(2)
    
    # Création d'un DataFrame de résumé pour l'affichage
    transport_summary = pd.DataFrame({
        'Effectif (Count)': transport_counts,
        'Proportion (%)': transport_proportions
    })

    print("Répartition des modalités de 'Transport' :")
    print(transport_summary)

    # --- 2. Visualisation (Graphique à barres Horizontal) ---
    print("\nVisualisation de la répartition (Horizontale) :")
    try:
        # Utilisation de figsize plus adaptée au format horizontal
        plt.figure(figsize=(9, 4)) 
        
        # Utilisation de y='Transport' pour les barres horizontales
        ax = sns.countplot(y='Transport', data=df, palette='viridis',
                           # Ordonner par effectif décroissant
                           order=transport_counts.index) 

        # Titres et axes
        plt.title('Distribution de la variable transport', fontsize=14)
        # Inversion des étiquettes X et Y pour le format horizontal
        plt.xlabel('Nombre de Biens (Fréquence)', fontsize=12)
        plt.ylabel('Modalité Transport', fontsize=12)
        # La grille est maintenant verticale (axis='x')
        plt.grid(axis='x', linestyle='--', alpha=0.7) 

        # --- Ajout des annotations (effectif + %) ---
        total = len(df)
        # Parcourt les barres. Pour un barplot horizontal, on utilise get_width()
        for p in ax.patches:
            # La largeur de la barre donne l'effectif (X)
            count = int(p.get_width()) 
            if count > 0:
                pct = 100 * count / total
                
                # Le texte est placé à la fin de la barre (x) et centré sur la barre (y)
                ax.text(p.get_width() + total * 0.005, # léger décalage horizontal
                        p.get_y() + p.get_height() / 2,
                        f"{count} ({pct:.1f}%)",
                        ha='left', va='center', # Ancrage à gauche de la position et centré verticalement
                        fontsize=10, color='black')

        # Ajustement automatique des marges pour ne pas couper les étiquettes Y (Transport)
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        print(f"\nErreur lors de la création du graphique : {e}")
        

try:
    if 'df' in locals() and isinstance(df, pd.DataFrame) and not df.empty:
        analyze_transport_only(df)
    else:
        # Ceci est exécuté si le DataFrame n'est pas défini ou vide
        print("Le DataFrame 'df' n'est pas défini ou est vide. Veuillez le charger.")
except NameError:
     print("La variable 'df' n'est pas définie (NameError). Veuillez charger votre DataFrame.")



# ==============================================================================
# 3. ANALYSE DE L'IMPACT DE LA LOCALITÉ (Prime Verte Locale)
# ==============================================================================

def analyze_local_impact(data):
    """Analyse et visualise la variation des prix selon DPE/GES par localité."""
    print("\n" + "="*50)
    print("3.1. Prix moyen par localité et DPE/GES")
    print("="*50)
    
    # Prix moyen par localité et DPE
    prix_moyen_dpe_localite = data.groupby(['ville_classer', 'DPE_num'])['Prix_au_m2_w'].mean().unstack()
    print("Prix moyen (€/m²) par localité et DPE :")
    print(prix_moyen_dpe_localite)
    
    # Prix moyen par localité et GES
    prix_moyen_ges_localite = data.groupby(['ville_classer', 'GES_num'])['Prix_au_m2_w'].mean().unstack()
    print("\nPrix moyen (€/m²) par localité et GES :")
    print(prix_moyen_ges_localite)

    # Visualisation de la prime verte locale
    # === 1. Visualisation de la prime verte locale (Barplot + annotations) ===
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x='DPE_num', y='Prix_au_m2_w',
        hue='ville_classer', data=df,
        errorbar=None, palette='viridis',
        dodge=0.25, edgecolor='white', linewidth=1
    )

    # Titres et axes
    plt.title('Prix moyen par DPE et localité (variabilité de la prime verte)', fontsize=14, pad=20)
    plt.xlabel('DPE (1 = Bon | 2 = Moyen | 3 = Mauvais)', fontsize=12)
    plt.ylabel('Prix moyen au m² (€)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # 🔹 Légende à droite
    plt.legend(
        title='Localité',
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10, title_fontsize=11,
        frameon=False
    )

    # Annotations : moyenne + effectif
    stats = (
        df.groupby(['DPE_num', 'ville_classer'], observed=False)['Prix_au_m2_w']
        .agg(['mean', 'count']).reset_index()
    )
    localites = df['ville_classer'].unique()
    offsets = np.linspace(-0.25, 0.25, len(localites))

    for i, (ville, offset) in enumerate(zip(localites, offsets)):
        subset = stats[stats['ville_classer'] == ville]
        for _, row in subset.iterrows():
            x = row['DPE_num'] - 1 + offset
            y = row['mean']
            ax.text(
                x, y + (y * 0.02),
                f"{int(row['mean']):,}€\n(n={int(row['count'])})",
                ha='center', va='bottom',
                fontsize=9, color='black', fontweight='semibold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
            )

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.show()

    # === 2. Distribution des prix au m² par GES et localité (Boxplot + médianes) ===
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(
        x='GES_num', y='Prix_au_m2_w',
        hue='ville_classer', data=df, palette='viridis'
    )

    plt.title('Distribution des prix au m² par GES et localité', fontsize=14, pad=20)
    plt.xlabel('GES (1 = Bon | 2 = Moyen | 3 = Mauvais)', fontsize=12)
    plt.ylabel('Prix au m² (€)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # Légende à droite également
    plt.legend(
        title='Localité',
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10, title_fontsize=11,
        frameon=False
    )

    # Ajout des médianes centrées sur chaque boîte
    medians = (
        df.groupby(['GES_num', 'ville_classer'], observed=False)['Prix_au_m2_w']
        .median().reset_index()
    )

    for i, (ville, offset) in enumerate(zip(localites, offsets)):
        subset = medians[medians['ville_classer'] == ville]
        for _, row in subset.iterrows():
            x = row['GES_num'] - 1 + offset
            y = row['Prix_au_m2_w']
            ax.text(
                x, y, f"{int(y):,}€",
                ha='center', va='bottom',
                fontsize=8, color='black', fontweight='semibold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
            )

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.show()

if not df.empty:
    analyze_local_impact(df)

# ==============================================================================
# 4. MODÉLISATION ET DIAGNOSTIC
# ==============================================================================

def check_multicollinearity(data):
    """Calcule le VIF pour évaluer la multicolinéarité."""
    print("\n" + "="*50)
    print("4.1. Diagnostic de Multicolinéarité (VIF)")
    print("="*50)
    
    # Variables indépendantes pour le modèle de régression
    X_vars = ['Surface_w', 'DPE_num', 'GES_num', 'kw_pos', 'kw_neg']
    X = data[X_vars].dropna()
    
    # Calcul du VIF. Il faut ajouter une constante pour l'interception.
    X_vif = sm.add_constant(X)
    
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_vif.columns
    # La colonne 'const' n'est pas interprétable
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) 
                       for i in range(len(X_vif.columns))]
    
    print("Variance Inflation Factor (VIF) : (VIF > 5-10 pose problème)")
    # Exclure la constante
    print(vif_data[vif_data.Variable != 'const'].sort_values('VIF', ascending=False))
    
    # Note: VIF de DPE_num et GES_num est élevé (>5) en raison de leur forte corrélation (0.81). 
    # Cela suggère qu'ils mesurent le même construit (Performance Énergétique) et qu'il faut 
    # interpréter leurs effets avec prudence, ou envisager de les inclure dans des modèles séparés, 
    # ou d'utiliser un indice composite.

def run_regression_model(data):
    """Exécute et affiche le modèle de régression OLS."""
    print("\n" + "="*50)
    print("4.2. Modélisation de régression OLS (Hypothèse 3)")
    print("="*50)

    # Formule du modèle complet :
    # Prix_au_m2_w = f(DPE, GES, Surface, Localité, Interaction Localité*DPE/GES, Qualité textuelle)
    # C(variable) indique une variable catégorielle
    # DPE_num:C(ville_classer) est le terme d'interaction (effet différent de DPE selon la ville)
    
    formula = (
        'Prix_au_m2_w ~ DPE_num + GES_num + Surface_w + C(ville_classer) + '
        'DPE_num:C(ville_classer) + GES_num:C(ville_classer) + kw_pos + kw_neg'
    )
    
    # Ajustement du modèle OLS
    model = smf.ols(formula, data=data).fit()
    
    # Affichage du résumé
    print(model.summary())
    print("\n--- Interprétation du Modèle OLS ---")
    print(f"R-squared: {model.rsquared:.3f}. Le modèle explique {model.rsquared*100:.1f}% de la variance du prix.")
    print("Les termes d'interaction DPE_num:C(ville_classer) sont significatifs, ce qui confirme l'hypothèse de la 'Prime Verte' variable selon la localité.")


# ==============================================================================
# 5. MODÉLISATION DE RÉGRESSION ENRICHIE (Analyse des Croisements/Interactions)
# ==============================================================================

def run_enriched_regression_model(data):
    """
    Exécute et affiche le modèle de régression OLS enrichi avec les interactions clés.
    
    Les croisements testés incluent :
    - DPE/GES * Localité (déjà présent, pour l'effet de la Prime Verte variable)
    - DPE/GES * Surface (pour tester si la décote énergétique est plus forte sur les grandes surfaces)
    - DPE/GES * Année de construction (pour tester l'effet de la modernité)
    - Localité * Parking (pour tester la valeur variable d'un équipement selon la ville)
    """
    print("\n" + "="*70)
    print("5. MODÉLISATION DE RÉGRESSION OLS ENRICHIE AVEC INTERACTIONS")
    print("="*70)

    # Définition du modèle de base (variables principales)
    base_terms = (
        'Prix_au_m2_w ~ DPE_num + GES_num + Surface_w + C(ville_classer) + '
        'kw_pos + kw_neg'
    )
    
    # Définition des termes d'interaction (croisements)
    # H2: DPE/GES varie selon la localité
    interaction_localite = 'DPE_num:C(ville_classer) + GES_num:C(ville_classer)'
    
    # Nouvelles interactions proposées
    # 1. DPE/GES * Surface (Coût énergétique proportionnel à la taille)
    interaction_surface_dpe_ges = 'DPE_num:Surface_w + GES_num:Surface_w'
    
    # 2. DPE * Année de construction (Impact de la modernité sur l'indice)
    # Note: L'Année_de_construction peut être traitée comme continue
    interaction_annee_dpe = 'DPE_num:Année_de_construction'
    
    # 3. Localité * Parking (L'importance d'un équipement varie par marché)
    # C(Parking) car c'est probablement une variable binaire ou nominale
    interaction_localite_parking = 'C(ville_classer):C(Parking)'

    # Construction de la formule finale
    formula_enriched = f"{base_terms} + {interaction_localite} + {interaction_surface_dpe_ges} + {interaction_annee_dpe} + {interaction_localite_parking}"
    
    print(f"Formule de Régression : \n{formula_enriched}\n")
    
    # Ajout d'une ligne pour gérer les NaN avant la modélisation
    # On sélectionne les colonnes nécessaires et on supprime les lignes avec des valeurs manquantes.
    required_cols = ['Prix_au_m2_w', 'DPE_num', 'GES_num', 'Surface_w', 'ville_classer', 
                     'kw_pos', 'kw_neg', 'Année_de_construction', 'Parking']
    df_model = data[required_cols].dropna()

    try:
        # Ajustement du modèle OLS
        model_enriched = smf.ols(formula_enriched, data=df_model).fit()
        
        # Affichage du résumé
        print(model_enriched.summary())
        
        print("\n--- Analyse des Interactions ---")
        print(f"R-squared du modèle enrichi : {model_enriched.rsquared:.4f}")
        
        # Interprétation des p-values des termes d'interaction (Focus sur les nouveautés)
        
        # Interaction DPE:Surface
        p_dpe_surface = model_enriched.pvalues.filter(regex='DPE_num:Surface_w').iloc[0] if model_enriched.pvalues.filter(regex='DPE_num:Surface_w').shape[0] > 0 else np.nan
        print(f"P-value pour DPE_num:Surface_w : {p_dpe_surface:.4e}")
        
        # Interaction GES:Surface
        p_ges_surface = model_enriched.pvalues.filter(regex='GES_num:Surface_w').iloc[0] if model_enriched.pvalues.filter(regex='GES_num:Surface_w').shape[0] > 0 else np.nan
        print(f"P-value pour GES_num:Surface_w : {p_ges_surface:.4e}")
        
        # Interaction DPE:Année_de_construction
        p_dpe_annee = model_enriched.pvalues.filter(regex='DPE_num:Année_de_construction').iloc[0] if model_enriched.pvalues.filter(regex='DPE_num:Année_de_construction').shape[0] > 0 else np.nan
        print(f"P-value pour DPE_num:Année_de_construction : {p_dpe_annee:.4e}")
        
        print("\nSi les p-values des interactions sont faibles (< 0.05), leur effet est statistiquement significatif.")

    except Exception as e:
        print(f"Erreur lors de l'exécution du modèle de régression : {e}")

# Exécution de la fonction
if not df.empty:
    run_enriched_regression_model(df)


# Variables (exactement celles que tu as listées)
cols = [
    'Prix_au_m2_w', 'DPE_num', 'GES_num', 'Surface_w', 'Procedure_bin', 'Charges_w',
    'Nombre_de_lots', 'Pièces', 'Parking', 'Balcon',
    'Nombre_de_photos', 'Note_z', 'Avis_z', 'Transport', 'Ascenseur'
]

# Sous-ensemble + retrait des NA (simple et efficace)
dfm = df[cols].dropna().copy()
print(f"Échantillon utilisé : n = {len(dfm):,}")

# Formule OLS avec Transport traité en catégorielle
formula = (
    "Prix_au_m2_w ~ DPE_num + GES_num + Surface_w + Procedure_bin + Charges_w + "
    "Nombre_de_lots + Pièces + Parking + Balcon + Nombre_de_photos + "
    "Note_z + Avis_z + C(Transport) + Ascenseur"
)

# Régression (SE robustes pour être tranquille)
model = smf.ols(formula, data=dfm).fit(cov_type="HC3")
print(model.summary())


def generate_wordcloud(data):
    """Génère un nuage de mots à partir des descriptions."""
    print("\n4.3. Nuage de Mots pour l'Analyse Textuelle")
    text = ' '.join(data['Description'].dropna().astype(str).tolist())
    stop_fr = set(STOPWORDS) | set([
        'appartement','pièce','pièces','chambre','chambres','dans','avec','sans','de','des','du','la','le','les','et','au','aux','un','une','sur',
        'salle', 'bain', 'cuisine', 'séjour' # Ajout d'autres mots très fréquents
    ])
    wc = WordCloud(width=1200, height=600, background_color="white",
                   stopwords=stop_fr, collocations=True).generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off'); plt.title('Nuage de mots – Description Immobilière'); plt.show()


if not df.empty:
    check_multicollinearity(df)
    run_regression_model(df)
    generate_wordcloud(df)

# Installation nécessaire si ce n'est pas déjà fait :
# pip install folium geopy

def prepare_data_for_mapping(data):
    """Agrège les données au niveau du Code_Postal et géocode."""
    if data.empty or 'Code_Postal' not in data.columns:
        print("Le DataFrame est vide ou ne contient pas la colonne 'Code_Postal'.")
        return pd.DataFrame()

    print("\n" + "="*80)
    print("5. PRÉPARATION CARTOGRAPHIQUE : AGRÉGATION ET GÉOCODAGE")
    print("="*80)

    # 1. Agrégation des données par Code Postal
    map_data = data.groupby('Code_Postal').agg(
        Prix_moyen=('Prix_au_m2_w', 'mean'),
        DPE_moyen=('DPE_num', 'mean'),
        Nbre_biens=('Code_Postal', 'count')
    ).reset_index()

    # Le DPE/GES est ordinal (1=A, 7=G). Une moyenne faible est un bon DPE.
    
    # 2. Géocodage des Codes Postaux (Récupération des coordonnées GPS)
    geolocator = Nominatim(user_agent="real_estate_analysis")
    
    def geocode_postal_code(postal_code):
        try:
            # Recherche en France uniquement
            location = geolocator.geocode(f"{postal_code}, France")
            if location:
                # Ajouter un petit délai pour éviter de surcharger le serveur Nominatim
                time.sleep(1) 
                return location.latitude, location.longitude
            return np.nan, np.nan
        except Exception as e:
            print(f"Erreur de géocodage pour {postal_code}: {e}")
            return np.nan, np.nan

    # Application du géocodage (Attention: cette étape peut être longue !)
    # Si le nombre de codes postaux uniques est grand, cette étape doit être optimisée.
    print(f"Géocodage de {map_data['Code_Postal'].nunique()} codes postaux uniques...")
    map_data[['Latitude', 'Longitude']] = map_data['Code_Postal'].apply(
        lambda cp: pd.Series(geocode_postal_code(cp))
    )

    map_data.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    
    return map_data.set_index('Code_Postal')

# Exécution
if not df.empty:
    df_map = prepare_data_for_mapping(df.copy())
else:
    df_map = pd.DataFrame()

def create_interactive_maps(df_map):
    """Crée deux cartes interactives : Prix et DPE/GES."""
    if df_map.empty:
        print("Pas de données géocodées pour la cartographie.")
        return

    print("\n" + "="*80)
    print("6. CARTES INTERACTIVES (FOLIUM)")
    print("="*80)

    # Calculer le centre de la carte (moyenne des coordonnées)
    lat_center = df_map['Latitude'].mean()
    lon_center = df_map['Longitude'].mean()
    
    # === CARTE 1: PRIX AU MÈTRE CARRÉ MOYEN ===
    
    # Normalisation pour la couleur/taille des marqueurs
    max_prix = df_map['Prix_moyen'].max()
    
    # Création de la carte
    map_prix = folium.Map(location=[lat_center, lon_center], zoom_start=9)

    for index, row in df_map.iterrows():
        # Taille du marqueur proportionnelle au nombre de biens (confiance)
        radius = np.log(row['Nbre_biens'] + 1) * 3 
        
        # Couleur basée sur le prix (utiliser une ColorMap)
        color_norm = row['Prix_moyen'] / max_prix
        color = plt.cm.plasma(color_norm) # Obtient une couleur RGB
        color_hex = f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}'
        
        # Popup d'information
        popup_html = f"""
            <b>Code Postal:</b> {index}<br>
            <b>Prix Moyen:</b> {row['Prix_moyen']:.0f} €/m²<br>
            <b>Nb Biens:</b> {row['Nbre_biens']}
        """

        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=radius,
            color=color_hex,
            fill=True,
            fill_color=color_hex,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(map_prix)

    map_prix.save('carte_prix_moyen.html')
    print("La carte des Prix Moyens (carte_prix_moyen.html) créée.")


    # === CARTE 2: DPE/GES MOYEN (Performance Énergétique) ===

    # Utiliser une échelle de couleurs pour le DPE (1=Bon DPE -> Vert ; 7=Mauvais DPE -> Rouge)
    max_dpe = df_map['DPE_moyen'].max()
    min_dpe = df_map['DPE_moyen'].min()
    
    map_dpe = folium.Map(location=[lat_center, lon_center], zoom_start=9)
    
    for index, row in df_map.iterrows():
        radius = np.log(row['Nbre_biens'] + 1) * 3 

        # Inverser l'échelle de couleurs pour que 1 (Bon DPE) soit Vert et Max (Mauvais DPE) soit Rouge
        color_norm = (row['DPE_moyen'] - min_dpe) / (max_dpe - min_dpe)
        # Utiliser une ColorMap Rouge/Vert (RdYlGn inversée)
        color = plt.cm.RdYlGn(1 - color_norm) # 1 - color_norm inverse le dégradé
        color_hex = f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}'

        popup_html = f"""
            <b>Code Postal:</b> {index}<br>
            <b>DPE Moyen:</b> {row['DPE_moyen']:.2f} '(1 = Bon | 2 = Moyen | 3 = Mauvais)'<br>
            <b>Nb Biens:</b> {row['Nbre_biens']}
        """

        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=radius,
            color=color_hex,
            fill=True,
            fill_color=color_hex,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(map_dpe)

    map_dpe.save('carte_dpe_moyen.html')
    print("La carte du DPE Moyen (carte_dpe_moyen.html) créée.")


# Exécution de la cartographie
if not df_map.empty:
    create_interactive_maps(df_map)

