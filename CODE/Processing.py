# Importer la bibliothèque pandas
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize 

# Chemin d'accès au fichier CSV 
file_path = 'C:/Users/Pc/OneDrive/Bureau/Dossier/IMSD/PROJET_PYTHON/DONNEES/Donnees_brutes_annonces_immobilieres.csv'  

# Importer le fichier CSV dans un DataFrame pandas
df = pd.read_csv(file_path, delimiter=';')
df.head()

# Afficher les informations générales sur le DataFrame
df.info()

#==============================================================
# Nettoyage et transformation des données
#==============================================================

# Renommer la colonne 'Ville demandée (slug)' en 'Ville'
df = df.rename(columns={'Ville demandée (slug)': 'Ville'})
# Supprimer la colonne 'Prix' 
df = df.drop(columns=['Prix'])
# Exemple de colonne 'Ville' contenant des valeurs comme 'paris-75'
df['Ville'] = df['Ville'].str.split('-').str[0]
# Extraire le nom de l'appartement sans la ville et le code postal 
df['Nom de l\'appartement'] = df['Titre'].str.replace(r'\s?\(.*\)', '', regex=True)

# Extraire le code postal à partir de la parenthèse (extrait les chiffres dans les parenthèses)
df['Code Postal'] = df['Titre'].str.extract(r'\((\d{5})\)')

# Convertir DPE en catégories selon les définitions données
df['DPE_cat'] = df['DPE'].apply(lambda x: 'Tres bonne performance energetique' if x == 'A' else 
                                            ('Bonne performance energetique' if x == 'B' else 
                                             ('Performance energetique correcte' if x == 'C' else 
                                              ('Performance energetique moyenne' if x == 'D' else 
                                               ('Faible performance energetique' if x == 'E' else 
                                                ('Tres faible performance energetique' if x == 'F or Logement à consommation énergétique excessive' else
                                                 'Pas de DPE'))))))

# Convertir GES en catégories selon les définitions données
df['GES_cat'] = df['GES'].apply(lambda x: 'Tres faible emission de gaz a effet de serre' if x == 'A' else 
                                             ('Faible emission de gaz a effet de serre' if x == 'B' else 
                                              ('Emission moyenne de gaz a effet de serre' if x == 'C' else 
                                               ('Emission elevee de gaz a effet de serre' if x == 'D' else 
                                                ('Tres elevee emission de gaz a effet de serre' if x == 'E' else 
                                                 ('Emission extremement elevee de gaz a effet de serre' if x == 'F' else
                                                  'Emission extremement elevee de gaz a effet de serre' if x == 'G' else
                                                  'Pas de GES'))))))

# Extraire l'année de construction si mentionnée dans la description
df['Année de construction'] = df['Description'].str.extract(r'(\d{4})')

# Extraire des infos depuis la description
df['Parking'] = df['Description'].str.contains('parking', case=False, na=False).astype(int)
df['Balcon'] = df['Description'].str.contains('balcon', case=False, na=False).astype(int)

# Prix de vente : ne garder que les chiffres
df['Prix de vente'] = pd.to_numeric(
    df['Prix de vente'].astype(str).str.replace(r'[^\d]', '', regex=True),
    errors='coerce'
)
# Nettoyer la colonne 'Surface' : retirer les unités, les termes 'de terrain', 'de surface brute', etc.
df['Surface'] = df['Surface'].replace({' m²': '', ' ': '', ',': '', 'de terrain': '', 'de surface brute': ''}, regex=True)
# Convertir les valeurs restantes en float
df['Surface'] = pd.to_numeric(df['Surface'], errors='coerce')
# Nettoyage de la colonne 'Prix au m²' : retirer les espaces, les symboles '€' et les unités '/ m²'
df['Prix au m²'] = df['Prix au m²'].replace({r'\u202f': '', r'\xa0': '', '€': '', '/ m²': '', ' ': ''}, regex=True)
# Convertir la colonne en float
df['Prix au m²'] = pd.to_numeric(df['Prix au m²'], errors='coerce')
# Nettoyage de la colonne 'Pièces'
df['Pièces'] = df['Pièces'].apply(lambda x: x.replace('Pas de pièces', '0') if isinstance(x, str) else x)  # Remplacer 'Pas de pièces' par '0'
df['Pièces'] = df['Pièces'].str.extract('(\d+)', expand=False)  # Extraire les chiffres
df['Pièces'] = pd.to_numeric(df['Pièces'], errors='coerce')  # Convertir en numérique
# Nettoyage de la colonne 'Nombre de lots'
df['Nombre de lots'] = df['Nombre de lots'].apply(lambda x: None if isinstance(x, str) and x == 'Non renseigné' else x)  # Remplacer 'Non renseigné' par NaN
df['Nombre de lots'] = df['Nombre de lots'].str.replace(r'\s+', '', regex=True)  # Enlever les espaces inutiles
# Conversion en numérique
df['Nombre de lots'] = pd.to_numeric(df['Nombre de lots'], errors='coerce')  # Convertir en numérique, NaN pour les valeurs non convertibles
# Nettoyage de la colonne 'Charges prévisionnelles'
df['Charges prévisionnelles'] = df['Charges prévisionnelles'].apply(lambda x: None if isinstance(x, str) and (x == 'Non renseigné' or x.lower() == 'non communiqué') else x)  # Remplacer 'Non renseigné' et 'non communiqué' par NaN
# Suppression de l'unité (€ / an) et conversion en nombres
df['Charges prévisionnelles'] = df['Charges prévisionnelles'].str.replace(r'[^\d]', '', regex=True)  # Enlever tout sauf les chiffres
df['Charges prévisionnelles'] = pd.to_numeric(df['Charges prévisionnelles'], errors='coerce')  # Conversion en numérique, NaN pour les valeurs non convertibles
# Exemple de la dataframe
df['Procédure en cours'] = df['Procédure en cours'].replace({'Non renseigné': None, 'oui': 1, 'aucune': 0}).astype(float)
df['Transport'] = df['Description'].str.contains('métro|station|RER|gare', case=False, na=False).astype(int)
df['Ascenseur'] = df['Description'].str.contains('ascenseur', case=False, na=False).astype(int)

print(df[['Prix de vente', 'Prix au m²','Pièces']].head())

# Suppression des colonnes "Titre", "Avis" et "Note"
df = df.drop(columns=['Titre', 'Nombre de photos','Avis', 'Note'])
# Renommage des colonnes
df = df.rename(columns={
    'Note_num': 'Note',
    'Avis_num': 'Avis',
    'Photos_num': 'Nombre de photos'
})
# Mise à jour de l'ordre des colonnes en fonction des noms exacts
nouvel_ordre = ['Ville', 'Code Postal', 'Année de construction', 'Nom de l\'appartement', 'Nombre de lots', 
                'Surface', 'Pièces', 'Parking', 'Balcon', 'Prix de vente', 'Prix au m²', 
                'Charges prévisionnelles', 'Procédure en cours', 'Nombre de photos', 'Note', 'Avis','Transport', 'Ascenseur', 
                'DPE_cat', 'GES_cat', 'DPE', 'GES', 'Agent', 'Description', 'Lien', 'Image']

# Réorganiser les colonnes en fonction de nouvel_ordre
df = df[nouvel_ordre]
# Renommer les colonnes : remplacement des espaces par des underscores
df.columns = [col.replace(" ", "_") for col in df.columns]
# Dictionnaire des unités pour les colonnes spécifiques
unites = {
    'Prix_de_vente': 'Prix de vente (€)',
    'Surface': 'Surface (m²)'
}

# Appliquer les unités devant les noms des colonnes spécifiées
df = df.rename(columns={col: unites.get(col, col) for col in df.columns})

# Convertir les valeurs en années (en format texte ou numérique) directement
df['Année_de_construction'] = pd.to_numeric(df['Année_de_construction'], errors='coerce')

# Remplacer les années en dehors de la plage 1700 à 2025 par NaN
df['Année_de_construction'] = df['Année_de_construction'].apply(lambda x: int(x) if 1700 <= x <= 2025 else np.nan)
# Fonction pour remplacer les NaN par 0
def remplacer_nan_par_a(df, colonne):
    df[colonne] = df[colonne].fillna('0')
    return df

# Exemple d'utilisation sur la colonne 'Année_de_construction'
df = remplacer_nan_par_a(df, 'Année_de_construction')
# ------------------------------------------------------------
# Helper : winsorisation qui conserve les NaN
# Coupe les 1% plus bas et 1% plus hauts pour limiter l'effet des outliers
# ------------------------------------------------------------
def winsor_keep_na(s, lo=0.01, hi=0.99):
    ql = s.dropna().quantile(lo); qh = s.dropna().quantile(hi)
    return s.clip(lower=ql, upper=qh)

# ------------------------------------------------------------
# CIBLE & TAILLES
# ------------------------------------------------------------

df["Prix_au_m2_w"] = winsor_keep_na(df["Prix_au_m²"])                # Prix/m² winsorisé (1–99%) → prix “robuste” sans extrêmes
df["Surface_w"]    = winsor_keep_na(df["Surface (m²)"])               # Surface winsorisée (1–99%) → réduit l’effet des très grands biens
df["log_prix_m2"]  = np.log(df["Prix_au_m2_w"])                       # Log du prix/m² winsorisé → rend l’analyse plus stable et interprétable en %

# ------------------------------------------------------------
# CHARGES & COPROPRIÉTÉ
# ------------------------------------------------------------

df["Charges_w"]   = df["Charges_prévisionnelles"].clip(lower=0)       # Charges tronquées à 0 (pas de valeurs négatives)
df["log_charges"] = np.log1p(df["Charges_w"])                         # Log(1 + charges) → linéarise et gère les zéros

df["Lots_w"]      = winsor_keep_na(df["Nombre_de_lots"])              # Nombre de lots winsorisé → proxy de taille/structure de copro
df["Procedure_bin"] = (df["Procédure_en_cours"].fillna(0) > 0) \
                        .astype(int)                                   # Procédure en cours (1 oui / 0 non) → risque juridique/financier

# ------------------------------------------------------------
# MARKETING / QUALITÉ D’ANNONCE
# ------------------------------------------------------------

df["Photos_w"] = winsor_keep_na(df["Nombre_de_photos"])               # Nb de photos winsorisé → intensité marketing / mise en valeur

# Standardisation (z-score) : effet lisible “par +1 écart-type”
for col in ["Note","Avis"]:
    if col in df.columns:
        m, s = df[col].mean(skipna=True), df[col].std(skipna=True)
        df[f"{col}_z"] = (df[col]-m)/s if (s and not np.isnan(s) and s>0) else df[col]-m
        # Note_z / Avis_z : variables standardisées (0 ≈ moyenne ; +1 = 1 écart-type au-dessus)

df["desc_len"] = df["Description"].fillna("").str.len()                # Longueur de la description (caractères) → soin éditorial / info

# ------------------------------------------------------------
# MOTS-CLÉS DANS LE TEXTE (SIGNAL + / -)
# ------------------------------------------------------------

KW_POS = ["refait","rénové","neuf","lumineux","calme","balcon","terrasse","vue","ascenseur"]   # Liste de mots “positifs”
KW_NEG = ["travaux","à rénover","sans ascenseur","bruyant","rez-de-chaussée"]                  # Liste de mots “négatifs”

def has_kw(s, kws):
    s = str(s).lower()
    return int(any(k in s for k in kws))

df["kw_pos"] = df["Description"].apply(lambda x: has_kw(x, KW_POS))    # 1 si la description contient au moins un mot-clé positif
df["kw_neg"] = df["Description"].apply(lambda x: has_kw(x, KW_NEG))    # 1 si la description contient au moins un mot-clé négatif

# Fonction pour classer les villes
def classer_ville(ville):
    ville = ville.lower()  # Mettre tout en minuscule pour éviter la casse
    if 'paris' in ville:  # Paris
        return 'Paris'
    elif any(sub in ville for sub in ['boulogne', 'nanterre', 'saint-denis', 'villejuif', 'versailles']):  # Banlieue proche
        return 'Banlieue proche'
    elif any(sub in ville for sub in ['creteil', 'palaiseau', 'melun']):  # Banlieue éloignée
        return 'Banlieue éloignée'

# Appliquer la fonction de classement à la colonne 'Ville' de votre DataFrame
df['ville_classer'] = df['Ville'].apply(classer_ville)

# Vérifier les résultats
print(df[['Ville', 'ville_classer']].head(10))

# Compter le nombre d'occurrences de chaque catégorie
ville_counts = df['ville_classer'].value_counts()
print(ville_counts)

# Regroupement des catégories DPE en trois groupes
dpe_mapping = {
    'Tres mauvaise performance energetique': 3,  # Mauvaise performance
    'Faible performance energetique': 3,
    'Performance energetique moyenne': 2,  # Performance moyenne
    'Performance energetique correcte': 1,
    'Bonne performance energetique': 1,  # Bonne performance
    'Très bonne performance energetique': 1
}

# Regroupement des catégories GES en trois groupes
ges_mapping = {
    'Tres faible emission de gaz à effet de serre': 1,  # Faible émission
    'Faible emission de gaz a effet de serre': 1,
    'Emission moyenne de gaz a effet de serre': 2,  # Moyenne émission
    'Emission elevee de gaz a effet de serre': 3,
    'Tres elevee emission de gaz a effet de serre': 3,  # Haute émission
    'Emission extremement elevee de gaz a effet de serre': 3,
    'Maximum d’emission de gaz a effet de serre': 3
}

# Appliquer le mappage pour créer les nouvelles colonnes regroupées
df['DPE_num'] = df['DPE_cat'].map(dpe_mapping)
df['GES_num'] = df['GES_cat'].map(ges_mapping)

# Appliquer la winsorisation sur DPE_num et GES_num (limiter les valeurs aux bornes 1% et 99%)
df['DPE_num'] = winsorize(df['DPE_num'], limits=[0.01, 0.01])  # Limite les valeurs aux percentiles 1% et 99%
df['GES_num'] = winsorize(df['GES_num'], limits=[0.01, 0.01])

# Vérification après winsorisation
print(df[['DPE_num', 'GES_num']].describe())

# Vérifier les valeurs uniques dans les nouvelles colonnes
print("Valeurs uniques dans DPE_num:", df['DPE_num'].unique())
print("Valeurs uniques dans GES_num:", df['GES_num'].unique())
# Vérifier les valeurs manquantes dans les nouvelles colonnes
print("Valeurs manquantes dans DPE_num:", df['DPE_num'].isnull().sum())
print("Valeurs manquantes dans GES_num:", df['GES_num'].isnull().sum())

# Vérifier les valeurs manquantes dans la colonne 'ville_classer'
print("Valeurs manquantes dans ville_classer:", df['ville_classer'].isnull().sum()) 
# Vérifier les valeurs uniques dans la colonne 'ville_classer'
print("Valeurs uniques dans ville_classer:", df['ville_classer'].unique())  
# Vérifier les types de données après les modifications
print(df.dtypes) 
df.info()
# Exporter les données vers un fichier CSV
df.to_csv('Processing_data.csv', index=False, encoding='utf-8-sig', sep=';')
print("Les données ont été exportées vers 'Processing_data.csv' avec un encodage adapté.")