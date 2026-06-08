#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
APP STREAMLIT • PRIME VERTE (DPE/GES → Prix au m²)
--------------------------------------------------
Lancer localement :
  streamlit run app.py
"""

# ============================
# 0. IMPORTS & CONFIG GLOBALE
# ============================
import io
import os
import time
from textwrap import dedent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy.stats import f_oneway, kruskal, ttest_ind, levene
from statsmodels.stats.outliers_influence import variance_inflation_factor
from wordcloud import WordCloud, STOPWORDS

import streamlit as st
from streamlit_folium import st_folium
import folium
from geopy.geocoders import Nominatim

# Thème général des plots
sns.set_style("whitegrid")
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 120,
})

# =====================================
# 0.1 PARAMÈTRES GLOBAUX
# =====================================
DEFAULT_FILE_PATH = 'C:/Users/Pc/OneDrive/Bureau/Dossier/IMSD/PROJET_PYTHON/DONNEES/Processing_data.csv'
ALPHA = 0.05  # Seuil de signification

# ===================
# 0.2 PAGE CONFIG UI
# ===================
st.set_page_config (
    page_title="Effet DPE/GES sur le prix au m² : aperçu, tests et modèles.",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/',
        #'Report a bug': "ekouraogo73@gmail.com",
        'About': "# Projet Streamlit • Analyse de l’impact de la performance énergétique (DPE/GES) sur le prix au m²."
    }
)

# ==========================
# STYLES GLOBAUX
# ==========================

st.markdown("""
<style>
/* Titres internes (section principale dans le body) */
.section-title{
  font-size:1.15rem; font-weight:700;
  padding:.55em .75em; margin:1.2em 0 .6em;
  border-left:6px solid #3ac47d;
  background:rgba(58,196,125,0.08);
  color:#3ac47d; border-radius:8px;
}
.section-title.missing{
  border-left:6px solid #d9534f;
  background:rgba(217,83,79,0.08);
  color:#d9534f;
}

/* Switchs verts */
[data-testid="stSwitch"] > label[data-baseweb="switch"] {
    background-color: #246c48 !important;
}
[data-testid="stSwitch"] > label[data-baseweb="switch"][aria-checked="true"] {
    background-color: #3ac47d !important;
}
[data-testid="stSwitch"] > label[data-baseweb="switch"]:hover {
    box-shadow: 0 0 0 2px rgba(58,196,125,0.3);
}
[data-testid="stSwitch"] + div p {
    color: #d4f3df !important;
}

/* Bande verte réutilisable pour titres de la sidebar */
.sideband {
  background: linear-gradient(180deg, #0e5b41 0%, #0a4b35 100%);
  border: 1px solid rgba(58,196,125,.25);
  border-radius: 14px;
  padding: 12px 14px;
  margin: 10px 0 12px;
  color: #eaf7f0;
  box-shadow: 0 6px 14px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.06);
  display: flex; align-items: center; gap: 10px;
}
.sideband .title {
  font-weight: 800; letter-spacing: .2px; font-size: 1.02rem;
}

/* Header principal de la sidebar */
.sidebar-header {
  background: linear-gradient(90deg, #054d35 0%, #0b6647 100%);
  border-radius: 8px;
  padding: 10px 10px;
  text-align: center;
  box-shadow: 0 2px 4px rgba(0,0,0,0.4);
  margin: 25px 0 14px 0;  
}
.sidebar-header h3 {
  color: #f8f9fa;
  font-weight: 800;
  font-size: 1.3rem;
  margin: 0;
}
.sidebar-sub {
  color: #cdeedc;
  font-size: 0.9rem;
  margin-top: 4px;
  line-height: 1.3;
}
</style>
""", unsafe_allow_html=True)


# ==========================
# FONCTION UTILITAIRE
# ==========================
def side_band(title_with_emoji: str):
    """Crée une bande verte pour chaque sous-section de la sidebar."""
    st.markdown(f"<div class='sideband'><span class='title'>{title_with_emoji}</span></div>", unsafe_allow_html=True)


# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    # ---- En-tête verte ----
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-title">🌿 LA PRIME VERTE</div>
        <div class="sidebar-sub">Analyse de l’impact DPE/GES sur le prix au m²</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

    # ---- 📦 Chargement des données ----
    side_band("📦 Chargement des données")
    up = st.file_uploader("CSV (délimiteur ';')", type=["csv"])
    use_default = st.toggle("Utiliser le chemin par défaut", value=not bool(up))

    # ---- 🎛️ Options d’affichage ----
    side_band("🎛️ Options d’affichage")
    show_annotations = st.toggle("Annoter les graphiques (n, médianes, etc.)", True)
    show_kde = st.toggle("Afficher la densité (KDE) sur histogrammes", True)

    # ---- 🧭 Sections ----
    side_band("🧭 Sections")
    with st.expander("Afficher/Masquer", expanded=True):
        sec_overview         = st.checkbox("Aperçu & distributions", True)
        sec_desc             = st.checkbox("Statistiques descriptives", True)
        sec_interact_1       = st.checkbox("Interactions DPE/GES × Caractéristiques", True)
        sec_comp             = st.checkbox("Comparaisons DPE/GES (box+violin)", True)
        sec_interact_2       = st.checkbox("Interactions Localité & Transport", True)
        sec_text             = st.checkbox("Qualité textuelle (kw_pos/kw_neg)", True)
        sec_transport_price  = st.checkbox("Transport → prix moyen au m²", True)
        sec_marketing        = st.checkbox("Interactions marketing & annexes", True)
        sec_tests            = st.checkbox("Corrélations & Tests H1-H3", True)
        sec_models           = st.checkbox("Régressions & VIF", True)
        sec_maps             = st.checkbox("Cartes (prix & DPE) par code postal", True)
        sec_wordcloud        = st.checkbox("Nuage de mots (Descriptions)", True)



# =========================
# 0.3 CHARGEMENT DES DONNÉES
# =========================
@st.cache_data(show_spinner=False)
def load_data(file_like, fallback_path: str) -> pd.DataFrame:
    try:
        if file_like is not None:
            return pd.read_csv(file_like, delimiter=';')
        if fallback_path and os.path.exists(fallback_path):
            return pd.read_csv(fallback_path, delimiter=';')
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur de lecture CSV : {e}")
        return pd.DataFrame()


file_path = DEFAULT_FILE_PATH if use_default else None
df = load_data(up, file_path)

# ---- En-tête / Hero + cartes ----
st.markdown("""
<style>
.hero      { font-size: 36px; font-weight: 800; margin: 0 0 4px 0; }
.subhero   { font-size: 16px; color: #a0a6ad; margin: 0 0 18px 0; }
.card      { padding: 16px 18px; border-radius: 14px;
             border: 1px solid rgba(255,255,255,.08);
             background: rgba(255,255,255,.02); margin-bottom: 12px; }
.card h4   { margin: 0 0 8px 0; }
.badge     { display:inline-block; padding:2px 8px; border-radius:999px;
             font-size:12px; font-weight:600; margin-right:8px;
             background: rgba(16,185,129,.15); color:#10b981; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.hero {
    font-family: 'Segoe UI', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    text-align: center;
    color: #eafbea; /* texte vert clair */
    background: linear-gradient(90deg, #0b3d2e 0%, #145a32 100%); /* dégradé vert profond */
    padding: 1.2em 0.5em;
    border-radius: 10px;
    margin-bottom: 1.5em;
    letter-spacing: 0.5px;
    box-shadow: 0 3px 8px rgba(0,0,0,0.4);
}
.hero span {
    display: block;
    font-weight: 400;
    font-size: 1rem;
    color: #cde7d8; /* vert clair/gris pour le sous-titre */
    margin-top: 0.4em;
}
</style>

<div class="hero">
    🌿 PROJET DE SCRAPING – ANALYSE DES DONNÉES IMMOBILIÈRES
    <span>Analyse de l’impact de la performance énergétique sur le prix au mètre carré en Ile de France</span>
</div>
""", unsafe_allow_html=True)


c1, c2 = st.columns([1.2, 1])

with c1:
    st.markdown("<div class='card'><h4>🎯 Objectif</h4>"
                "<p>Mesurer la <b>prime verte</b> : l’impact de la performance énergétique "
                "(<i>DPE</i>, <i>GES</i>) sur le <b>prix au m²</b>, en tenant compte de la "
                "surface, la localisation et les caractéristiques du bien.</p></div>",
                unsafe_allow_html=True)

with c2:
    st.markdown("<div class='card'><h4>🧪 Hypothèses</h4>"
                "<div class='badge'>H1</div> Les logements ayant un bon DPE et un bon GES se vendent à un prix plus élevé.<br>"
                "<div class='badge'>H2</div> La prime verte est <i>variable</i> selon la localité<br>"
                "<div class='badge'>H3</div> Le prix dépend d’un <i>ensemble</i> de facteurs</div>",
                unsafe_allow_html=True)


if df.empty:
    st.warning("Aucun fichier chargé. Importez votre CSV ou cochez l'option chemin par défaut.")
    st.stop()

# Normalisation douce des noms de colonnes
orig_cols = df.columns.copy()
df.columns = [c.replace(" ", "_").replace("-", "_") for c in df.columns]

# Alias pour compatibilité
if 'Prix_au_m²' in df.columns and 'Prix_au_m2' not in df.columns:
    df['Prix_au_m2'] = df['Prix_au_m²']

if 'Prix_au_m2_w' not in df.columns and 'Prix_au_m2' in df.columns:
    df['Prix_au_m2_w'] = df['Prix_au_m2']  # fallback

# =====================
# 0.4 OUTILS UTILITAIRES
# =====================
def have(*cols) -> bool:
    return all(c in df.columns for c in cols)


def section_header(title: str, ok: bool=True):
    # On récupère proprement les colonnes requises (évite les KeyError)
    needed = required.get(title, [])
    missing = [c for c in needed if c not in df.columns]

    css_class = "section-title" if ok else "section-title missing"
    icon = "✅" if ok else "⚠️"

    # Titre coloré (vert si ok, rouge sinon)
    st.markdown(f'<div class="{css_class}">{icon} {title}</div>', unsafe_allow_html=True)

    # Message d’info s’il manque des colonnes
    if not ok and missing:
        st.info("Colonnes manquantes : " + ", ".join(missing))


# Titre de graphe sur une seule ligne (nowrap)
def title_line(text: str):
    st.markdown(f"<div style='font-weight:600; margin:4px 0 6px; white-space:nowrap'>{text}</div>", unsafe_allow_html=True)

# Dictionnaire des colonnes requises par section
# Dictionnaire des colonnes requises par section (aligné avec les titres actuels)
# Dictionnaire des colonnes requises par section (synchronisé avec tes libellés de sidebar/sections)
required = {
    "1) Aperçu et distributions": ['Prix_au_m2_w', 'DPE_num', 'GES_num', 'Surface_w', 'ville_classer'],
    "2) Statistiques descriptives": ['Prix_au_m2_w'],
    "3) Interactions DPE/GES × Caractéristiques": ['Prix_au_m2_w', 'DPE_num', 'GES_num', 'Surface_w', 'Ascenseur', 'Balcon', 'Année_de_construction'],
    "4) Comparaisons DPE/GES (box+violin)": ['Prix_au_m2_w', 'DPE_num', 'GES_num'],
    "5) Interactions Localité & Transport": ['Prix_au_m2_w', 'Surface_w', 'ville_classer', 'Parking', 'Transport'],
    "6) Qualité textuelle (kw_pos/kw_neg)": ['Prix_au_m2_w', 'DPE_num', 'kw_pos', 'kw_neg', 'ville_classer'],
    "7) Transport → prix moyen au m²": ['Transport', 'Prix_au_m2_w'],
    "8) Corrélations & Tests H1-H3": ['Prix_au_m2_w', 'DPE_num', 'GES_num', 'Surface_w'],
    "9) Régressions & VIF": ['Prix_au_m2_w', 'DPE_num', 'GES_num', 'Surface_w', 'ville_classer', 'kw_pos', 'kw_neg'],
    "10) Cartes (prix & DPE) par code postal": ['Code_Postal', 'Prix_au_m2_w', 'DPE_num'],
    "11) Nuage de mots (Descriptions)": ['Description'],
    "12) Interactions marketing & annexes": []  # pas strictement nécessaire mais évite les surprises
}


# ===============================
# 1) APERÇU ET DISTRIBUTIONS (EDA)
# ===============================
if sec_overview:
    ok = have(*required["1) Aperçu et distributions"])
    section_header("1) Aperçu et distributions", ok)
    with st.container(border=True):
        st.subheader("Aperçu rapide")
        st.dataframe(df.head(10), width='stretch')
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("n (lignes)", f"{len(df):,}")
        with c2:
            st.metric("Colonnes", f"{df.shape[1]}")
        with c3:
            st.metric("Localités uniques", f"{df['ville_classer'].nunique() if 'ville_classer' in df.columns else 0}")

        if have('Prix_au_m2_w'):
            title_line("Distribution du prix au m²")
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.histplot(df['Prix_au_m2_w'], bins=30, kde=show_kde, ax=ax)
            ax.set(xlabel='Prix au m² (€)', ylabel='Fréquence', title='Distribution du prix au m²')
            if show_annotations:
                for p in ax.patches:
                    h = p.get_height()
                    if h > 0:
                        ax.text(p.get_x()+p.get_width()/2, h, f"{int(h)}", ha='center', va='bottom', fontsize=8)
            st.pyplot(fig, use_container_width=True, clear_figure=True)

        if have('Prix_au_m2_w') and (have('Prix_au_m2') or have('Prix_au_m²')):
    # — utilitaire : boxplot horizontal + annotations 5 chiffres —
            def draw_box_with_stats(ax, series, title):
                s = pd.to_numeric(series.dropna(), errors="coerce").dropna()
                if s.empty:
                    ax.set_title(title + " (pas de données)")
                    ax.axis("off")
                    return

                ax.boxplot(s, vert=False)
                ax.set_title(title)
                ax.set_xlabel("Prix au m² (€)")

                # chiffres clés
                q1 = s.quantile(0.25)
                med = s.median()
                q3 = s.quantile(0.75)
                smin = s.min()
                smax = s.max()

                # y = 1 sur un boxplot 1-ligne (léger décalage vertical pour lisibilité)
                y_top = 1.05
                y_low = 0.95

                ax.text(med,  y_top, f"Médiane: {med:.0f}",  ha="center", va="bottom", fontsize=9, color="black")
                ax.text(q1,   y_low, f"Q1: {q1:.0f}",       ha="center", va="top",    fontsize=8, color="gray")
                ax.text(q3,   y_low, f"Q3: {q3:.0f}",       ha="center", va="top",    fontsize=8, color="gray")
                ax.text(smin, y_top, f"Min: {smin:.0f}",    ha="left",   va="bottom", fontsize=8, color="gray")
                ax.text(smax, y_top, f"Max: {smax:.0f}",    ha="right",  va="bottom", fontsize=8, color="gray")

                # petites marges pour éviter les labels coupés
                xmin, xmax = ax.get_xlim()
                span = xmax - xmin
                ax.set_xlim(xmin - 0.02*span, xmax + 0.02*span)

            # — affichage côte à côte
            colA, colB = st.columns(2)
            with colA:
                title_line("Avant winsorisation (boxplot)")
                fig, ax = plt.subplots(figsize=(11, 4))
                # Supporte 'Prix_au_m²' (accent) ou 'Prix_au_m2'
                s_raw = df['Prix_au_m²'] if 'Prix_au_m²' in df.columns else df['Prix_au_m2']
                draw_box_with_stats(ax, s_raw, "Prix au m² (Avant winsorisation)")
                st.pyplot(fig, use_container_width=True, clear_figure=True)

            with colB:
                title_line("Après winsorisation (boxplot)")
                fig, ax = plt.subplots(figsize=(11, 4))
                draw_box_with_stats(ax, df['Prix_au_m2_w'], "Prix au m² (Après winsorisation)")
                st.pyplot(fig, use_container_width=True, clear_figure=True)


        # --- DPE & GES : même logique, couleurs coolwarm ---
        if have('DPE_num', 'GES_num'):
            col1, col2 = st.columns(2)
            with col1:
                title_line("Distribution DPE")
                fig, ax = plt.subplots(figsize=(11, 5))
                ax = sns.countplot(x='DPE_num', data=df, palette='coolwarm')
                ax.set(xlabel='DPE (1=Bon 2=Moyen 3=Mauvais)')
                total = sum(p.get_height() for p in ax.patches)
                for p in ax.patches:
                    h = p.get_height()
                    if h > 0:
                        ax.text(p.get_x() + p.get_width()/2, h,
                                f"{int(h)}\n({h/total*100:.1f}%)",
                                ha='center', va='bottom', fontsize=9)
                st.pyplot(fig, width='stretch', clear_figure=True)

            with col2:
                title_line("Distribution GES")
                fig, ax = plt.subplots(figsize=(11, 5))
                ax = sns.countplot(x='GES_num', data=df, palette='coolwarm')
                ax.set(xlabel='GES (1=Bon 2=Moyen 3=Mauvais)')
                total = sum(p.get_height() for p in ax.patches)
                for p in ax.patches:
                    h = p.get_height()
                    if h > 0:
                        ax.text(p.get_x() + p.get_width()/2, h,
                                f"{int(h)}\n({h/total*100:.1f}%)",
                                ha='center', va='bottom', fontsize=9)
                st.pyplot(fig, width='stretch', clear_figure=True)


# =======================================
# 2) STATISTIQUES DESCRIPTIVES 
# =======================================
if sec_desc:
    ok = 'Prix_au_m2_w' in df.columns
    section_header("2) Statistiques descriptives", ok)
    with st.container(border=True):
        if not ok:
            st.info("Colonnes nécessaires manquantes : Prix_au_m2_w")
        else:
            # ---------- Filtres rapides ----------
            f1, f2, f3 = st.columns(3)
            with f1:
                villes = sorted(df['ville_classer'].dropna().unique()) if 'ville_classer' in df.columns else []
                ville_sel = st.multiselect("Filtre — Localités", villes)
            with f2:
                dpe_vals = sorted(df['DPE_num'].dropna().unique()) if 'DPE_num' in df.columns else []
                dpe_sel = st.multiselect("Filtre — DPE", dpe_vals)
            with f3:
                ges_vals = sorted(df['GES_num'].dropna().unique()) if 'GES_num' in df.columns else []
                ges_sel = st.multiselect("Filtre — GES", ges_vals)

            mask = pd.Series(True, index=df.index)
            if ville_sel:
                mask &= df['ville_classer'].isin(ville_sel)
            if dpe_sel:
                mask &= df['DPE_num'].isin(dpe_sel)
            if ges_sel:
                mask &= df['GES_num'].isin(ges_sel)

            df_view = df[mask].copy()

            # ---------- KPI (cartes) ----------
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Prix moyen € / m²", f"{df_view['Prix_au_m2_w'].mean():,.0f}".replace(",", " "))
            with k2:
                st.metric("Médiane € / m²", f"{df_view['Prix_au_m2_w'].median():,.0f}".replace(",", " "))
            with k3:
                q75 = df_view['Prix_au_m2_w'].quantile(.75)
                q25 = df_view['Prix_au_m2_w'].quantile(.25)
                st.metric("IQR (p75–p25)", f"{(q75 - q25):,.0f}".replace(",", " "))
            with k4:
                st.metric("n (observations)", f"{len(df_view):,}".replace(",", " "))

            # ---------- Tableau global formaté ----------
            num_cols = [c for c in [
                'Prix_au_m2_w','Prix_au_m2','Surface_w','Charges_w','Nombre_de_lots',
                'Pièces','Nombre_de_photos','Note_z','Avis_z','DPE_num','GES_num'
            ] if c in df_view.columns]

            with st.expander("Résumé global (quartiles, min/max, manquants)", expanded=True):
                desc = df_view[num_cols].describe(percentiles=[.05,.25,.5,.75,.95]).T
                desc = desc.rename(columns={
                    'count':'n','mean':'moyenne','std':'écart-type','min':'min',
                    '5%':'p05','25%':'p25','50%':'p50','75%':'p75','95%':'p95','max':'max'
                })
                desc['n_manquants'] = df_view[num_cols].isna().sum().reindex(desc.index)
                desc['%_manquants'] = (desc['n_manquants'] / len(df_view) * 100)

                st.dataframe(
                    desc.round(2),
                    use_container_width=True,
                    hide_index=False,
                    column_config={
                        "n": st.column_config.NumberColumn(format="%.0f"),
                        "moyenne": st.column_config.NumberColumn(format="%.0f"),
                        "écart-type": st.column_config.NumberColumn(format="%.0f"),
                        "min": st.column_config.NumberColumn(format="%.0f"),
                        "p05": st.column_config.NumberColumn(format="%.0f"),
                        "p25": st.column_config.NumberColumn(format="%.0f"),
                        "p50": st.column_config.NumberColumn(format="%.0f"),
                        "p75": st.column_config.NumberColumn(format="%.0f"),
                        "p95": st.column_config.NumberColumn(format="%.0f"),
                        "max": st.column_config.NumberColumn(format="%.0f"),
                        "n_manquants": st.column_config.NumberColumn(format="%.0f"),
                        "%_manquants": st.column_config.ProgressColumn(
                            "Manquants (%)", format="%.0f%%", min_value=0, max_value=100
                        ),
                    },
                )

                csv_desc = desc.round(2).to_csv().encode("utf-8")
                st.download_button("Télécharger — stats globales (CSV)", csv_desc,
                                   "stats_globales.csv", "text/csv")

            # ---------- Par localité (tri médiane) ----------
            if 'ville_classer' in df_view.columns:
                title_line("Prix au m² par localité (médiane, trié)")
                loc_stats = (
                    df_view.groupby('ville_classer')['Prix_au_m2_w']
                    .agg(médiane='median', moyenne='mean', n='count')
                    .sort_values('médiane', ascending=False)
                )
                st.dataframe(
                    loc_stats.round(0),
                    use_container_width=True,
                    column_config={
                        "médiane": st.column_config.NumberColumn(format="%.0f"),
                        "moyenne": st.column_config.NumberColumn(format="%.0f"),
                        "n": st.column_config.NumberColumn(format="%.0f"),
                    }
                )
                csv_loc = loc_stats.round(0).to_csv().encode("utf-8")
                st.download_button("Télécharger — par localité (CSV)", csv_loc,
                                   "prix_par_localite.csv", "text/csv")

            # ---------- Répartition DPE / GES ----------
            col_dpe, col_ges = st.columns(2)

            if 'DPE_num' in df_view.columns:
                with col_dpe:
                    title_line("Répartition des DPE")
                    tab = (df_view['DPE_num'].value_counts().rename('n').to_frame()
                           .assign(pct=lambda t: (t['n'] / len(df_view) * 100)))
                    st.dataframe(
                        tab.sort_index(),
                        use_container_width=True,
                        column_config={
                            "n": st.column_config.NumberColumn(format="%.0f"),
                            "pct": st.column_config.NumberColumn(format="%.1f %%"),
                        }
                    )

            if 'GES_num' in df_view.columns:
                with col_ges:
                    title_line("Répartition des GES")
                    tab = (df_view['GES_num'].value_counts().rename('n').to_frame()
                           .assign(pct=lambda t: (t['n'] / len(df_view) * 100)))
                    st.dataframe(
                        tab.sort_index(),
                        use_container_width=True,
                        column_config={
                            "n": st.column_config.NumberColumn(format="%.0f"),
                            "pct": st.column_config.NumberColumn(format="%.1f %%"),
                        }
                    )


# =====================================================
# 3) INTERACTIONS DPE/GES × CARACTÉRISTIQUES DU BIEN
# =====================================================
if sec_interact_1:
    ok = have(*required["3) Interactions DPE/GES × Caractéristiques"])
    section_header("3) Interactions DPE/GES × Caractéristiques", ok)
    with st.container(border=True):
        if not ok:
            st.stop()

        # 3.1 DPE × Année de construction (boxplot par catégorie ancienneté)
        df['Anciennete_Cat'] = pd.cut(
            df['Année_de_construction'],
            bins=[0, 1975, df['Année_de_construction'].max() if df['Année_de_construction'].notna().any() else 2100],
            labels=['Avant 1975 (Isolation faible)', 'Après 1975 (Isolation réglementée)'],
            right=False
        )

        title_line("Graphique 3.1 — Prix au m² par DPE, selon ancienneté")
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.boxplot(x='DPE_num', y='Prix_au_m2_w', hue='Anciennete_Cat', data=df, ax=ax)
        ax.set(xlabel='DPE (1=Bon 2=Moyen 3=Mauvais)', ylabel='Prix au m² (€)')

        # === Annotations médiane + effectif ===
        group_stats = (
            df.groupby(['DPE_num', 'Anciennete_Cat'], observed=False)['Prix_au_m2_w']
            .agg(med='median', n='count')
            .reset_index()
        )

        # Correction ici : convertir en liste
        x_order = sorted(df['DPE_num'].dropna().unique())
        if hasattr(df['Anciennete_Cat'], 'cat'):
            hue_order = list(df['Anciennete_Cat'].cat.categories)
        else:
            hue_order = sorted(df['Anciennete_Cat'].dropna().unique())

        width = 0.8
        n_hues = len(hue_order)
        ymin, ymax = ax.get_ylim()
        y_offset = (ymax - ymin) * 0.012

        for _, r in group_stats.iterrows():
            if pd.isna(r['DPE_num']) or pd.isna(r['Anciennete_Cat']):
                continue
            i = x_order.index(r['DPE_num'])
            j = hue_order.index(r['Anciennete_Cat'])
            x_pos = i + (j - (n_hues - 1) / 2) * (width / n_hues)
            y_pos = r['med']
            ax.text(
                x_pos, y_pos + y_offset,
                f"{y_pos:.0f}€\n(n={int(r['n'])})",
                ha='center', va='bottom', fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2')
            )

        ax.legend(title='Ancienneté', bbox_to_anchor=(1.02, 1), loc='upper left')
        st.pyplot(fig, width='stretch', clear_figure=True)


        # 3.2 Surface × DPE (regplots + stats β, R², n)
        title_line("Graphique 3.2 — Prix vs Surface par niveau de DPE")
        mask_good = df['DPE_num'].isin([1, 2, 3])
        mask_mid  = df['DPE_num'].isin([4])
        mask_bad  = df['DPE_num'].isin([5, 6, 7])
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.regplot(x='Surface_w', y='Prix_au_m2_w', data=df[mask_good], scatter_kws={'alpha':0.3}, line_kws={'color':'green'}, label='DPE Bon (A–C)', ax=ax)
        sns.regplot(x='Surface_w', y='Prix_au_m2_w', data=df[mask_mid],  scatter_kws={'alpha':0.3}, line_kws={'color':'orange'}, label='DPE Moyen (D)', ax=ax)
        sns.regplot(x='Surface_w', y='Prix_au_m2_w', data=df[mask_bad],  scatter_kws={'alpha':0.3}, line_kws={'color':'red'}, label='DPE Mauvais (E–G)', ax=ax)
        ax.set(xlabel='Surface winsorisée (m²)', ylabel='Prix au m² (€)')

        def fit_stats(x, y):
            x = np.asarray(x); y = np.asarray(y)
            m = np.isfinite(x) & np.isfinite(y)
            x = x[m]; y = y[m]
            n = x.size
            if n < 2: return None
            slope, _ = np.polyfit(x, y, 1)
            r = np.corrcoef(x, y)[0, 1] if n > 1 else np.nan
            r2 = r**2 if np.isfinite(r) else np.nan
            return slope, r2, n

        ypos = 0.96
        for mask, label, color in [
            (mask_good, "DPE Bon (A–C)", "green"),
            (mask_mid,  "DPE Moyen (D)", "orange"),
            (mask_bad,  "DPE Mauvais (E–G)", "red"),
        ]:
            stats = fit_stats(df.loc[mask, 'Surface_w'], df.loc[mask, 'Prix_au_m2_w'])
            if stats:
                slope, r2, n = stats
                ax.text(0.02, ypos, f"{label} : n={n}  β≈{slope*10:.1f} €/m² / +10 m²  R²={r2:.2f}", transform=ax.transAxes, ha='left', va='top', fontsize=9, color=color)
                ypos -= 0.08
        ax.legend()
        st.pyplot(fig, use_container_width=True, clear_figure=True)

        # 3.3 Équipements (Ascenseur, Balcon) × DPE
        for equip in ['Ascenseur', 'Balcon']:
            if equip in df.columns:
                title_line(f"Graphique 3.3 — Prix moyen par DPE, avec/sans {equip.lower()}")
                fig, ax = plt.subplots(figsize=(11, 5))
                sns.pointplot(x='DPE_num', y='Prix_au_m2_w', hue=equip, data=df, errorbar=None, capsize=0.1, palette='Set2', ax=ax)
                ax.set(xlabel='DPE (1=Bon 2=Moyen 3=Mauvais)', ylabel='Prix moyen au m² (€)')
                if show_annotations:
                    grouped = df.groupby(['DPE_num', equip], observed=False)['Prix_au_m2_w']
                    hue_order = sorted(df[equip].dropna().unique())
                    for (dpe, h), sub in grouped:
                        if len(sub) == 0: continue
                        mean, n = sub.mean(), len(sub)
                        hi = hue_order.index(h)
                        x = (dpe - 1) + (hi - (len(hue_order)-1)/2) * 0.2
                        ax.text(x, mean, f"{mean:.0f}€\n(n={n})", ha='center', va='bottom', fontsize=8)
                st.pyplot(fig, use_container_width=True, clear_figure=True)

# ============================================================
# 4) COMPARAISONS DPE/GES (BOXPLOTS + VIOLINS + ANNOTATIONS)
# ============================================================
if sec_comp:
    ok = have('Prix_au_m2_w', 'DPE_num', 'GES_num')
    section_header("4) Comparaisons DPE/GES (box+violin)", ok)
    with st.container(border=True):
        if not ok:
            st.info("Colonnes manquantes : Prix_au_m2_w, DPE_num, GES_num")
        else:
            # utilitaire d'annotation (médiane + n)
            def annotate_box_violin(ax, frame, xcol, ycol):
                ticks = [t.get_text() for t in ax.get_xticklabels()]
                if not ticks:
                    return
                ymin, ymax = ax.get_ylim()
                yoff = (ymax - ymin) * 0.02
                # cast en str pour matcher les tick labels
                key = frame[xcol].astype(str)
                for i, lab in enumerate(ticks):
                    sub = frame.loc[key == lab, ycol].dropna()
                    if sub.empty:
                        continue
                    med = sub.median(); n = len(sub)
                    ax.text(i, med + yoff, f"{med:.0f}€\n(n={n})",
                            ha='center', va='bottom', fontsize=9,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1))

            title_line("Graphiques 4.1 — Relations DPE/GES avec le prix (4 vues)")
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
            axes[0, 1].set_ylabel('')
            annotate_box_violin(axes[0, 1], df, 'GES_num', 'Prix_au_m2_w')

            # --- Violin DPE
            sns.violinplot(x='DPE_num', y='Prix_au_m2_w', data=df, palette='Set2', ax=axes[1, 0])
            axes[1, 0].set_title('Violin plot prix au m² vs DPE')
            axes[1, 0].set_xlabel('DPE (1=Bon 2=Moyen 3=Mauvais)')
            axes[1, 0].set_ylabel('Prix au m² (€)')
            annotate_box_violin(axes[1, 0], df, 'DPE_num', 'Prix_au_m2_w')

            # --- Violin GES
            sns.violinplot(x='GES_num', y='Prix_au_m2_w', data=df, palette='Set2', ax=axes[1, 1])
            axes[1, 1].set_title('Violin plot prix au m² vs GES')
            axes[1, 1].set_xlabel('GES (1=Bon 2=Moyen 3=Mauvais)')
            axes[1, 1].set_ylabel('')
            annotate_box_violin(axes[1, 1], df, 'GES_num', 'Prix_au_m2_w')

            plt.tight_layout()
            st.pyplot(fig, width='stretch', clear_figure=True)


# ===============================================
# 5) INTERACTIONS LOCALITÉ & TRANSPORT / PARKING
# ===============================================
if sec_interact_2:
    ok = have(*required["5) Interactions Localité & Transport"])
    section_header("5) Interactions Localité & Transport", ok)
    with st.container(border=True):
        if not ok:
            st.stop()

        # 5.1 Localité × Surface (style lmplot)
        title_line("Graphique 5.1 — Prix vs Surface par localité")
        fig, ax = plt.subplots(figsize=(12, 5))
        for loc, sub in df.groupby('ville_classer'):
            sns.regplot(x='Surface_w', y='Prix_au_m2_w', data=sub, scatter_kws={'alpha':0.35}, label=str(loc), ax=ax)
        ax.set(xlabel='Surface winsorisée (m²)', ylabel='Prix au m² (€)')
        ax.legend(title='Localité', bbox_to_anchor=(1.02, 1), loc='upper left')
        st.pyplot(fig, use_container_width=True, clear_figure=True)

        # 5.2 Localité × Parking
        if have('Parking'):
            title_line("Graphique 5.2 — Prix moyen par localité, avec/sans parking")
            fig, ax = plt.subplots(figsize=(12, 5))
            x_order = sorted(df['ville_classer'].dropna().unique())
            hue_order = sorted(df['Parking'].dropna().unique())
            sns.barplot(x='ville_classer', y='Prix_au_m2_w', hue='Parking', data=df, errorbar=None, palette='pastel', order=x_order, hue_order=hue_order, ax=ax)
            ax.set(xlabel='Localité', ylabel='Prix moyen au m² (€)')
            if show_annotations:
                counts = df.groupby(['ville_classer', 'Parking'], observed=False)['Prix_au_m2_w'].size().to_dict()
                idx = 0
                for xi, _x in enumerate(x_order):
                    for hi, _h in enumerate(hue_order):
                        p = ax.patches[idx]; idx += 1
                        h = p.get_height(); x = p.get_x() + p.get_width()/2
                        n = counts.get((_x, _h), 0)
                        ax.text(x, h, f"{h:.0f}€\n(n={n})", ha='center', va='bottom', fontsize=9)
            ax.legend(title='Parking', bbox_to_anchor=(1.02, 1), loc='upper left')
            st.pyplot(fig, use_container_width=True, clear_figure=True)

        # 5.3 Transport – distribution horizontale + %
        if have('Transport'):
            title_line("Répartition des modalités de transport")
            fig, ax = plt.subplots(figsize=(12, 5))
            order = df['Transport'].value_counts().index
            sns.countplot(y='Transport', data=df, order=order, ax=ax)
            ax.set(xlabel='Nombre de biens', ylabel='Modalité')
            if show_annotations:
                total = len(df)
                for p in ax.patches:
                    count = int(p.get_width())
                    if count > 0:
                        pct = 100 * count / total
                        ax.text(p.get_width()+total*0.005, p.get_y()+p.get_height()/2, f"{count} ({pct:.1f}%)", ha='left', va='center', fontsize=9)
            st.pyplot(fig, use_container_width=True, clear_figure=True)

        # 5.4 Transport → Prix moyen au m² (barres horizontales + annotations)
        if sec_transport_price and have('Transport', 'Prix_au_m2_w'):
            title_line("Graphique 5.4 — Prix moyen au m² par modalité de transport")

            # Prépare stats (moyenne, médiane, effectif)
            stats_tp = (
                df.groupby('Transport', dropna=False)['Prix_au_m2_w']
                  .agg(mean='mean', median='median', n='count')
                  .reset_index()
            )
            # Nettoyage étiquettes et tri par prix moyen (desc)
            stats_tp['Transport'] = stats_tp['Transport'].astype(str)
            stats_tp = stats_tp.sort_values('mean', ascending=True)  # ordre vertical bas→haut

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(
                data=stats_tp, y='Transport', x='mean',
                palette='viridis', ax=ax, orient='h', errorbar=None
            )

            ax.set(
                title='Prix moyen au m² par modalité de transport',
                xlabel='Prix moyen au m² (€)',
                ylabel='Modalité Transport'
            )
            ax.grid(axis='x', linestyle='--', alpha=0.7)

            # Annotations: valeur moyenne + effectif
            xlim = ax.get_xlim()
            x_range = xlim[1] - xlim[0]
            for p, (_, row) in zip(ax.patches, stats_tp.iterrows()):
                val = row['mean']
                n = int(row['n'])
                # position à la fin de la barre
                ax.text(
                    p.get_width() + 0.01 * x_range,
                    p.get_y() + p.get_height() / 2,
                    f"{val:,.0f} €  (n={n})".replace(",", " "),
                    ha='left', va='center', fontsize=10, color='black'
                )

            plt.tight_layout()
            st.pyplot(fig, width='stretch', clear_figure=True)
        elif sec_transport_price:
            st.info("Colonnes nécessaires manquantes : Transport, Prix_au_m2_w")


# ==========================================
# 6) QUALITÉ TEXTUELLE : kw_pos / kw_neg
# ==========================================
if sec_text:
    ok = have(*required["6) Qualité textuelle (kw_pos/kw_neg)"])
    section_header("6) Qualité textuelle (kw_pos/kw_neg)", ok)
    with st.container(border=True):
        if not ok:
            st.stop()

        # DPE_Qualite à 3 classes
        df['DPE_Qualite'] = df['DPE_num'].apply(lambda x: 'Mauvais DPE (E-G)' if x > 2 else ('Moyen DPE (D)' if x == 2 else 'Bon DPE (A-C)'))
        df['DPE_Qualite'] = pd.Categorical(df['DPE_Qualite'], categories=['Bon DPE (A-C)', 'Moyen DPE (D)', 'Mauvais DPE (E-G)'], ordered=True)

        # Terciles kw_pos (fallback médiane si nécessaire)
        q33, q66 = df['kw_pos'].quantile([0.33, 0.66])
        eps = 1e-6
        bins = sorted(set([df['kw_pos'].min()-eps, q33, q66, df['kw_pos'].max()+eps]))
        labels = ['kw_pos Faible', 'kw_pos Moyen', 'kw_pos Élevé'][:len(bins)-1]
        if len(bins) >= 3:
            df['kw_pos_cat'] = pd.cut(df['kw_pos'], bins=bins, labels=labels, include_lowest=True, right=False)
        else:
            med = df['kw_pos'].median()
            df['kw_pos_cat'] = np.where(df['kw_pos']>med, 'kw_pos > Médiane', 'kw_pos ≤ Médiane')

        # 4.1 Boxplot Prix ~ DPE_Qualite × kw_pos_cat + annotations
        title_line("Graphique 4.1 — Prix au m² par DPE, segmenté par niveau de mots positifs")
        fig, ax = plt.subplots(figsize=(12, 5.5))
        sns.boxplot(x='DPE_Qualite', y='Prix_au_m2_w', hue='kw_pos_cat', data=df, palette='Set3', ax=ax)
        ax.set(xlabel='Performance énergétique', ylabel='Prix au m² (€)')
        ax.legend(title='Niveau de mots positifs', bbox_to_anchor=(1.02, 1), loc='upper left')
        if show_annotations and 'kw_pos_cat' in df.columns:
            group_stats = (
                df.groupby(['DPE_Qualite', 'kw_pos_cat'], observed=False)['Prix_au_m2_w']
                  .agg(Min='min', Q1=lambda x: x.quantile(0.25), Median='median', Q3=lambda x: x.quantile(0.75), Max='max', n='count')
                  .reset_index()
            )
            hue_levels = [h for h in df['kw_pos_cat'].dropna().unique()]
            offsets = np.linspace(-0.3, 0.3, len(hue_levels)) if len(hue_levels)>0 else [0]
            for i, dpe in enumerate(df['DPE_Qualite'].cat.categories):
                for j, hval in enumerate(hue_levels):
                    sub = group_stats[(group_stats['DPE_Qualite']==dpe) & (group_stats['kw_pos_cat']==hval)]
                    if sub.empty: continue
                    s = sub.iloc[0]; x_pos = i + offsets[j]
                    for k in ['Min','Q1','Median','Q3','Max']:
                        ax.text(x_pos, s[k], ("Med" if k=='Median' else f"{k}")+f"{int(s[k]):,}€", ha='center', va='bottom', fontsize=7)
                    ax.text(x_pos, s['Max']*1.03, f"(n={s['n']})", ha='center', va='bottom', fontsize=8)
        st.pyplot(fig, use_container_width=True, clear_figure=True)
    
        # 4.2 Barplot négatifs par localité
        title_line("Graphique 4.2 — Décote des mots négatifs par localité")
        fig, ax = plt.subplots(figsize=(12, 5))
        df['kw_neg_bin'] = np.where(df['kw_neg']>0, 'kw_neg > 0', 'kw_neg = 0')
        sns.barplot(x='ville_classer', y='Prix_au_m2_w', hue='kw_neg_bin', data=df, errorbar=None, palette='Set1', ax=ax)
        ax.set(xlabel='Localité', ylabel='Prix moyen au m² (€)')
        ax.legend(title='Mots négatifs', bbox_to_anchor=(1.02, 1), loc='upper left')
        if show_annotations:
            for p in ax.patches:
                h = p.get_height()
                if np.isfinite(h) and h>0:
                    x = p.get_x()+p.get_width()/2
                    ax.text(x, h*1.01, f"{int(h):,}€", ha='center', va='bottom', fontsize=8)
        st.pyplot(fig, use_container_width=True, clear_figure=True)

# ================================================
# 7) INTERACTIONS MARKETING & AUTRES VARIABLES
# ================================================
if sec_marketing:
    # Colonnes requises (on affiche l’avertissement si besoin)
    needed_any = [
        ('Nombre_de_photos','DPE_num','Prix_au_m2_w'),
        ('Charges_w','DPE_num','Prix_au_m2_w'),
        ('Pièces','DPE_num','Prix_au_m2_w'),
        ('Procedure_bin','DPE_num','Prix_au_m2_w'),
        ('Note_z','DPE_num','Prix_au_m2_w'),
    ]
    ok_any = any(all(c in df.columns for c in t) for t in needed_any)
    section_header("7) Interactions marketing & annexes", ok_any)
    with st.container(border=True):
        if not ok_any:
            st.info("Certaines colonnes nécessaires manquent (photos, charges, pièces, procédure, note). Les sous-graphes s’afficheront seulement si leurs colonnes existent.")
        
        # --------- Sécurité DPE_Qualite (A–C / D / E–G) ----------
        if 'DPE_num' in df.columns and 'DPE_Qualite' not in df.columns:
            df['DPE_Qualite'] = df['DPE_num'].apply(
                lambda x: 'Mauvais DPE (E-G)' if x > 2 else ('Moyen DPE (D)' if x == 2 else 'Bon DPE (A-C)')
            )
            df['DPE_Qualite'] = pd.Categorical(
                df['DPE_Qualite'],
                categories=['Bon DPE (A-C)', 'Moyen DPE (D)', 'Mauvais DPE (E-G)'],
                ordered=True
            )

        # =========================
        # 7.1 DPE/GES × Nb de photos
        # =========================
        if all(c in df.columns for c in ['Nombre_de_photos','DPE_num','Prix_au_m2_w','DPE_Qualite']):
            title_line("Graphique 7.1 — Prix au m² par qualité DPE, segmenté par nombre de photos")
            q_med = df['Nombre_de_photos'].median()
            df['Photos_Cat'] = np.where(df['Nombre_de_photos'] > q_med, 'Photos > Médiane', 'Photos ≤ Médiane')
            hue_order = ['Photos ≤ Médiane', 'Photos > Médiane']
            df['Photos_Cat'] = pd.Categorical(df['Photos_Cat'], categories=hue_order, ordered=True)
            x_order = list(df['DPE_Qualite'].cat.categories)

            fig, ax = plt.subplots(figsize=(11.5, 5.5))
            sns.boxplot(
                x='DPE_Qualite', y='Prix_au_m2_w',
                hue='Photos_Cat', data=df,
                order=x_order, hue_order=hue_order, palette='Set2', ax=ax
            )
            ax.set(xlabel='Performance énergétique', ylabel='Prix au m² (€)')
            ax.legend(title='Nombre de photos', bbox_to_anchor=(1.02, 1), loc='upper left')

            if show_annotations:
                stats = (df.groupby(['DPE_Qualite','Photos_Cat'], observed=False)['Prix_au_m2_w']
                           .agg(median='median', n='count').reset_index())
                x_idx = {lvl:i for i,lvl in enumerate(x_order)}
                h_idx = {lvl:j for j,lvl in enumerate(hue_order)}
                m = len(hue_order); width = 0.8
                ymin,ymax = ax.get_ylim(); yoff=(ymax-ymin)*0.012
                for _,r in stats.iterrows():
                    i = x_idx.get(r['DPE_Qualite']); j = h_idx.get(r['Photos_Cat'])
                    if i is None or j is None: continue
                    x_pos = i + (j - (m - 1)/2)*(width/m)
                    y_pos = r['median']
                    ax.text(x_pos, y_pos + yoff, f"{y_pos:.0f}€\n(n={int(r['n'])})",
                            ha='center', va='bottom', fontsize=8,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=2))
            st.pyplot(fig, width='stretch', clear_figure=True)

        # ===============================
        # 7.2 DPE/GES × Charges (Charges_w)
        # ===============================
        if all(c in df.columns for c in ['Charges_w','DPE_num','Prix_au_m2_w']):
            title_line("Graphique 7.2 — Prix moyen par DPE, segmenté par niveau de charges")
            q75 = df['Charges_w'].quantile(0.75)
            df['Charges_Cat'] = np.where(df['Charges_w'] > q75, 'Charges Élevées (>75e pct)', 'Charges Faibles/Moyennes')
            hue_order = ['Charges Faibles/Moyennes', 'Charges Élevées (>75e pct)']
            df['Charges_Cat'] = pd.Categorical(df['Charges_Cat'], categories=hue_order, ordered=True)

            stats = (df.groupby(['DPE_num','Charges_Cat'], observed=False)['Prix_au_m2_w']
                       .agg(mean='mean', n='count').reset_index())

            fig, ax = plt.subplots(figsize=(11.5, 5.5))
            offsets = {'Charges Faibles/Moyennes': -0.15, 'Charges Élevées (>75e pct)': 0.15}
            colors = {'Charges Faibles/Moyennes': 'royalblue', 'Charges Élevées (>75e pct)': 'tomato'}

            for cat in hue_order:
                sub = stats[stats['Charges_Cat'] == cat]
                x = sub['DPE_num'] + offsets[cat]
                y = sub['mean']
                ax.plot(x, y, marker='o', linestyle='-', label=cat, color=colors[cat])
                if show_annotations:
                    for xi, yi, n in zip(x, y, sub['n']):
                        ax.text(xi, yi, f"{yi:.0f}€\n(n={int(n)})", ha='center', va='bottom', fontsize=8,
                                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1))
            ax.set(xlabel='DPE (1=Bon 2=Moyen 3=Mauvais)', ylabel='Prix moyen au m² (€)')
            ax.legend(title='Niveau de charges', bbox_to_anchor=(1.02, 1), loc='upper left')
            ax.grid(alpha=0.3)
            st.pyplot(fig, width='stretch', clear_figure=True)

        # ===========================
        # 7.3 DPE/GES × Nombre de pièces
        # ===========================
        if all(c in df.columns for c in ['Pièces','DPE_num','Prix_au_m2_w']):
            title_line("Graphique 7.3 — Prix moyen par DPE, segmenté par nombre de pièces")
            df['Pieces_Cat'] = df['Pièces'].apply(lambda x: 'Grand Bien (> 3 Pièces)' if x > 3 else 'Petit/Moyen Bien (≤ 3 Pièces)')
            hue_order = ['Petit/Moyen Bien (≤ 3 Pièces)', 'Grand Bien (> 3 Pièces)']
            df['Pieces_Cat'] = pd.Categorical(df['Pieces_Cat'], categories=hue_order, ordered=True)

            fig, ax = plt.subplots(figsize=(11.5, 5.5))
            sns.barplot(x='DPE_num', y='Prix_au_m2_w', hue='Pieces_Cat', data=df,
                        errorbar=None, palette='pastel', hue_order=hue_order, ax=ax)
            ax.set(xlabel='DPE (1=Bon 2=Moyen 3=Mauvais)', ylabel='Prix moyen au m² (€)')

            if show_annotations:
                stats = (df.groupby(['DPE_num','Pieces_Cat'], observed=False)['Prix_au_m2_w']
                           .agg(mean='mean', n='count').reset_index())
                x_order = sorted(df['DPE_num'].dropna().unique())
                m = len(hue_order); width = 0.8
                ymin,ymax = ax.get_ylim(); yoff=(ymax-ymin)*0.015
                for _, r in stats.iterrows():
                    i = x_order.index(r['DPE_num'])
                    j = hue_order.index(r['Pieces_Cat'])
                    x_pos = i + (j - (m - 1)/2) * (width / m)
                    y_pos = r['mean']
                    ax.text(x_pos, y_pos + yoff, f"{y_pos:.0f}€\n(n={int(r['n'])})",
                            ha='center', va='bottom', fontsize=9,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1))
            ax.legend(title='Catégorie de pièces', bbox_to_anchor=(1.02, 1), loc='upper left')
            st.pyplot(fig, width='stretch', clear_figure=True)

        # ===========================
        # 7.4 DPE/GES × Procédure en cours
        # ===========================
        if all(c in df.columns for c in ['Procedure_bin','DPE_num','Prix_au_m2_w','DPE_Qualite']):
            title_line("Graphique 7.4 — Prix au m² par qualité DPE, segmenté par procédure (0/1)")
            hue_order = [0, 1]  # 0=Non, 1=Oui
            fig, ax = plt.subplots(figsize=(11.5, 5.5))
            sns.boxplot(x='DPE_Qualite', y='Prix_au_m2_w', hue='Procedure_bin', data=df,
                        order=list(df['DPE_Qualite'].cat.categories), hue_order=hue_order, ax=ax)
            ax.set(xlabel='Performance énergétique', ylabel='Prix au m² (€)')
            ax.legend(title='Procédure', bbox_to_anchor=(1.02, 1), loc='upper left')

            if show_annotations:
                stats = (df.groupby(['DPE_Qualite','Procedure_bin'], observed=False)['Prix_au_m2_w']
                           .agg(median='median', n='count').reset_index())
                x_order = list(df['DPE_Qualite'].cat.categories)
                m = len(hue_order); width = 0.8
                ymin,ymax = ax.get_ylim(); yoff=(ymax-ymin)*0.012
                x_idx = {lvl:i for i,lvl in enumerate(x_order)}
                for _, r in stats.iterrows():
                    i = x_idx[r['DPE_Qualite']]; j = hue_order.index(r['Procedure_bin'])
                    x_pos = i + (j - (m - 1)/2)*(width/m)
                    y_pos = r['median']
                    ax.text(x_pos, y_pos + yoff, f"{y_pos:.0f}€\n(n={int(r['n'])})",
                            ha='center', va='bottom', fontsize=8,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1))
            st.pyplot(fig, width='stretch', clear_figure=True)

        # ===========================
        # 7.5 DPE/GES × Note/Avis (Note_z)
        # ===========================
        if all(c in df.columns for c in ['Note_z','DPE_num','Prix_au_m2_w']):
            title_line("Graphique 7.5 — Prix moyen par DPE, segmenté par note agent (Z-score)")
            df['Note_z_Cat'] = np.where(df['Note_z'] < 0, 'Note < Moyenne',
                                 np.where(df['Note_z'] >= 0, 'Note ≥ Moyenne', 'Note manquante'))
            hue_order = ['Note < Moyenne', 'Note ≥ Moyenne', 'Note manquante']
            df['Note_z_Cat'] = pd.Categorical(df['Note_z_Cat'], categories=hue_order, ordered=True)

            fig, ax = plt.subplots(figsize=(11.5, 5.5))
            sns.pointplot(x='DPE_num', y='Prix_au_m2_w', hue='Note_z_Cat', data=df,
                          errorbar=None, capsize=0.1, palette='Set2', hue_order=hue_order, ax=ax)
            ax.set(xlabel='DPE (1=Bon 2=Moyen 3=Mauvais)', ylabel='Prix moyen au m² (€)')

            if show_annotations:
                stats = (df.groupby(['DPE_num','Note_z_Cat'], observed=False)['Prix_au_m2_w']
                           .agg(mean='mean', n='count').reset_index())
                x_order = sorted(df['DPE_num'].dropna().unique())
                m = len(hue_order); width = 0.8
                ymin,ymax = ax.get_ylim(); yoff=(ymax-ymin)*0.015
                for _, r in stats.iterrows():
                    if r['Note_z_Cat'] not in hue_order: 
                        continue
                    i = x_order.index(r['DPE_num']); j = hue_order.index(r['Note_z_Cat'])
                    x_pos = i + (j - (m - 1)/2)*(width/m)
                    y_pos = r['mean']
                    ax.text(x_pos, y_pos + yoff, f"{y_pos:.0f}€\n(n={int(r['n'])})",
                            ha='center', va='bottom', fontsize=8,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1))
            ax.legend(title='Note (Z)', bbox_to_anchor=(1.02, 1), loc='upper left')
            st.pyplot(fig, width='stretch', clear_figure=True)

# =============================================
# 8) CORRÉLATIONS & TESTS H1–H3 (ANOVA, etc.)
# =============================================
if sec_tests:
    ok = have(*required["8) Corrélations & Tests H1-H3"])
    section_header("8) Corrélations & Tests H1-H3", ok)
    with st.container(border=True):
        if not ok:
            st.stop()

        corr_vars = [c for c in ['Prix_au_m2_w','DPE_num','GES_num','Surface_w','Procedure_bin','Charges_w','Nombre_de_lots','Pièces','Parking','Balcon','Nombre_de_photos','Note_z','Avis_z','Transport','Ascenseur'] if c in df.columns]
        title_line("Corrélation (Spearman) – robuste pour ordinal/non-linéaire")
        spearman_corr = df[corr_vars].corr(method='spearman')
        fig, ax = plt.subplots(figsize=(12, 6.5))
        sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
        ax.set(title='Carte de chaleur des corrélations (Spearman)')
        st.pyplot(fig, use_container_width=True, clear_figure=True)

        # Tests d'hypothèse — sécurisation des tailles d'échantillons
        title_line("Tests d’hypothèse")
        min_obs = 30
        df_dpe = df[df.groupby('DPE_num')['Prix_au_m2_w'].transform('count') >= min_obs]
        df_ges = df[df.groupby('GES_num')['Prix_au_m2_w'].transform('count') >= min_obs]

        groups_dpe = [g.dropna() for _, g in df_dpe.groupby('DPE_num')['Prix_au_m2_w'] if len(g.dropna()) >= 3]
        groups_ges = [g.dropna() for _, g in df_ges.groupby('GES_num')['Prix_au_m2_w'] if len(g.dropna()) >= 3]

        # Init rows
        lev_dpe = None; anova_row = None; lev_ges = None; kruskal_row = None; welch_row = None

        if len(groups_dpe) >= 2:
            stat_l_dpe, p_l_dpe = levene(*groups_dpe)
            an = f_oneway(*groups_dpe)
            lev_dpe = {"Test":"Levene (DPE)", "stat": stat_l_dpe, "p_value": p_l_dpe, "seuil": ALPHA, "commentaire":"Variances non homogènes si p < seuil"}
            anova_row = {"Test":"ANOVA (DPE)", "stat": an.statistic, "p_value": an.pvalue, "seuil": ALPHA, "commentaire":"Comparaison des moyennes par niveaux de DPE"}
        else:
            st.info("DPE : effectifs trop faibles par groupe (n<3) pour Levene/ANOVA.")

        if len(groups_ges) >= 2:
            stat_l_ges, p_l_ges = levene(*groups_ges)
            kw = kruskal(*groups_ges)
            lev_ges = {"Test":"Levene (GES)", "stat": stat_l_ges, "p_value": p_l_ges, "seuil": ALPHA, "commentaire":"Variances non homogènes si p < seuil"}
            kruskal_row = {"Test":"Kruskal–Wallis (GES)", "stat": kw.statistic, "p_value": kw.pvalue, "seuil": ALPHA, "commentaire":"Comparaison des distributions par niveaux de GES"}
        else:
            st.info("GES : effectifs trop faibles par groupe (n<3) pour Levene/Kruskal.")

        if have('DPE_num'):
            high = df[df['DPE_num'].isin([1,2,3])]['Prix_au_m2_w'].dropna()
            low  = df[df['DPE_num'].isin([4,5,6,7])]['Prix_au_m2_w'].dropna()
            if len(high) >= 3 and len(low) >= 3:
                t_stat, p_val = ttest_ind(high, low, equal_var=False)
                welch_row = {"Test":"t de Welch (A–C vs D–G)", "stat": t_stat, "p_value": p_val, "seuil": ALPHA, "commentaire":"Différence de moyennes entre groupes"}
            else:
                st.info("t-test DPE : échantillons trop petits (n<3) pour comparer A–C vs D–G.")

        # Tableau récap propre
        rows = [r for r in [lev_dpe, lev_ges, anova_row, kruskal_row, welch_row] if r is not None]
        if rows:
            out = pd.DataFrame(rows)
            out['stat'] = out['stat'].round(3)
            out['p_value'] = out['p_value'].round(4)
            st.dataframe(out, width='stretch')

# =====================================
# 9) RÉGRESSIONS (OLS) & MULTICOLLINEARITÉ
# =====================================
if sec_models:
    ok = have(*required["9) Régressions & VIF"])
    section_header("9) Régressions & VIF", ok)
    with st.container(border=True):
        if not ok:
            st.stop()

        # -------- Helpers --------
        def tidy_model(m):
            ci = m.conf_int()
            df_tidy = pd.DataFrame({
                'Terme': m.params.index,
                'Coef': m.params.values,
                'StdErr': m.bse.values,
                't': m.tvalues.values,
                'p': m.pvalues.values,
                'CI_low': ci[0].values,
                'CI_high': ci[1].values,
            })
            return df_tidy

        def kpi_row(r2, r2a, n):
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("R²", f"{r2:.3f}")
            with c2: st.metric("Adj R²", f"{r2a:.3f}")
            with c3: st.metric("n", f"{int(n)}")

        def style_vif_table(df_vif: pd.DataFrame):
            dfv = df_vif.copy()
            if 'Variable' in dfv.columns:
                dfv = dfv[dfv['Variable'] != 'const']
            dfv = dfv.sort_values('VIF', ascending=False).reset_index(drop=True)
            return (
                dfv.style
                   .format({'VIF': '{:.3f}'})
                   .apply(lambda s: [
                       ('background-color:#3b1d1d' if (pd.notna(v) and float(v)>10)
                        else 'background-color:#2a2626' if (pd.notna(v) and float(v)>5)
                        else '') for v in s
                    ], subset=['VIF'])
            )

        def style_coef_table(df_coef: pd.DataFrame):
            fmt = {'Coef':'{:.3f}','StdErr':'{:.3f}','t':'{:.3f}','p':'{:.4f}','CI_low':'{:.3f}','CI_high':'{:.3f}'}
            def color_sign(val):
                try:
                    v = float(val)
                    return 'color:#28a745' if v>0 else ('color:#e55353' if v<0 else '')
                except:
                    return ''
            def highlight_p(col):
                out = []
                for v in col:
                    try:
                        p = float(v)
                        if p < 0.01: out.append('font-weight:700;background-color:#243a2b')
                        elif p < 0.05: out.append('font-weight:600;background-color:#2a2f2a')
                        else: out.append('')
                    except:
                        out.append('')
                return out
            return (
                df_coef.style
                       .format(fmt, na_rep='—')
                       .applymap(color_sign, subset=['Coef'])
                       .apply(highlight_p, subset=['p'])
            )

        # -------- 9.1 VIF ----------
        vif_data = pd.DataFrame()
        X_vars = [c for c in ['Surface_w','DPE_num','GES_num','kw_pos','kw_neg'] if c in df.columns]
        X = df[X_vars].dropna()
        if not X.empty:
            Xc = sm.add_constant(X)
            vif_data = pd.DataFrame({
                'Variable': Xc.columns,
                'VIF': [variance_inflation_factor(Xc.values, i) for i in range(len(Xc.columns))]
            })
        st.markdown("**VIF (Variance Inflation Factor)** – >5/10 à surveiller")
        if not vif_data.empty:
            st.dataframe(style_vif_table(vif_data), use_container_width=True)
            st.download_button("Télécharger VIF (CSV)",
                               vif_data.to_csv(index=False).encode('utf-8'),
                               file_name="vif.csv", mime="text/csv")
        else:
            st.info("Pas assez de données complètes pour calculer le VIF.")

        # -------- 9.2 OLS principal (localité & interactions) ----------
        model = None
        base_cols = ['Prix_au_m2_w','DPE_num','GES_num','Surface_w','ville_classer','kw_pos','kw_neg']
        if all(c in df.columns for c in base_cols):
            formula = (
                'Prix_au_m2_w ~ DPE_num + GES_num + Surface_w + C(ville_classer) + '
                'DPE_num:C(ville_classer) + GES_num:C(ville_classer) + kw_pos + kw_neg'
            )
            model = smf.ols(formula, data=df.dropna(subset=base_cols)).fit()

        # -------- 9.3 OLS enrichi ----------
        model_enriched = None
        enrich_needed = ['Année_de_construction','Parking']
        if all(c in df.columns for c in (base_cols + enrich_needed)):
            formula_enriched = (
                'Prix_au_m2_w ~ DPE_num + GES_num + Surface_w + C(ville_classer) + kw_pos + kw_neg + '
                'DPE_num:C(ville_classer) + GES_num:C(ville_classer) + '
                'DPE_num:Surface_w + GES_num:Surface_w + DPE_num:Année_de_construction + C(ville_classer):C(Parking)'
            )
            dfm = df[['Prix_au_m2_w','DPE_num','GES_num','Surface_w','ville_classer',
                      'kw_pos','kw_neg','Année_de_construction','Parking']].dropna()
            if not dfm.empty:
                model_enriched = smf.ols(formula_enriched, data=dfm).fit()

        # -------- 9.4 OLS variante Transport (HC3) ----------
        model_alt = None
        cols_alt = [c for c in ['Prix_au_m2_w','DPE_num','GES_num','Surface_w','Procedure_bin','Charges_w',
                                'Nombre_de_lots','Pièces','Parking','Balcon','Nombre_de_photos',
                                'Note_z','Avis_z','Transport','Ascenseur'] if c in df.columns]
        if all(c in df.columns for c in cols_alt):
            df_alt = df[cols_alt].dropna().copy()
            if not df_alt.empty:
                formula_alt = (
                    "Prix_au_m2_w ~ DPE_num + GES_num + Surface_w + Procedure_bin + Charges_w + "
                    "Nombre_de_lots + Pièces + Parking + Balcon + Nombre_de_photos + "
                    "Note_z + Avis_z + C(Transport) + Ascenseur"
                )
                model_alt = smf.ols(formula_alt, data=df_alt).fit(cov_type="HC3")

        # ===== Présentation (onglets + KPIs) =====
        tabs = st.tabs([
            "OLS (localité & interactions)",
            "OLS enrichi (surface/année/parking)",
            "OLS variante (Transport • SE robustes)"
        ])

        with tabs[0]:
            if model is not None:
                st.subheader("OLS – modèle avec interactions (localité)")
                df_main = tidy_model(model)
                st.dataframe(style_coef_table(df_main), use_container_width=True)
                kpi_row(model.rsquared, model.rsquared_adj, model.nobs)
                st.download_button("Télécharger les coefficients (CSV)",
                                   df_main.to_csv(index=False).encode('utf-8'),
                                   file_name="ols_localite_coefs.csv", mime="text/csv")
            else:
                st.info("Colonnes insuffisantes pour le modèle OLS avec interactions de localité.")

        with tabs[1]:
            if model_enriched is not None:
                st.subheader("OLS – modèle enrichi (interactions surface/année/parking)")
                df_enr = tidy_model(model_enriched)
                st.dataframe(style_coef_table(df_enr), use_container_width=True)
                kpi_row(model_enriched.rsquared, model_enriched.rsquared_adj, model_enriched.nobs)
                st.download_button("Télécharger les coefficients (CSV)",
                                   df_enr.to_csv(index=False).encode('utf-8'),
                                   file_name="ols_enrichi_coefs.csv", mime="text/csv")
            else:
                st.info("Modèle enrichi indisponible (jeu de données réduit après dropna).")

        with tabs[2]:
            if model_alt is not None:
                st.subheader("OLS – Variante avec Transport (SE robustes HC3)")
                df_altc = tidy_model(model_alt)
                st.dataframe(style_coef_table(df_altc), use_container_width=True)
                kpi_row(model_alt.rsquared, model_alt.rsquared_adj, model_alt.nobs)
                st.download_button("Télécharger les coefficients (CSV)",
                                   df_altc.to_csv(index=False).encode('utf-8'),
                                   file_name="ols_transport_coefs.csv", mime="text/csv")
            else:
                st.info("Variante Transport indisponible (colonnes manquantes).")


# ==================================
# 10) CARTES INTERACTIVES – FOLIUM
# ==================================
@st.cache_resource(show_spinner=False)
def geocoder():
    return Nominatim(user_agent="real_estate_analysis_streamlit")

@st.cache_data(show_spinner=False)
def prepare_map_data(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty or not have('Code_Postal','Prix_au_m2_w','DPE_num'):
        return pd.DataFrame()
    map_df = data.groupby('Code_Postal').agg(
        Prix_moyen=('Prix_au_m2_w','mean'),
        DPE_moyen=('DPE_num','mean'),
        Nbre_biens=('Code_Postal','count')
    ).reset_index()
    return map_df

@st.cache_data(show_spinner=True)
def geocode_postal_codes(map_df: pd.DataFrame) -> pd.DataFrame:
    if map_df.empty:
        return map_df
    g = geocoder()
    lats, lons = [], []
    for cp in map_df['Code_Postal']:
        try:
            loc = g.geocode(f"{cp}, France")
            if loc:
                lats.append(loc.latitude); lons.append(loc.longitude)
            else:
                lats.append(np.nan); lons.append(np.nan)
            time.sleep(1)  # Politesse Nominatim
        except Exception:
            lats.append(np.nan); lons.append(np.nan)
    map_df['Latitude'] = lats; map_df['Longitude'] = lons
    map_df = map_df.dropna(subset=['Latitude','Longitude'])
    map_df.set_index('Code_Postal', inplace=True)
    return map_df

if sec_maps:
    ok = have(*required["10) Cartes (prix & DPE) par code postal"])
    section_header("10) Cartes (prix & DPE) par code postal", ok)
    with st.container(border=True):
        if not ok:
            st.stop()
        map_data = prepare_map_data(df)
        if map_data.empty:
            st.info("Pas de données agrégées pour la carte.")
        else:
            with st.spinner("Géocodage des codes postaux (Nominatim)…"):
                geo = geocode_postal_codes(map_data.copy())
            if geo.empty:
                st.info("Géocodage indisponible. Vérifiez la connectivité ou réduisez le nombre de CP.")
            else:
                latc, lonc = geo['Latitude'].mean(), geo['Longitude'].mean()

                # Carte 1 : Prix moyen
                fmap1 = folium.Map(location=[latc, lonc], zoom_start=9)
                max_prix = geo['Prix_moyen'].max()
                for cp, row in geo.iterrows():
                    radius = np.log(row['Nbre_biens'] + 1) * 3
                    color = plt.cm.plasma(row['Prix_moyen']/max_prix)
                    color_hex = f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}'
                    popup = folium.Popup(dedent(f"""
                        <b>Code Postal:</b> {cp}<br>
                        <b>Prix Moyen:</b> {row['Prix_moyen']:.0f} €/m²<br>
                        <b>Nb Biens:</b> {row['Nbre_biens']}
                    """), max_width=300)
                    folium.CircleMarker(location=[row['Latitude'], row['Longitude']], radius=radius, color=color_hex, fill=True, fill_color=color_hex, fill_opacity=0.7, popup=popup).add_to(fmap1)
                st.markdown("**Carte – Prix moyen au m²**")
                st_folium(fmap1, width='stretch', height=500)

                # Carte 2 : DPE moyen (vert=bon → rouge=mauvais)
                fmap2 = folium.Map(location=[latc, lonc], zoom_start=9)
                max_dpe, min_dpe = geo['DPE_moyen'].max(), geo['DPE_moyen'].min()
                for cp, row in geo.iterrows():
                    radius = np.log(row['Nbre_biens'] + 1) * 3
                    norm = (row['DPE_moyen'] - min_dpe) / (max_dpe - min_dpe if max_dpe>min_dpe else 1)
                    color = plt.cm.RdYlGn(1 - norm)
                    color_hex = f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}'
                    popup = folium.Popup(dedent(f"""
                        <b>Code Postal:</b> {cp}<br>
                        <b>DPE Moyen:</b> {row['DPE_moyen']:.2f} (1=Bon → 7=Mauvais)<br>
                        <b>Nb Biens:</b> {row['Nbre_biens']}
                    """), max_width=300)
                    folium.CircleMarker(location=[row['Latitude'], row['Longitude']], radius=radius, color=color_hex, fill=True, fill_color=color_hex, fill_opacity=0.7, popup=popup).add_to(fmap2)
                st.markdown("**Carte – DPE moyen**")
                st_folium(fmap2, width='stretch', height=500)

# ==============================
# 11) NUAGE DE MOTS (Descriptions)
# ==============================
if sec_wordcloud:
    ok = have(*required["11) Nuage de mots (Descriptions)"])
    section_header("11) Nuage de mots (Descriptions)", ok)
    with st.container(border=True):
        if not ok:
            st.stop()
        text = ' '.join(df['Description'].dropna().astype(str).tolist())
        stop_fr = set(STOPWORDS) | set(['appartement','pièce','pièces','chambre','chambres','dans','avec','sans','de','des','du','la','le','les','et','au','aux','un','une','sur','salle','bain','cuisine','séjour'])
        wc = WordCloud(width=1600, height=600, background_color="white", stopwords=stop_fr, collocations=True).generate(text)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Nuage de mots – Descriptions immobilières')
        st.pyplot(fig, use_container_width=True, clear_figure=True)

# ======================
# FOOTER – SOURCES & AIDE
# ======================
st.caption(
    """
    Conseils : si des colonnes manquent, les blocs s'affichent en ⚠️. Vous pouvez masquer/afficher les annotations
    pour des visuels plus épurés. Les modèles OLS utilisent `statsmodels`; les SE robustes sont indiquées quand pertinent.
    Géocodage assuré par Nominatim (OpenStreetMap) – gentil throttling.
    """
)
