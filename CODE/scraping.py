## SCRAPING DES DONNEES DU SITE IADFRANCE.FR
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time, re, pandas as pd

# --- Villes & config ---
villes = [
    'paris-75', 'boulogne-billancourt-92100', 'nanterre-92000',
    'creteil-94000', 'saint-denis-93200', 'villejuif-94800',
    'palaiseau-91120', 'versailles-78000', 'melun-77000'
]

annonces_data = []

# --- Helpers Note / Avis ---
def _parse_avis(txt: str):
    if not txt: return None
    txt = txt.replace('\xa0', ' ')
    m = re.search(r'\((\d+)\s*avis\)', txt, flags=re.IGNORECASE) or \
        re.search(r'\b(\d+)\s*avis\b', txt, flags=re.IGNORECASE) or \
        re.search(r'avis\s*\(?\s*(\d+)\s*\)?', txt, flags=re.IGNORECASE)
    return m.group(1) if m else None

def _parse_note(txt: str):
    if not txt: return None
    txt = txt.replace('\xa0', ' ')
    m = re.search(r'(\d+(?:[.,]\d+)?)/5\b', txt)
    return m.group(1).replace(',', '.') if m else None


def get_annonces_from_page(driver, ville, page_number):
    """Retourne le nombre d'annonces trouvées pour cette page (0 => fin)."""
    url = f'https://www.iadfrance.fr/annonces/{ville}/vente?page={page_number}'
    driver.get(url)
    time.sleep(4)  # laisser charger

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    annonces = soup.find_all('article', class_='overflow-clip')
    if not annonces:
        return 0

    for annonce in annonces:
        # --- Carte (listing) ---
        titre = annonce.find('h2').get_text(strip=True) if annonce.find('h2') else 'Pas de titre'

        prix = 'Pas de prix'
        for p in annonce.find_all('p'):
            txt = p.get_text(" ", strip=True)
            if re.search(r'\d[\d\s.,]*\s*€', txt):
                prix = txt; break

        pieces, surface = 'Pas de pièces', 'Pas de surface'
        for li in annonce.find_all('li'):
            txt = li.get_text(" ", strip=True)
            if 'pièce' in txt.lower(): pieces = txt
            elif 'm²' in txt or 'm2' in txt.lower(): surface = txt

        a = annonce.find('a')
        lien = a['href'] if a and a.has_attr('href') else None
        lien_complet = f'https://www.iadfrance.fr{lien}' if lien else 'Pas de lien'

        img_tag = annonce.find('img')
        image = (img_tag.get('src') or img_tag.get('data-src')) if img_tag else "Pas d'image"

        agent = annonce.find('figcaption').get_text(strip=True) if annonce.find('figcaption') else "Pas d'agent"

        card_text = annonce.get_text(" ", strip=True).replace('\xa0', ' ')
        note_value = _parse_note(card_text) or 'Pas de note'
        avis_value = _parse_avis(card_text) or None

        # --- Page détail ---
        if lien_complet.startswith('http'):
            driver.get(lien_complet)
            time.sleep(3)
            soup_details = BeautifulSoup(driver.page_source, 'html.parser')
        else:
            soup_details = BeautifulSoup("", 'html.parser')

        if note_value == 'Pas de note':
            note_from_detail = _parse_note(soup_details.get_text(" ", strip=True))
            if note_from_detail: note_value = note_from_detail
        if avis_value is None:
            avis_from_detail = _parse_avis(soup_details.get_text(" ", strip=True))
            if avis_from_detail: avis_value = avis_from_detail
        if avis_value is None: avis_value = "Pas d'avis"

        description = (
            soup_details.find('div', {'class': 'line-clamp-[8]'}).get_text(" ", strip=True)
            if soup_details.find('div', {'class': 'line-clamp-[8]'})
            else 'Pas de description'
        )

        # --- Copro ---
        copro_details = {}
        for item in soup_details.find_all('dl', {'class': 'flex gap-m justify-between'}):
            dt = item.find('dt').get_text(strip=True) if item.find('dt') else ''
            dd = item.find('dd').get_text(" ", strip=True) if item.find('dd') else ''
            if 'Nombre de lots' in dt:
                copro_details['Nombre de lots'] = dd
            elif 'Charges prévisionnelles' in dt:
                copro_details['Charges prévisionnelles'] = dd
            elif 'Procédure en cours' in dt:
                copro_details['Procédure en cours'] = dd

        prix_vente = (
            soup_details.find('dt', string="Prix de vente").find_next('dd').get_text(" ", strip=True)
            if soup_details.find('dt', string="Prix de vente") else 'Pas de prix de vente'
        )
        prix_m2 = (
            soup_details.find('dt', string="Prix au m²").find_next('dd').get_text(" ", strip=True)
            if soup_details.find('dt', string="Prix au m²") else 'Pas de prix au m²'
        )

        # --- DPE & GES (lettre si nécessaire) ---
        dpe = 'Pas de DPE'
        ges = 'Pas de GES'

        dpe_span = soup_details.find('span', string=lambda t: t and 'consommation énergétique excessive' in t.lower())
        if dpe_span:
            dpe = dpe_span.get_text(strip=True)
        else:
            dpe_letter = soup_details.find('span', class_='font-semibold text-2xl')
            if dpe_letter:
                dpe = dpe_letter.get_text(strip=True)

        ges_span = soup_details.find('span', string=lambda t: t and 'gaz à effet de serre' in t.lower())
        if ges_span:
            ges = ges_span.get_text(strip=True)
        else:
            ges_letter_candidates = soup_details.find_all('span', class_='font-semibold text-2xl')
            if len(ges_letter_candidates) >= 2:
                ges = ges_letter_candidates[1].get_text(strip=True)

        # --- Photos / Visite virtuelle -> juste le nombre ---
        nb_photos = '0'
        photos_span = soup_details.find('span', string=lambda t: t and 'photo' in t.lower())
        if photos_span:
            m = re.search(r'\d+', photos_span.get_text())
            if m: nb_photos = m.group(0)
        else:
            alt = soup_details.find(
                lambda tag: tag.name in ['div', 'p', 'li']
                and tag.get_text(strip=True)
                and 'photo' in tag.get_text(strip=True).lower()
            )
            if alt:
                m = re.search(r'\d+', alt.get_text())
                if m: nb_photos = m.group(0)

        # --- Push ---
        annonces_data.append({
            'Ville demandée (slug)': ville,
            'Titre': titre,
            'Prix': prix,
            'Surface': surface,
            'Pièces': pieces,
            'Lien': lien_complet,
            'Image': image,
            'Agent': agent,
            'Description': description,
            'Nombre de lots': copro_details.get('Nombre de lots', 'Non renseigné'),
            'Charges prévisionnelles': copro_details.get('Charges prévisionnelles', 'Non renseigné'),
            'Procédure en cours': copro_details.get('Procédure en cours', 'Non renseigné'),
            'Prix de vente': prix_vente,
            'Prix au m²': prix_m2,
            'DPE': dpe,
            'GES': ges,
            'Nombre de photos': nb_photos,
            'Note': note_value,
            'Avis': avis_value
        })

        # retour à la liste (pour la carte suivante)
        if lien_complet.startswith('http'):
            driver.back()
            time.sleep(1)

    return len(annonces)


# --- Run multi-villes + pagination ---
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
try:
    for ville in villes:
        print(f"\n=== Extraction pour {ville} ===")
        page = 1
        while True:
            print(f"  → Page {page} ...")
            try:
                count = get_annonces_from_page(driver, ville, page)
            except Exception as e:
                print(f"    !! Erreur page {page}: {e}")
                break
            if count == 0:
                print(f"  Fin de pagination pour {ville} (aucune annonce).")
                break
            page += 1
finally:
    driver.quit()

# --- DataFrame & petits nettoyages utiles ---
df = pd.DataFrame(annonces_data)

# Convertir en numériques quand possible
df["Note_num"] = pd.to_numeric(df["Note"], errors="coerce")
df["Avis_num"] = pd.to_numeric(df["Avis"], errors="coerce")
df["Photos_num"] = pd.to_numeric(df["Nombre de photos"], errors="coerce")


print("Extraction terminée !")

print(f"\nExtraction terminée : {len(df)} annonces collectées.")

# Exporter les données vers un fichier CSV
df.to_csv('Donnees_brutes_annonces_immobilieres.csv', index=False, encoding='utf-8-sig', sep=';')
print("Les données ont été exportées vers 'annonces_immobilieres.csv' avec un encodage adapté.")
