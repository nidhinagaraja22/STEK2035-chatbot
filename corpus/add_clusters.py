# -*- coding: utf-8 -*-
"""
Fold the 6 official STEK 2035 cluster pages (scraped from heidelberg.de) INTO the
canonical corpus (corpus_v2). These describe the city's official development
goals and were missed in the original scrape.

Reconciliation note (2026-09): the clusters were previously kept in a separate
corpus_v2_1. They are now appended directly onto the cleaned corpus_v2 so the
project has ONE corpus. Clusters are appended at the END, so the existing
embedding rows stay aligned to the first N chunks; only the appended rows are
new. This script is idempotent (re-running does nothing once clusters exist) and
backs up the pre-cluster corpus before writing.

Authority: these pages present the official STEK strategy goals directly, so they
are tagged Level 1 (official strategy).

AFTER RUNNING: RE-EMBED the corpus (chunk count changed) before retrieval — see
the printed instructions.
"""
import json
import re
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CORPUS = BASE / "corpus/corpus_v2/corpus_v2_chunks.jsonl"        # canonical corpus (in place)
BACKUP = BASE / "corpus/corpus_v2/corpus_v2_chunks_precluster.jsonl"
CHANGELOG = BASE / "corpus/corpus_v2/clusters_changelog.md"

AUTHORITY_LABELS = {1: "Official STEK strategy (approved policy)"}
BASE_URL = "https://www.heidelberg.de/HD/Rathaus/cluster+{n}.html"

# clean verbatim German body per cluster (nav/contact/footer stripped)
CLUSTERS = {
1: ("Dynamische Entwicklung + Sparsame Flächennutzung",
"Die Stadt soll wachsen und sich entwickeln. Attraktive Arbeits- und Bildungsangebote und hohe Lebensqualität ziehen viele Menschen, Unternehmen, Einrichtungen und Institutionen nach Heidelberg. Sie sollen aktiv gefördert werden. Trotz dieses Wachstums geht die Stadt weiterhin sparsam mit der Ressource Fläche um und trägt so auch zukünftig dem Umweltschutz Rechnung. Zusätzlich benötigte Flächen für Wohnungsbau, wirtschaftliche Entwicklung und soziale Infrastruktur werden im Bestand sowie an neuen Standorten in Heidelberg und in Kooperation mit der Region bereitgestellt und entwickelt. Das Bauen auf bereits versiegelten Flächen und die Nutzung bestehender Gebäude stehen im Fokus. Umsetzungsprogramme für Wirtschaft und Wohnen stellen sicher, dass jederzeit attraktive Entwicklungsflächen für Wirtschaft, Wissenschaft und Wohnen zur Verfügung stehen. Weithin sichtbares Beispiel für nachhaltige Stadtentwicklung ist die Konversionsfläche des Patrick-Henry-Villages (PHV): Im Zentrum der Metropolregion wächst ein besonderer Ort der Innovation. Es entsteht ein Explo-Areal, ein Experimentierfeld, in dem Forschende, Beschäftigte und Bewohnerinnen und Bewohner im Alltag des Stadtteils gemeinsam neue Lösungen entwickeln und erproben können. Heidelberg spielt dabei insbesondere seine Stärken als Spitzenstandort für Gesundheit und Lebenswissenschaften aus. Die Entwicklung im Außenbereich konzentriert sich auf die im Flächennutzungsplan ausgewiesenen Entwicklungsflächen. Die grünen Stadtkonturen setzen klare Grenzen für den Siedlungsraum und bleiben unangetastet, um wertvolle Freiflächen zu bewahren. Für die künftige Stadtentwicklung liegt der Fokus auf dem Innenbereich. Hier verfolgt Heidelberg eine Strategie der mehrfachen Innenentwicklung, die mit besonderer Sorgfalt umgesetzt wird, um eine nachhaltige und ausgewogene Nutzung sicherzustellen. Das Stadtentwicklungskonzept ist eng mit dem Modell Räumliche Ordnung verzahnt und die Inhalte sind aufeinander abgestimmt."),
2: ("Freiraum Nutzung + Freiraum Schutz",
"Der Stadtwald und das grüne Neckartal bilden eine weltweit berühmte Kulturlandschaft. Ihre vielfältige Natur bietet für alle Heidelbergerinnen und Heidelberger vor der Haustür Möglichkeiten zur Erholung und Freizeitgestaltung. Heidelberg verbindet Freiraumschutz und -entwicklung. Gemeinsam mit den Nachbarkommunen werden die Grünzüge als verbindender und erlebbarer Landschaftsraum mit Neckar, Odenwald und Rheinebene erhalten mit Zugang aus allen Stadtteilen. Der Biotopverbund spielt eine wichtige Rolle, um Schutzgebiete zu vernetzen. Gleichzeitig wird das Gebiet der Metropolregion Rhein-Neckar für Landwirtschaft, Energieversorgung und technische Infrastrukturen benötigt. Um die Zerschneidung der Landschaft zu minimieren, setzt sich Heidelberg dafür ein, dass neue Schienenwege und Energiekorridore gebündelt neben bereits vorhandenen Infrastrukturtrassen verlaufen. Das Stadtentwicklungskonzept ist eng mit der kommunalen Biotopverbundplanung verknüpft; die Inhalte beider Planwerke sind aufeinander abgestimmt. Im Rahmen der Biotopverbundplanung werden unter anderem zentrale Schwerpunktbereiche für Maßnahmen sowie bedeutende Verbundachsen identifiziert."),
3: ("Inklusives und soziales Miteinander + Individuell geprägte Orte",
"In Heidelberg wird soziales Miteinander gelebt. Keimzelle hierfür sind die Stadtteile. Für den täglichen Bedarf, die Gesundheitsversorgung und Freizeit ist ebenso gesorgt wie für bezahlbaren, barrierefreien Wohnraum und niederschwellige Begegnungsorte. Kurze Wege gibt es auch zu den vielfältigen Bildungseinrichtungen. Sie werden gestärkt und vernetzt, um gute Startchancen und lebenslanges Lernen für alle zu ermöglichen – von Kindertagesstätten über Schulen bis zu Erwachsenenbildung. Die Stadt bekennt sich klar zu Chancengleichheit, Inklusion, Integration, Vielfalt und gleichberechtigter Teilhabe. Vereinbarkeit von Privat- und Berufsleben sowie die Fähigkeit zur Selbstbestimmung werden gefördert. Die unterschiedlichen Charaktere der Stadtteile und ihre individuell geprägten Gebiete bestimmen die lebendige Vielfalt der gesamten Stadt. Die Entwicklung der Stadtteile mit Räumen für Wohnen, Wirtschaft und Handel, Wissenschaft, Kultur, Sport und Soziales sorgt für eine lebenswerte Stadt. Heidelberg versteht sich als Stadt der kurzen Wege. Das Ziel des STEK besteht darin, eine vielfältige Nutzung insbesondere in den Zentren und zentralen Bereichen der Stadt zu fördern. Im Sinne der Stadt der kurzen Wege werden Wohnen, Arbeiten, Einkaufen und Begegnungsorte eng miteinander verknüpft. Gleichzeitig wird großer Wert auf die individuelle Identität einzelner Standorte gelegt. So werden Gewerbegebiete in ihrer Funktion gezielt gestärkt, um ihre Beständigkeit zu sichern und sie vor Verdrängung durch andere Nutzungen zu schützen. Auch der Wissenschaft wird ausreichender Raum für Entwicklung eingeräumt, um ihre Innovationskraft und Wettbewerbsfähigkeit langfristig zu gewährleisten."),
4: ("Energie- und Mobilitätswende + Teilhabe an Veränderungen",
"Heidelberg als eine der jüngsten Städte Deutschlands hat die nächsten Generationen im Blick und macht große Schritte auf dem Weg zur Klimaneutralität. Dafür werden notwendige Investitionen in die Energie- und Mobilitätswende getätigt und Flächen für die nachhaltige Energieproduktion im regionalen Kontext entwickelt – etwa für Windkraft im Odenwald, Geothermie, Flusswärme und Abwasserwärmenutzung. Heidelberg will die Kreislaufwirtschaft im Baubereich stärken. Bildung für Nachhaltigkeit informiert und sensibilisiert für den Klimaschutz. Bürgerinnen und Bürger sollen weitere Angebote bekommen, um sich auch finanziell an der Energiewende beteiligen zu können. Das Stadtentwicklungskonzept ist eng mit dem Verkehrsentwicklungsplan/Klimamobilitätsplan (VEP/KMP) verknüpft und die Inhalte sind aufeinander abgestimmt. In Heidelberg werden die Verkehrsströme in Nord-Süd- und Ost-West-Richtung neu geordnet, wobei der Individualverkehr auf die Ernst-Walz-Brücke, die Lessingstraße und die B 37 konzentriert wird, während in zentralen Bereichen wie der Bergheimer Straße, der Kurfürsten-Anlage und der heutigen B 3 der Umweltverbund im Fokus steht."),
5: ("International vernetzte Stadt + Lokale Bedürfnisse",
"Heidelberg schöpft seine Stärke aus Internationalität und Weltoffenheit. Wirtschaft und Arbeit, Wissenschaft und Forschung, Bildung, Kultur und soziales Miteinander sind die starken Motoren der dynamischen Stadtentwicklung, die Heidelberg auszeichnet. Menschen mit unterschiedlichen Qualifikationen und Ausbildungswünschen aus der ganzen Welt werden gezielt angeworben. Das Miteinander von Bewohnerinnen und Bewohnern, die lange hier leben und denen, die neu zuziehen, wird niederschwellig unterstützt."),
6: ("Gesunde, resiliente und sichere Stadt + Stadt im Stress",
"Ein sicheres und gesundes Leben wird weiterhin gezielt gefördert – auch mit attraktiven Bewegungs- und Sportangeboten. Der gesellschaftliche Zusammenhalt und die Demokratie sollen gestärkt werden. Bürgerinnen und Bürger können sich auf vielfältige Weise einbringen, die digitale städtische Beteiligungsplattform baut diesen Weg weiter aus. Heidelberg reagiert auf herausfordernde Veränderungen wie den Klimawandel und globale Krisen mit Frühwarnsystemen und klimaangepasster Stadtentwicklung. Auch die Sicherstellung der öffentlichen Ordnung und der Schutz der Versorgungsinfrastruktur hat eine hohe Bedeutung."),
}


def chunk_words(text, size=250):
    w = text.split()
    return [" ".join(w[i:i + size]) for i in range(0, len(w), size)] or [text]


def build_cluster_chunks():
    out = []
    for n, (title, body) in CLUSTERS.items():
        body = re.sub(r"\s+", " ", body).strip()
        for ci, ch in enumerate(chunk_words(body)):
            out.append({
                "chunk_uid": f"website_html::cluster_{n}::{ci}",
                "source": "website_html",
                "origin": BASE_URL.format(n=n),
                "chunk_id": ci,
                "text": ch,
                "topics": [],
                "relevance_score": None,
                "lda_topic": None,
                "document_id": f"cluster_{n}",
                "document_title": f"STEK 2035 Cluster {n}: {title}",
                "source_url": BASE_URL.format(n=n),
                "publication_date": "2024",
                "doc_type": "official_strategy",
                "authority_level": 1,
                "authority_label": AUTHORITY_LABELS[1],
                "citation_prefix": f"{AUTHORITY_LABELS[1]} — STEK 2035 Cluster {n}: {title} (2024)",
                "is_citizen_opinion": False,
                "topic": None,
                "section": ci,
            })
    return out


def main():
    existing = [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines()]
    have = {c["chunk_uid"] for c in existing}
    clusters = build_cluster_chunks()

    if all(c["chunk_uid"] in have for c in clusters):
        print(f"Clusters already present in corpus_v2 ({len(existing)} chunks). Nothing to do.")
        return

    # back up the pre-cluster corpus once
    if not BACKUP.exists():
        shutil.copy2(CORPUS, BACKUP)
        print(f"Backed up pre-cluster corpus -> {BACKUP.name}")

    to_add = [c for c in clusters if c["chunk_uid"] not in have]
    combined = existing + to_add
    with CORPUS.open("w", encoding="utf-8") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    CHANGELOG.write_text(
        f"# corpus_v2 clusters changelog\n\n"
        f"- Base (cleaned corpus_v2): {len(existing)} chunks\n"
        f"- Added: {len(to_add)} official STEK cluster chunks "
        f"(heidelberg.de/HD/Rathaus/cluster+N.html), document_id cluster_1..6\n"
        f"- Total: {len(combined)} chunks\n"
        f"- Position: appended at the END (rows {len(existing)}..{len(combined) - 1}), "
        f"so existing embedding rows stay aligned to chunks 0..{len(existing) - 1}\n"
        f"- Authority: L1 (official strategy)\n"
        f"- ACTION REQUIRED: re-embed the corpus so all rows have embeddings.\n",
        encoding="utf-8")

    print(f"Added {len(to_add)} cluster chunks. corpus_v2: {len(existing)} -> {len(combined)}")
    print(f"Backup: {BACKUP.name}  |  changelog: {CHANGELOG.name}")
    print("\nNEXT — re-embed on a GPU box (needs sentence-transformers + torch):")
    print("  python stek_reembed_cleaned_corpus.py         # regenerates embeddings_v2_e5base.npy")
    print("  python stek_generate_e5large_bgem3.py         # e5-large + bge-m3 (point its CHUNKS_PATH at corpus_v2_chunks.jsonl)")


if __name__ == "__main__":
    main()
