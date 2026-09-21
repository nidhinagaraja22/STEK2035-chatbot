# -*- coding: utf-8 -*-
"""
Generate a DRAFT German Q&A benchmark set (>=100) across sectors, split into two
files: questions (no answers) + answers (reference answers, kept separate).

IMPORTANT: the reference answers are a DRAFT starting point. Per the professor's
requirement, recruited annotators must VERIFY and CORRECT them against the corpus
before they count as gold-standard ground truth. Answerability was sanity-checked
against corpus coverage, but individual answers are not yet verified.
"""
import json
from pathlib import Path

OUT = Path("eval")
OUT.mkdir(exist_ok=True)

# category, q_type, answerable, is_citizen_opinion, expected_authority, question, draft_answer
E = []
def add(cat, qt, ans, cit, auth, q, a):
    E.append({"category": cat, "q_type": qt, "answerable": ans, "is_citizen_opinion": cit,
              "expected_authority": auth, "question": q, "reference_answer": a})

ABSTAIN = "Die vorliegenden STEK-Dokumente enthalten keine ausreichenden Informationen, um diese Frage zu beantworten."

# ---------------- Wohnen ----------------
add("Wohnen","policy",True,False,1,"Wie schafft Heidelberg bezahlbaren Wohnraum?","Durch ein Wohnungsentwicklungsprogramm, Förderprogramme, Belegungsrechte und die städtische GGH sowie das Bündnis für bezahlbaren Wohnraum.")
add("Wohnen","policy",True,False,2,"Welche Rolle spielt die GGH beim Wohnungsbau?","Die städtische Wohnungsbaugesellschaft GGH errichtet und bewirtschaftet geförderten und preisgünstigen Wohnraum in Heidelberg.")
add("Wohnen","factual",True,False,2,"Was ist das Bündnis für bezahlbaren Wohnraum?","Ein 2013 von der Stadt initiiertes Bündnis, das Akteure zusammenbringt, um bezahlbaren Wohnraum in Heidelberg zu sichern.")
add("Wohnen","policy",True,False,2,"Wie geht Heidelberg gegen überhöhte Mieten vor?","Seit Herbst 2024 lässt die Stadt Online-Wohnungsanzeigen auf überhöhte Mieten prüfen; Betroffene können überhöhte Mieten über ein Formular melden.")
add("Wohnen","factual",True,False,2,"Was ist der Wohnberechtigungsschein und wo bekommt man ihn?","Ein Nachweis für den Zugang zu geförderten Wohnungen, den das Bürgeramt ausstellt.")
add("Wohnen","policy",True,False,3,"Was sind gemeinschaftliche Wohnprojekte in Heidelberg?","Von der Stadt unterstützte Wohnformen, bei denen Gruppen gemeinsam Wohnraum planen und bewohnen; Infos bietet die Wohnplattform.")
add("Wohnen","policy",True,False,1,"Wie wird preisgünstiger Wohnraum auf neuen Flächen gesichert?","Auf Konversionsflächen und in der Bahnstadt hat die Entwicklung preisgünstigen Wohnraums Priorität.")
add("Wohnen","cross-document",True,False,2,"Welche Förderprogramme gibt es im Bereich Wohnen?","Förderungen für Familien, Baugruppen sowie barrierefreien oder umweltbewussten Ausbau.")
add("Wohnen","factual",True,True,4,"Was wünschen sich Bürgerinnen und Bürger beim Thema Wohnen?","Laut Beteiligung wünschen sich Bürger mehr bezahlbaren und familiengerechten Wohnraum sowie den Erhalt von Grün in Wohngebieten.")

# ---------------- Mobilität ----------------
add("Mobilität","policy",True,False,1,"Wie wird der ÖPNV in Heidelberg ausgebaut?","Durch bessere Anbindung neuer Quartiere und Verknüpfung mit Rad- und Fußverkehr im Rahmen des Klimamobilitätsplans 2035.")
add("Mobilität","policy",True,False,1,"Wie fördert Heidelberg den Radverkehr?","Durch Ausbau des Radwegnetzes, Radschnellwege, bessere Abstellanlagen und die RadKULTUR-Initiative.")
add("Mobilität","factual",True,False,2,"Was ist der Klimamobilitätsplan 2035?","Ein Plan, der die Verkehrswende in Heidelberg hin zu klimafreundlicher Mobilität steuert.")
add("Mobilität","policy",True,False,1,"Wie soll der Parkraum in der Innenstadt entwickelt werden?","In verkehrsberuhigten Bereichen und der Innenstadt sollen Parkflächen zugunsten von Aufenthaltsqualität reduziert werden.")
add("Mobilität","policy",True,False,2,"Was tut Heidelberg für barrierefreie Mobilität?","Die Stadt verfolgt barrierefreie Wege, abgesenkte Bordsteine und barrierefreie Haltestellen für alle Verkehrsteilnehmenden.")
add("Mobilität","factual",True,False,2,"Was ist die RadKULTUR-Initiative?","Eine Initiative zur Förderung der Fahrradkultur und zur Steigerung des Radverkehrsanteils in Heidelberg.")
add("Mobilität","policy",True,False,2,"Wie sollen Fußgänger in der Stadt gefördert werden?","Durch freie Gehwege, verkehrsberuhigte Bereiche und mehr Aufenthaltsqualität im öffentlichen Raum.")
add("Mobilität","factual",True,False,2,"Was zeigt die Mobilitätsbefragung SrV 2023?","Eine repräsentative Erhebung des Verkehrsverhaltens in Heidelberg (Steckbrief SrV 2023).")
add("Mobilität","factual",True,True,4,"Was wünschen sich Bürger zur Mobilität in Heidelberg?","Laut Beteiligung wünschen sich Bürger besseren ÖPNV, mehr sichere Radwege und weniger Autoverkehr in der Stadt.")

# ---------------- Umwelt/Klima ----------------
add("Umwelt","policy",True,False,1,"Wie reagiert Heidelberg auf Klimawandel und Hitzewellen?","Durch Frischluftkorridore, Dach- und Fassadenbegrünung, mehr Stadtbäume und entsiegelte, wasserdurchlässige Böden.")
add("Umwelt","policy",True,False,2,"Wie schützt die Stadt Grünflächen und Bäume?","Durch Biotoppflege, Biotopvernetzung, Artenschutzprogramme, die Baumschutzsatzung und den Erhalt des Stadtwalds.")
add("Umwelt","factual",True,False,2,"Was regelt die Baumschutzsatzung?","Sie regelt den Schutz und Erhalt von Bäumen im Stadtgebiet Heidelberg.")
add("Umwelt","policy",True,False,2,"Was tut Heidelberg für den Klimaschutz?","Heidelberg verfolgt Klimaneutralität, u. a. über den Masterplan Klimaschutz und kommunale Wärmeplanung.")
add("Umwelt","factual",True,False,2,"Was ist an der Bahnstadt aus Umweltsicht besonders?","Die Bahnstadt ist die größte Passivhaussiedlung der Welt und erhielt 2014 den Passive House Award.")
add("Umwelt","policy",True,False,1,"Welche Bedeutung hat der Neckar für die Stadtentwicklung?","Der Neckar ist ein zentraler Frei- und Erholungsraum, der durch bessere Zugänglichkeit und Uferpflege gestärkt werden soll.")
add("Umwelt","policy",True,False,2,"Wie geht Heidelberg mit Hochwasser um?","Durch Hochwasservorsorge, Freihaltung von Überschwemmungsflächen und wassersensible Stadtgestaltung.")
add("Umwelt","factual",True,False,2,"Was ist der Erholungswald in Heidelberg?","Der Heidelberger Stadtwald ist als Erholungswald zertifiziert und dient Naherholung und Naturschutz.")
add("Umwelt","policy",True,False,2,"Wie fördert Heidelberg biologische Vielfalt?","Über Artenschutzprogramme, Offenland-Biotopkartierung, Schutzgebiete und Biotopvernetzung.")
add("Umwelt","factual",True,True,4,"Welche Umweltthemen sind Bürgern besonders wichtig?","Laut Beteiligung sind Bürgern Erhalt von Grünflächen, Bäumen, Neckarwiese und Klimaanpassung besonders wichtig.")

# ---------------- Wirtschaft ----------------
add("Wirtschaft","policy",True,False,1,"Wie stärkt Heidelberg den Wirtschaftsstandort?","Durch Förderung von Innovation, Wissenschaft und Gewerbeflächen sowie regionale Vernetzung in der Metropolregion.")
add("Wirtschaft","factual",True,False,2,"Welche Rolle spielt die Wissenschaft für Heidelbergs Wirtschaft?","Universität und Forschungseinrichtungen prägen Heidelberg als Wissenschafts- und Innovationsstandort.")
add("Wirtschaft","policy",True,False,2,"Wie geht Heidelberg mit Gewerbeflächen um?","Die Stadt sichert und entwickelt Gewerbe- und Innovationsflächen im Rahmen einer flächensparenden Entwicklung.")
add("Wirtschaft","policy",True,False,2,"Wie verbindet Heidelberg Wirtschaft und Nachhaltigkeit?","Ziel ist eine wirtschaftlich starke und zugleich klimaneutrale und ressourcenschonende Stadtentwicklung.")
add("Wirtschaft","factual",True,False,2,"Was bedeutet die Metropolregion Rhein-Neckar für Heidelberg?","Sie bildet den regionalen Kooperationsraum für Wirtschaft, Verkehr und Planung, in den Heidelberg eingebunden ist.")
add("Wirtschaft","policy",True,False,2,"Wie soll Innovation in Heidelberg gefördert werden?","Durch innovative Arbeitsorte, Wissenstransfer und die Verbindung von Forschung und Wirtschaft.")
add("Wirtschaft","cross-document",True,False,1,"Wie sollen Wohnen und Arbeiten in Heidelberg ausbalanciert werden?","Ziel ist eine gemischte Stadt mit kurzen Wegen, in der Wohnen und Arbeiten räumlich ausgeglichen werden.")
add("Wirtschaft","factual",True,True,4,"Welche wirtschaftlichen Anliegen nennen Bürger?","Laut Beteiligung nennen Bürger u. a. faire Arbeitsbedingungen, lokale Wirtschaft und wohnortnahe Arbeitsplätze.")

# ---------------- Kultur ----------------
add("Kultur","policy",True,False,2,"Welche Bedeutung hat Kultur für Heidelberg?","Kulturelle Vielfalt, Theater, Museen und internationales Flair werden als prägende Merkmale Heidelbergs gefördert.")
add("Kultur","factual",True,False,2,"Was macht die Heidelberger Altstadt einzigartig?","Historische Bauten wie Schloss, Alte Brücke und Universität prägen die Altstadt und stehen unter Denkmalschutz.")
add("Kultur","policy",True,False,2,"Wie schützt Heidelberg sein kulturelles Erbe?","Durch Denkmalschutz und sorgsame Sanierung historischer Gebäude und Ensembles.")
add("Kultur","policy",True,False,2,"Wie fördert die Stadt kulturelle Veranstaltungen?","Durch Unterstützung von Kulturangeboten, Veranstaltungen und Einrichtungen wie Theater und Museen.")
add("Kultur","factual",True,False,2,"Welche Rolle spielt die Universität für Heidelbergs Identität?","Die renommierte Universität und die Wissenschaft prägen Heidelberg als international bekannte Stadt.")
add("Kultur","policy",True,False,2,"Wie wird kulturelle Vielfalt in Heidelberg unterstützt?","Durch Förderung von Offenheit, internationaler Gemeinschaft und Teilhabe unterschiedlicher Gruppen.")
add("Kultur","cross-document",True,False,1,"Wie verbindet das STEK Kultur und Stadtentwicklung?","Kultur gilt als Standortfaktor und Teil einer lebenswerten, vielfältigen Stadtentwicklung.")
add("Kultur","factual",True,True,4,"Was macht Heidelberg für Bürger einzigartig?","Laut Beteiligung nennen Bürger historisches Erbe, Natur am Neckar, kulturelle Vielfalt und die Universität.")

# ---------------- Soziales ----------------
add("Soziales","policy",True,False,1,"Was wird für Familien und ältere Menschen angeboten?","Mehr Kinderbetreuung, kinderfreundliche Räume, altersgerechtes Wohnen und barrierefreie Infrastruktur.")
add("Soziales","policy",True,False,2,"Wie fördert Heidelberg soziale Teilhabe?","Durch soziale Infrastruktur, Angebote für vulnerable Gruppen und Maßnahmen für Integration und Zusammenhalt.")
add("Soziales","policy",True,False,2,"Was tut die Stadt für Kinderbetreuung?","Ausbau von Kindertagesbetreuung und kinderfreundlicher Infrastruktur als Teil der sozialen Daseinsvorsorge.")
add("Soziales","policy",True,False,2,"Wie unterstützt Heidelberg Integration?","Durch Angebote für internationale Gemeinschaft, Teilhabe und Maßnahmen gegen Diskriminierung.")
add("Soziales","policy",True,False,2,"Wie sorgt Heidelberg für Bildungsgerechtigkeit?","Über Schulen, Bildungsangebote und den Anspruch, alle Menschen auf dem Weg des Wandels mitzunehmen.")
add("Soziales","policy",True,False,2,"Was bedeutet altersgerechtes Wohnen in Heidelberg?","Barrierefreie und seniorengerechte Wohn- und Infrastrukturangebote für ältere Menschen.")
add("Soziales","cross-document",True,False,1,"Wie will Heidelberg niemanden im Wandel zurücklassen?","Ziel ist eine sozial gerechte Stadtentwicklung, die vulnerable Gruppen besonders berücksichtigt.")
add("Soziales","factual",True,False,2,"Was versteht das STEK unter sozialem Zusammenhalt?","Gesellschaftliche Stabilität, Teilhabe und ein gutes Miteinander in den Stadtteilen.")
add("Soziales","factual",True,True,4,"Auf welche Menschen muss die Stadt laut Bürgern besonders achten?","Laut Beteiligung nennen Bürger vulnerable Gruppen wie ältere Menschen, Kinder und sozial Benachteiligte.")

# ---------------- Stadtentwicklung ----------------
add("Stadtentwicklung","factual",True,False,2,"Was ist die Bahnstadt und welche Rolle spielt sie?","Ein neues Quartier auf ehemaliger Bahnfläche, die größte Passivhaussiedlung der Welt und Vorbild nachhaltiger Entwicklung.")
add("Stadtentwicklung","factual",True,False,1,"Was passiert mit den Konversionsflächen?","Konversionsflächen wie Patrick-Henry-Village werden zu gemischten Quartieren mit Wohnen und Gewerbe entwickelt.")
add("Stadtentwicklung","factual",True,False,2,"Was ist das Modell Räumliche Ordnung (MRO)?","Ein informelles strategisches Planungsinstrument, das die räumliche Entwicklung Heidelbergs bis 2035+ steuert.")
add("Stadtentwicklung","policy",True,False,1,"Wie soll die Innenstadt weiterentwickelt werden?","Durch Stärkung von Aufenthaltsqualität, Grün, Handel und Kultur bei Erhalt der historischen Struktur.")
add("Stadtentwicklung","policy",True,False,1,"Wie geht Heidelberg mit dem Flächenverbrauch um?","Ziel ist Netto-Null-Flächenverbrauch durch Innenentwicklung, Nachverdichtung und Entsiegelung.")
add("Stadtentwicklung","policy",True,False,2,"Wie sollen Stadtteile gestärkt werden?","Durch wohnortnahe Angebote, kurze Wege und lebendige, gemischte Quartiere.")
add("Stadtentwicklung","factual",True,False,2,"Was ist das Ziel 'gemischte Stadt'?","Eine Stadt mit Mischung aus Wohnen, Arbeiten und Versorgung und kurzen Wegen für alle.")
add("Stadtentwicklung","cross-document",True,False,1,"Wie hängen MRO und STEK 2035 zusammen?","Das MRO liefert die räumliche Strategie, das STEK 2035 den übergreifenden Wegweiser nachhaltiger Stadtentwicklung.")
add("Stadtentwicklung","factual",True,True,4,"Was soll aus Bürgersicht in Heidelberg unbedingt erhalten bleiben?","Laut Beteiligung Grünflächen, Neckarwiese, Altstadt, Wälder und der besondere Charakter der Stadtteile.")

# ---------------- Bürgerbeteiligung ----------------
add("Bürgerbeteiligung","factual",True,False,4,"Wie können sich Bürger am STEK beteiligen?","Über Online-Beteiligung, aufsuchende Formate an Festen und Märkten sowie thematische Arbeitstreffen.")
add("Bürgerbeteiligung","factual",True,False,4,"Was sind aufsuchende Beteiligungsformate?","Formate, bei denen die Stadt Menschen direkt im öffentlichen Raum anspricht, z. B. mit einem digitalen Glücksrad.")
add("Bürgerbeteiligung","factual",True,False,4,"Was ist die Online-Beteiligung zum STEK 2035?","Eine digitale Beteiligung, bei der Bürger zu Zukunftsfragen Beiträge einreichen konnten.")
add("Bürgerbeteiligung","factual",True,False,4,"Was sind die Arbeitskreise (AK) zum STEK?","Arbeitskreise, in denen in Sitzungen thematische Grundlagen des STEK erarbeitet und dokumentiert wurden.")
add("Bürgerbeteiligung","policy",True,False,2,"Warum ist Beteiligung für das STEK wichtig?","Beteiligung bindet Wissen und Anliegen der Stadtgesellschaft ein und stärkt die Legitimität der Planung.")
add("Bürgerbeteiligung","cross-document",True,False,4,"Welche Themen wurden in der Beteiligung besonders oft kommentiert?","Laut Dokumentation u. a. Freiraum, Netto-Null-Flächenverbrauch, lebenswerte und inklusive Stadt.")
add("Bürgerbeteiligung","factual",True,False,4,"Wie wurden Bürgerbeiträge geprüft und veröffentlicht?","Beiträge wurden auf diskriminierende oder beleidigende Aussagen geprüft und dann auf der Webseite veröffentlicht.")

# ---------------- Sicherheit ----------------
add("Sicherheit","policy",True,False,2,"Was tut Heidelberg für die Sicherheit im öffentlichen Raum?","Maßnahmen für Sicherheitsgefühl, Beleuchtung und die Vermeidung von Angsträumen.")
add("Sicherheit","factual",True,True,5,"Wo fühlen sich Menschen in Heidelberg unsicher?","Laut Bürgerbeiträgen werden Orte wie Bismarckplatz und die Neckarwiese nachts teils als unsicher empfunden.")
add("Sicherheit","policy",True,False,2,"Wie soll das Sicherheitsgefühl verbessert werden?","Durch bessere Beleuchtung, belebte öffentliche Räume und Gestaltung sicherer Wege.")
add("Sicherheit","factual",True,True,5,"Was sagen Bürger zur Sicherheit in der Innenstadt?","Einige Bürger wünschen sich mehr Sicherheit und weniger Angsträume in der Innenstadt und Altstadt.")
add("Sicherheit","cross-document",True,False,2,"Wie hängt Sicherheit mit der Gestaltung öffentlicher Räume zusammen?","Belebte, gut beleuchtete und einsehbare Räume erhöhen das Sicherheitsempfinden.")

# ---------------- Digitalisierung ----------------
add("Digitalisierung","policy",True,False,2,"Wie treibt Heidelberg die Digitalisierung voran?","Durch digitale Verwaltungsangebote und die Weiterentwicklung als digitale, smarte Stadt.")
add("Digitalisierung","policy",True,False,2,"Welche digitalen Behördenangebote gibt es?","Zahlreiche Behördengänge sollen digitalisiert werden, ergänzt durch persönliche Ansprechpartner für komplexe Fragen.")
add("Digitalisierung","cross-document",True,False,2,"Wie verbindet Heidelberg Digitalisierung und Beteiligung?","Digitale Beteiligungsformate wie die Online-Beteiligung ermöglichen breite Mitwirkung.")
add("Digitalisierung","factual",True,True,4,"Was wünschen Bürger im Bereich Digitalisierung?","Laut Beteiligung wünschen Bürger mehr digitale Behördengänge bei gleichzeitigem Erhalt persönlicher Ansprechpartner.")
add("Digitalisierung","policy",True,False,2,"Sollen persönliche Ansprechpartner trotz Digitalisierung erhalten bleiben?","Ja, neben digitalen Angeboten sollen weiterhin Menschen als Ansprechpartner für komplexe Fragen verfügbar sein.")

# ---------------- Strategie/STEK ----------------
add("Strategie","factual",True,False,1,"Was ist das STEK 2035?","Das Stadtentwicklungskonzept 2035 ist der Wegweiser für eine nachhaltige Entwicklung Heidelbergs.")
add("Strategie","policy",True,False,1,"Welche Nachhaltigkeitsziele verfolgt das STEK 2035?","Eine klimaneutrale, sozial gerechte und wirtschaftlich starke Stadtentwicklung, dokumentiert im Nachhaltigkeitsbericht 2025.")
add("Strategie","factual",True,False,1,"An welchen übergeordneten Zielen orientiert sich das STEK?","Es orientiert sich an den Nachhaltigkeitszielen der Vereinten Nationen.")
add("Strategie","factual",True,False,2,"Was ist der Statusbericht zum STEK 2035?","Ein Bericht, der Auswirkungen jüngster Krisen und Ereignisse auf die Stadtentwicklung dokumentiert.")
add("Strategie","cross-document",True,False,1,"Welche großen Herausforderungen adressiert das STEK 2035?","U. a. Klimawandel, Flächenknappheit, bezahlbares Wohnen, sozialer Zusammenhalt und Mobilitätswende.")
add("Strategie","factual",True,False,1,"Welchen Zeithorizont hat das STEK?","Das STEK blickt auf das Zieljahr 2035 und teils darüber hinaus.")

# ---------------- Unanswerable (not in corpus) ----------------
UA = [
 ("Wohnen","Wie hoch ist das genaue Budget der Stadt für Wohnungsbau im Jahr 2026?"),
 ("Mobilität","Wie viele Kilometer Radweg wurden 2025 exakt neu gebaut?"),
 ("Wirtschaft","Wie hoch ist die Arbeitslosenquote in Heidelberg im September 2026?"),
 ("Strategie","Welche konkreten Beschlüsse fasst der Gemeinderat im Dezember 2026?"),
 ("Kultur","Wie viele Besucher hatte das Heidelberger Schloss im Jahr 2025 genau?"),
 ("Umwelt","Wie viele Bäume wird die Stadt bis 2040 exakt pflanzen?"),
 ("Soziales","Wie viele Kita-Plätze fehlen in Heidelberg aktuell genau?"),
 ("Stadtentwicklung","Wann genau wird Patrick-Henry-Village vollständig fertiggestellt?"),
 ("Wirtschaft","Welche Unternehmen ziehen 2027 nach Heidelberg?"),
 ("Mobilität","Wie hoch werden die Parkgebühren in der Altstadt 2027 sein?"),
 ("Strategie","Wie schneidet Heidelberg im Vergleich zu München bei CO2-Emissionen ab?"),
 ("Soziales","Wie viele Sozialwohnungen wird die GGH bis 2035 exakt bauen?"),
]
for cat,q in UA:
    add(cat,"unanswerable",False,False,None,q,ABSTAIN)

# ---------------- Vague (need clarification) ----------------
CLARIFY = "Diese Frage ist zu allgemein. Bitte präzisieren – z. B. Wohnen, Mobilität, Wirtschaft, Umwelt, Kultur oder Soziales?"
VG = [
 "Erzähl mir etwas über Heidelberg.",
 "Was ist geplant?",
 "Was passiert mit der Stadt?",
 "Was sagt das STEK?",
 "Worum geht es hier?",
 "Was ist wichtig?",
 "Kannst du mir helfen?",
 "Was gibt es Neues?",
]
for q in VG:
    add("Vage","vague",False,False,None,q,CLARIFY)

# ------------------------------------------------------------------ write out
questions, answers = [], []
for i, e in enumerate(E, 1):
    qid = f"g{i:03d}"
    questions.append({"id": qid, "category": e["category"], "q_type": e["q_type"],
                      "answerable": e["answerable"], "question": e["question"]})
    answers.append({"id": qid, "reference_answer": e["reference_answer"],
                    "expected_authority": e["expected_authority"],
                    "is_citizen_opinion": e["is_citizen_opinion"],
                    "status": "DRAFT - verify against corpus"})

(OUT / "questions_100.json").write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "answers_100.json").write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")

# stats
import collections
cat = collections.Counter(e["category"] for e in E)
qt = collections.Counter(e["q_type"] for e in E)
print(f"Total Q&A: {len(E)}")
print("By category:", dict(cat))
print("By type:", dict(qt))
print("Answerable:", sum(1 for e in E if e['answerable']), "| Unanswerable:", sum(1 for e in E if not e['answerable'] and e['q_type']=='unanswerable'), "| Vague:", qt['vague'])
print("Citizen-opinion questions:", sum(1 for e in E if e['is_citizen_opinion']))
print("\nWrote eval/questions_100.json + eval/answers_100.json")
