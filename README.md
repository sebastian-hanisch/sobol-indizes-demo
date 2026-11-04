# Sobol-Indizes: welche Eingabe treibt die Varianz? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-sobol-indizes-demo.streamlit.app/)** -->

Zehntes Stück der Linie **„Monte-Carlo-Schätzung und Varianzreduktion“** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (die Linie wird erst vollständig veröffentlicht). Nachfolger von [quasi-monte-carlo-demo](https://github.com/sebastian-hanisch/quasi-monte-carlo-demo) (Stück 9, dort die exakte ANOVA und die mittlere Dimension): Hier werden dieselben Größen aus Modellläufen **geschätzt**. Der Pick-Freeze-Entwurf liefert Erst- und Gesamtindizes mit (p + 2) N Läufen; gemessen wird, wie genau, für welche Indexgröße und zu welchem Preis. Vehikel sind die Netz-Integranden der Reihe (Zusammenhang eines Gitternetzes oder einer Hantel, glatt oder als Sprung) und ein Begleiter exp(Σ aᵢuᵢ); alle Indizes sind exakt bekannt.

## Warum dieses Problem

Ein Sobol-Index sagt, welcher Anteil der Ausgabevarianz an einer Eingabe hängt: der **Erstordnungsindex** (die Eingabe allein) und der **Gesamtindex** (mit allen Wechselwirkungen). Wer ein teures Modell hat, will wissen, welche Eingaben sich zu messen lohnen. Die Kernfrage: **Wie genau und zu welchem Preis misst man, welche Eingabe die Varianz treibt, und wo geht ein Index im Rauschen unter?**

## Befunde und Korrekturen gegenüber dem Plan

Vor dem Bau wurden die Hypothesen der Gliederung an Messreihen geprüft (Vormessung im Website-Repo, `docs/planung/monte-carlo-planung/vormessung/`). Was sich geändert hat oder beim Bau dazukam:

- **„Kleine Indizes werden vom Rauschen verschluckt“ stimmt bei großem N nur für Erstordnungsindizes.** Der Fehler des Gesamtindex-Schätzers von Jansen wächst mit dem Index (er mittelt Quadrate kleiner Differenzen); der Fehler der Erstordnungsschätzer von Jansen und Janon hängt kaum vom Index ab, der von Saltelli wächst mit ihm. Bei kleinem N geht auch ein kleiner Gesamtindex unter (Gitter 3×3, T = 0,0659, N = 100: RMSE 0,093, größer als der Index).
- **Eine Abhilfe für kleine Erstordnungsindizes gibt es:** Owens Drei-Matrizen-Schätzer (2013), bei gleichen Läufen 2- bis 30-mal genauer für kleine Indizes; bei den gemessenen großen Indizes (0,53 bis 0,63) ist er schlechter als Jansen und Janon.
- **Bei wenigen diskreten Eingaben ist die Aufzählung billiger** als jede Stichprobe.
- **Die Näherung mit festem Nenner ist zu konservativ:** Sie macht den Erstordnungsbalken etwa 1,8-mal so breit (Überdeckung 0,99 statt 0,90); der Bootstrap muss den ganzen Schätzer neu rechnen.
- **Der Messwert „Anteil mit Varianz“ ist keine Verzerrung des Schätzers,** sondern ein eigener Befund bei seltenen Ausgaben (Gitter 3×4 bei p = 0,99).
- **Der Engpass-Integrand aus Stück 9 entfiel** (keine exakten Indizes); der glatte Integrand läuft nur bis 12 Kanten (Zeit), der Sprung bis 17.
- **Quasi-Zufallszahlen sind eine Option im Entwurf** (A und B als zwei Hälften eines 2p-dimensionalen Owen-Sobol-Punktes), kein eigenes Experiment.

## Modell

- **Indizes:** ANOVA f = Σ_S f_S, σ²_S = Var f_S; Erstordnung S_i = σ²_{i}/σ², Gesamt T_i = Σ_{S∋i} σ²_S/σ². Es gilt S_i ≤ T_i, ΣS_i ≤ 1 ≤ ΣT_i und **Σ T_i = mittlere Dimension** (Stück 9).
- **Pick-Freeze:** zwei unabhängige Matrizen A und B (N × p), AB_i = A mit Spalte i aus B; Läufe f(A), f(B), f(AB_1), …, f(AB_p): **(p + 2) N**; für die Gesamtindizes allein (p + 1) N.
- **Schätzer** (V = Varianz der 2N Werte f(A), f(B)): Saltelli 2010 mean(f_B (f_AB − f_A))/V; Jansen 1 − mean((f_B − f_AB)²)/(2V) (Erstordnung) und mean((f_A − f_AB)²)/(2V) (Gesamt); Janon/Monod (Korrelation von f_B und f_AB, symmetrisiert); Homma–Saltelli (nicht zentriert); Owen 2013 „Correlation 2“ mit drei Matrizen, (2p + 2) N Läufe.
- **Integranden:** *Sprung* = Netz hängt mit den Kanten {u_e < p} zusammen; *glatt* = Zuverlässigkeit bei unsicheren Haltewahrscheinlichkeiten p_e = 0,7 + 0,4 (u_e − ½) (ein Polynom); *Begleiter* exp(Σ aᵢuᵢ) mit Gewichten 1, 2/i² oder 1/i und d bis 32.
- **Exakte Indizes:** Netz-Integranden aus den ANOVA-Koeffizienten (Aufzählung aller 2^p Kantenmengen, Bernoulli-Basis), Begleiter in geschlossener Form. Mathematik im 📐-Expander der App.

## Methodik

Zufallszahlen aus festen SplitMix64-Strömen (nie ein einzelner Seed); A, B unabhängig oder als Hälften eines Owen-Sobol-Punktes (aus Stück 9). Genauigkeitskurve N = 100 bis 10 000 mit 100 Wiederholungen je N; Einzelmessungen mit 100 Wiederholungen. RMSE und Verzerrung je für den größten und den kleinsten exakten Index; „verschluckt“ heißt RMSE größer als der Index.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Wenn nicht anders angegeben: Pick-Freeze mit unabhängigen Zufallsmatrizen, RMSE über 100 Wiederholungen, ein einzelner RMSE ist auf etwa 10 bis 15 % genau (bei schweren Rändern wie der seltenen Ausgabe schlechter).

| Frage | Befund |
|---|---|
| Exakte Indizes, Hantel (Sprung p = 0,9) | Erstordnung: Brücke **0,631**, jede der sechs Dreieckskanten **0,0175** (Summe 0,736); Gesamt: Brücke **0,668**, Dreieckskanten **0,111** (Summe **1,336**) |
| Exakte Indizes, Gitter 3×3 (Sprung p = 0,9) | zwei Kantenklassen: 8 Kanten S = **0,0283**, T = **0,225**; 4 Kanten S = **0,0024**, T = **0,0659** (Summe S 0,2365, Summe T **2,067**) |
| Exakte Indizes, Begleiter aᵢ = 1/i, d = 8 | S von **0,630** bis **0,0100**, T von **0,658** bis **0,0113**; Summe S 0,967, Summe T **1,034** |
| Fehler gegen N, Hantel, RMSE größter Erstordnungsindex (Jansen), N = 100 / 300 / 1000 / 3000 / 10 000 | **0,111 / 0,056 / 0,032 / 0,019 / 0,010**: fällt mit 1/√N |
| **Kleinster Erstordnungsindex** des Gitters 3×3 (0,0024), N = 10 000 (140 000 Läufe) | RMSE Saltelli **0,0175**, Jansen **0,0134**, Janon **0,0108**: das **4,5- bis 7-Fache** des Index; **43 bis 50 %** der Schätzungen sind negativ |
| **Kleinster Gesamtindex**, RMSE bei N = 10 000, Jansen | Gitter 3×3 (T = 0,0659) **0,0075**; Hantel (T = 0,111) **0,0065**; Begleiter (T = 0,0113) **0,0002**; Schätzungen nie negativ |
| Nicht zentrierter Schätzer, Hantel, N = 1000 | Homma–Saltelli: größter S **0,150** gegen Jansen **0,032** (das 4,7-Fache); Gesamtindex **0,098** gegen **0,036** |
| **Schätzerwahl nach Indexgröße**, Begleiter, N = 1000, RMSE größter / kleinster S | Saltelli **0,113 / 0,0150**; Jansen **0,0208 / 0,0289**; Janon **0,0197 / 0,0294**: für den großen Index Jansen und Janon, für den kleinen Saltelli (am Begleiter; am Gitter 3×3 ist Saltelli am kleinsten Index der schlechteste der drei) |
| **Owens Schätzer bei gleichen Läufen** (N Saltelli = 1000, N Owen 538 bis 562), RMSE kleinster / größter S | Gitter 3×3 (0,0024 / 0,0283): **0,0086 / 0,0352** gegen Jansen 0,0432 / 0,0589; Begleiter (0,010 / 0,63): **0,0010 / 0,0538** gegen 0,0292 / 0,0202; glatte Hantel (0,0747 / 0,531): **0,0077 / 0,0477** gegen 0,0316 / 0,0197; Hantel (0,0175 / 0,631): **0,0152 / 0,0829** gegen 0,0362 / 0,0345 |
| dasselbe bei N Saltelli = 10 000, Verhältnis Jansen : Owen am kleinsten Index | Gitter 3×3 **3,9**, Begleiter **32**, Hantel **2,4** (bei N = 1000 gemessen); Anteil negativer Schätzungen am Gitter: Owen unter 5 %, Jansen 43 % |
| **Summen der Indizes**, N = 10 000, relativer RMSE der Summe | Σ T (Jansen): Gitter 3×3 **2,7 %** (RMSE 0,056 bei 2,067), Hantel **1,3 %**, Begleiter **1,1 %**; Σ S (Janon): Gitter **46 %** (0,109 bei 0,2365), Hantel 8 %, Begleiter 7,7 % |
| **Rangfolge**, Gitter 3×3 (8 Kanten T = 0,225 gegen 4 mit 0,0659), Jansen-Gesamtindex, N = 30 / 100 / 300 / 1000 / 3000 | Anteil der Läufe, in denen alle 8 wirksamen vor allen 4 anderen stehen: **0,00 / 0,04 / 0,43 / 0,97 / 1,00** (390 / 1300 / 3900 / 13 000 / 39 000 Läufe) |
| Rangfolge, Gitter 3×4 (8 obere gegen 9 untere Kanten) | **0,00 / 0,02 / 0,24 / 0,87 / 1,00** (540 bis 54 000 Läufe) |
| Rangfolge, Begleiter aᵢ = 1/i, d = 8 | die wichtigste Eingabe wird schon bei N = 30 (270 Läufe) in allen Läufen erkannt |
| **Seltene Ausgabe**, Gitter 3×4, p = 0,99 (P(Ausfall) = 4,2·10⁻⁴) | exakt: größter T **0,244**, kleinster 0,0026, **Σ S nur 0,0202**; Anteil der Läufe mit Varianz bei N = 100 / 300 / 1000 / 3000 / 10 000 etwa **0,1 / 0,2 / 0,6 / 0,9 / 1,0** (Formel 1 − (1 − P)^(2N) − P^(2N); gemessen 0,13 / 0,18 / 0,40 / 0,94 / 1,00); selbst bei N = 10 000 ist der RMSE des größten Gesamtindex **0,158**, das 0,65-Fache des Index |
| **Quasi-Zufallszahlen im Entwurf**, N = 1000, RMSE Zufall : Owen-Sobol (S Saltelli / T Jansen, größter Index) | Hantel Sprung **3,2 / 2,8**; Gitter 3×3 Sprung **1,5 / 1,5**; Begleiter **13 / 4,7**; glatte Hantel **27 / 8,2** |
| **Fehlerbalken**, Gitter 3×3, N = 1000 (100 Wiederholungen, 100 Bootstrap-Ziehungen), größter Erstordnungsindex | Nenner neu gerechnet: Überdeckung **0,90**, Halbbreite **0,100**; fester Nenner: Überdeckung **0,99**, Halbbreite **0,176** (das **1,8-Fache**); kleinster Erstordnungsindex: Halbbreite **0,074**, mehr als das Zehnfache des Index; Gesamtindex: Überdeckung 0,89 bis 0,93, beide Varianten gleich breit |
| **Kosten**, N = 1000, Läufe der Aufzählung gegen Pick-Freeze (p + 2) N | Hantel (p = 7) **128** gegen 9000; Gitter 3×3 (p = 12) **4096** gegen 14 000; Gitter 3×4 (p = 17) **131 072** gegen 19 000; nur Gesamtindizes (p + 1) N: 8000 / 13 000 / 18 000; Owen (2p + 2) N: 16 000 / 26 000 / 36 000; Gleichstand bei N = 2^p/(p + 2) = **14 / 293 / 6899** |

## Ehrliche Grenzen

- **Nur drei Netze und ein Begleiter** (d bis 32) gemessen; Sprung bei p = 0,9 und 0,99 (Messreihen) und 0,7 (Regler), glatt nur an der Hantel und am Gitter 3×3.
- **Die Eingaben sind unabhängig;** für abhängige Eingaben sind Erst- und Gesamtindizes nicht mehr eindeutig. Keine Indizes zweiter Ordnung, keine Gruppenindizes.
- **Owens Schätzer läuft mit unabhängigen Zufallszahlen** (3p Dimensionen), auch wenn A und B quasi-zufällig sind.
- **Intervalle nur für den Jansen-Schätzer und nur bis N = 1000** (Kosten); für kleine Indizes sind sie breiter als der Index selbst.
- **Eine Messung mit 100 Wiederholungen rauscht:** ein einzelner RMSE ist auf etwa 10 bis 15 % genau (bei schweren Rändern wie der seltenen Ausgabe schlechter), Überdeckungen auf etwa ±0,05 bis ±0,09; die Bänder der Tests sind entsprechend weit.
- **Kosten sind Modellläufe;** die Tabellenauswertung des Netzes ist fast kostenlos, bei einem echten Modell ist es die Läufezahl, die zählt.
- **Quellenlage:** Saltelli 2010, Jansen und Janon/Monod sind nur als Formeln aus dem Gedächtnis eingebaut (das Abstract von Saltelli 2010 war nicht erreichbar); Owen 2013 ist im Abstract und in Abschnitt 3 gelesen (die Erwartungstreue der Owen-Formel ist dennoch exakt nachgerechnet, siehe Verifikation), Azzini u. a. 2020 nur im Abstract.

## Verifikation

- **Kern** (`tests/test_algorithm.py`): Zufallsstrom gegen Ganzzahl-Referenz; die ersten vier Sobol-Punkte von Hand; **Pick-Freeze-Läufe und alle Schätzer von Hand** an f(u) = u₁ + 2u₂ mit N = 2 (V = 0,3225; Saltelli 0,33/V und −0,96/V; Jansen; Homma; Janon); Owens Schätzer von Hand (7); Varianz null macht alle Schätzer unbestimmt; exakte Indizes des Pfads von Hand (⅓, ⅔, Summe 4/3); Begleiter in einer und zwei Dimensionen; Varianzanteile; Zusammenhangstabelle am Dreieck; Bootstrap bei konstantem Unterschied.
- **Orakel** (`tests/test_oracle_sbi.py`): Sobol-Richtungszahlen ganzzahlig gleich `scipy.stats.qmc.Sobol` (d = 64, N = 4096); **exakte Indizes gegen die Aufzählung aller bedingten Erwartungen** (Hantel bei p = 0,5 / 0,7 / 0,9, Gitter 3×3); **Σ T = mittlere Dimension und Σ S = additiver Anteil aus Stück 9** für drei Netze und drei Integranden; Begleiter-Indizes gegen Gauß–Legendre-Quadratur; **Erwartungstreue der Zähler exakt** als Summe über alle 2^14 Zustandspaare (Saltelli, Jansen in beiden Formen, Homma) und über alle **2^21 Tripel für Owens Schätzer**; Schätzer bei N = 150 000 gegen die exakten Indizes (Sprung, glatt, Begleiter); der Saltelli-Zähler mit Owen-Sobol-Entwurf ist im Mittel exakt; Zusammenhangstabelle gegen `networkx`.
- **Eigenschaften** (`tests/test_properties.py`): S ≤ T und die Summen, Fehler fällt mit 1/√N, Homma–Saltelli deutlich schlechter, kleine Erstordnungsindizes verschluckt und kleine Gesamtindizes nicht, Owen gegen Jansen bei kleinen und großen Indizes, Saltelli gegen Jansen bei kleinen Indizes glatter Funktionen, Rangfolge, Anteil der Läufe mit Varianz gleich der Formel, Summen, fester Nenner, Quasi-Zufallszahlen, Reproduzierbarkeit, Kosten.
- **Randfälle** (`tests/test_edge_cases.py`): eine Eingabe, kleinste Zeilenzahl, Entwurfsfehler, konstanter Integrand, nicht rechenbare Einstellungen, seltene Ausgabe ohne Absturz, Eingabeprüfung der Spielwiese.
- **Verdikte** (`tests/test_verdicts.py`): jede Schwelle und jede genannte Zahl an von Hand gesetzten Werten, deutsches Zahlenformat.
- **Presets, Layout, Oberfläche:** Presets vollständig und jedes mit dem erwarteten Verdikt; Schnellstart oben, fester Kopf, aktives Preset hervorgehoben; AppTest für jeden Button, jede Auswahl jedes Reglers, die Experimente, die Spielwiese, den PDF-Export und den Permalink.
- **Laufzeit-Importe, Footer, Claims** (`tests/test_claims.py`).

Zahl der Tests: siehe `pytest --collect-only -q tests`.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `sbi_algorithm.py` | Kern: Strom, Sobol-Folge (aus Stück 9), Integranden, exakte Indizes, Entwurf, Schätzer, Owen 2013, Bootstrap |
| `sbi_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `sbi_evaluation.py` | Kennzahlen, Verdikt-Logik, Experimente über feste Ströme, Kosten |
| `sbi_layout.py` | Fester Kopf (Stylesheet) |
| `sbi_pdf_export.py` | PDF der aktuellen Einstellung |
| `sbi_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufallsstrom-Knopf |
| `sbi_scenario.py` | Gitter, Hantel und eigene Kantenlisten |
| `sbi_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest, networkx und scipy |
| `requirements.txt` | Laufzeit-Bibliotheken |
| `tests/` | Kern von Hand, Orakel, Eigenschaften, Randfälle, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe, Claims |

## Bewusst nicht umgesetzt

| Annahme | Verweis |
|---|---|
| Exakte ANOVA und mittlere Dimension | [quasi-monte-carlo-demo](https://github.com/sebastian-hanisch/quasi-monte-carlo-demo) (Stück 9) |
| Der additive Anteil bestimmt den Gewinn von Latin Hypercube | [latin-hypercube-demo](https://github.com/sebastian-hanisch/latin-hypercube-demo) (Stück 5) |
| Gesamtindex des herausintegrierten Teils = Varianzgewinn | [bedingtes-monte-carlo-demo](https://github.com/sebastian-hanisch/bedingtes-monte-carlo-demo) (Stück 6) |
| Seltene Ausgaben gezielt abtasten | Stück 7 und 8 der Linie (Importance Sampling, Kreuzentropie) |
| Bootstrap als Thema | Stück 15 der Linie |
| Morris und Elementareffekte, Shapley-Effekte, momentfreie Indizes (PAWN, Delta) | nicht gebaut |
| Metamodell-Verfahren (Polynomchaos, Gauß-Prozess) | nicht gebaut |
| Abhängige Eingaben, Indizes zweiter Ordnung, Gruppenindizes | nicht gebaut |
| Die Schätzer von Azzini u. a. (2020) | nicht gebaut (nur Abstract gelesen) |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
