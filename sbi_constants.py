"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

NETZE = {"hantel": ("hantel", 0, 0), "3x3": ("gitter", 3, 3), "3x4": ("gitter", 3, 4)}
DEFAULT_NETZ = "3x3"
LIVE_EDGE_LIMIT = 17          # Aufzählung aller Kantenmengen live bis hierher (2^17 = 131 072)
CUSTOM_EDGE_LIMIT = 17
CUSTOM_NODE_LIMIT = 14

INTEGRANDS = {"sprung": "Sprung: das Netz hängt zusammen", "glatt": "glatt: unsichere Haltewahrscheinlichkeiten", "exp": "Begleiter: exp(Σ aᵢuᵢ)"}
DEFAULT_INTEGRAND = "sprung"

P_CHOICES = (0.7, 0.9, 0.99)                                 # Haltewahrscheinlichkeit je Kante (nur Sprung)
P_MIN, P_MAX, DEFAULT_P = P_CHOICES[0], P_CHOICES[-1], 0.9
P_SMOOTH, DELTA_SMOOTH = 0.7, 0.4                            # glatter Integrand: p_e = P_SMOOTH + DELTA_SMOOTH (u_e - 1/2)
D_CHOICES = (4, 8, 16, 32)                                   # Dimension des Begleiters (2 d <= 64 für die Option Owen-Sobol)
D_MIN, D_MAX, DEFAULT_D = D_CHOICES[0], D_CHOICES[-1], 8
WEIGHTS = {"gleich": "aᵢ = 1", "quadratisch": "aᵢ = 2/i²", "harmonisch": "aᵢ = 1/i"}
DEFAULT_WEIGHT = "harmonisch"
N_CHOICES = (100, 300, 1000, 3000, 10000)                    # größtes N der Genauigkeitskurve und der Einzelmessungen
N_MIN, N_MAX, DEFAULT_N = N_CHOICES[0], N_CHOICES[-1], 3000
DESIGNS = {"mc": "unabhängig (Zufall)", "owen": "Owen-Sobol-Folge"}
DEFAULT_DESIGN = "mc"
SEED_MAX, DEFAULT_SEED = 999999, 35

REPS = 100                                # Wiederholungen je Punktzahl der Genauigkeitskurve (feste Ströme)
GRID_REPS = 100                           # Wiederholungen der Einzelmessungen (kleine Indizes, Rangfolge, Summen)
CI_REPS = 100                             # Wiederholungen der Intervall-Messung
BOOT = 100                                # Bootstrap-Ziehungen je Intervall
CI_N_MAX = 1000                           # Obergrenze von N in der Intervall-Messung (Kosten)
RANK_NS = (30, 100, 300, 1000, 3000)      # Punktzahlen der Rangfolge-Messung
RANK_FRACTION = 0.5                       # wirksam: Gesamtindex größer als dieser Anteil des größten
BITS = 32                                 # Stellen der Sobol-Ganzzahlen
MAX_DIM = 64                              # eingebettete Richtungszahlen reichen bis hierher

Z95 = 1.959963984540054

COLORS = {"S_saltelli": "#4c78a8", "S_jansen": "#2F6B65", "S_janon": "#8e6bb0", "S_homma": "#C0392B", "S_owen": "#E08E0B", "T_jansen": "#2F6B65", "T_homma": "#C0392B"}
LABELS = {"S_saltelli": "Saltelli 2010", "S_jansen": "Jansen", "S_janon": "Janon / Monod", "S_homma": "Homma–Saltelli", "S_owen": "Owen 2013", "T_jansen": "Jansen (Gesamtindex)", "T_homma": "Homma–Saltelli (Gesamtindex)"}
MAIN_ESTIMATORS = ("S_saltelli", "S_jansen", "S_janon", "S_homma", "T_jansen", "T_homma")
COLOR_TRUTH = "#222222"
COLOR_BAD = "#C0392B"
COLOR_OK = "#2F6B65"

PRESETS = {
    "Standardfall (Voreinstellung)": {"integrand": DEFAULT_INTEGRAND, "netz": DEFAULT_NETZ, "p": DEFAULT_P, "d": DEFAULT_D, "weight": DEFAULT_WEIGHT, "n": DEFAULT_N, "design": DEFAULT_DESIGN, "seed": DEFAULT_SEED},
    "Hantel: eine Brücke": {"integrand": "sprung", "netz": "hantel", "p": 0.9, "d": DEFAULT_D, "weight": DEFAULT_WEIGHT, "n": DEFAULT_N, "design": DEFAULT_DESIGN, "seed": DEFAULT_SEED},
    "Kleine Indizes am glatten Begleiter": {"integrand": "exp", "netz": DEFAULT_NETZ, "p": DEFAULT_P, "d": 8, "weight": "harmonisch", "n": DEFAULT_N, "design": DEFAULT_DESIGN, "seed": DEFAULT_SEED},
    "Seltene Ausgabe": {"integrand": "sprung", "netz": "3x4", "p": 0.99, "d": DEFAULT_D, "weight": DEFAULT_WEIGHT, "n": 10000, "design": DEFAULT_DESIGN, "seed": DEFAULT_SEED},
    "Quasi-Zufallszahlen im Entwurf": {"integrand": "glatt", "netz": "hantel", "p": DEFAULT_P, "d": DEFAULT_D, "weight": DEFAULT_WEIGHT, "n": 1000, "design": "owen", "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Gitter 3×3, Sprung »das Netz hängt zusammen« bei p = 0,9: zwei Kantenklassen, eine mit Index 0,0024. Die großen Indizes sind gut zu messen, der kleine geht im Rauschen unter.",
    "Hantel: eine Brücke": "Zwei Dreiecke und eine Brücke: die Brücke trägt fast den ganzen Erstordnungsindex, die sechs Dreieckskanten je 0,018.",
    "Kleine Indizes am glatten Begleiter": "exp(Σ aᵢuᵢ) mit abfallenden Gewichten: Indizes von 0,63 bis 0,01. Hier zeigt sich, welcher Schätzer bei kleinen Indizes taugt.",
    "Seltene Ausgabe": "Gitter 3×4 bei p = 0,99: Ausfälle sind so selten, dass viele Läufe gar keinen sehen; auch 10 000 Zeilen reichen nicht.",
    "Quasi-Zufallszahlen im Entwurf": "Glatte Hantel mit den zwei Hälften eines Owen-Sobol-Punktes als A und B: bei glatten Integranden um ein Vielfaches genauer.",
}
