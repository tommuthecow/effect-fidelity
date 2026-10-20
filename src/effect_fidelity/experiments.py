"""Real experiments with public ground truth, plus the text each simulator sees.

Every experiment has two arms, A (reference) and B (treatment); effect = mean(B) - mean(A)
in the experiment's native unit (percentage points for binary outcomes, scale points otherwise).
Ground truth is computed from respondent-level public microdata; nothing is hand-entered.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
ML1 = RAW / "ml1" / "Data" / "Tab.delimited.Cleaned.dataset.WITH.variable.values.dat"
GSS = RAW / "GSS2018.dta"
HILL = RAW / "hillstrom.csv"


@dataclass
class Item:
    """One question. `text(arm)` is the stimulus; options are shown as labels; values map labels to the DV."""
    text: Callable[[str], str]
    labels: list[str]
    options: list[str] | None  # option texts shown next to labels (None for pure numeric scales)
    values: dict[str, float]
    scale_note: str = ""
    pre: "Item | None" = None  # optional earlier question (order experiments): only asked in arms listed in pre_arms
    pre_arms: tuple = ()


@dataclass
class Experiment:
    id: str
    name: str
    family: str
    source: str
    license: str
    unit: str  # "pp" or "points"
    arm_names: dict
    items: list[Item]
    persona_pool: pd.DataFrame  # rows used to build personas (real respondents from the same study)
    persona_text: Callable[[pd.Series], str]
    blind_text: str
    real: dict = field(default_factory=dict)  # arm -> np.ndarray of respondent DV values
    scale: float = 1.0  # multiply DV by this (100 for proportions -> pp)
    real_cov: dict = field(default_factory=dict)  # arm -> DataFrame of persona covariates aligned row-for-row with real[arm]


def _num(s):
    return pd.to_numeric(s, errors="coerce")


# ---------------------------------------------------------------- Many Labs 1 (Klein et al. 2014), US sites
POL = {3: "strongly liberal", 2: "moderately liberal", 1: "slightly liberal", 0: "politically moderate",
       -1: "slightly conservative", -2: "moderately conservative", -3: "strongly conservative"}


def _ml1_persona(r):
    sex = {"f": "woman", "m": "man"}.get(r["sex"], "person")
    age = int(r["age"]) if pd.notna(r["age"]) else 20
    pol = POL.get(int(r["politicalid"]), "politically moderate") if pd.notna(r["politicalid"]) else "politically moderate"
    return f"You are a {age}-year-old {sex} living in the United States, {pol}, taking part in a university research study."


ML1_BLIND = "You are a participant in a university research study in the United States."

AGREE9 = [str(i) for i in range(1, 10)]
AGREE7 = [str(i) for i in range(1, 8)]

FLAG_ITEMS = [
    ("Laws designed to protect the environment pose too high a cost on businesses that contribute to the economy.", False),
    ("A woman should have the right to choose what to do with her body, even if that means getting an abortion.", True),
    ("The United States should not have invaded Iraq.", True),
    ("Homosexuals should have the same right to marriage as anyone else.", True),
    ("Affirmative action gives those groups with a history of oppression a chance to get ahead.", True),
    ("Gun control laws are not nearly strict enough.", True),
    ("It is important for our legal system to use the death penalty as punishment for heinous crimes.", False),
    ("Stem Cell research has important implications for medical advances, and should be pursued at all costs.", True),
]
SYSJ_ITEMS = [
    ("In general, I find society to be fair.", False),
    ("In general, the American political system operates as it should.", False),
    ("American society needs to be radically restructured.", True),
    ("The United States is the best country in the world to live in.", False),
    ("Most policies serve the greater good.", False),
    ("Everyone has a fair shot at wealth and happiness.", False),
    ("Our society is getting worse every year.", True),
    ("Society is set up so that people usually get what they deserve.", False),
]
CONTACT_ITEMS = [
    ("How much do you intend to interact with Muslims in the future?", "1 = not at all, 9 = very much"),
    ("How much time do you think you might spend learning about Islam in the future?", "1 = none at all, 9 = a lot of time"),
    ("How important do you think interacting with Muslims is?", "1 = not at all important, 9 = highly important"),
    ("How willing would you be to attend a mosque gathering to learn more about Islamic beliefs and practices?", "1 = not at all willing, 9 = very willing"),
]
FLAG_PRIME = {"A": "Earlier in this study you looked at four photographs of city scenes and estimated what time of day each was taken.",
              "B": "Earlier in this study you looked at four photographs of city scenes, each showing an American flag, and estimated what time of day each was taken."}
MONEY_PRIME = {"A": "Earlier in this study you filled in a short demographic form on a plain web page.",
               "B": "Earlier in this study you filled in a short demographic form on a web page whose background was a faint image of US hundred-dollar bills."}
CONTACT_PRIME = {"A": "Earlier in this study you spent one minute imagining you were walking in the outdoors (a beach, a forest, hills, the horizon) and then wrote a description of the scene.",
                 "B": "Earlier in this study you spent one minute imagining meeting a Muslim stranger for the first time and finding out some interesting and unexpected things about them, and then wrote a description of the scene."}


def _likert_items(stems, prime, n, anchors, reverse_flags=None):
    items = []
    for i, st in enumerate(stems):
        stem, extra = (st if isinstance(st, tuple) else (st, None))
        rev = reverse_flags[i] if reverse_flags else False
        labs = [str(k) for k in range(1, n + 1)]
        vals = {l: float(n + 1 - int(l)) if rev else float(l) for l in labs}
        note = extra if isinstance(extra, str) else anchors
        items.append(Item(text=(lambda arm, s=stem, p=prime: f"{p[arm]}\n\nNow: {s}"), labels=labs, options=None,
                          values=vals, scale_note=note))
    return items


def load_ml1() -> list[Experiment]:
    d = pd.read_csv(ML1, sep="\t", low_memory=False, encoding="latin1")
    d = d[d.us_or_international == 0].copy()
    for c in ["age", "politicalid"]:
        d[c] = _num(d[c])
    d = d[d.age.between(16, 90)]
    COV = ["age", "sex", "politicalid"]
    covs = []
    pool = d[["age", "sex", "politicalid"]].dropna()
    pool = pool[pool.sex.isin(["f", "m"])].reset_index(drop=True)
    src = "Many Labs 1 (Klein et al., 2014, Social Psychology 45:142-152), OSF wx7ck, US sites"
    lic = "CC0 (data released under CC0 per OSF 'Data released under CC0 license.txt')"
    E = []

    def two(name, g, v, a_val, b_val, transform=lambda x: x):
        x = transform(_num(d[v]))
        gg = _num(d[g])
        out = {"A": x[(gg == a_val) & x.notna()].to_numpy(float), "B": x[(gg == b_val) & x.notna()].to_numpy(float)}
        covs.append({"A": d.loc[(gg == a_val) & x.notna(), COV], "B": d.loc[(gg == b_val) & x.notna(), COV]})
        return out

    # 1 sunk cost: A=free, B=paid; DV 1-9 likelihood of going
    disease_base = ("Imagine that your favorite football team is playing an important game. You have a ticket to the game that you have {how}. "
                    "However, on the day of the game, it happens to be freezing cold. What do you do?")
    E.append(Experiment("ml1_sunk_cost", "Sunk cost (paid vs free ticket)", "Many Labs 1", src, lic, "points",
                        {"A": "ticket received free", "B": "ticket paid for handsomely"},
                        [Item(lambda arm: disease_base.format(how="paid handsomely for" if arm == "B" else "received for free from a friend"),
                              AGREE9, None, {l: float(l) for l in AGREE9}, "1 = definitely stay at home, 9 = definitely go to the game")],
                        pool, _ml1_persona, ML1_BLIND, two("", "sunkgroup", "sunkDV", 1, 0)))
    # 2 Asian disease framing: A=gain frame, B=loss frame; DV = chose risky program
    gain = ("Imagine that the United States is preparing for the outbreak of an unusual disease, which is expected to kill 600 people. "
            "Two alternative programs to fight the disease have been proposed. Assume that the exact scientific estimate of the consequences of the programs are as follows:\n")
    opts = {"A": ["If Program A is adopted, 200 people will be saved.",
                  "If Program B is adopted, there is 1/3 probability that 600 people will be saved and 2/3 probability that no people will be saved."],
            "B": ["If Program A is adopted, 400 people will die.",
                  "If Program B is adopted, there is 1/3 probability that no people will die and 2/3 probability that 600 people will die."]}
    da, db = _num(d.diseaseframinga), _num(d.diseaseframingb)  # a = 'saved' frame, b = 'die' frame; 2 = risky
    covs.append({"A": d.loc[da.dropna().index, COV], "B": d.loc[db.dropna().index, COV]})
    E.append(Experiment("ml1_gain_loss", "Gain vs loss framing (Asian disease)", "Many Labs 1", src, lic, "pp",
                        {"A": "lives-saved frame", "B": "deaths frame"},
                        [Item(lambda arm: gain + "\n".join(opts[arm]) + "\nWhich program would you choose?", ["A", "B"],
                              ["Program A", "Program B"], {"A": 0.0, "B": 1.0})],
                        pool, _ml1_persona, ML1_BLIND,
                        {"A": (da.dropna() == 2).to_numpy(float), "B": (db.dropna() == 2).to_numpy(float)}, scale=100))
    # 3 low vs high category scales: DV = reports > 2.5 h TV/day
    low = ["Up to a half hour", "Half an hour to an hour", "One to one and a half hours", "One and a half to two hours",
           "Two to two and a half hours", "More than two and a half hours"]
    high = ["Up to two and a half hours", "Two and a half to three hours", "Three to three and a half hours",
            "Three and a half to four hours", "Four to four and a half hours", "More than four and a half hours"]
    L6 = list("ABCDEF")
    sc = {"A": (low, {l: (1.0 if l == "F" else 0.0) for l in L6}), "B": (high, {l: (0.0 if l == "A" else 1.0) for l in L6})}
    E.append(Experiment("ml1_scales", "Low vs high category scale (TV hours)", "Many Labs 1", src, lic, "pp",
                        {"A": "low-frequency scale", "B": "high-frequency scale"},
                        [Item(lambda arm: "How much television do you watch per day?", L6, None, {}, "")],
                        pool, _ml1_persona, ML1_BLIND, two("", "scalesgroup", "scales", 0, 1), scale=100))
    E[-1].items[0].arm_options = {a: sc[a][0] for a in sc}
    E[-1].items[0].arm_values = {a: sc[a][1] for a in sc}
    # 4 norm of reciprocity: DV = yes, let North Korean reporters in. A = asked first, B = asked after the NK-lets-US question
    q_us = "Do you think the United States should let newspaper reporters from North Korea come in here and send back to their papers the news as they see it?"
    q_nk = "Do you think North Korea should let American newspaper reporters come in and send back to America the news as they see it?"
    pre = Item(lambda arm: q_nk, ["A", "B"], ["Yes", "No"], {"A": 1.0, "B": 0.0})
    E.append(Experiment("ml1_reciprocity", "Norm of reciprocity (question order)", "Many Labs 1", src, lic, "pp",
                        {"A": "asked first", "B": "asked after the reciprocal question"},
                        [Item(lambda arm: q_us, ["A", "B"], ["Yes", "No"], {"A": 1.0, "B": 0.0}, pre=pre, pre_arms=("B",))],
                        pool, _ml1_persona, ML1_BLIND, two("", "reciprocitygroup", "reciprocityus", 1, 0), scale=100))
    # 5 allowed / forbidden: DV = against the speeches (forbid = Yes, allow = No). A = 'forbid?' wording, B = 'allow?' wording
    afa, afb = _num(d.allowedforbiddena), _num(d.allowedforbiddenb)  # a: forbid? 1=Yes 2=No ; b: allow? 1=Yes 2=No
    covs.append({"A": d.loc[afa.dropna().index, COV], "B": d.loc[afb.dropna().index, COV]})
    E.append(Experiment("ml1_allowed_forbidden", "Allowed vs forbidden wording", "Many Labs 1", src, lic, "pp",
                        {"A": "'forbid?' wording", "B": "'allow?' wording"},
                        [Item(lambda arm: f"Do you think the United States should {'allow' if arm == 'B' else 'forbid'} public speeches against democracy?",
                              ["A", "B"], ["Yes", "No"], {})],
                        pool, _ml1_persona, ML1_BLIND,
                        {"A": (afa.dropna() == 1).to_numpy(float), "B": (afb.dropna() == 2).to_numpy(float)}, scale=100))
    E[-1].items[0].arm_values = {"A": {"A": 1.0, "B": 0.0}, "B": {"A": 0.0, "B": 1.0}}
    # 6 quote attribution: DV agreement 1-9 (9 = strongly agree after recode). A = Bin Laden, B = Washington
    qt = 'A quote similar to what follows was once spoken by {who}: "I have sworn to only live free, even if I find bitter the taste of death." How much do you agree with this quote?'
    E.append(Experiment("ml1_quote", "Quote attribution (liked vs disliked source)", "Many Labs 1", src, lic, "points",
                        {"A": "attributed to Osama Bin Laden", "B": "attributed to George Washington"},
                        [Item(lambda arm: qt.format(who="George Washington" if arm == "B" else "Osama Bin Laden"), AGREE9, None,
                              {l: float(10 - int(l)) for l in AGREE9}, "1 = strongly agree, 5 = neither agree nor disagree, 9 = strongly disagree")],
                        pool, _ml1_persona, ML1_BLIND, two("", "quoteGroup", "quote", 0, 1)))
    # 7 flag priming (conservatism, 1-7), 8 money priming (system justification, 1-7), 9 imagined contact (1-9)
    E.append(Experiment("ml1_flag", "Flag priming (political conservatism)", "Many Labs 1", src, lic, "points",
                        {"A": "no flag", "B": "US flag in photos"},
                        _likert_items([s for s, _ in FLAG_ITEMS], FLAG_PRIME, 7, "1 = completely disagree, 7 = completely agree",
                                      [r for _, r in FLAG_ITEMS]),
                        pool, _ml1_persona, ML1_BLIND, two("", "flagGroup", "flagdv", 0, 1)))
    E.append(Experiment("ml1_money", "Money priming (system justification)", "Many Labs 1", src, lic, "points",
                        {"A": "plain background", "B": "money background"},
                        _likert_items([s for s, _ in SYSJ_ITEMS], MONEY_PRIME, 7, "1 = strongly disagree, 4 = neither, 7 = strongly agree",
                                      [r for _, r in SYSJ_ITEMS]),
                        pool, _ml1_persona, ML1_BLIND, two("", "MoneyGroup", "Sysjust", 0, 1)))
    E.append(Experiment("ml1_contact", "Imagined contact (intentions toward Muslims)", "Many Labs 1", src, lic, "points",
                        {"A": "imagine outdoors", "B": "imagine meeting a Muslim"},
                        _likert_items(CONTACT_ITEMS, CONTACT_PRIME, 9, ""),
                        pool, _ml1_persona, ML1_BLIND, two("", "ContactGroup", "Imagineddv", 0, 1)))
    assert len(covs) == len(E)
    for e, c in zip(E, covs):
        e.real_cov = c
    return E


# ---------------------------------------------------------------- GSS 2018 split-ballot wording experiments
GSS_STEM = ("We are faced with many problems in this country, none of which can be solved easily or inexpensively. "
            "I'm going to name some of these problems, and for each one I'd like you to tell me whether you think we're "
            "spending too much money on it, too little money, or about the right amount. Are we spending too much, too little, "
            "or about the right amount on {topic}?")
GSS_PAIRS = [  # (id, name, standard var, standard wording, version-Y var, Y wording)
    ("gss_welfare", "'Welfare' vs 'assistance to the poor'", "natfare", "welfare", "natfarey", "assistance to the poor"),
    ("gss_crime", "'Halting the rising crime rate' vs 'law enforcement'", "natcrime", "halting the rising crime rate", "natcrimy", "law enforcement"),
    ("gss_cities", "'Solving problems of big cities' vs 'assistance to big cities'", "natcity", "solving the problems of the big cities", "natcityy", "assistance to big cities"),
    ("gss_race", "'Improving conditions of Black people' vs 'assistance to Black people'", "natrace", "improving the conditions of Black people", "natracey", "assistance to Black people"),
    ("gss_space", "'Space exploration program' vs 'space exploration'", "natspac", "the space exploration program", "natspacy", "space exploration"),
    ("gss_environment", "'Improving & protecting the environment' vs 'the environment'", "natenvir", "improving and protecting the environment", "natenviy", "the environment"),
]
POLV = {1: "extremely liberal", 2: "liberal", 3: "slightly liberal", 4: "moderate, middle of the road", 5: "slightly conservative",
        6: "conservative", 7: "extremely conservative"}
DEG = {0: "did not finish high school", 1: "has a high school diploma", 2: "has a junior college degree", 3: "has a bachelor's degree", 4: "has a graduate degree"}
RACE = {1: "White", 2: "Black", 3: "of another race (not White or Black)"}


def _gss_persona(r):
    sex = {1: "man", 2: "woman"}[int(r["sex"])]
    pol = POLV.get(int(r["polviews"]), "moderate") if pd.notna(r["polviews"]) else "moderate"
    return (f"You are a {int(r['age'])}-year-old {RACE[int(r['race'])]} {sex} living in the United States who {DEG[int(r['degree'])]} "
            f"and describes your political views as {pol}. You are being interviewed for the General Social Survey.")


def load_gss() -> list[Experiment]:
    d = pd.read_stata(GSS, convert_categoricals=False)
    pool = d[["age", "sex", "race", "degree", "polviews"]].dropna().reset_index(drop=True)
    E = []
    for eid, name, a, aw, b, bw in GSS_PAIRS:
        real = {"A": (d[a].dropna() == 1).to_numpy(float),
                "B": (d[b].dropna() == 1).to_numpy(float)}
        E.append(Experiment(eid, name, "GSS split ballot", "General Social Survey 2018 (NORC), cross-section, unweighted; "
                            "split-ballot random assignment of standard vs 'Version Y' wording", "Free public-use data (NORC GSS); cite as GSS 2018",
                            "pp", {"A": f"'{aw}'", "B": f"'{bw}'"},
                            [Item(lambda arm, aw=aw, bw=bw: GSS_STEM.format(topic=bw if arm == "B" else aw), ["A", "B", "C"],
                                  ["Too little", "About right", "Too much"], {"A": 1.0, "B": 0.0, "C": 0.0})],
                            pool, _gss_persona, "You are an adult living in the United States being interviewed for the General Social Survey.",
                            real, scale=100))
        E[-1].real_cov = {"A": d.loc[d[a].dropna().index, list(pool.columns)], "B": d.loc[d[b].dropna().index, list(pool.columns)]}
    return E


# ---------------------------------------------------------------- Hillstrom MineThatData email RCT (2008)
def _hill_persona(r):
    lines = [f"months since your last purchase: {int(r['recency'])}",
             f"amount you spent with the retailer in the past year: ${r['history']:.0f}",
             f"you have bought men's merchandise from them: {'yes' if r['mens'] else 'no'}",
             f"you have bought women's merchandise from them: {'yes' if r['womens'] else 'no'}",
             f"you live in a {r['zip_code'].lower().replace('surburban', 'suburban')} area",
             f"you became a new customer in the past year: {'yes' if r['newbie'] else 'no'}",
             f"you have purchased through: {r['channel'].lower()}"]
    return "You are a customer of an online apparel retailer. About you: " + "; ".join(lines) + "."


HILL_ARMS = {"No E-Mail": "In the past week you did not receive any marketing email from the retailer.",
             "Mens E-Mail": "In the past week you received a marketing email from the retailer featuring its men's merchandise.",
             "Womens E-Mail": "In the past week you received a marketing email from the retailer featuring its women's merchandise."}


def load_hillstrom() -> list[Experiment]:
    d = pd.read_csv(HILL)
    pool = d.drop(columns=["segment", "visit", "conversion", "spend"]).reset_index(drop=True)
    q = ("{arm}\n\nDuring the two weeks after that, did you visit the retailer's website?")
    E = []
    for eid, nm, bseg in [("hill_mens", "Men's-merchandise email vs no email (site visits)", "Mens E-Mail"),
                          ("hill_womens", "Women's-merchandise email vs no email (site visits)", "Womens E-Mail")]:
        arms = {"A": "No E-Mail", "B": bseg}
        E.append(Experiment(eid, nm, "Hillstrom email RCT", "Kevin Hillstrom MineThatData E-Mail Analytics challenge (2008), 64,000 customers randomized",
                            "Publicly released by the author for analysis (MineThatData blog); widely redistributed",
                            "pp", {"A": "no email", "B": bseg.lower()},
                            [Item(lambda arm, arms=arms: q.format(arm=HILL_ARMS[arms[arm]]), ["A", "B"], ["Yes", "No"], {"A": 1.0, "B": 0.0})],
                            pool, _hill_persona, "You are a customer of an online apparel retailer.",
                            {"A": d.visit[d.segment == "No E-Mail"].to_numpy(float), "B": d.visit[d.segment == bseg].to_numpy(float)},
                            scale=100))
        E[-1].real_cov = {"A": pool[(d.segment == "No E-Mail").to_numpy()], "B": pool[(d.segment == bseg).to_numpy()]}
    return E


def load_all() -> list[Experiment]:
    return load_ml1() + load_gss() + load_hillstrom()
