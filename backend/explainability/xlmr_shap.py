"""
SHAP explainability for XLM-RoBERTa text classification.

This module is intentionally model-path-agnostic because it reuses the runtime model
bundle from `inference.xlmr.loader`. Replacing the model artifacts path/config should
not require SHAP code changes.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from collections import OrderedDict
from typing import Any

import numpy as np

from core.config import (
    shap_cache_enabled,
    shap_cache_maxsize,
    shap_max_evals,
    shap_max_words,
    shap_top_k,
    xlmr_inference_temperature,
)
from inference.xlmr.loader import XLMRBundle
from schemas.models import ExplanationIndicator, ExplanationResult, ExplanationTopToken

_PUNCT_RE = re.compile(r"^\W+$")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")

_EXPLANATION_NOTE = (
    "SHAP identifies token contributions to the model prediction. "
    "It does not verify factual correctness."
)

# IMPORTANT: This dictionary is used only to group SHAP token contributions into
# interpretable UI indicators. It does NOT influence model prediction and does NOT
# perform factual verification/fact-checking.
_INDICATOR_KEYWORDS: dict[str, list[str]] = {
    "Linguistic Tone": [
        "angry", "anger", "fear", "fearful", "worried", "worry", "sad", "sadness",
        "happy", "emotional", "emotive", "alarming", "alarm", "threat", "threatening",
        "danger", "dangerous", "panic", "panicked", "controversial", "criticized",
        "criticize", "condemned", "condemn", "praised", "praise", "mocked", "mock",
        "insulted", "insult", "accused", "accuse", "blamed", "blame", "outrage",
        "outraged", "furious", "shame", "shameful", "disappointed", "disappointing",
        "concern", "concerned", "warning", "warned", "warn",
        "horrific", "horrifying", "terrifying", "terrified", "disturbing", "disturbed",
        "devastating", "tragic", "tragedy", "heartbreaking", "heartbroken",
        "stress", "stressed", "anxious", "anxiety", "frustrated", "frustrating",
        "disgusting", "disgusted", "infuriating", "enraged", "rage",
        "livid", "upset", "shocked", "shockingly", "appalled", "appalling",
        "suspicious", "suspicion", "outrageous", "hateful", "hate",
        "support", "supported", "supporting", "admire", "admired", "admiration",
        "respect", "respected", "disrespect", "disrespected",
        "celebrate", "celebrated", "celebration", "critic", "criticism",
        "attack", "attacked", "slammed", "blast", "blasted",
        "call out", "called out", "denounce", "denounced",
        "protest", "protested", "protesting", "uproar",
        "galit", "nagagalit", "kinagalit", "takot", "natakot", "nakakatakot",
        "pangamba", "nangangamba", "kabado", "nakakabahala", "banta", "nagbabanta",
        "delikado", "panganib", "mapanganib", "gulat", "nagulat", "nakakagulat",
        "malungkot", "lungkot", "masaya", "emosyonal", "umiyak", "iyak",
        "kinondena", "kondena", "pinuna", "batikos", "binatikos", "puna",
        "pinuri", "papuri", "inis", "naiinis", "napahiya", "kahihiyan",
        "sinisi", "sisi", "nakakainis", "nakakabigla", "babala", "nagbabala",
        "nakakagalit", "galit na galit", "gigigil", "naiirita", "irita",
        "nakakabwisit", "bwisit", "badtrip", "nakakabadtrip",
        "kabog", "grabe", "grabeh", "grabe naman", "grabeng", "sobrang", "sobra",
        "nakakaloka", "loka", "nakakainis", "nakakairita",
        "nakakagigil", "nakakahiya", "hiya", "nakakahiya naman",
        "nakakapanlumo", "panlumo", "nakakadepress", "depress", "depressed",
        "nakakaawa", "awa", "kawawa", "nakakaiyak", "iyak na iyak",
        "nakakatuwa", "tuwa", "aliw", "nakakaaliw",
        "shook", "shookt", "OMG", "omg", "wtf", "lmao", "lol",
        "sus", "suss", "hmm", "hays", "hay", "grrr",
        "pikon", "napikon", "pikon na pikon", "as in",
        "nakakatakot", "nakakakaba", "kaba", "nakakakilabot", "kilabot",
        "nakakapangilabot", "nakakabahala", "nakakabother",
        "nakaka stress", "stressful", "nakakapraning", "praning",
        "nakakagago", "gigil much", "iyak tawa", "nakakaiyak na lang",
        "sumabog ang damdamin", "nanlumo", "nakakapanikot ng ulo",
        "dismayado", "dismaya", "nakakadismaya", "napapamura",
        "nakakagulat talaga", "sakit sa ulo", "nakaka frustrate",
        "toxic", "nasty", "vile", "vicious", "bitter", "spiteful", "sarcastic",
        "sarcasm", "derogatory", "disparaging", "vilify", "smear", "slur",
        "obscene", "vulgar", "profanity", "cursing", "curse", "filthy",
        "moron", "idiot", "stupid", "dumb", "pathetic", "loser", "scumbag",
        "jerk", "worthless", "useless", "freak", "weirdo", "dirtbag",
        "despicable", "revolting", "foul mouthed", "trashy", "trash talk",
        "burn", "burned", "roasted", "owning", "clowned", "pathetic loser",
        "gago", "gaga", "ulol", "leche", "punyeta", "piste", "yawa",
        "yawa ka", "kupal", "pakshet", "pucha", "puta", "pota",
        "tang ina", "tangina", "taena", "putang", "putangina",
        "tangina mo", "putang ina mo", "pakyo", "pakyu", "fk", "fvck",
        "bobo", "boba", "tanga", "engot", "gunggong", "tarantado",
        "inutil", "bubo", "bubo ka", "gungong", "tarantadu",
        "walang silbi", "walang kwenta", "walang modo", "walang hiya",
        "walang respeto", "bastos", "salbahe", "mapang api",
        "mapanglait", "manlait", "pambabastos", "mang insulto", "insultuhin",
        "mang asar", "asar", "nakakaasar", "nakakabwisit na", "nakakaburat",
        "mema", "mema lang", "satsat", "puro dakdak", "clout", "clout chaser",
        "epal", "sipsip", "trapo", "trapong", "buwaya", "mangongotong",
        "sinungaling", "sinungaling ka", "mang sinungaling", "mandaraya",
        "manloloko", "budol", "budolero", "budol king", "plastik", "peke",
        "doble kara", "dobleng mukha", "balimbing", "balimbingan",
        "taksil", "traydor", "taksil ka", "hunyango", "manggagamit",
        "mapanlinlang", "linlang", "niloloko", "niloko", "garapal",
        "walang konsensya", "evil", "masamang tao", "masamang loob",
        "demons", "demonyo", "demonsyo", "satanas", "impyerno",
        "hayop", "hayop ka", "animal", "animal ka", "mga hayop",
        "aso", "baboy", "baboy ka", "daga", "halimaw", "linta",
        "sipsip na linta", "pusit", "pusit ka", "squid", "reptile",
        "reptilya", "basura", "basurang tao", "basurang pulitiko",
        "adik", "adek", "adek boy", "drug adik", "shabu", "shabs",
        "bangag", "bangag ka", "high ka", "lasing", "lasheng", "stoned",
        "basag", "lutang", "lutang ka", "coke head", "sniffer",
        "patayin", "papapatay", "papatayin", "saksakin", "suntukin",
        "bugbugin", "sampalin", "barilin", "sunugin", "itumba", "itumba mo",
        "ibitay", "bitayin", "pugutan", "balatan", "wasakin",
        "dutae", "dugyot", "dugyot ka", "dilawan", "pinklawan",
        "kakampink", "yellowtard", "dds", "ddsz", "apologist",
        "bayarang", "bayaran", "paid troll", "trolls", "troll farm",
        "junjun", "magnanakaw", "magnanakaw ka", "dictator",
        "diktador", "oppressor", "salot", "sumpa",
        "kanser", "kanser sa lipunan", "balakid", "panggulo",
        "demonyo ng", "halimaw na", "abomination", "disgrace",
        "kakahiyan", "nakakahiya ka", "shame on you", "how dare",
        "kupal ka", "bobo mo", "tanga mo",
    ],
    "Claim Certainty": [
        "confirmed", "confirm", "proven", "proved", "proof", "definitely",
        "certainly", "surely", "guaranteed", "guarantee", "undeniable",
        "undoubtedly", "always", "never", "must", "will", "cannot", "clearly",
        "obviously", "true", "false", "fake", "real", "claim", "claimed",
        "claims", "alleged", "allegedly", "reportedly", "supposedly",
        "rumored", "rumour", "rumor", "possible", "possibly", "may", "might",
        "could", "likely", "unlikely", "said to be", "believed", "according",
        "assert", "asserted", "assertion", "insist", "insisted", "insisting",
        "deny", "denied", "denial", "refute", "refuted", "debunk", "debunked",
        "fact", "facts", "factually", "no doubt", "without doubt",
        "certain", "certainty", "uncertain", "uncertainty", "unclear",
        "seems", "seemingly", "appears", "apparently", "suggests", "suggested",
        "estimate", "estimated", "estimates", "roughly", "approximately",
        "about", "around", "at least", "at most",
        "impossible", "definitive", "conclusive", "inconclusive",
        "confirmed by", "unconfirmed", "not confirmed",
        "kumpirmado", "kinumpirma", "patunay", "napatunayan", "pinatunayan",
        "sigurado", "tiyak", "tiyak na", "talaga", "totoo", "hindi totoo",
        "peke", "huwad", "di umano", "umano", "diumano", "sinasabing",
        "sabi", "sinabi", "ayon", "ayon sa", "pinaniniwalaan", "maaaring",
        "posible", "malamang", "hindi maaari", "dapat", "hindi dapat",
        "walang duda", "klaro", "malinaw", "inaangkin", "pahayag",
        "balitang", "usap usapan", "kumakalat",
        "katiyakan", "siguradong", "tiyak na tiyak", "totoong", "tunay",
        "hindi raw", "raw", "daw", "sabi daw", "diumano", "di-umano",
        "pinapaniwalaan", "pinaniniwalaan", "inaakala", "akala", "parang",
        "mukhang", "tila", "posibleng", "maaaring", "baka", "siguro",
        "malinaw na", "klarong", "walang alinlangan", "walang duda",
        "pawang", "totoo ba", "legit", "legit ba", "peke ba", "fake ba",
        "confirmed na", "kumpirmado na",
        "for sure", "sure na", "sureball", "sure ball",
        "100 percent", "100%", "walang kaso", "solid na",
        "hindi pa kumpirmado", "unverified", "di verified",
        "parang totoo", "mukhang totoo", "mukhang fake",
        "baka naman", "pwede rin", "posibleng totoo", "hindi sigurado",
        "syempre", "syempre naman", "obvious", "obvious naman",
        "plainly", "plain as day", "no brainer", "case closed",
        "periodt", "period", "tapos ang usapan", "tapos na ang usapan",
        "wala nang debate", "ayaw na mag debate", "open and shut",
        "irrefutable", "irrefutably", "indisputable", "indisputably",
        "unequivocal", "unequivocally", "unquestionable", "unquestionably",
        "absolutely", "absolutely true", "absolutely false", "categorically",
        "categorically false", "categorically true", "patently false",
        "patently fake", "blatant lie", "bare faced lie", "big lie",
        "hoax", "bogus", "sham", "fraud", "fraudulent", "fabricated",
        "fabrication", "made up", "invented story", "false narrative",
        "disinformation", "misinformation", "malicious lie", "sinungaling",
        "sinungaling talaga", "sinungaling sila", "propaganda",
        "black propaganda", "fake news yarn", "fake news talaga",
        "legit na peke", "peke talaga", "to naman", "totoo naman",
        "hindi mapagdudahan", "di na mapagdudahan", "walang pagdududa",
        "di na pinagtatalunan", "hindi na pinagtatalunan", "final answer",
        "final na", "tapos na", "solved na", "klarong klaro",
        "linaw na linaw", "panigurado", "sigurado yun", "sigurado ako",
        "sure ako", "ako sure", "hindi ako nagkakamali",
        "hindi ako nagkakamali dito", "imposibleng hindi", "imposibleng oo",
        "truth bomb", "hard truth", "capital t truth",
        "undeniable truth", "gospel truth", "word for word",
        "verbatim", "as a matter of fact", "matter of fact",
        "in reality", "in fact", "fact of the matter", "bottom line",
        "truth is", "reality check", "get real",
        "hindi makatotohanan", "hindi tugma sa datos", "fake account",
        "di verified", "hindi validated", "unvalidated claim",
        "unsupported claim", "baseless accusation",
        "walang batayan", "walang pundasyon", "hindi credible",
        "walang credibility", "discredited na", "debunk na",
    ],
    "Presence of Evidence-related Language": [
        "evidence", "proof", "data", "record", "records", "document",
        "documents", "report", "reports", "study", "studies", "research",
        "investigation", "investigated", "findings", "analysis", "survey",
        "statistics", "statistical", "source", "sources", "statement",
        "official statement", "press release", "certificate", "court record",
        "medical record", "police report", "audit", "verified", "verification",
        "fact check", "factcheck", "fact checked", "based on", "according to",
        "dataset", "methodology", "methods", "peer reviewed", "peer-reviewed",
        "journal", "publication", "published", "preprint", "meta analysis", "meta-analysis",
        "clinical trial", "trial", "randomized", "double blind", "double-blind",
        "case study", "case report", "laboratory", "lab", "test results",
        "eeg", "mri", "ct scan", "x ray", "x-ray", "blood test",
        "court filing", "complaint", "affidavit", "sworn statement", "testimony",
        "transcript", "minutes", "resolution", "memorandum", "circular",
        "policy", "guidelines", "protocol", "standard operating procedure", "sop",
        "invoice", "receipt", "contract", "agreement",
        "election return", "er", "certificate of canvass", "coc", "cocp",
        "official tally", "canvass", "canvassing",
        "evidence shows", "documented", "documentation", "records show",
        "confirmed in", "as per", "based upon",
        "ebidensya", "patunay", "datos", "rekord", "tala", "dokumento",
        "ulat", "pag aaral", "saliksik", "imbestigasyon", "sinisiyasat",
        "natuklasan", "resulta", "pagsusuri", "sarvey", "estadistika",
        "pinagmulan", "sanggunian", "pahayag", "opisyal na pahayag",
        "sertipiko", "rekord ng korte", "ulat ng pulis",
        "beripikado", "beripikasyon", "batay sa",
        "ayon sa ulat", "base sa", "basehan", "katibayan",
        "beripikasyon", "beripikado", "na-verify", "na verify", "verify",
        "dokyumento", "dokyu", "resibo", "kontrata", "kasunduan",
        "testigo", "patotoo", "salaysay", "sinumpaang salaysay",
        "resolusyon", "memorandum", "circular", "polisiya", "patakaran",
        "gabayan", "guidelines", "protocol", "proseso", "pamamaraan",
        "rekord", "talaan", "tala", "tala ng korte", "kaso", "demanda",
        "lab test", "resulta ng test", "resulta ng pagsusuri",
        "statement of assets", "saln", "statement of campaign contributions",
        "soce", "certificate of candidacy", "candidacy", "coc filing",
        "barangay blotter", "incident report", "forensic", "forensics",
        "satellite image", "geotag", "metadata", "source code",
        "open data", "foi", "freedom of information", "foi request",
        "official gazette", "gazette", "government data",
        "screenshot", "screenshots", "screencap", "screen cap", "screencapture",
        "screengrab", "screen grab", "photo evidence", "larawan ng ebidensya",
        "cctv", "cctv footage", "body cam", "bodycam", "dashcam", "dash cam",
        "audio leak", "leaked audio", "audio recording", "voice recording",
        "video evidence", "video bilang patunay", "raw footage", "uncut video",
        "metadata", "exif", "timestamp", "time stamp", "digital trail",
        "paper trail", "chain of custody", "evidence bag", "exhibit a",
        "exhibit b", "annex", "attachment", "attachments", "hyperlink",
        "permalink", "archived link", "wayback", "wayback machine",
        "primary source", "secondary source", "firsthand", "first hand",
        "eyewitness", "eye witness", "witness account", "witness statement",
        "sworn testimony", "under oath", "sinumpaang salaysay",
        "corroborate", "corroborates", "corroborated", "corroboration",
        "disprove", "disproves", "disproved", "invalidates", "invalidated",
        "forensic report", "autopsy", "ballistics", "dna", "dna test",
        "fingerprint", "tally sheet", "precinct count", "election data",
        "recount", "audit trail", "log file", "server logs", "ip logs",
        "geotagged", "geolocation", "triangulated", "triangulation",
        "ebidensyang", "mga ebidensya", "resibong", "may resibo",
        "may patunay", "may katibayan", "may dokumento", "may rekord",
        "walang ebidensya", "walang patunay", "walang proweba",
        "walang katibayan", "walang dokumento", "walang resibo",
        "hindi napatunayan", "hindi napatotohanan", "hindi beripikado",
        "hindi supported ng datos", "walang basehang datos",
        "base sa datos", "base sa ebidensya", "sinusuportahan ng datos",
        "pinapatunayan ng", "nagpapakita ng patunay",
        "redacted document", "unredacted", "leaked docs", "leaked memo",
        "internal memo", "confidential memo", "classified", "declassified",
    ],
    "Textual Source Attribution Mentions": [
        "said", "says", "stated", "announced", "according", "according to",
        "reported by", "spokesperson", "official", "officials", "agency",
        "department", "government", "president", "senator", "mayor",
        "governor", "police", "authority", "authorities", "expert", "experts",
        "researcher", "researchers", "scientist", "scientists", "journalist",
        "news agency", "court", "supreme court", "congress", "senate",
        "house", "doh", "deped", "dilg", "comelec", "pnp", "nbi", "doj",
        "pna", "philippine news agency", "vera files", "rappler", "gma",
        "abs cbn", "cnn philippines", "inquirer", "philstar", "manila bulletin",
        "dswd", "dti", "dost", "da", "denr", "dpwh", "dot", "dof", "doh", "dole",
        "ltfrb", "ltto", "lto", "mmda", "pagasa", "phivolcs", "neda",
        "doh", "who", "un", "unesco", "unicef",
        "sec", "bir", "bsp", "philhealth", "sss", "gsis",
        "pcso", "prc", "ched", "tesda",
        "afp", "pnp", "pnpa", "pcg", "philippine coast guard",
        "dotr", "ltfrb", "caap", "marina",
        "drrmo", "bdrrmo", "lgu", "barangay", "city hall",
        "malacañang", "malacanang", "op", "office of the president",
        "iATF", "iatf", "inter-agency task force",
        "supreme court", "court of appeals", "sandiganbayan",
        "ombudsman", "commission on audit", "coa",
        "gma news", "gma network", "gma integrated news",
        "abs cbn news", "abscbn news", "tv patrol",
        "cnnph", "cnn philippines", "one news", "tv5",
        "inquirer.net", "philstar.com", "mb.com.ph", "manila bulletin",
        "manila times", "the manila times", "businessworld",
        "sunstar", "sun star", "cebu daily news",
        "interaksyon", "news5", "ptv", "ptv4", "radyo pilipinas",
        "dzmm", "dzbb", "dwiz", "dzrh", "rmn",
        "fact check", "factcheck", "verafiles fact check", "vera files fact check",
        "tsek.ph", "tsekph", "poynter",
        "mindanews", "mindanao times", "davao today", "bombo radyo",
        "brigada news", "abante", "bandera", "tempo", "journal online",
        "cebuano", "iloilo", "baguio", "pampanga", "zamboanga",
        "ppcrv", "namfrel", "eleksyon", "election watchdog",
        "barangay captain", "sk chairman", "city council", "sangguniang bayan",
        "sangguniang panlungsod", "governor s office", "mayor s office",
        "house of representatives", "commission on elections",
        "office of civil defense", "ocd", "ndrrmc",
        "sinabi", "ayon", "ayon kay", "ayon sa", "pahayag", "ipinahayag",
        "iniulat", "ulat", "anunsyo", "inanunsyo", "tagapagsalita",
        "opisyal", "ahensya", "kagawaran", "gobyerno", "pangulo",
        "senador", "alkalde", "gobernador", "pulis", "awtoridad",
        "eksperto", "mananaliksik", "mamamahayag", "hukuman", "korte",
        "kongreso", "senado", "kamara", "barangay", "lgu", "lokal na pamahalaan",
        "ayon sa mga ulat", "ayon sa report", "ayon sa pahayag",
        "sa panayam", "panayam", "interview", "iniinterview",
        "presscon", "press con", "press conference",
        "sinabi ni", "sabi ni", "sabi ng", "pahayag ni", "pahayag ng",
        "inihayag", "inihayag ni", "inihayag ng",
        "iniulat ni", "iniulat ng", "ayon sa kanya", "ayon sa kanila",
        "quotes", "quoted", "quoting", "cites", "cited", "citation",
        "writes", "wrote", "author", "byline", "columnist",
        "per source", "unnamed source", "anonymous source", "insider",
        "whistleblower", "leaked by", "obtained by", "exclusive to",
        "speaks on condition", "on background", "on the record",
        "off the record", "statement from", "released by",
        "marcos", "marcos jr", "bbm", "pbbm", "bongbong", "imee",
        "duterte", "sara", "inday sara", "rodrigo", "bato", "bato dela rosa",
        "leni", "robredo", "kiko", "pangilinan", "isko", "moreno",
        "trillanes", "de lima", "leila", "zubiri", "escudero", "chiz",
        "cayetano", "mikey", "villar", "mark villar", "manny villar",
        "diokno", "chel diokno", "kiko pangilinan", "poe", "grace poe",
        "hontiveros", "risa", "lapid", "lito lapid", "revilla", "bong revilla",
        "enrile", "juan ponce", "gonzaga", "toni", "kuya wil", "vice ganda",
        "philippine star", "the philippine star", "bulletin online",
        "journal online ph", "standard", "daily tribune",
        "manila standard", "business mirror", "malaya business insight",
        "daily tribune online", "tribune.net", "bworld online",
        "altermidya", "bulatlat", "kodao", "pinoyweekly",
        "bilyonaryo", "get real philippines", "showbiz insider",
        "sonshine", "smni", "net 25", "net25",
        "dzxl", "dwsb", "super radyo", "tele radyo", "bombo radyo nationwide",
        "comelec commissioner", "poll body", "namfrel volunteer",
        "watchdog group", "civil society", "ngo", "international observers",
        "commission on human rights", "chr",
        "kontra daya", "legal network", "ibp", "integrated bar",
    ],
    "Sensationalism": [
        "viral", "shocking", "shock", "exposed", "expose", "unbelievable",
        "breaking", "must see", "must watch", "secret", "scandal", "leaked",
        "leak", "watch", "share", "click", "click here", "urgent", "alert",
        "warning", "bombshell", "explosive", "truth revealed", "revealed",
        "finally revealed", "hidden truth", "banned", "censored", "you wont believe",
        "wow", "amazing", "miracle", "instant", "destroyed", "humiliated",
        "caught", "caught on camera", "exclusive", "latest", "trending", "omg",
        "here s why", "heres why", "what happens next", "what happened next",
        "this is why", "this changes everything", "mind blown", "mindblown",
        "jaw dropping", "jaw-dropping", "insane", "crazy", "wild",
        "epic", "massive", "huge", "biggest", "worst", "best ever",
        "unreal", "no one saw this coming", "shocking truth", "truth bomb",
        "warning!", "alert!", "breaking!", "exclusive!", "just in",
        "must read", "must-watch", "must-see", "watch now", "share now",
        "goes viral", "trending now", "everyone is talking about",
        "kumalat", "kalat", "nakakagulat", "gulat", "ibinunyag",
        "binunyag", "pasabog", "eskandalo", "sekreto", "lihim",
        "kumalat na video", "panoorin", "i share",
        "pindutin", "alerto", "grabe",
        "di kapani paniwala", "hindi kapani paniwala", "malupit", "matindi",
        "wasak", "pinahiya", "nahuli", "huli sa camera", "eksklusibo",
        "pinakabago", "mainit na balita", "abangan",
        "alam niyo ba", "hindi mo aakalain", "ikakagulat mo",
        "panoorin ngayon", "panoorin na", "panuorin", "panuorin na",
        "i-share ngayon", "ishare", "ishare mo", "share mo", "share natin",
        "i-like", "like", "i-comment", "comment", "subscribe",
        "abangan ang susunod", "abangan mo", "huwag palampasin",
        "grabe to", "grabe ito", "sobra na", "sobra to",
        "hindi ka maniniwala", "di ka maniniwala", "di mo aakalain",
        "nakakaloka", "nakakagulantang", "gulantang", "pasabog",
        "explosive", "bombahan", "bombshell", "mainit", "hot",
        "full video", "watch till the end", "watch till end", "part 1", "part 2",
        "must try", "life changing", "life-changing", "secret revealed",
        "exclusive leak", "breaking update", "teaser", "teaser lang",
        "netizens react", "netizens furious", "netizens shocked",
        "kakaiba", "sarap panuorin", "grabe ang nangyari", "sumabog sa internet",
        "pinag uusapan", "usap usapan ngayon", "kumatok sa social media",
        "viral ngayon", "nag viral", "nag-viral",
        "plot twist", "mega plot twist", "twist ending", "blowing up",
        "blew up online", "social media erupted", "internet meltdown",
        "firestorm", "media frenzy", "feeding frenzy", "chaos online",
        "went nuts", "losing their minds", "cannot believe", "unsee",
        "watch before deleted", "before it gets deleted", "takedown",
        "restored footage", "leaked tape", "alleged video",
        "blind item", "blind items", "chismis", "tsismis", "tea", "spill the tea",
        "bardagulan", "savage", "savage af", "clapback", "clap back",
        "ratio", "ratioed", "owns", "owned", "destroyed him", "destroyed her",
        "mic drop", "karma", "instant karma", "served", "justice served",
        "you wont see this", "never before seen", "first time leak",
        "limited time", "act now", "last chance", "do not scroll",
        "stop scrolling", "wait for it", "wait till the end", "ending tho",
        "pilipinas trending", "trending sa pilipinas", "pilipinas talk",
        "number 1 trending", "top trending topic", "trendlist",
        "usap-usapan ng bayan", "pinag uusapan ng bayan",
        "usap-usapan ngayon", "pinaguusapan ng lahat",
        "nagliyab ang komento", "sumabog ang comment section",
        "nagulantang ang netizens", "nagulantang mga netizens",
        "kumatok ang balita", "pumatok ang post",
        "kumakalat sa facebook", "kumakalat sa tiktok", "tiktok trending",
        "facebook viral", "nagpapakulo", "mainit na isyu",
        "wag kalimutan panoorin", "wag palampasin", "wag palampasin to",
        "huling parte", "huling bahagi", "part 3", "to be continued",
        "abangan susunod", "abangan mamaya",
        "hindi ako makapaniwala grabe", "grabe ang trending",
        "viral-worthy", "viral worthy", "click worthy", "clout chasing",
        "for the views", "for engagement", "bait engagement",
        "outrage bait", "rage bait", "ragebait",
    ],
}

_INDICATOR_PRIORITY: list[str] = [
    "Textual Source Attribution Mentions",
    "Presence of Evidence-related Language",
    "Sensationalism",
    "Claim Certainty",
    "Linguistic Tone",
]

_INDICATOR_SUMMARY: dict[str, str] = {
    "Linguistic Tone": (
        "These highlighted words suggest emotionally loaded tone that influenced the prediction."
    ),
    "Claim Certainty": (
        "These words reflect certainty or hedging cues that influenced the prediction."
    ),
    "Presence of Evidence-related Language": (
        "These words indicate evidence-related framing that influenced the prediction."
    ),
    "Textual Source Attribution Mentions": (
        "These words mention sources/attribution patterns that influenced the prediction."
    ),
    "Sensationalism": (
        "These highlighted words suggest attention-grabbing or sensational wording that influenced the prediction."
    ),
}


@dataclass
class _TokenContribution:
    text: str
    score: float


def _truncate_text_by_words(text: str, max_words: int | None = None) -> str:
    max_words = max_words or shap_max_words()
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _normalize_token(raw: str) -> str:
    token = (raw or "").strip()
    if not token:
        return ""
    # SentencePiece/BPE markers commonly used by XLM-R tokenizers.
    token = token.replace("▁", " ").replace("Ġ", " ").strip()
    # Collapse whitespace from merged pieces.
    token = re.sub(r"\s+", " ", token)
    return token


def _readable_token_text(raw: str) -> str:
    """
    Improve readability for glued token strings from subword/tokenizer artifacts.

    This is intentionally conservative: we only add boundaries for very common
    machine-token artifacts (URL joins, camel-case joins, alpha<->digit joins).
    We avoid aggressive dictionary-based splitting to prevent corrupting words.
    """
    text = _normalize_token(raw)
    if not text or " " in text:
        return text

    # Break URL-like concatenations.
    text = re.sub(r"(?i)(https?)(?=[A-Za-z0-9])", r" \1", text).strip()
    text = re.sub(r"(?i)(www)(?=[A-Za-z0-9])", r" \1", text).strip()

    # Split CamelCase boundaries.
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)

    # Add boundaries between numbers and letters when tokenizer output glues them.
    text = re.sub(r"(?<=[0-9])(?=[A-Za-z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])(?=[0-9])", " ", text)

    # Light cleanup for occasional repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _collapse_subword_contributions(
    raw_tokens: list[Any],
    raw_scores: np.ndarray,
    *,
    token_to_text: Any | None = None,
) -> list[_TokenContribution]:
    """
    Collapse contiguous XLM-R/GPT-style subword pieces into cleaner word-level units.

    Why this exists:
    - SHAP returns tokenizer pieces, not always user-facing words.
    - UI expects readable "token-like" chunks for ranking/highlighting.
    - We preserve contribution math by summing scores of continuation pieces.
    """
    collapsed: list[_TokenContribution] = []
    current_pieces: list[str] = []
    current_score = 0.0

    def _flush_current() -> None:
        nonlocal current_pieces, current_score
        if current_pieces:
            if token_to_text:
                try:
                    # Prefer tokenizer-native reconstruction for language-safe spacing.
                    text = _normalize_token(str(token_to_text(current_pieces)))
                except Exception:
                    # Fallback keeps behavior stable if tokenizer helper is unavailable.
                    text = _normalize_token("".join(current_pieces))
            else:
                text = _normalize_token("".join(current_pieces))
            collapsed.append(_TokenContribution(text=text, score=current_score))
            current_pieces = []
            current_score = 0.0

    for raw_token, raw_score in zip(raw_tokens, raw_scores):
        raw_text = str(raw_token or "")
        has_leading_boundary = bool(re.match(r"^[\s▁Ġ]+", raw_text))
        piece = _normalize_token(raw_text)
        if not piece:
            _flush_current()
            continue

        starts_new_word = (
            raw_text.startswith("▁")
            or raw_text.startswith("Ġ")
            or has_leading_boundary
        )
        piece_is_punct = bool(_PUNCT_RE.match(piece))

        if piece_is_punct:
            _flush_current()
            continue

        if starts_new_word or not current_pieces:
            _flush_current()
            current_pieces = [raw_text]
            current_score = float(raw_score)
        else:
            # No leading word-boundary marker means this is a continuation piece.
            current_pieces.append(raw_text)
            current_score += float(raw_score)

    _flush_current()
    return collapsed


def _normalize_match_text(raw: str) -> str:
    text = (raw or "").lower().strip()
    if not text:
        return ""
    text = (
        text.replace("’", "'")
        .replace("`", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("-", " ")
    )
    text = re.sub(r"[\"'`]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


_NORMALIZED_KEYWORDS: dict[str, list[str]] = {
    name: sorted(
        {_normalize_match_text(k) for k in kws if _normalize_match_text(k)},
        key=len,
        reverse=True,
    )
    for name, kws in _INDICATOR_KEYWORDS.items()
}


def _is_useful_token(token: str) -> bool:
    if not token.strip():
        return False
    if _PUNCT_RE.match(token):
        return False
    return bool(_WORD_RE.search(token))


def _match_indicator(token_text: str) -> str | None:
    # Phrase-aware matching with deterministic priority for overlapping keywords.
    low = _normalize_match_text(token_text)
    padded = f" {low} "
    for indicator_name in _INDICATOR_PRIORITY:
        for kw in _NORMALIZED_KEYWORDS.get(indicator_name, []):
            if low == kw or f" {kw} " in padded:
                return indicator_name
    return None


def _predict_proba_fn(bundle: XLMRBundle):
    """
    Return a prediction callable for SHAP:
    input list[str] -> ndarray[n_samples, n_labels] of softmax probabilities.
    """
    import torch

    tokenizer = bundle.tokenizer
    model = bundle.model
    device = bundle.device

    def _predict(texts) -> np.ndarray:
        # SHAP may pass numpy/object arrays; normalize to plain list[str] for tokenizer.
        if isinstance(texts, np.ndarray):
            batch = texts.reshape(-1).tolist()
        elif isinstance(texts, (list, tuple)):
            batch = list(texts)
        else:
            batch = [texts]
        batch = [x if isinstance(x, str) else str(x) for x in batch]

        encoded = tokenizer(
            batch,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits
            probs = torch.softmax(logits / xlmr_inference_temperature(), dim=-1)
        return probs.detach().cpu().numpy()

    return _predict


@lru_cache(maxsize=1)
def _get_shap_explainer() -> Any:
    import shap
    from inference.xlmr.loader import load_bundle

    bundle = load_bundle()
    predict_fn = _predict_proba_fn(bundle)
    masker = shap.maskers.Text(bundle.tokenizer)
    return shap.Explainer(predict_fn, masker)


def _extract_class_values(values: Any, predicted_class_index: int) -> np.ndarray:
    arr = np.asarray(values)
    # Common shape for text classifier explanations: [samples, tokens, classes].
    if arr.ndim == 3:
        return arr[0, :, predicted_class_index]
    # Fallback shape: [tokens, classes]
    if arr.ndim == 2 and arr.shape[1] > predicted_class_index:
        return arr[:, predicted_class_index]
    # Last fallback: flat contributions already selected for one class.
    return arr.reshape(-1)


def build_shap_explanation(
    text: str,
    predicted_class_index: int,
    *,
    max_evals: int | None = None,
) -> ExplanationResult:
    """
    Build SHAP explanation payload for one text sample.

    Raises exceptions if SHAP cannot run; caller should handle fallback behavior.
    """
    clipped = _truncate_text_by_words(text)
    if not clipped.strip():
        return ExplanationResult(note=_EXPLANATION_NOTE, top_tokens=[], indicators=[])

    # Fast path: cache exact SHAP output for repeated selections.
    cached = _get_cached_explanation(
        clipped,
        predicted_class_index=predicted_class_index,
        max_evals=max_evals or shap_max_evals(),
        max_words=shap_max_words(),
        top_k=shap_top_k(),
        temperature=xlmr_inference_temperature(),
    )
    if cached is not None:
        return cached

    explainer = _get_shap_explainer()
    # Keep evaluation budget bounded for API latency.
    used_max_evals = max_evals or shap_max_evals()
    shap_values = explainer([clipped], max_evals=used_max_evals)

    raw_tokens = list(np.asarray(shap_values.data[0]).tolist())
    raw_scores = _extract_class_values(shap_values.values, predicted_class_index)

    merged_scores: dict[str, float] = {}
    token_to_text = getattr(getattr(explainer, "masker", None), "tokenizer", None)
    convert_tokens_to_string = getattr(token_to_text, "convert_tokens_to_string", None)
    collapsed_tokens = _collapse_subword_contributions(
        raw_tokens,
        raw_scores,
        token_to_text=convert_tokens_to_string if callable(convert_tokens_to_string) else None,
    )
    for item in collapsed_tokens:
        token = _readable_token_text(item.text)
        if not _is_useful_token(token):
            continue
        merged_scores[token] = merged_scores.get(token, 0.0) + float(item.score)

    ranked = sorted(
        (_TokenContribution(text=t, score=s) for t, s in merged_scores.items()),
        key=lambda x: abs(x.score),
        reverse=True,
    )[: shap_top_k()]

    top_tokens: list[ExplanationTopToken] = []
    grouped_abs_sum: dict[str, float] = {}
    grouped_tokens: dict[str, set[str]] = {}

    for item in ranked:
        indicator = _match_indicator(item.text)
        top_tokens.append(
            ExplanationTopToken(
                text=item.text,
                score=round(abs(item.score), 6),
                direction="supports_predicted_class" if item.score >= 0 else "opposes_predicted_class",
                indicator=indicator,
            )
        )
        if indicator:
            grouped_abs_sum[indicator] = grouped_abs_sum.get(indicator, 0.0) + abs(item.score)
            grouped_tokens.setdefault(indicator, set()).add(item.text)

    total_grouped = sum(grouped_abs_sum.values())
    indicators: list[ExplanationIndicator] = []
    if total_grouped > 0 and math.isfinite(total_grouped):
        for indicator_name, abs_sum in sorted(grouped_abs_sum.items(), key=lambda x: x[1], reverse=True):
            pct = (abs_sum / total_grouped) * 100.0
            indicators.append(
                ExplanationIndicator(
                    name=indicator_name,
                    contribution_percent=round(pct, 2),
                    tokens=sorted(grouped_tokens.get(indicator_name, set())),
                    summary=_INDICATOR_SUMMARY[indicator_name],
                )
            )

    result = ExplanationResult(note=_EXPLANATION_NOTE, top_tokens=top_tokens, indicators=indicators)
    _put_cached_explanation(
        clipped,
        predicted_class_index=predicted_class_index,
        max_evals=used_max_evals,
        max_words=shap_max_words(),
        top_k=shap_top_k(),
        temperature=xlmr_inference_temperature(),
        value=result,
    )
    return result


# -----------------------------------------------------------------------------
# SHAP result caching
# -----------------------------------------------------------------------------

_SHAP_CACHE: "OrderedDict[str, ExplanationResult]" = OrderedDict()


def _cache_key(
    text: str,
    *,
    predicted_class_index: int,
    max_evals: int,
    max_words: int,
    top_k: int,
    temperature: float,
) -> str:
    # Hash the text to avoid keeping long strings as keys.
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return f"{h}|c={predicted_class_index}|e={max_evals}|w={max_words}|k={top_k}|t={temperature}"


def _get_cached_explanation(
    text: str,
    *,
    predicted_class_index: int,
    max_evals: int,
    max_words: int,
    top_k: int,
    temperature: float,
) -> ExplanationResult | None:
    if not shap_cache_enabled():
        return None
    maxsize = shap_cache_maxsize()
    if maxsize <= 0:
        return None
    key = _cache_key(
        text,
        predicted_class_index=predicted_class_index,
        max_evals=max_evals,
        max_words=max_words,
        top_k=top_k,
        temperature=temperature,
    )
    value = _SHAP_CACHE.get(key)
    if value is None:
        return None
    # LRU touch
    _SHAP_CACHE.move_to_end(key)
    return value


def _put_cached_explanation(
    text: str,
    *,
    predicted_class_index: int,
    max_evals: int,
    max_words: int,
    top_k: int,
    temperature: float,
    value: ExplanationResult,
) -> None:
    if not shap_cache_enabled():
        return
    maxsize = shap_cache_maxsize()
    if maxsize <= 0:
        return
    key = _cache_key(
        text,
        predicted_class_index=predicted_class_index,
        max_evals=max_evals,
        max_words=max_words,
        top_k=top_k,
        temperature=temperature,
    )
    _SHAP_CACHE[key] = value
    _SHAP_CACHE.move_to_end(key)
    while len(_SHAP_CACHE) > maxsize:
        _SHAP_CACHE.popitem(last=False)


def explanation_unavailable(note: str | None = None) -> ExplanationResult:
    """Fallback payload when SHAP is disabled, fails, or is unavailable."""
    return ExplanationResult(
        note=note or "SHAP explanation is currently unavailable. Classification result is still returned.",
        top_tokens=[],
        indicators=[],
    )
