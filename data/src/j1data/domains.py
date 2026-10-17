"""Transparent measurement heuristics: topical domain, spelling variety and
structured-format presence.

These labels exist to *measure* the mixture. They are lexicon heuristics with
known error, not trained classifiers, and they never decide admission.
"""

from __future__ import annotations

import re

import numpy as np

from j1data.ngrams import U64, word_hashes

VERSION = "j1.domains.v1"

DOMAIN_LEXICON: dict[str, frozenset[str]] = {
    "mathematics": frozenset(
        [
            "theorem",
            "lemma",
            "proof",
            "equation",
            "integral",
            "derivative",
            "polynomial",
            "matrix",
            "vector",
            "algebra",
            "calculus",
            "geometry",
            "triangle",
            "probability",
            "fraction",
            "numerator",
            "denominator",
            "logarithm",
            "exponent",
            "function",
            "prime",
            "integer",
            "coefficient",
            "quadratic",
            "theorem",
            "corollary",
            "eigenvalue",
            "topology",
            "arithmetic",
            "sqrt",
            "frac",
            "solve",
            "inequality",
            "factor",
            "divisible",
            "congruent",
            "hypotenuse",
            "radius",
            "circumference",
            "perpendicular",
            "parabola",
            "series",
            "converges",
        ]
    ),
    "computing": frozenset(
        [
            "code",
            "function",
            "variable",
            "python",
            "javascript",
            "java",
            "compiler",
            "algorithm",
            "programming",
            "software",
            "database",
            "query",
            "server",
            "api",
            "library",
            "class",
            "method",
            "array",
            "string",
            "integer",
            "loop",
            "recursion",
            "debug",
            "runtime",
            "git",
            "linux",
            "kernel",
            "http",
            "json",
            "html",
            "css",
            "framework",
            "install",
            "command",
            "terminal",
            "syntax",
            "repository",
            "deploy",
            "thread",
            "memory",
            "pointer",
            "stack",
            "queue",
            "hash",
            "binary",
            "bash",
            "shell",
            "docker",
            "sql",
            "github",
            "npm",
            "pip",
            "import",
        ]
    ),
    "physical_science": frozenset(
        [
            "physics",
            "energy",
            "force",
            "mass",
            "velocity",
            "acceleration",
            "momentum",
            "quantum",
            "electron",
            "proton",
            "neutron",
            "atom",
            "molecule",
            "chemical",
            "reaction",
            "chemistry",
            "compound",
            "element",
            "ion",
            "bond",
            "oxidation",
            "acid",
            "base",
            "gravity",
            "wavelength",
            "photon",
            "thermodynamics",
            "entropy",
            "temperature",
            "pressure",
            "magnetic",
            "electric",
            "voltage",
            "current",
            "circuit",
            "particle",
            "orbit",
            "planet",
            "star",
            "galaxy",
            "astronomy",
            "telescope",
        ]
    ),
    "life_science": frozenset(
        [
            "cell",
            "cells",
            "biology",
            "protein",
            "dna",
            "rna",
            "gene",
            "genes",
            "genetic",
            "species",
            "evolution",
            "organism",
            "bacteria",
            "virus",
            "enzyme",
            "tissue",
            "organ",
            "membrane",
            "mitochondria",
            "photosynthesis",
            "ecosystem",
            "plant",
            "plants",
            "animal",
            "animals",
            "metabolism",
            "chromosome",
            "mutation",
            "neuron",
            "immune",
            "hormone",
        ]
    ),
    "health_medicine": frozenset(
        [
            "patient",
            "patients",
            "disease",
            "symptoms",
            "treatment",
            "diagnosis",
            "clinical",
            "medical",
            "medicine",
            "therapy",
            "doctor",
            "health",
            "hospital",
            "infection",
            "cancer",
            "diabetes",
            "vaccine",
            "drug",
            "dose",
            "blood",
            "heart",
            "surgery",
            "chronic",
            "nutrition",
        ]
    ),
    "engineering": frozenset(
        [
            "engineering",
            "design",
            "structure",
            "load",
            "beam",
            "material",
            "materials",
            "manufacturing",
            "mechanical",
            "electrical",
            "sensor",
            "control",
            "system",
            "signal",
            "power",
            "engine",
            "motor",
            "turbine",
            "voltage",
            "torque",
            "efficiency",
            "prototype",
            "specification",
        ]
    ),
    "humanities_social": frozenset(
        [
            "history",
            "war",
            "century",
            "king",
            "empire",
            "government",
            "political",
            "politics",
            "economy",
            "economic",
            "society",
            "social",
            "culture",
            "religion",
            "church",
            "philosophy",
            "literature",
            "novel",
            "poem",
            "author",
            "language",
            "law",
            "court",
            "rights",
            "election",
            "market",
            "trade",
            "art",
            "music",
        ]
    ),
}

_WORD_DOMAINS: dict[str, tuple[str, ...]] = {}
for _domain, _words in DOMAIN_LEXICON.items():
    for _word in _words:
        _WORD_DOMAINS[_word] = (*_WORD_DOMAINS.get(_word, ()), _domain)

# American / British spelling pairs counted as whole words.
_SPELLING_PAIRS = [
    ("color", "colour"),
    ("colors", "colours"),
    ("behavior", "behaviour"),
    ("favorite", "favourite"),
    ("center", "centre"),
    ("centers", "centres"),
    ("meter", "metre"),
    ("meters", "metres"),
    ("liter", "litre"),
    ("analyze", "analyse"),
    ("analyzed", "analysed"),
    ("organize", "organise"),
    ("organized", "organised"),
    ("organization", "organisation"),
    ("realize", "realise"),
    ("realized", "realised"),
    ("recognize", "recognise"),
    ("optimize", "optimise"),
    ("optimization", "optimisation"),
    ("defense", "defence"),
    ("license", "licence"),
    ("gray", "grey"),
    ("program", "programme"),
    ("traveled", "travelled"),
    ("labor", "labour"),
    ("neighbor", "neighbour"),
    ("honor", "honour"),
    ("humor", "humour"),
    ("catalog", "catalogue"),
    ("modeling", "modelling"),
    ("aluminum", "aluminium"),
    ("fiber", "fibre"),
    ("theater", "theatre"),
    ("characterize", "characterise"),
    ("minimize", "minimise"),
    ("maximize", "maximise"),
    ("utilize", "utilise"),
]
_US = {us: idx for idx, (us, _) in enumerate(_SPELLING_PAIRS)}
_UK = {uk: idx for idx, (_, uk) in enumerate(_SPELLING_PAIRS)}

_FORMAT_PATTERNS: dict[str, re.Pattern[str]] = {
    # Horizontal whitespace only: a cross-line \s* would rescan long blank runs from every line start.
    "markdown": re.compile(
        r"^(?:#{1,6} \S|```|[ \t]*[-*] \S.*\n[ \t]*[-*] \S|\|.+\|[ \t]*\n\|[ \t:|-]+\|)", re.MULTILINE
    ),
    "json": re.compile(r"^[ \t]*[\[{][ \t]*\n[ \t]*\"[^\"\n]+\"[ \t]*:", re.MULTILINE),
    "yaml": re.compile(r"^(?:---[ \t]*\n)?[A-Za-z_][\w-]*:[ \t]*(?:\n[ \t]+[-\w]|\S)", re.MULTILINE),
    "xml_html": re.compile(
        r"<(?:\?xml|!DOCTYPE|html|div|span|[a-z]+:[a-z]+|[A-Z][A-Za-z]+)[\s>/]", re.MULTILINE
    ),
    "shell": re.compile(
        r"^(?:\$ \S|#!/(?:usr/)?bin/(?:env )?(?:ba|z)?sh|sudo |apt(?:-get)? install |pip install )",
        re.MULTILINE,
    ),
    "latex": re.compile(r"\\(?:frac|begin\{|sum_|int_|mathbb|sqrt\{)|\$\$"),
    "table": re.compile(r"^\|.+\|[ \t]*$", re.MULTILINE),
}


_DOMAIN_NAMES = sorted(DOMAIN_LEXICON)
_LEXICON_HASHES, _LEXICON_DOMAIN = (
    np.concatenate([word_hashes(sorted(DOMAIN_LEXICON[d])) for d in _DOMAIN_NAMES]),
    np.concatenate([np.full(len(DOMAIN_LEXICON[d]), i) for i, d in enumerate(_DOMAIN_NAMES)]),
)
_LEXICON_ORDER = np.argsort(_LEXICON_HASHES, kind="stable")
_LEXICON_HASHES, _LEXICON_DOMAIN = _LEXICON_HASHES[_LEXICON_ORDER], _LEXICON_DOMAIN[_LEXICON_ORDER]
_US_HASHES = np.unique(word_hashes(sorted(_US)))
_UK_HASHES = np.unique(word_hashes(sorted(_UK)))


def domain_label(signature_hashes: U64, kind: str, min_hits_per_kword: float = 6.0) -> str:
    """Lexicon domain of a document from the word hashes of its matching signature.

    A word listed under several domains counts for each of them."""
    if kind == "code":
        return "code"
    if signature_hashes.size == 0:
        return "general"
    left = np.searchsorted(_LEXICON_HASHES, signature_hashes, side="left")
    right = np.searchsorted(_LEXICON_HASHES, signature_hashes, side="right")
    hits = right > left
    if not hits.any():
        return "general"
    counts = np.zeros(len(_DOMAIN_NAMES), dtype=np.int64)
    for start, stop in zip(left[hits], right[hits], strict=True):
        np.add.at(counts, _LEXICON_DOMAIN[start:stop], 1)
    best = int(np.argmax(counts))
    if counts[best] >= 3 and counts[best] / (signature_hashes.size / 1000.0) >= min_hits_per_kword:
        return _DOMAIN_NAMES[best]
    return "general"


def spelling_counts(signature_hashes: U64) -> tuple[int, int]:
    return (
        int(np.count_nonzero(np.isin(signature_hashes, _US_HASHES))),
        int(np.count_nonzero(np.isin(signature_hashes, _UK_HASHES))),
    )


def formats_present(text: str) -> list[str]:
    return [name for name, pattern in _FORMAT_PATTERNS.items() if pattern.search(text)]
