"""Map public drug/biotech tickers to their ClinicalTrials.gov sponsor
name, for cases where the registry's lead-sponsor name differs from the
company name carried in our records.

For most filers the cleaned `company_name` matches the registry sponsor
well enough (ClinicalTrials.gov does a lenient text match on
`query.spons`). This map is the override layer for the long tail where
the legal entity that sponsors trials differs from the listed company
(subsidiaries, post-merger names, common abbreviations).

Coverage focuses on large-cap pharma + active biotech; more can be added
incrementally. A ticker absent from this map falls back to its cleaned
company name in scripts/enrich_pipeline.py.
"""

TICKER_TO_SPONSOR = {
    # Large-cap pharma — registry uses specific legal-entity names
    "LLY": "Eli Lilly and Company",
    "MRK": "Merck Sharp & Dohme LLC",
    "JNJ": "Janssen Research & Development, LLC",
    "ABBV": "AbbVie",
    "BMY": "Bristol-Myers Squibb",
    "PFE": "Pfizer",
    "AZN": "AstraZeneca",
    "NVS": "Novartis Pharmaceuticals",
    "GSK": "GlaxoSmithKline",
    "SNY": "Sanofi",
    "NVO": "Novo Nordisk A/S",
    "RHHBY": "Hoffmann-La Roche",
    "AMGN": "Amgen",
    "GILD": "Gilead Sciences",
    "REGN": "Regeneron Pharmaceuticals",
    "VRTX": "Vertex Pharmaceuticals Incorporated",
    "BIIB": "Biogen",
    "MRNA": "ModernaTX, Inc.",
    "BNTX": "BioNTech SE",
    "TAK": "Takeda",
    # Active mid-cap biotech
    "INCY": "Incyte Corporation",
    "EXEL": "Exelixis",
    "ALNY": "Alnylam Pharmaceuticals",
    "BMRN": "BioMarin Pharmaceutical",
    "SRPT": "Sarepta Therapeutics",
    "NBIX": "Neurocrine Biosciences",
    "JAZZ": "Jazz Pharmaceuticals",
    "HALO": "Halozyme Therapeutics",
}
