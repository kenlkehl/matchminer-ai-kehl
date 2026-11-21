"""Guideline features of the app: guideline-space retrieval and workup review for one patient (the Patient tab's
Guidelines), and patient populations with their guideline concordance checks (the Population tab).

The clinical behavior stays in the inference package; this package holds the app's settings for it
(``settings``), the inference adapters (``services``), populations (``population``, ``population_cache``) and
extraction-based workup reviews (``extraction_workup``). Its routes are ``routes.guidelines`` and
``routes.population``.
"""
