"""Instability act of the tutorial (tutorial §1, "How well can models reason?").

Parses a corpus of ReasonBench/CacheSaver eval logs (``logs/``, ``models.parquet``)
to show how flaky LLM reasoning is run-to-run, and how the scoring protocol you
pick reshapes perceived capability. Pure file/parquet parsing — no model calls.

Modules: ``attempts`` (log parsing + Game24 re-grading), ``consistency`` (the
seed-variance / scoring-protocol grid), ``game24_tree`` and ``protocols_app``
(self-contained HTML visualizers), and ``serve`` (the unified demo launcher).
"""
