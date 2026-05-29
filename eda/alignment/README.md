# eda/alignment/

Exploratory analysis that consumes **labeling / alignment** output — i.e. EDA
performed *after* ground-truth labeling (and, later, the algorithmic aligner).
Distinct from [eda/corpus_empirics/](../corpus_empirics/), which analyzes the
ingested corpus upstream of alignment.

`eda/` is a cross-cutting leaf in the pipeline DAG
(`core · scrape → ingest → analysis → labeling ⟶ alignment`): it has inbound
edges from multiple stages, organized here by the stage each analysis reads
from. Empty for now — the first post-alignment analyses land here once
ground-truth write-back exists.
