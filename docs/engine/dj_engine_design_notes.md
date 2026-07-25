Claude Code, while maximizing short-term productivity, to some extent does slow down long-term productivity...
This was particularly evident in our tracklist engine, and this is meant as a means of counteracting it. specifically, working with pseudocode i can understand and stop claude code from misalignment... and patch-work fixes. 


So we theoretically have three DAGS. One for the alignment algorithm, one for the labeling algorithm, and
one for the cotrianing phase. I think the latter may encapsulate the former actually...

So the DAG is:

web crawl -> tokenize -> ingest -> analyze -> ... ?


There is an honest fork at analyze, because we go to both labeling and alignment, which both circle
back to ingest based on whether the correct version was downloaded


label -> ableton interpreter -> ground truth

We have made a webassembly interpreter, so this should be digestable we just need to give it some time.



NB: Use:
frozen dataclasses or Pydantic models,
Protocol for interfaces,
enums rather than free-text strings,
opaque typed IDs,
strict static checking with Pyright,
schema validation at process boundaries,
PostgreSQL constraints for durable invariants.