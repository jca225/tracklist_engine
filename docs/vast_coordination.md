# Vast.ai / cluster coordination (multi-agent)

Multiple agents run in parallel and share **one Vast.ai account** + the pi-storage
cluster. This is the registry + protocol so no two collide. **Read and update this
file before renting a Vast box or starting a long cluster job.**

## Protocol

1. **List before create.** `vastai show instances` first. **Never reuse, stop, or
   destroy a box you did not create** — another agent may be mid-job on it.
2. **One box per agent, distinctly labeled.** Tag your instance (see the registry
   below) so ownership is unambiguous. Tear down **only your labeled box**.
3. **Namespace your outputs.** Write to a path/DB no other agent writes. Declare it
   in the registry. Never write another agent's namespace.
4. **Budget/instance cap.** Agree a ceiling so two boxes don't blow spend or hit the
   account instance limit. Default: at most 1 box per agent at a time.
5. **Separate IP = separate SoundCloud/YouTube rate-limit** — a dedicated box per
   agent avoids a shared download ban.

## Ownership registry

| Agent / task | Instance label | Reads | Writes (namespace) | Touches canonical DB? | Status |
|---|---|---|---|---|---|
| **taste-embed** (this session) | `taste-embed` | SoundCloud (fresh DL) | `data/taste/tail_track_embeds.pkl` (local pickle) | **No** | Mac batch running; Vast TBD |
| analysis / BB10–17 (parallel agent) | *(unknown — confirm)* | pi-storage `objects/` | `music_database.db` (`track_mert_measures`, …), pi-storage `stems/` | **Yes** | running |

## Collision surface, and why taste-embed is isolated

The taste-embed job downloads from SoundCloud and writes a **local pickle** — it does
**not** touch `music_database.db`, pi-storage `objects/`/`stems/`, or any shared job
queue. So at the data layer there is **no write contention** with the analysis agent.
The only shared resource is the **Vast account/instances** — covered by the protocol
above (separate labeled box, list-before-create, destroy-only-yours).

> If taste-embed ever persists to a canonical DB, it must use the **taste warehouse**
> (`data/taste/taste_warehouse.db`, table `sc_track_mert`) — NOT `music_database.db` —
> to stay out of the analysis agent's namespace.
