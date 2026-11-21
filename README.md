# Sokoban solver (MUHSAF build)

Push-optimal Sokoban solver for the make.fun competition
"Build the best Sokoban solver, 21 November 2026"
(`sokoban-solver-01-run-20261121`).

## Design

- The level is split into one hallway row (the row holding `@`) and
  1–4 rooms. Each room has a single door cell next to a hallway cell
  (its entry).
- Every room is solved **independently** with push-optimal A\*:
  - state = sorted box cells + the player's reachable region
    (flood fill over the room + its entry cell, boxes as obstacles);
    one action = one push (cost 1). The player starts on the entry
    cell, so a box sitting on the door cell is pushed inward as the
    first push.
  - heuristic = minimum-cost matching of boxes to goals over
    wall-only BFS distances (admissible and consistent, so A\* is
    push-optimal).
  - deadlock pruning: dead-square precomputation (reverse push-BFS
    from the goals), wall-only freeze, frozen 2×2 blocks.
  - goal test: every box on a goal AND the player can still reach
    the door cell (so the player can leave for the next room). The
    last room is exempt from the exit condition.
- Rooms are stitched in door-column order: walk the hallway, replay
  the room's pushes (BFS walks between pushes), walk out through
  the door, continue.
- No wall clock is read anywhere; the search budget is a fixed
  expansion count per room, so two runs are byte-identical.

## Run

```sh
node main.ts
```

Reads `/input/<id>/level.txt`, writes `/output/<id>/solution.txt`.
Env overrides: `MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR`.

## Test (practice set, from the kit dir next to this workdir)

```sh
node ../sokoban-solver-01-run-20261121-muhsaf/test.mjs
```

It imports `solveLevel` from `repo/main.ts`, runs all five practice
levels, replays each answer with the kit checker, and prints pushes
vs P\* (13/16/29/47/60), expansions per room, and milliseconds —
plus a second run to check determinism.

## License

MIT — see `LICENSE`.
