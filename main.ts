// Sokoban solver — push-optimal A* per room (MUHSAF build).
//
// Design (written from the competition spec; no copied solver code):
//  * A level is one hallway row (the row holding '@') plus 1-4 rooms.
//    Each room has a single door cell next to a hallway cell; the
//    hallway cell beside it is the room's entry.
//  * Every room is solved INDEPENDENTLY with push-optimal A*:
//      - a state is the sorted box cells plus the player's reachable
//        region (flood fill over the room + its entry cell, boxes as
//        obstacles); one action is one push (cost 1). The player
//        starts on the ENTRY cell, so a box sitting on the door cell
//        is pushed inward as the first push — that push counts.
//      - the heuristic is the minimum-cost matching of boxes to
//        goals over wall-only BFS distances (admissible, and
//        consistent: one push changes it by at most 1), so A* is
//        push-optimal.
//      - deadlock pruning: dead-square precomputation (reverse
//        push-BFS from the goals), wall-only freeze, frozen 2x2
//        blocks.
//      - goal test: every box on a goal AND the player can still
//        reach the door cell (so the player can leave for the next
//        room). The last room is exempt from the exit condition:
//        once its boxes are on goals the level is solved.
//  * Rooms are stitched in door-column order: walk the hallway to
//    the entry, replay the room's pushes (BFS walks between pushes,
//    boxes as obstacles), walk out through the door, continue.
//  * No wall clock is read anywhere; the search budget is a fixed
//    expansion count per room, so two runs are byte-identical.
//
// Contract: `node main.ts` reads /input/<id>/level.txt (or
// MAKEFUN_INPUT_DIR) and writes /output/<id>/solution.txt (or
// MAKEFUN_OUTPUT_DIR). `solveLevel(levelText)` is the pure core,
// exported for tests; the IO wrapper only runs when this file is
// executed directly as a script.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

interface DirStep {
  dx: number;
  dy: number;
  ch: string;
}

const DIRS: DirStep[] = [
  { dx: 0, dy: -1, ch: 'u' },
  { dx: 0, dy: 1, ch: 'd' },
  { dx: -1, dy: 0, ch: 'l' },
  { dx: 1, dy: 0, ch: 'r' },
];

const MAX_EXPANSIONS_PER_ROOM = 50000;
// Practice worst case is 1,104 expansions in a room, so 50,000 is ~45x
// headroom; it also bounds worst-case wall time on the grader's 1 CPU.
// A room that exhausts it no longer returns null outright: solveRoom
// retries with weighted A* and greedy best-first (shared fallback
// budget per level) before giving up on the room.

export interface SolverStats {
  expansions: number[];
  pushes: number[];
  rooms: number;
  // Rooms solved only by the weighted/greedy fallback (non-optimal).
  fallbacks: number;
  // Boxes no room claimed (e.g. a room without a door): attempted via
  // the whole-level fallback and reported here instead of dropped.
  orphans: number;
  // True when the whole-level fallback produced the answer.
  wholeLevel: boolean;
}

export const lastStats: SolverStats = { expansions: [], pushes: [], rooms: 0, fallbacks: 0, orphans: 0, wholeLevel: false };

interface Parsed {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

// Parse XSB into a grid padded with a one-cell wall border, so neighbour
// arithmetic never walks off the array.
function parseLevel(text: string): Parsed {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const w0 = Math.max(...rows.map((r) => r.length));
  const h0 = rows.length;
  const w = w0 + 2;
  const h = h0 + 2;
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  for (let y = 0; y < h0; y++) {
    const row = rows[y] as string;
    for (let x = 0; x < w0; x++) {
      const ch = row[x] ?? '#';
      const i = (y + 1) * w + (x + 1);
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  }
  return { w, h, wall, goals, boxes, player };
}

function boxSetOf(boxes: number[], n: number): Uint8Array {
  const a = new Uint8Array(n);
  for (const b of boxes) a[b] = 1;
  return a;
}

// Cells reachable from `start` walking only on cells where ok is 1 and
// that hold no box. `ok` is 0 on the padded border, so this never
// indexes outside the grid.
function flood(start: number, ok: Uint8Array, boxAt: Uint8Array, w: number): Uint8Array {
  const reach = new Uint8Array(ok.length);
  if (!ok[start] || boxAt[start]) return reach;
  const q: number[] = [start];
  reach[start] = 1;
  for (let head = 0; head < q.length; head++) {
    const c = q[head] as number;
    for (const d of DIRS) {
      const nb = c + d.dy * w + d.dx;
      if (!ok[nb] || boxAt[nb] || reach[nb]) continue;
      reach[nb] = 1;
      q.push(nb);
    }
  }
  return reach;
}

// Shortest walk (u/d/l/r) from start to target over `ok` cells, treating
// boxes as obstacles. Deterministic neighbour order (u, d, l, r).
// Returns null when unreachable.
function bfsWalk(start: number, target: number, ok: Uint8Array, boxAt: Uint8Array, w: number): string | null {
  if (start === target) return '';
  if (!ok[start] || !ok[target] || boxAt[target]) return null;
  const prev = new Int32Array(ok.length).fill(-1);
  const prevCh: string[] = new Array<string>(ok.length).fill('');
  const q: number[] = [start];
  prev[start] = start;
  for (let head = 0; head < q.length; head++) {
    const c = q[head] as number;
    for (const d of DIRS) {
      const nb = c + d.dy * w + d.dx;
      if (!ok[nb] || boxAt[nb] || prev[nb] !== -1) continue;
      prev[nb] = c;
      prevCh[nb] = d.ch;
      if (nb === target) {
        let s = '';
        let cur = nb;
        while (cur !== start) {
          s = (prevCh[cur] as string) + s;
          cur = prev[cur] as number;
        }
        return s;
      }
      q.push(nb);
    }
  }
  return null;
}

function stepChar(from: number, to: number, w: number): string {
  const d = to - from;
  if (d === 1) return 'r';
  if (d === -1) return 'l';
  if (d === w) return 'd';
  return 'u';
}

interface Room {
  cells: number[];
  mask: Uint8Array;
  solveMask: Uint8Array;
  goals: number[];
  boxes: number[];
  door: number;
  entry: number;
}

// Rooms are the connected floor components once the hallway row (the row
// holding '@') is removed. Each room's door is the room cell next to a
// hallway cell (the spec guarantees exactly one); entry is that hallway
// cell. solveMask is the room plus its entry cell: the player region the
// room search is allowed to use.
function findRooms(level: Parsed): Room[] {
  const w = level.w;
  const n = w * level.h;
  const hallY = Math.floor(level.player / w);
  const seen = new Uint8Array(n);
  const rooms: Room[] = [];
  for (let i = 0; i < n; i++) {
    if (level.wall[i] || Math.floor(i / w) === hallY || seen[i]) continue;
    const cells: number[] = [];
    const q: number[] = [i];
    seen[i] = 1;
    for (let head = 0; head < q.length; head++) {
      const c = q[head] as number;
      cells.push(c);
      for (const d of DIRS) {
        const nb = c + d.dy * w + d.dx;
        if (level.wall[nb] || Math.floor(nb / w) === hallY || seen[nb]) continue;
        seen[nb] = 1;
        q.push(nb);
      }
    }
    cells.sort((a, b) => a - b);
    const mask = new Uint8Array(n);
    for (const c of cells) mask[c] = 1;
    let door = -1;
    let entry = -1;
    let bestX = Infinity;
    let bestY = Infinity;
    for (const c of cells) {
      const cx = c % w;
      const cy = Math.floor(c / w);
      for (const d of DIRS) {
        const nb = c + d.dy * w + d.dx;
        if (level.wall[nb] || Math.floor(nb / w) !== hallY) continue;
        if (cx < bestX || (cx === bestX && cy < bestY)) {
          bestX = cx;
          bestY = cy;
          door = c;
          entry = nb;
        }
      }
    }
    if (door === -1) continue;
    const solveMask = mask.slice();
    solveMask[entry] = 1;
    const goals = level.goals.filter((g) => mask[g] === 1);
    const boxes = level.boxes.filter((b) => mask[b] === 1);
    rooms.push({ cells, mask, solveMask, goals, boxes, door, entry });
  }
  return rooms;
}

// All injective assignments of k distinct goal indices out of m.
function permutations(m: number, k: number): number[][] {
  const out: number[][] = [];
  if (k > m) return out;
  const used: boolean[] = new Array<boolean>(m).fill(false);
  const cur: number[] = [];
  const rec = (): void => {
    if (cur.length === k) {
      out.push(cur.slice());
      return;
    }
    for (let j = 0; j < m; j++) {
      if (used[j]) continue;
      used[j] = true;
      cur.push(j);
      rec();
      cur.pop();
      used[j] = false;
    }
  };
  rec();
  return out;
}

interface PushStep {
  from: number;
  dir: number;
}

interface RoomResult {
  pushes: PushStep[];
  expansions: number;
  // True when the pushes came from the weighted/greedy fallback, i.e.
  // they are legal but may be longer than the room's push-optimum.
  fallback: boolean;
}

interface SearchOutcome {
  pushes: PushStep[] | null;
  expansions: number;
  // True only when the search stopped because it hit `cap`, as opposed
  // to exhausting the state space (which means no solution exists).
  capped: boolean;
}

const FALLBACK_EXPANSIONS_PER_ATTEMPT = 25000;
const FALLBACK_BUDGET_PER_LEVEL = 50000;
// Fallback searches share one budget per level so a pathological room
// cannot eat the whole 60 s wall budget on the grader.
export interface FallbackBudget {
  remaining: number;
}

// Push-optimal A* for one room (weight 1), or a weighted/greedy search
// (weight > 1 / Infinity) used as a fallback. The player starts on the
// entry cell. `relaxed` (last room only) drops the "player must reach
// the door" goal condition: once the boxes are on goals the level is
// solved. `cap` bounds expansions; hitting it reports `capped` so the
// caller can retry with a different weight instead of giving up.
function solveRoomSearch(room: Room, level: Parsed, relaxed: boolean, cap: number, weight: number): SearchOutcome {
  const w = level.w;
  const n = w * level.h;
  const nBoxes = room.boxes.length;
  const nGoals = room.goals.length;
  if (nBoxes === 0) return { pushes: [], expansions: 0, capped: false };
  if (nGoals === 0) return { pushes: null, expansions: 0, capped: false };

  // Wall-only BFS distance from each goal over room cells.
  const dist: Int32Array[] = [];
  for (const g of room.goals) {
    const d = new Int32Array(n).fill(-1);
    const q: number[] = [g];
    d[g] = 0;
    for (let head = 0; head < q.length; head++) {
      const c = q[head] as number;
      for (const dir of DIRS) {
        const m = c + dir.dy * w + dir.dx;
        if (!room.mask[m] || d[m] !== -1) continue;
        d[m] = (d[c] as number) + 1;
        q.push(m);
      }
    }
    dist.push(d);
  }

  // Dead squares: reverse push-BFS from the goals. A box at x came from
  // y = x - off, pushed in direction off, with the player standing at
  // y - off; y must be a room cell and y - off a standable cell (a room
  // cell OR the entry cell — a box on the door is pushed in from the
  // entry). Cells never reached can never be pushed onto a goal
  // (ignoring other boxes, which only makes it harder), so a box
  // there is dead.
  const live = new Uint8Array(n);
  const dq: number[] = [];
  for (const g of room.goals) {
    live[g] = 1;
    dq.push(g);
  }
  for (let dh = 0; dh < dq.length; dh++) {
    const x = dq[dh] as number;
    for (const dir of DIRS) {
      const off = dir.dy * w + dir.dx;
      const y = x - off;
      const s = y - off;
      if (y < 0 || y >= n || s < 0 || s >= n) continue;
      if (!room.mask[y] || !room.solveMask[s] || live[y]) continue;
      live[y] = 1;
      dq.push(y);
    }
  }
  const dead = new Uint8Array(n);
  for (const c of room.cells) if (!live[c]) dead[c] = 1;

  const goalSet = new Uint8Array(n);
  for (const g of room.goals) goalSet[g] = 1;

  const perms = nBoxes <= 4 ? permutations(nGoals, nBoxes) : [];
  if (nBoxes <= 4 && perms.length === 0) return { pushes: null, expansions: 0, capped: false };

  // Minimum-cost box->goal matching over wall-only distances. Admissible
  // and consistent: one push moves one box one cell, so the matching
  // cost changes by at most 1 per push. Rooms have at most 4 boxes, so
  // the brute-force matching is cheap; for larger box counts (only the
  // whole-level fallback) use the sum of per-box nearest-goal distances
  // instead — also admissible (it can only undercount the matching),
  // so the optimal search stays optimal and the fallback stays valid.
  const useMatching = nBoxes <= 4 && perms.length > 0;
  const heuristic = (boxes: number[]): number => {
    if (useMatching) {
      let best = Infinity;
      for (const p of perms) {
        let sum = 0;
        for (let i = 0; i < nBoxes; i++) {
          const d = (dist[p[i] as number] as Int32Array)[boxes[i] as number] as number;
          if (d < 0) {
            sum = Infinity;
            break;
          }
          sum += d;
        }
        if (sum < best) best = sum;
      }
      return best;
    }
    let sum = 0;
    for (const b of boxes) {
      let m = Infinity;
      for (const d of dist) {
        const v = d[b] as number;
        if (v >= 0 && v < m) m = v;
      }
      if (m === Infinity) return Infinity;
      sum += m;
    }
    return sum;
  };

  // Canonical player region: the smallest cell reachable from the
  // player. Given fixed boxes, that cell identifies the whole region.
  const regionMin = (player: number, boxes: number[], boxAt?: Uint8Array): number => {
    const reach = flood(player, room.solveMask, boxAt ?? boxSetOf(boxes, n), w);
    for (const c of room.cells) if (reach[c]) return c;
    if (reach[room.entry]) return room.entry;
    return -1;
  };

  const isDeadState = (boxes: number[], boxAt?: Uint8Array): boolean => {
    const at = boxAt ?? boxSetOf(boxes, n);
    for (const b of boxes) {
      if (dead[b]) return true;
      if (goalSet[b]) continue;
      // Wall-only freeze: no direction has a room-cell target and a
      // standable stand cell (a room cell or the entry cell — a box on
      // the door is pushed in from the entry). (Cells blocked only by
      // other boxes do not count: those boxes may move away later.)
      let canMove = false;
      for (const dir of DIRS) {
        const off = dir.dy * w + dir.dx;
        const t = b + off;
        const s = b - off;
        if (t < 0 || t >= n || s < 0 || s >= n) continue;
        if (room.mask[t] && room.solveMask[s]) {
          canMove = true;
          break;
        }
      }
      if (!canMove) return true;
      // Frozen 2x2: a block whose four cells are all wall-or-box holds
      // any box inside it forever, because every push of such a box
      // needs a free cell inside the block (as target or as stand).
      // A cell counts as blocking only if no player can stand on it
      // (not a room cell and not the entry) or it holds a box.
      const bx = b % w;
      const by = Math.floor(b / w);
      for (let ox = -1; ox <= 0; ox++) {
        for (let oy = -1; oy <= 0; oy++) {
          const x0 = bx + ox;
          const y0 = by + oy;
          const c00 = y0 * w + x0;
          const c10 = c00 + 1;
          const c01 = c00 + w;
          const c11 = c01 + 1;
          const blocked = (c: number): boolean => !room.solveMask[c] || at[c] === 1;
          if (blocked(c00) && blocked(c10) && blocked(c01) && blocked(c11)) return true;
        }
      }
    }
    return false;
  };

  const startBoxes = room.boxes.slice().sort((a, b) => a - b);
  if (isDeadState(startBoxes)) return { pushes: null, expansions: 0, capped: false };
  const h0 = heuristic(startBoxes);
  if (h0 === Infinity) return { pushes: null, expansions: 0, capped: false };

  const sBoxes: number[][] = [startBoxes];
  const sPlayer: number[] = [room.entry];
  const sParent: number[] = [-1];
  const sFrom: number[] = [-1];
  const sDir: number[] = [-1];
  const sRegion: number[] = [regionMin(room.entry, startBoxes)];
  const best = new Map<string, number>();
  best.set(startBoxes.join(',') + '|' + (sRegion[0] as number), 0);

  type HeapEntry = [number, number, number, number];
  const heap: HeapEntry[] = [];
  let seq = 0;
  const heapLess = (a: HeapEntry, b: HeapEntry): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: HeapEntry): void => {
    heap.push(e);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i] as HeapEntry, heap[p] as HeapEntry)) break;
      const tmp = heap[i] as HeapEntry;
      heap[i] = heap[p] as HeapEntry;
      heap[p] = tmp;
      i = p;
    }
  };
  const heapPop = (): HeapEntry => {
    const top = heap[0] as HeapEntry;
    const last = heap.pop() as HeapEntry;
    if (heap.length > 0) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1] as HeapEntry, heap[c] as HeapEntry)) c++;
        if (!heapLess(heap[c] as HeapEntry, heap[i] as HeapEntry)) break;
        const tmp = heap[i] as HeapEntry;
        heap[i] = heap[c] as HeapEntry;
        heap[c] = tmp;
        i = c;
      }
    }
    return top;
  };

  heapPush([h0, 0, seq++, 0]);
  let expansions = 0;
  while (heap.length > 0) {
    const entry = heapPop();
    const g = entry[1];
    const u = entry[3];
    const boxes = sBoxes[u] as number[];
    const key = boxes.join(',') + '|' + (sRegion[u] as number);
    if (best.get(key) !== g) continue;
    const player = sPlayer[u] as number;
    const boxAt = boxSetOf(boxes, n);
    const reach = flood(player, room.solveMask, boxAt, w);
    let allGoals = true;
    for (const b of boxes) {
      if (!goalSet[b]) {
        allGoals = false;
        break;
      }
    }
    // Exit check (non-last rooms only): the player must be able to
    // reach the door cell from where the last push left them. A box
    // sitting on the door blocks the flood, so a state with the door
    // occupied is never accepted here — correct, because that box
    // would bar the exit to the next room. (If the door cell itself
    // were a goal, the room would be unsolvable as a non-last room;
    // the generator never produces that: its solved states need the
    // player to leave through the door.) The last room is relaxed:
    // boxes on goals already solve the level, even if the player
    // ends trapped inside.
    if (allGoals && (relaxed || reach[room.door])) {
      const pushes: PushStep[] = [];
      let cur = u;
      while ((sParent[cur] as number) !== -1) {
        pushes.push({ from: sFrom[cur] as number, dir: sDir[cur] as number });
        cur = sParent[cur] as number;
      }
      pushes.reverse();
      return { pushes, expansions, capped: false };
    }
    if (expansions >= cap) return { pushes: null, expansions, capped: true };
    expansions++;
    for (let i = 0; i < nBoxes; i++) {
      const b = boxes[i] as number;
      for (let d = 0; d < 4; d++) {
        const dir = DIRS[d] as DirStep;
        const off = dir.dy * w + dir.dx;
        const t = b + off;
        const s = b - off;
        if (t < 0 || t >= n || s < 0 || s >= n) continue;
        // The box must land inside the room; the player must be able to
        // stand behind it (stand may be the entry cell).
        if (!room.mask[t] || boxAt[t] || !room.solveMask[s] || !reach[s]) continue;
        const nb = boxes.slice();
        nb[i] = t;
        nb.sort((a, b2) => a - b2);
        const nbSet = boxSetOf(nb, n);
        if (isDeadState(nb, nbSet)) continue;
        const h = heuristic(nb);
        if (h === Infinity) continue;
        const reg = regionMin(b, nb, nbSet);
        const nkey = nb.join(',') + '|' + reg;
        const ng = g + 1;
        const prev = best.get(nkey);
        if (prev !== undefined && prev <= ng) continue;
        best.set(nkey, ng);
        const idx = sBoxes.length;
        sBoxes.push(nb);
        sPlayer.push(b);
        sParent.push(u);
        sFrom.push(b);
        sDir.push(d);
        sRegion.push(reg);
        const f = weight === Infinity ? h : ng + weight * h;
        heapPush([f, ng, seq++, idx]);
      }
    }
  }
  return { pushes: null, expansions, capped: false };
}

// Per-room solve: push-optimal A* first. If that search hits its
// expansion cap (not "unsolvable" — just unexplored), retry with
// weighted A* (weight 3) and then greedy best-first, both capped and
// sharing the level's fallback budget. Those retries return a LEGAL
// push sequence that may be longer than P*, which still scores
// 500*P*/pushes instead of 0. A room the optimal search proves
// unsolvable (dead start, exhausted heap) returns null immediately —
// no retry can help there.
function solveRoom(
  room: Room,
  level: Parsed,
  relaxed: boolean,
  budget?: FallbackBudget,
  optCap: number = MAX_EXPANSIONS_PER_ROOM,
  fbCap: number = FALLBACK_EXPANSIONS_PER_ATTEMPT,
): RoomResult | null {
  const first = solveRoomSearch(room, level, relaxed, optCap, 1);
  if (first.pushes !== null) return { pushes: first.pushes, expansions: first.expansions, fallback: false };
  if (!first.capped) return null;
  const b = budget ?? { remaining: 2 * FALLBACK_EXPANSIONS_PER_ATTEMPT };
  let total = first.expansions;
  for (const w of [3, Infinity]) {
    if (b.remaining <= 0) break;
    const cap = Math.min(fbCap, b.remaining);
    const attempt = solveRoomSearch(room, level, relaxed, cap, w);
    total += attempt.expansions;
    b.remaining -= attempt.expansions;
    if (attempt.pushes !== null) return { pushes: attempt.pushes, expansions: total, fallback: true };
    if (!attempt.capped) return null;
  }
  return null;
}

// Replay a push sequence on the full floor: walk to each stand cell
// (boxes as obstacles), push, update. Returns the move string, or null
// if any walk is impossible. Used by the whole-level fallback.
function replayPushes(start: number, pushes: PushStep[], ok: Uint8Array, boxes: number[], w: number, n: number): string | null {
  let moves = '';
  let cur = start;
  let curBoxes = boxes.slice().sort((a, b) => a - b);
  for (const p of pushes) {
    const dir = DIRS[p.dir] as DirStep;
    const off = dir.dy * w + dir.dx;
    const stand = p.from - off;
    const boxAt = boxSetOf(curBoxes, n);
    const walk = bfsWalk(cur, stand, ok, boxAt, w);
    if (walk === null) return null;
    moves += walk;
    moves += dir.ch;
    cur = p.from;
    curBoxes = curBoxes.map((b) => (b === p.from ? p.from + off : b)).sort((a, b) => a - b);
  }
  return moves;
}

export function solveLevel(levelText: string): string {
  lastStats.expansions = [];
  lastStats.pushes = [];
  lastStats.rooms = 0;
  lastStats.fallbacks = 0;
  lastStats.orphans = 0;
  lastStats.wholeLevel = false;
  if (!levelText || !levelText.trim()) return '';
  const level = parseLevel(levelText);
  if (level.player < 0) return '';
  const w = level.w;
  const n = w * level.h;
  const hallY = Math.floor(level.player / w);
  const hallMask = new Uint8Array(n);
  const floorMask = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (level.wall[i]) continue;
    floorMask[i] = 1;
    if (Math.floor(i / w) === hallY) hallMask[i] = 1;
  }
  const rooms = findRooms(level);
  rooms.sort((a, b) => (a.door % w) - (b.door % w) || Math.floor(a.door / w) - Math.floor(b.door / w));
  lastStats.rooms = rooms.length;

  // Boxes no room claims (a room without a door is skipped by
  // findRooms, a box may also sit on the hallway itself) used to be
  // dropped silently and the level scored 0. Try the whole level as
  // one room instead — with <=4 boxes it uses the exact matching
  // heuristic, with more it uses the nearest-goal heuristic. If that
  // also fails, fall through to the per-room pass below; the orphan
  // count stays visible in lastStats instead of vanishing.
  const covered = new Uint8Array(n);
  for (const r of rooms) for (const c of r.cells) covered[c] = 1;
  const orphanBoxes = level.boxes.filter((b) => !covered[b] && !level.goals.includes(b));
  lastStats.orphans = orphanBoxes.length;
  if (orphanBoxes.length > 0) {
    const wholeCells: number[] = [];
    for (let i = 0; i < n; i++) if (floorMask[i]) wholeCells.push(i);
    const whole: Room = {
      cells: wholeCells,
      mask: floorMask,
      solveMask: floorMask,
      goals: level.goals,
      boxes: level.boxes.slice().sort((a, b) => a - b),
      door: level.player,
      entry: level.player,
    };
    const wholeSol = solveRoom(whole, level, true, { remaining: 10000 }, 10000, 5000);
    if (wholeSol !== null) {
      const wholeMoves = replayPushes(level.player, wholeSol.pushes, floorMask, level.boxes, w, n);
      if (wholeMoves !== null) {
        lastStats.wholeLevel = true;
        lastStats.expansions.push(wholeSol.expansions);
        lastStats.pushes.push(wholeSol.pushes.length);
        if (wholeSol.fallback) lastStats.fallbacks++;
        return wholeMoves;
      }
    }
  }

  const boxAt = new Uint8Array(n);
  for (const b of level.boxes) boxAt[b] = 1;

  let moves = '';
  let cur = level.player;
  const fbBudget: FallbackBudget = { remaining: FALLBACK_BUDGET_PER_LEVEL };
  for (let ri = 0; ri < rooms.length; ri++) {
    const room = rooms[ri] as Room;
    const movesBeforeRoom = moves.length;
    // Walk the hallway to this room's entry (fall back to a full-grid
    // walk with boxes as obstacles if the hallway row is split).
    const toEntry =
      bfsWalk(cur, room.entry, hallMask, boxAt, w) ?? bfsWalk(cur, room.entry, floorMask, boxAt, w);
    if (toEntry === null) break;
    moves += toEntry;
    cur = room.entry;
    // A box sitting on the door cell is normal (i001, i003, i005): the
    // room search starts with the player on the entry cell and its
    // first push shoves that box inward. No guard here.
    const sol = solveRoom(room, level, ri === rooms.length - 1, fbBudget);
    if (sol === null) break;
    lastStats.expansions.push(sol.expansions);
    lastStats.pushes.push(sol.pushes.length);
    if (sol.fallback) lastStats.fallbacks++;
    // Replay the room's pushes: walk to the stand cell, then push.
    // The player starts on the entry cell, so a door box is pushed
    // inward by the first push when one sits on the door.
    let rBoxes = room.boxes.slice().sort((a, b) => a - b);
    let ok = true;
    for (const p of sol.pushes) {
      const dir = DIRS[p.dir] as DirStep;
      const off = dir.dy * w + dir.dx;
      const stand = p.from - off;
      const rBoxAt = boxSetOf(rBoxes, n);
      const walk = bfsWalk(cur, stand, room.solveMask, rBoxAt, w);
      if (walk === null) {
        ok = false;
        break;
      }
      moves += walk;
      moves += dir.ch;
      cur = p.from;
      boxAt[p.from] = 0;
      boxAt[p.from + off] = 1;
      rBoxes = rBoxes.map((b) => (b === p.from ? p.from + off : b)).sort((a, b) => a - b);
    }
    if (!ok) break;
    // Independent check of the room result: every room box must sit on
    // a room goal after the replay. If the replay ever diverged from
    // the search, drop this room's moves instead of shipping a
    // half-solved level (which scores 0 either way).
    const roomGoals = new Set<number>(room.goals);
    if (!rBoxes.every((b) => roomGoals.has(b))) {
      moves = moves.slice(0, movesBeforeRoom);
      break;
    }
    if (sol.pushes.length === 0) continue;
    // Walk out through the door to the entry cell. The goal test
    // already guaranteed the player can reach the door for every
    // non-last room (a box sitting on the door blocks the flood, so
    // such a state is never accepted there), so this walk can only
    // fail for the last room — whose goal test is relaxed on purpose.
    // Keeping the moves then is correct: all boxes are already on
    // goals and the level is solved, even if the player ends trapped
    // inside the room.
    const rBoxAt = boxSetOf(rBoxes, n);
    const toDoor = bfsWalk(cur, room.door, room.solveMask, rBoxAt, w);
    if (toDoor === null) break;
    moves += toDoor;
    moves += stepChar(room.door, room.entry, w);
    cur = room.entry;
  }
  return moves;
}

function runMain(): void {
  const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
  const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
  for (const id of readdirSync(I).sort()) {
    const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
    mkdirSync(join(O, id), { recursive: true });
    writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
  }
}

if (resolve(process.argv[1] ?? '') === fileURLToPath(import.meta.url)) {
  runMain();
}
