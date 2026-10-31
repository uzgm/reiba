globalThis.tf = {}; // lm.js가 tfjs를 require하지 않도록
const fs = require("fs"), K = require("./lm.js"), { tools, mem } = K;

const rng = seed => { let a = seed >>> 0; const f = () => { a = (a + 0x6D2B79F5) >>> 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  f.int = (lo, hi) => lo + Math.floor(f() * (hi - lo + 1)); f.pick = a => a[Math.floor(f() * a.length)]; f.chance = p => f() < p;
  f.shuf = a => { a = [...a]; for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(f() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }; return f; };

const hasJ = w => { const c = w.charCodeAt(w.length - 1) - 0xAC00; return c >= 0 && c < 11172 && c % 28 !== 0; };
const eun = w => w + (hasJ(w) ? "은" : "는"), iga = w => w + (hasJ(w) ? "이" : "가"), eul = w => w + (hasJ(w) ? "을" : "를");
const ida = w => w + (hasJ(w) ? "이다" : "다"), iya = w => w + (hasJ(w) ? "이야" : "야"), ieyo = w => w + (hasJ(w) ? "이에요" : "예요");

const TAX = [
  { path: ["생물", "동물", "포유류"], leaves: ["고양이", "개", "사자", "호랑이", "고래", "토끼", "말", "소", "돼지", "원숭이", "코끼리", "곰"] },
  { path: ["생물", "동물", "새"], leaves: ["참새", "비둘기", "독수리", "까치", "펭귄", "타조", "앵무새"] },
  { path: ["생물", "동물", "물고기"], leaves: ["고등어", "상어", "연어", "잉어", "붕어"] },
  { path: ["생물", "식물", "꽃"], leaves: ["장미", "튤립", "국화", "백합"] },
  { path: ["생물", "식물", "나무"], leaves: ["소나무", "참나무", "은행나무", "벚나무"] },
  { path: ["음식", "과일"], leaves: ["사과", "배", "포도", "딸기", "귤", "수박"] },
  { path: ["음식", "채소"], leaves: ["당근", "오이", "배추", "양파", "감자"] },
  { path: ["탈것", "자동차"], leaves: ["택시", "버스", "트럭", "승용차"] },
  { path: ["탈것", "항공기"], leaves: ["헬리콥터", "비행기", "열기구"] },
];
const HOLD_LEAF = new Set(["곰", "앵무새", "붕어", "백합", "귤", "감자", "헬리콥터", "참나무"]);
const NAMES = ["철수", "영희", "민수", "지민", "수진", "하준", "지우", "민준", "서윤", "예준"], HOLD_NAMES = ["도윤", "서연", "시우"];
const THINGS = ["책", "가방", "시계", "차", "집", "펜", "모자", "신발"], HOLD_THINGS = ["우산", "열쇠", "지갑"];
const FLY = ["펭귄", "타조"];
const PROPS = [
  ["새", "날 수 있다", "날 수 없다", "날 수 있어", FLY], ["물고기", "헤엄칠 수 있다", "헤엄칠 수 없다", "헤엄칠 수 있어", []],
  ["나무", "뿌리가 있다", "뿌리가 없다", "뿌리가 있어", []], ["꽃", "향기가 난다", "향기가 나지 않는다", "향기가 나", []],
];
const DIM = [
  ["크다", "작다", "커", "작아", "p"], ["빠르다", "느리다", "빨라", "느려", "p"], ["무겁다", "가볍다", "무거워", "가벼워", "p"],
  ["높다", "낮다", "높아", "낮아", "p"], ["길다", "짧다", "길어", "짧아", "t"], ["많다", "적다", "많아", "적어", "t"], ["비싸다", "싸다", "비싸", "싸", "t"],
];
const COND = [
  ["비가 오", "비가 온다", "땅이 젖는다", "땅이 젖지 않는다", "땅이 젖을까", "비가 올까"],
  ["알람이 울리", "알람이 울린다", "철수가 일어난다", "철수가 일어나지 않는다", "철수가 일어날까", "알람이 울릴까"],
  ["철수가 버튼을 누르", "철수가 버튼을 누른다", "문이 열린다", "문이 열리지 않는다", "문이 열릴까", "철수가 버튼을 누를까"],
  ["물이 끓으", "물이 끓는다", "수증기가 난다", "수증기가 나지 않는다", "수증기가 날까", "물이 끓을까"],
  ["눈이 오", "눈이 온다", "길이 막힌다", "길이 막히지 않는다", "길이 막힐까", "눈이 올까", true],
  ["불이 나", "불이 난다", "연기가 난다", "연기가 나지 않는다", "연기가 날까", "불이 날까"],
  ["영희가 문을 두드리", "영희가 문을 두드린다", "민수가 나온다", "민수가 나오지 않는다", "민수가 나올까", "영희가 문을 두드릴까"],
  ["바람이 불", "바람이 분다", "나뭇잎이 흔들린다", "나뭇잎이 흔들리지 않는다", "나뭇잎이 흔들릴까", "바람이 불까"],
  ["수진이 씨앗을 심으", "수진이 씨앗을 심는다", "싹이 난다", "싹이 나지 않는다", "싹이 날까", "수진이 씨앗을 심을까"],
  ["얼음이 녹으", "얼음이 녹는다", "물이 생긴다", "물이 생기지 않는다", "물이 생길까", "얼음이 녹을까"],
  ["하준이 비를 맞", "하준이 비를 맞는다", "옷이 젖는다", "옷이 젖지 않는다", "옷이 젖을까", "하준이 비를 맞을까"],
  ["서윤이 약을 먹으", "서윤이 약을 먹는다", "열이 내린다", "열이 내리지 않는다", "열이 내릴까", "서윤이 약을 먹을까"],
  ["서연이 창문을 열", "서연이 창문을 연다", "바람이 들어온다", "바람이 들어오지 않는다", "바람이 들어올까", "서연이 창문을 열까", true],
  ["시우가 전화를 걸", "시우가 전화를 건다", "벨이 울린다", "벨이 울리지 않는다", "벨이 울릴까", "시우가 전화를 걸까", true],
];

const reset = () => { for (const k of Object.keys(mem)) delete mem[k]; };
const teach = text => { reset(); try { const r = tools(text); return r && r.src === "추론" ? r.a : null; } catch { return null; } };
const verdict = (a, gold) => !!a && (gold === "yes" ? a.startsWith("응") : gold === "no" ? a.startsWith("아니야") : gold === "unk" ? /^(내가 아는 것만|알려준 비교만으로는)/.test(a) : gold.startsWith("top:") ? a.startsWith(gold.slice(4) + "!") : false);

const S = (c, ...v) => ({ c, v });
const isaF = (a, b) => S(`${eun(a)} ${ida(b)}.`, `${eun(a)} ${iya(b)}.`, `${eun(a)} ${ieyo(b)}.`, `${iga(a)} ${ida(b)}.`, `${eun(a)} ${b}에 속한다.`);
const isaQ = (a, b) => S(`${eun(a)} ${b}일까?`, `${eun(a)} ${iya(b)}?`, `${iga(a)} ${iya(b)}?`, `${eun(a)} ${b}인가요?`, `${eun(a)} ${b}에 속해?`);

function make(R, hold) {
  const leaves = t => t.leaves.filter(l => hold === HOLD_LEAF.has(l)), nm = hold ? HOLD_NAMES : NAMES, th = hold ? HOLD_THINGS : THINGS;
  const pickLeaf = () => { const t = R.pick(TAX.filter(t => leaves(t).length)); const l = R.pick(leaves(t)); return { l, chain: [l, ...[...t.path].reverse()] }; };
  const distract = (n, skip) => { const o = []; for (let i = 0; i < n; i++) { const t = R.pick(TAX), c = [R.pick(t.leaves.filter(x => !HOLD_LEAF.has(x) || hold)), ...[...t.path].reverse()]; if (c.some(x => skip.includes(x))) continue; const j = R.int(0, c.length - 2); o.push(isaF(c[j], c[j + 1])); } return o; };
  const fam = R.pick(["isa", "isa", "isa", "prop", "prop", "cmp", "cmp", "cmp", "cond", "cond"]);
  if (fam === "isa") {
    const { l, chain } = pickLeaf(), mode = R.pick(["yes", "yes", "yes", "unk", "no"]);
    if (mode === "yes") { const a = R.int(0, chain.length - 2), b = R.int(a + 1, chain.length - 1), F = []; for (let i = a; i < b; i++) F.push(isaF(chain[i], chain[i + 1])); return { fam, kind: "yes", F: [...F, ...distract(R.int(0, 2), chain)], Q: isaQ(chain[a], chain[b]) }; }
    const other = R.pick(TAX.filter(t => !t.path.includes(chain[1]) && t.path[0] !== chain[chain.length - 1])), tgt = R.pick(other.path.slice(1)), F = [];
    for (let i = 0; i < chain.length - 1; i++) F.push(isaF(chain[i], chain[i + 1]));
    if (mode === "no") return { fam, kind: "no", F: [...F, S(`${eun(l)} ${iga(tgt)} 아니다.`, `${eun(l)} ${iga(tgt)} 아니야.`)], Q: isaQ(l, tgt) };
    return { fam, kind: "unk", F, Q: isaQ(l, tgt) };
  }
  if (fam === "prop") {
    const P = R.pick(PROPS), t = TAX.find(t => t.path.at(-1) === P[0]), pool = t.leaves.filter(x => hold === HOLD_LEAF.has(x) || (hold && !HOLD_LEAF.has(x) && false));
    const ls = pool.length ? pool : t.leaves, l = R.pick(ls), mode = R.pick(["yes", "yes", "unk", FLY.includes(l) ? "no" : "yes"]), gen = S(`${eun(P[0])} ${P[1]}.`, `${P[0]}들은 ${P[1]}.`);
    if (mode === "unk") { const o = R.pick(TAX.filter(x => x.path.at(-1) !== P[0] && x.path.at(-1) !== "물고기")), ol = R.pick(o.leaves.filter(x => !HOLD_LEAF.has(x) || hold)); return { fam, kind: "unk", F: [gen, isaF(ol, o.path.at(-1))], Q: S(`${eun(ol)} ${P[3]}?`, `${iga(ol)} ${P[3]}?`) }; }
    const F = [gen, isaF(l, P[0])]; if (mode === "no") F.push(S(`${eun(l)} ${P[2]}.`)); else if (R.chance(.4)) F.push(...distract(1, [l, P[0]]));
    return { fam, kind: mode, F, Q: S(`${eun(l)} ${P[3]}?`, `${iga(l)} ${P[3]}?`) };
  }
  if (fam === "cmp") {
    const D = R.pick(DIM), pool = D[4] === "p" ? nm : th, n = Math.min(R.int(3, 4), pool.length), E = R.shuf(pool).slice(0, n), F = [];
    for (let i = 0; i < n - 1; i++) { const hi = E[i], lo = E[i + 1]; F.push(S(`${lo}보다 ${iga(hi)} ${D[0]}.`, `${eun(hi)} ${lo}보다 ${D[0]}.`, `${eun(lo)} ${hi}보다 ${D[1]}.`, `${lo}보다 ${iga(hi)} 더 ${D[0]}.`)); }
    const mode = R.pick(["top", "bot", "pair", "pair"]);
    if (mode === "top") return { fam, kind: "top:" + E[0], F, Q: S(`누가 제일 ${D[2]}?`, `누가 가장 ${D[2]}?`, `${D[4] === "p" ? "누가" : "뭐가"} 제일 ${D[2]}?`) };
    if (mode === "bot") return { fam, kind: "top:" + E[n - 1], F, Q: S(`누가 제일 ${D[3]}?`, `누가 가장 ${D[3]}?`) };
    const i = R.int(0, n - 2), j = R.int(i + 1, n - 1), flip = R.chance(.5), x = flip ? E[j] : E[i], y = flip ? E[i] : E[j];
    return { fam, kind: flip ? "no" : "yes", F, Q: S(`${eun(x)} ${y}보다 ${D[2]}?`, `${iga(x)} ${y}보다 ${D[2]}?`, `${eun(x)} ${y}보다 ${D[2]}?`) };
  }
  const C = R.pick(COND.filter(c => !!c[6] === hold)), mode = R.pick(["fwd", "fwd", "back"]), rule = S(`${C[0]}면 ${C[2]}.`, `만약 ${C[0]}면 ${C[2]}.`);
  return mode === "fwd" ? { fam: "cond", kind: "yes", F: [rule, S(`${C[1]}.`)], Q: S(`${C[4]}?`) } : { fam: "cond", kind: "no", F: [rule, S(`${C[3]}.`)], Q: S(`${C[5]}?`) };
}

function render(R, ex, canon) {
  const pick = s => canon || !s.v.length || R.chance(.35) ? s.c : R.pick(s.v);
  const F = canon ? ex.F : (R.chance(.7) ? R.shuf(ex.F) : ex.F), pre = canon ? "" : R.pick(["", "", "", "그럼 ", "자, ", "그러면 "]);
  return F.map(pick).join(" ") + " " + pre + pick(ex.Q);
}

function build(n, hold, seed, seen) {
  const R = rng(seed), out = [], st = {}, bump = (f, k) => { (st[f] = st[f] || { made: 0, noTeacher: 0, mismatch: 0, kept: 0, teacherOnVariant: 0 })[k]++; };
  for (let guard = 0; out.length < n && guard < n * 30; guard++) {
    const ex = make(R, hold); bump(ex.fam, "made");
    const a = teach(render(R, ex, true));
    if (!a) { bump(ex.fam, "noTeacher"); continue; }
    if (!verdict(a, ex.kind)) { bump(ex.fam, "mismatch"); st._bad = st._bad || []; if (st._bad.length < 8) st._bad.push({ q: render(R, ex, true), gold: ex.kind, got: a.split("\n")[0] }); continue; }
    const q = render(R, ex, false); if (seen.has(q)) continue; seen.add(q);
    const t = verdict(teach(q), ex.kind) ? 1 : 0; bump(ex.fam, "kept"); if (t) bump(ex.fam, "teacherOnVariant");
    out.push({ q, a, fam: ex.fam, kind: ex.kind, t });
  }
  return { out, st };
}

if (require.main === module) {
  const [N = 24000, M = 400, seed = 7] = process.argv.slice(2).map(Number), seen = new Set(), t0 = Date.now();
  fs.mkdirSync("data", { recursive: true });
  const tr = build(N, false, seed, seen), ev = build(M, true, seed + 1, seen), w = (f, a) => fs.writeFileSync(f, a.map(x => JSON.stringify(x)).join("\n") + "\n");
  w("data/reason_train.jsonl", tr.out); w("data/reason_eval.jsonl", ev.out);
  fs.writeFileSync("data/reason_stats.json", JSON.stringify({ train: tr.st, eval: ev.st, n: { train: tr.out.length, eval: ev.out.length }, sec: (Date.now() - t0) / 1000 }, null, 1));
  console.log("train", tr.out.length, "eval", ev.out.length, `${((Date.now() - t0) / 1000).toFixed(1)}s`); console.log(JSON.stringify(tr.st, null, 1)); console.log("eval", JSON.stringify(ev.st));
}
module.exports = { build };
