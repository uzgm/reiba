#!/usr/bin/env node
/* train_free.js — 자유 생성(free generation) 학습 스크립트

   lm.js의 fit()은 brain에 있는 '고정된 답' 하나를 고르는 방식이라, 이름·추론 사슬이 매번 달라지는
   gen_reason.js 데이터(추론 답)는 배울 수 없다. 이 스크립트는 답 전체를 토큰 단위로 직접 써 내려가는
   (다음 토큰 예측) 모델을 학습한다. lm.js의 Tok(BPE) · LM(트랜스포머) · Fast(추론)를 그대로 쓴다.

   사용법
     node gen_reason.js 24000 400 7                  # 먼저 data/reason_train.jsonl, reason_eval.jsonl 생성
     node train_free.js train [--steps 6000 --B 16 --D 96 --H 4 --L 3 --merges 300 --maxT 256 --cmpw 0.4 --cp 0.10]
     node train_free.js eval  [--model model_free.json --n 400]
     node train_free.js ask "동물은 생물이다. 고양이는 동물이다. 고양이는 생물일까?" [--teacher]

   학습 시퀀스:  Q:{질문}\nA:{답}<eos>   (손실은 답 토큰에 1, 질문 토큰에 --pw(기본 0.05))
   배치 구성:    추론 76% (cmp --cmpw 40%, isa·prop·cond 각 20%) · brain 10% · 모름(OOD) 4% · 글자 복사 --cp(기본 10%)
   글자 복사:    무작위 음절 1~2개 단어를 그대로 따라 쓰기. 단어 길이는 2~5글자(평가 단어는 최대 5글자), 조사 글자도 포함한다.
   저장 형식:    { kind:"free-v1", tok, lm, ... } — lm 부분은 lm.js의 Fast가 그대로 읽는다.
   tfjs:         @tensorflow/tfjs-node가 있으면 그것을(가장 빠름), 없으면 tfjs-backend-wasm, 그것도 없으면 순수 JS cpu를 쓴다.
                 (--backend auto|wasm|cpu)  설치:  npm i @tensorflow/tfjs @tensorflow/tfjs-node   (또는 tfjs-backend-wasm) */
"use strict";
const fs = require("fs");
let tf, isNode = false; try { tf = require("@tensorflow/tfjs-node"); isNode = true; } catch { tf = require("@tensorflow/tfjs"); }
globalThis.tf = tf; // lm.js가 같은 tf를 쓰게 한다
const K = require("./lm.js"), { Tok, LM, Fast } = K;

/* ---------- 인자 ---------- */
const argv = process.argv.slice(2), cmd = argv[0], pos = [];
const opt = { steps: 6000, B: 16, D: 96, H: 4, L: 3, merges: 300, lr: 3e-3, maxT: 256, pw: .05, seed: 7, save: 500, log: 25, n: 400, reserve: 48, cmpw: .4, cp: .10,
  backend: "auto", data: "data", brain: "brain.json", out: "model_free.json", model: "model_free.json", resume: "", teacher: false };
for (let i = 1; i < argv.length; i++) {
  const a = argv[i];
  if (!a.startsWith("--")) { pos.push(a); continue; }
  const k = a.slice(2); if (!(k in opt)) { console.error("알 수 없는 옵션:", a); process.exit(1); }
  opt[k] = typeof opt[k] === "boolean" ? true : typeof opt[k] === "number" ? +argv[++i] : argv[++i];
}

const rng = seed => { let a = seed >>> 0; const f = () => { a = (a + 0x6D2B79F5) >>> 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; f.pick = x => x[Math.floor(f() * x.length)]; return f; };
const verdict = (a, gold) => !!a && (gold === "yes" ? a.startsWith("응") : gold === "no" ? a.startsWith("아니야") : gold === "unk" ? /^(내가 아는 것만|알려준 비교만으로는)/.test(a) : gold.startsWith("top:") ? a.startsWith(gold.slice(4) + "!") : false);

const OOD = ["오늘 날씨 어때", "주식 추천해줘", "점심 뭐 먹지", "맛집 알려줘", "영화 추천해줘", "노래 불러줘", "사랑이 뭐야", "서울 인구는 얼마야", "대통령이 누구야", "파이썬 코드 짜줘", "번역해줘", "농담 하나 해줘", "what is the weather", "tell me a joke", "who won the game", "capital of france"];
const PRE = ["", "", "", "혹시 ", "저기 ", "음 ", "그럼 "], SUF = ["", "", "?", "?", " 알려줘", " 좀", "!"];
const FAMS = ["isa", "prop", "cmp", "cond"];

/* lm.js의 LM과 같은 모델. 단 wasm 백엔드는 tf.gather의 기울기(UnsortedSegmentSum)가 없어서, 임베딩 조회를 oneHot×행렬곱(수학적으로 같음)으로 바꾼다. */
class FLM extends LM {
  static from(o) { const m = new FLM(o.V, o.cfg); m.vars.forEach((v, i) => v.assign(tf.tensor(o.w[i], v.shape))); return m; }
  forward(ids) {
    const [B, T] = ids.shape, { D, H, V } = this, dh = D / H, flat = ids.flatten();
    const emb = tf.getBackend() === "wasm" ? tf.oneHot(flat, V).matMul(this.wte) : tf.gather(this.wte, flat);
    let x = emb.reshape([B, T, D]).add(this.wpe.slice([0, 0], [T, D]));
    const bias = tf.sub(1, tf.linalg.bandPart(tf.ones([T, T]), -1, 0)).mul(-1e9);
    for (const b of this.blocks) {
      const qkv = this.ln(x).reshape([B * T, D]).matMul(b.qkv).reshape([B, T, 3, H, dh]);
      const part = i => qkv.slice([0, 0, i, 0, 0], [B, T, 1, H, dh]).reshape([B, T, H, dh]).transpose([0, 2, 1, 3]);
      const att = part(0).matMul(part(1), false, true).div(Math.sqrt(dh)).add(bias).softmax();
      x = x.add(att.matMul(part(2)).transpose([0, 2, 1, 3]).reshape([B * T, D]).matMul(b.o).reshape([B, T, D]));
      x = x.add(this.ln(x).reshape([B * T, D]).matMul(b.w1).add(b.b1).relu().matMul(b.w2).add(b.b2).reshape([B, T, D]));
    }
    return this.ln(x).reshape([B * T, D]).matMul(this.wte, false, true);
  }
}
async function initBackend() {
  if (!isNode && opt.backend !== "cpu") { try { require("@tensorflow/tfjs-backend-wasm"); await tf.setBackend("wasm"); } catch (e) { if (opt.backend === "wasm") throw e; } }
  await tf.ready();
}

/* ---------- 데이터 ---------- */
function loadData() {
  const rd = f => fs.readFileSync(f, "utf8").split("\n").filter(Boolean).map(l => JSON.parse(l));
  const f1 = `${opt.data}/reason_train.jsonl`, f2 = `${opt.data}/reason_eval.jsonl`;
  if (!fs.existsSync(f1)) { console.error(`${f1} 이(가) 없어요. 먼저 node gen_reason.js 를 실행하세요.`); process.exit(1); }
  const tr = rd(f1), ev = fs.existsSync(f2) ? rd(f2) : [];
  const P = K.prep(JSON.parse(fs.readFileSync(opt.brain, "utf8")));
  return { tr, ev, bq: P.pats.map(([q, i]) => ({ q, a: P.ans[i], fam: "brain" })), oq: OOD.map(q => ({ q, a: K.UNKA, fam: "ood" })) };
}
const promptOf = q => "Q:" + q + "\nA:";

function buildTok(D) { // BPE는 학습 데이터로만 만들고, 평가 세트에만 있는 글자는 '글자'로만 어휘에 넣는다(합쳐진 토큰은 만들지 않는다)
  const text = [...D.tr, ...D.bq, ...D.oq].map(e => promptOf(e.q) + e.a).concat([PRE.join(" ") + " " + SUF.join(" ")]).join("\n");
  const tok = new Tok(text, opt.merges), extra = new Set();
  for (const e of D.ev) for (const c of promptOf(e.q) + e.a) if (!tok.id.has(c)) extra.add(c);
  for (const c of extra) { tok.id.set(c, tok.vocab.length); tok.vocab.push(c); }
  return { tok, extra: [...extra] };
}

/* ---------- 생성 (순수 JS · KV 캐시) ---------- */
function makeGen(o) {
  const tok = Tok.fromJSON(o.tok), fast = new Fast(o.lm), T = o.lm.cfg.T;
  return (q, { max = 190, reserve = opt.reserve } = {}) => {
    let qq = q.trim(), ids; // 너무 길면 앞쪽을 자른다(질문은 맨 끝에 있다)
    while ((ids = tok.encode(promptOf(qq))).length > T - reserve && qq.length > 1) qq = qq.slice(1);
    const st = fast.state(); let lg; for (const id of ids) lg = fast.step(st, id);
    const out = [];
    while (out.length < max && st.n < T) {
      lg[1] = -Infinity; let b = 0; for (let i = 1; i < lg.length; i++) if (lg[i] > lg[b]) b = i;
      if (b === 0) break; out.push(b); if (st.n >= T) break; lg = fast.step(st, b);
    }
    return tok.decode(out);
  };
}

/* ---------- 평가 ---------- */
function evaluate(o, D, n = opt.n) {
  const gen = makeGen(o), R = rng(99), res = {}, fails = [], nf = {};
  const run = (name, rows, kindOf) => {
    const s = res[name] = {};
    for (const e of rows) {
      const f = s[e.fam] = s[e.fam] || { n: 0, verdict: 0, ground: 0, exact: 0 }, a = gen(e.q), gold = kindOf ? kindOf(e) : null, last = x => x.split("\n").pop();
      f.n++; if (a === e.a) f.exact++; const vOk = gold ? verdict(a, gold) : a === e.a; if (vOk) f.verdict++; if (vOk && last(a) === last(e.a)) f.ground++; // ground: 판정이 맞고 ⇒ 결론 줄(또는 한 줄 답 전체)까지 같다 = 우연히 맞힌 게 아님
      const key = name + "/" + e.fam; nf[key] = nf[key] || 0; // 틀린 예는 (평가 묶음, 유형)마다 3개까지, 답 전체(여러 줄)를 남긴다
      if (a !== e.a && fails.length < 40 && nf[key] < 3) { nf[key]++; fails.push({ set: name, fam: e.fam, q: e.q, gold: e.a, got: a, pass: gold ? verdict(a, gold) : false }); }
    }
  };
  const sample = (a, k) => { const c = [...a], out = []; while (out.length < k && c.length) out.push(c.splice(Math.floor(R() * c.length), 1)[0]); return out; };
  if (D.ev.length) run("hold-out(새 이름·사물)", D.ev.slice(0, n), e => e.kind);
  run("train 샘플(본 문제)", sample(D.tr, Math.min(200, n)), e => e.kind);
  run("brain 암기", D.bq, null); run("모름(OOD)", D.oq, null);
  const tot = {}; for (const [name, s] of Object.entries(res)) { let N = 0, V = 0, G = 0, X = 0; for (const f of Object.values(s)) { N += f.n; V += f.verdict; G += f.ground; X += f.exact; } tot[name] = { n: N, verdict: V / N, ground: G / N, exact: X / N }; }
  return { res, tot, fails };
}
function printEval(r) {
  const p = x => (100 * x).toFixed(1).padStart(5) + "%", ind = (t, pad) => String(t).replace(/\n/g, "\n" + pad);
  for (const [name, s] of Object.entries(r.res)) {
    console.log(`\n[${name}]  판정 ${p(r.tot[name].verdict)} · 판정+결론줄 ${p(r.tot[name].ground)} · 완전일치 ${p(r.tot[name].exact)}  (n=${r.tot[name].n})`);
    for (const [f, v] of Object.entries(s)) console.log(`  ${f.padEnd(5)} n=${String(v.n).padStart(3)}  판정 ${p(v.verdict / v.n)}  판정+결론줄 ${p(v.ground / v.n)}  완전일치 ${p(v.exact / v.n)}`);
  }
  if (r.fails.length) { console.log("\n틀린 예 (일부, 답 전체):"); for (const f of r.fails) console.log(`  [${f.set} · ${f.fam}] ${f.q}\n    정답: ${ind(f.gold, "          ")}\n    모델: ${ind(f.got, "          ")}${f.pass ? "\n    (판정은 맞고 뒷줄이 다름)" : ""}`); }
}

/* ---------- 학습 ---------- */
async function train() {
  const t0 = Date.now(), D = loadData(), R = rng(opt.seed);
  let tok, m, extra = [], T;
  if (opt.resume) { const o = JSON.parse(fs.readFileSync(opt.resume, "utf8")); tok = Tok.fromJSON(o.tok); m = FLM.from(o.lm); T = o.lm.cfg.T; console.log(`이어서 학습: ${opt.resume} (이전 ${o.steps}스텝)`); }
  else ({ tok, extra } = buildTok(D));
  const enc = e => { const p = tok.encode(promptOf(e.q)); return { ids: [...p, ...tok.encode(e.a), 0], np: p.length, fam: e.fam }; };
  const all = { reason: D.tr.map(enc) };
  if (!opt.resume) { // 가장 긴 시퀀스에 맞춰 T를 정한다(상한 maxT)
    let mx = 0; for (const e of [...all.reason, ...D.bq.map(enc), ...D.oq.map(enc)]) mx = Math.max(mx, e.ids.length);
    T = Math.min(opt.maxT, mx + 8);
  }
  const byFam = {}; for (const f of FAMS) byFam[f] = all.reason.filter(e => e.fam === f && e.ids.length <= T);
  const dropped = all.reason.length - FAMS.reduce((a, f) => a + byFam[f].length, 0);
  const syl = tok.vocab.filter(t => t.length === 1 && /[가-힣]/.test(t)); // 글자 복사 과제용 음절(평가 전용 글자 포함 — 임베딩이 학습되게. 조사 글자도 들어 있다)
  // (변경) 복사할 단어는 2~5글자(평가 단어는 최대 5글자: 행글라이더). 짧은 쪽이 조금 더 자주 나온다.
  const clen = () => R.pick([2, 2, 3, 3, 3, 4, 4, 5]);
  const copyEx = () => { const w = [...Array(1 + (R() < .4))].map(() => [...Array(clen())].map(() => R.pick(syl)).join("")).join(" "); return enc({ q: R.pick(["따라 해: ", "이렇게 말해: ", "반복해: "]) + w, a: w, fam: "copy" }); };
  const aug = q => { let s = q; if (s.length > 3 && R() < .12) { const i = 1 + Math.floor(R() * (s.length - 2)); s = s.slice(0, i) + s.slice(i + 1); } return R.pick(PRE) + s + R.pick(SUF); };
  // 추론 문제는 유형별 비중을 둔다: cmp(정렬이 필요해 가장 어렵다)를 --cmpw, 나머지 셋은 균등 분배
  const live = FAMS.filter(f => byFam[f].length), fw = f => f === "cmp" ? opt.cmpw : (1 - opt.cmpw) / 3, tw = live.reduce((a, f) => a + fw(f), 0);
  const pickFam = () => { let r = R() * tw; for (const f of live) { r -= fw(f); if (r < 0) return f; } return live[live.length - 1]; };
  const draw = () => { // 배치 하나는 한 종류(추론/brain/모름/복사)에서만 뽑는다
    const aq = rows => () => { const e = R.pick(rows); return enc({ q: aug(e.q), a: e.a, fam: e.fam }); };
    const pr = 1 - .14 - opt.cp, r = R(), one = r < pr ? () => R.pick(byFam[pickFam()]) : r < pr + .10 ? aq(D.bq) : r < pr + .14 ? aq(D.oq) : copyEx;
    return [...Array(opt.B)].map(one).filter(e => e.ids.length <= T);
  };
  if (!m) { if (tf.disposeVariables) tf.disposeVariables(); m = new FLM(tok.size, { D: opt.D, H: opt.H, L: opt.L, T }); }
  console.log(`tfjs ${tf.version.tfjs} backend=${tf.getBackend()} | V=${tok.size} T=${T} D=${m.D} H=${m.H} L=${m.L} 파라미터=${m.nParams.toLocaleString()}`);
  console.log(`학습 ${D.tr.length} · brain ${D.bq.length} · OOD ${D.oq.length} · 평가 ${D.ev.length} | T 초과로 제외 ${dropped} | 평가 전용 글자 ${extra.length}개를 어휘·복사과제에 포함`);
  console.log(`배치 구성: 추론 ${(100 * (1 - .14 - opt.cp)).toFixed(0)}% (cmp ${(100 * opt.cmpw).toFixed(0)}%) · brain 10% · OOD 4% · 복사 ${(100 * opt.cp).toFixed(0)}% | 유형별 문제 수 ${FAMS.map(f => `${f} ${byFam[f].length}`).join(" · ")}`);

  const save = (steps, loss) => fs.writeFileSync(opt.out, JSON.stringify({ kind: "free-v1", tok: tok.toJSON(), lm: m.toJSON(), steps, loss, opt: { ...opt, T }, made: new Date().toISOString() }));
  let avg = null;
  for (let s = 1; s <= opt.steps; s++) {
    const bt = draw(); if (!bt.length) continue;
    const Tb = Math.max(...bt.map(e => e.ids.length)), xs = [], ys = [], ws = [];
    for (const { ids, np } of bt) for (let t = 0; t < Tb; t++) { xs.push(ids[t] ?? 0); ys.push(ids[t + 1] ?? 0); ws.push(t + 1 >= ids.length ? 0 : t + 1 < np ? opt.pw : 1); }
    const warm = Math.min(1, s / 100), cos = .1 + .9 * .5 * (1 + Math.cos(Math.PI * s / opt.steps));
    m.opt.learningRate = opt.lr * warm * cos;
    const l = m.step(xs, ys, bt.length, Tb, ws); avg = avg === null ? l : avg * .95 + l * .05;
    if (!isFinite(l)) { console.error("손실이 발산했어요(NaN). --lr 을 낮춰보세요."); process.exit(1); }
    if (s % opt.log === 0 || s === opt.steps) { const el = (Date.now() - t0) / 1000; console.log(`step ${s}/${opt.steps}  loss ${avg.toFixed(4)}  lr ${m.opt.learningRate.toExponential(1)}  ${(s / el).toFixed(2)} step/s  남은 시간 ${Math.round(el / s * (opt.steps - s) / 60)}분`); }
    if (opt.save > 0 && s % opt.save === 0 && s < opt.steps) save(s, avg);
  }
  save(opt.steps, avg); console.log(`저장: ${opt.out} (${(fs.statSync(opt.out).size / 1e6).toFixed(1)}MB)`);
  const o = JSON.parse(fs.readFileSync(opt.out, "utf8")), r = evaluate(o, D); printEval(r);
  fs.mkdirSync(opt.data, { recursive: true }); fs.writeFileSync(`${opt.data}/free_eval.json`, JSON.stringify({ tot: r.tot, res: r.res, steps: opt.steps, loss: avg, fails: r.fails }, null, 1));
}

/* ---------- 실행 ---------- */
async function main() {
  if (cmd === "train") { await initBackend(); return train(); }
  if (cmd === "eval") { const o = JSON.parse(fs.readFileSync(opt.model, "utf8")); return printEval(evaluate(o, loadData())); }
  if (cmd === "ask") {
    const q = pos.join(" "); if (!q) { console.error('질문을 넣어주세요: node train_free.js ask "질문"'); process.exit(1); }
    const o = JSON.parse(fs.readFileSync(opt.model, "utf8")), gen = makeGen(o);
    console.log(gen(q));
    if (opt.teacher) { for (const k of Object.keys(K.mem)) delete K.mem[k]; const t = K.tools(q); console.log("\n--- 교사(lm.js tools) ---\n" + (t ? t.a : "(답 없음)")); }
    return;
  }
  console.log("사용법: node train_free.js train|eval|ask  (파일 위쪽 주석 참고)");
}
if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
module.exports = { makeGen, evaluate, verdict };
