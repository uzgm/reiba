/* 진짜 작은 LLM: BPE 토크나이저 + 디코더 전용 트랜스포머 (tf.js로 브라우저에서 직접 학습) */
(function (root) {
  let tf = root.tf || (typeof require !== "undefined" ? require("@tensorflow/tfjs") : null);
  const SEP = "\u0001";

  function mergePair(s, a, b) {
    const o = [];
    for (let i = 0; i < s.length; i++) {
      if (i < s.length - 1 && s[i] === a && s[i + 1] === b) { o.push(a + b); i++; } else o.push(s[i]);
    }
    return o;
  }
  const pieces = t => t.match(/\s|[^\s]+/g) || [];

  /* ---------- BPE 토크나이저: 자주 붙는 글자쌍을 합쳐 서브워드를 만든다 ---------- */
  class Tok {
    constructor(text, nMerges) {
      const freq = new Map();
      for (const p of pieces(text)) freq.set(p, (freq.get(p) || 0) + 1);
      let words = [...freq].map(([w, c]) => ({ s: [...w], c }));
      this.merges = [];
      for (let i = 0; i < nMerges; i++) {
        const cnt = new Map();
        for (const { s, c } of words)
          for (let j = 0; j < s.length - 1; j++) { const k = s[j] + SEP + s[j + 1]; cnt.set(k, (cnt.get(k) || 0) + c); }
        let best = null, bc = 1;
        for (const [k, c] of cnt) if (c > bc) { best = k; bc = c; }
        if (!best) break;
        const [a, b] = best.split(SEP);
        this.merges.push([a, b]);
        for (const w of words) w.s = mergePair(w.s, a, b);
      }
      this.vocab = ["<eos>", "<unk>", ...new Set([...text]), ...this.merges.map(([a, b]) => a + b)];
      this.vocab = [...new Set(this.vocab)];
      this.id = new Map(this.vocab.map((t, i) => [t, i]));
      this.cache = new Map();
    }
    get size() { return this.vocab.length; }
    toJSON() { return { merges: this.merges, vocab: this.vocab }; }
    static fromJSON(o) { const t = Object.create(Tok.prototype); t.merges = o.merges; t.vocab = o.vocab; t.id = new Map(o.vocab.map((x, i) => [x, i])); t.cache = new Map(); return t; }
    encode(text) {
      const out = [];
      for (const p of pieces(text)) {
        let ids = this.cache.get(p);
        if (!ids) {
          let s = [...p];
          for (const [a, b] of this.merges) if (s.length > 1) s = mergePair(s, a, b);
          ids = s.map(x => this.id.has(x) ? this.id.get(x) : 1);
          this.cache.set(p, ids);
        }
        out.push(...ids);
      }
      return out;
    }
    decode(ids) { return ids.map(i => (i < 2 ? "" : this.vocab[i])).join(""); }
  }

  /* ---------- 트랜스포머 ---------- */
  class LM {
    constructor(V, { D = 64, H = 4, L = 2, T = 128 } = {}) {
      Object.assign(this, { V, D, H, L, T });
      const r = (...s) => tf.variable(tf.randomNormal(s, 0, 0.02));
      const z = (...s) => tf.variable(tf.zeros(s));
      this.wte = r(V, D); this.wpe = r(T, D);
      this.blocks = [...Array(L)].map(() => ({ qkv: r(D, 3 * D), o: r(D, D), w1: r(D, 4 * D), b1: z(4 * D), w2: r(4 * D, D), b2: z(D) }));
      this.vars = [this.wte, this.wpe, ...this.blocks.flatMap(b => Object.values(b))];
      this.opt = tf.train.adam(3e-3);
    }
    toJSON() { return { cfg: { D: this.D, H: this.H, L: this.L, T: this.T }, V: this.V, w: this.vars.map(v => Array.from(v.dataSync(), x => +x.toPrecision(5))) }; }
    static fromJSON(o) { const m = new LM(o.V, o.cfg); m.vars.forEach((v, i) => v.assign(tf.tensor(o.w[i], v.shape))); return m; }
    get nParams() { return this.vars.reduce((a, v) => a + v.size, 0); }
    ln(x) { const { mean, variance } = tf.moments(x, 2, true); return x.sub(mean).div(variance.add(1e-5).sqrt()); }
    forward(ids) { // ids: int32 [B,T] -> logits [B*T,V]
      const [B, T] = ids.shape, { D, H } = this, dh = D / H;
      let x = tf.gather(this.wte, ids.flatten()).reshape([B, T, D]).add(this.wpe.slice([0, 0], [T, D]));
      const bias = tf.sub(1, tf.linalg.bandPart(tf.ones([T, T]), -1, 0)).mul(-1e9); // 미래 토큰 가리기
      for (const b of this.blocks) {
        const qkv = this.ln(x).reshape([B * T, D]).matMul(b.qkv).reshape([B, T, 3, H, dh]);
        const part = i => qkv.slice([0, 0, i, 0, 0], [B, T, 1, H, dh]).reshape([B, T, H, dh]).transpose([0, 2, 1, 3]);
        const att = part(0).matMul(part(1), false, true).div(Math.sqrt(dh)).add(bias).softmax();
        const o = att.matMul(part(2)).transpose([0, 2, 1, 3]).reshape([B * T, D]).matMul(b.o).reshape([B, T, D]);
        x = x.add(o);
        x = x.add(this.ln(x).reshape([B * T, D]).matMul(b.w1).add(b.b1).relu().matMul(b.w2).add(b.b2).reshape([B, T, D]));
      }
      return this.ln(x).reshape([B * T, D]).matMul(this.wte, false, true);
    }
    step(xs, ys, B, T, ws) { // 다음 토큰 예측 손실을 줄이는 방향으로 가중치 업데이트
      return tf.tidy(() => {
        const cost = this.opt.minimize(() => {
          const logits = this.forward(tf.tensor2d(xs, [B, T], "int32"));
          return tf.losses.softmaxCrossEntropy(tf.oneHot(tf.tensor1d(ys, "int32"), this.V), logits, ws ? tf.tensor1d(ws) : undefined);
        }, true, this.vars);
        return cost.dataSync()[0];
      });
    }
    async generate(prompt, { temp = 0.7, topK = 8, max = 160 } = {}, onTok) {
      const ctx = prompt.slice(-this.T), out = [];
      for (let i = 0; i < max; i++) {
        const win = ctx.slice(-this.T), n = win.length;
        const pt = tf.tidy(() => {
          const lg = this.forward(tf.tensor2d(win, [1, n], "int32")).slice([n - 1, 0], [1, this.V]).flatten();
          return temp < 0.05 ? lg : lg.div(temp).softmax();
        });
        const p = Array.from(await pt.data()); pt.dispose();
        p[1] = -Infinity; if (temp >= 0.05) p[1] = 0;
        let id;
        const idx = p.map((v, i) => i).sort((a, b) => p[b] - p[a]);
        if (temp < 0.05) id = idx[0];
        else {
          const top = idx.slice(0, topK), sum = top.reduce((a, i) => a + p[i], 0);
          let r = Math.random() * sum; id = top[top.length - 1];
          for (const i of top) { r -= p[i]; if (r <= 0) { id = i; break; } }
        }
        if (id === 0) break;
        out.push(id); ctx.push(id);
        if (onTok) onTok(id);
        await new Promise(r => setTimeout(r));
      }
      return out;
    }
  }

    /* ---------- 지식 정리 · 학습 데이터 증강 ---------- */
  const UNKA = "그건 아직 배우지 못했어. brain에 지식 JSON을 추가하고 다시 학습시키면 답할 수 있어.";
  const OOD = ["오늘 날씨 어때", "주식 추천해줘", "점심 뭐 먹지", "맛집 알려줘", "영화 추천해줘", "노래 불러줘", "사랑이 뭐야", "서울 인구는 얼마야", "대통령이 누구야", "파이썬 코드 짜줘", "번역해줘", "농담 하나 해줘", "what is the weather", "tell me a joke", "who won the game", "capital of france", "how tall is everest", "recommend a movie", "what is love", "write me a poem", "비트코인 가격", "로또 번호 알려줘", "내일 비 와", "축구 경기 결과", "지구는 얼마나 커", "공룡이 뭐야", "커피 만드는 법", "가장 높은 산은", "what is the meaning of life", "how old are you"];
  const PRE = ["", "", "", "혹시 ", "저기 ", "음 ", "그럼 ", "please ", "hey "], SUF = ["", "", "?", "?", " 알려줘", " 좀", " 해줘", "!", " please"];
  const SYL = "가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허고노도로모보소오조초코토포호구누두루무부수우주추쿠투푸후";
  const pick = a => a[Math.floor(Math.random() * a.length)];
  const gib = () => [...Array(1 + Math.floor(Math.random() * 3))].map(() => [...Array(2 + Math.floor(Math.random() * 3))].map(() => pick([...SYL])).join("")).join(" ");
  function aug(q) { // 말투·오타를 섞어 같은 질문을 다양하게 보여준다
    let s = q; const r = Math.random();
    if (s.length > 3) { const i = 1 + Math.floor(Math.random() * (s.length - 2)); if (r < .12) s = s.slice(0, i) + s.slice(i + 1); else if (r < .2) s = s.slice(0, i) + s[i] + s.slice(i); }
    return pick(PRE) + s + pick(SUF);
  }
  function prep(brain) {
    const ans = [], tags = [], pats = [];
    for (const d of Object.values(brain)) for (const e of d.entries || []) { const i = ans.length; ans.push(e.answer); tags.push(e.tags || []); for (const p of e.patterns || []) pats.push([p, i]); }
    ans.push(UNKA); tags.push([]);
    return { ans, tags, pats, unk: ans.length - 1 };
  }
  const sig = P => { let h = 5381; for (const a of P.ans) for (let i = 0; i < a.length; i++) h = (h * 33 + a.charCodeAt(i)) >>> 0; return h + ":" + P.ans.length; };

  /* ---------- 검색(글자 2-gram TF-IDF): 새 지식을 학습 전에도 바로 쓰게 해주는 보조 신호 ---------- */
  const clean = s => s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();
  const grams = s => { const g = []; for (const w of clean(s).split(" ")) { if (!w) continue; const p = "^" + w + "$"; for (let i = 0; i < p.length - 1; i++) g.push(p.slice(i, i + 2)); } return g; };
  class Retr {
    constructor(P) {
      this.P = P; this.D = []; const df = new Map();
      for (const [p, i] of P.pats) { const g = grams(p); this.D.push({ p, i, g }); for (const k of new Set(g)) df.set(k, (df.get(k) || 0) + 1); }
      const N = this.D.length || 1; this.idf = k => Math.log(1 + N / (1 + (df.get(k) || 0)));
      for (const d of this.D) d.v = this.vec(d.g);
    }
    vec(g) { const m = new Map(); for (const k of g) m.set(k, (m.get(k) || 0) + 1); let n = 0; for (const [k, c] of m) { const w = (1 + Math.log(c)) * Math.pow(this.idf(k), Retr.pow); m.set(k, w); n += w * w; } m.n = Math.sqrt(n) || 1; return m; }
    search(q) {
      const qv = this.vec(grams(q)), cq = clean(q), words = cq.split(" "), sc = new Array(this.P.ans.length).fill(0), bp = [];
      for (const d of this.D) { let s = 0; for (const [k, w] of qv) if (d.v.has(k)) s += w * d.v.get(k); s /= qv.n * d.v.n; if (s > sc[d.i]) { sc[d.i] = s; bp[d.i] = d.p; } }
      this.P.tags.forEach((tags, i) => { for (const k of tags) { const c = clean(k); if (c.length > 1 && (/^[a-z0-9 ]+$/.test(c) ? (c.includes(" ") ? cq.includes(c) : words.includes(c)) : cq.includes(c))) { sc[i] = Math.min(1, sc[i] + .3); break; } } });
      return { sc, bp };
    }
  }

  /* ---------- 답변 트라이: 신경망이 고른 토큰이 항상 ‘실제 지식’ 안에 머물게 한다 ---------- */
  function trie(A) {
    const root = { k: new Map(), a: [], end: -1, d: 0 }; let bd = 0;
    A.forEach((ids, ai) => { let n = root; n.a.push(ai); ids.forEach((id, d) => { if (!n.k.has(id)) n.k.set(id, { k: new Map(), a: [], end: -1, d: d + 1 }); n = n.k.get(id); n.a.push(ai); }); n.end = ai; });
    (function walk(n) { if (n.k.size > 1) bd = Math.max(bd, n.d); n.k.forEach(walk); })(root);
    return { root, bd };
  }

  /* ---------- 추론 전용 트랜스포머: tf.js 없이 순수 JS + KV 캐시 (토큰당 O(길이)) ---------- */
  const ln = x => { const n = x.length; let m = 0; for (const v of x) m += v; m /= n; let s = 0; for (const v of x) s += (v - m) * (v - m); s = Math.sqrt(s / n + 1e-5); return x.map(v => (v - m) / s); };
  const mv = (h, W, n, m) => { const o = new Float32Array(m); for (let i = 0; i < n; i++) { const hi = h[i]; for (let j = 0; j < m; j++) o[j] += hi * W[i * m + j]; } return o; };
  class Fast {
    constructor(o) {
      const c = o.cfg; Object.assign(this, { V: o.V, D: c.D, H: c.H, L: c.L, T: c.T });
      const w = o.w.map(a => Float32Array.from(a)); this.wte = w[0]; this.wpe = w[1];
      this.B = [...Array(this.L)].map((_, l) => { const k = 2 + l * 6; return { qkv: w[k], o: w[k + 1], w1: w[k + 2], b1: w[k + 3], w2: w[k + 4], b2: w[k + 5] }; });
    }
    state() { const z = () => this.B.map(() => new Float32Array(this.T * this.D)); return { n: 0, K: z(), Vv: z() }; }
    step(st, id) { // 토큰 하나를 넣고 다음 토큰 점수(logits)를 돌려준다
      const { D, H, V } = this, dh = D / H, pos = st.n++, x = new Float32Array(D);
      for (let i = 0; i < D; i++) x[i] = this.wte[id * D + i] + this.wpe[pos * D + i];
      this.B.forEach((b, l) => {
        const qkv = mv(ln(x), b.qkv, D, 3 * D), K = st.K[l], Vv = st.Vv[l];
        K.set(qkv.subarray(D, 2 * D), pos * D); Vv.set(qkv.subarray(2 * D), pos * D);
        const a = new Float32Array(D), s = new Float32Array(pos + 1);
        for (let hh = 0; hh < H; hh++) {
          const o = hh * dh; let mx = -1e9, z = 0;
          for (let t = 0; t <= pos; t++) { let d = 0; for (let i = 0; i < dh; i++) d += qkv[o + i] * K[t * D + o + i]; s[t] = d / Math.sqrt(dh); if (s[t] > mx) mx = s[t]; }
          for (let t = 0; t <= pos; t++) { s[t] = Math.exp(s[t] - mx); z += s[t]; }
          for (let t = 0; t <= pos; t++) { const p = s[t] / z; for (let i = 0; i < dh; i++) a[o + i] += p * Vv[t * D + o + i]; }
        }
        const ao = mv(a, b.o, D, D); for (let i = 0; i < D; i++) x[i] += ao[i];
        const f = mv(ln(x), b.w1, D, 4 * D); for (let i = 0; i < 4 * D; i++) f[i] = Math.max(0, f[i] + b.b1[i]);
        const g = mv(f, b.w2, 4 * D, D); for (let i = 0; i < D; i++) x[i] += g[i] + b.b2[i];
      });
      const h = ln(x), lg = new Float32Array(V);
      for (let v = 0; v < V; v++) { let d = 0; for (let i = 0; i < D; i++) d += h[i] * this.wte[v * D + i]; lg[v] = d; }
      return lg;
    }
  }

  /* ---------- Bot: 신경망 점수 + 검색 점수를 합쳐 트라이 위에서 토큰을 하나씩 고른다 ---------- */
    class Bot {
    constructor(brain, model) {
      this.P = prep(brain); this.R = new Retr(this.P); this.model = model || null; this.fast = null;
      try {
        if (model) { this.tok = Tok.fromJSON(model.tok); this.fast = new Fast(model.lm); this.T = model.lm.cfg.T; this.K = model.K; this.A = this.P.ans.map(a => this.tok.encode(a)); this.trie = trie(this.A); this.stale = model.sig !== sig(this.P); }
      } catch { this.fast = null; this.model = null; }
    }
    reply(q) {
      const t0 = performance.now(), { sc, bp } = this.R.search(q), P = this.P, top = Math.max(0, ...sc.slice(0, -1));
      sc[P.unk] = top < Bot.cfg.ut ? Bot.cfg.uv : 0;
      const sug = sc.map((s, i) => i).filter(i => i !== P.unk && sc[i] > .08 && bp[i]).sort((a, b) => sc[b] - sc[a]).slice(0, 3).map(i => bp[i]);
      const done = (text, ai, n, lm) => ({ text, ai, n, lm, ret: top, sug, unk: ai === P.unk, ms: performance.now() - t0 });
      if (!this.fast) { let bi = 0; sc.forEach((s, i) => { if (s > sc[bi]) bi = i; }); return done(P.ans[bi], bi, 0, null); }
      const { tok, fast, T, K } = this; let qq = q.trim().slice(0, 80), ids;
      while ((ids = tok.encode("Q:" + qq + "\nA:")).length > T - K - 2 && qq.length > 1) qq = qq.slice(0, -1);
      const st = fast.state(); let lg = null; for (const id of ids) lg = fast.step(st, id);
      let n = this.trie.root; const out = []; let lp = 0, nb = 0;
      for (;;) {
        if (n.d > 0 && n.a.length === 1) { out.push(...this.A[n.a[0]].slice(n.d)); break; } // 남은 길이 하나뿐이면 그대로 이어 쓴다
        const c = [...n.k]; if (n.end >= 0) c.push([0, null]);
        if (!c.length) break;
        let pk = c[0];
        if (c.length > 1) {
          let lse = 0;
          if (lg) { let mx = -1e9; for (const v of lg) if (v > mx) mx = v; for (const v of lg) lse += Math.exp(v - mx); lse = mx + Math.log(lse); }
          let best = -1e9;
          for (const x of c) { const s = Bot.cfg.b * (lg ? lg[x[0]] - lse : 0) + Bot.cfg.a * (x[1] ? Math.max(...x[1].a.map(a => sc[a])) : 0); if (s > best) { best = s; pk = x; } }
          if (lg) { lp += lg[pk[0]] - lse; nb++; }
        }
        if (!pk[1]) break;
        out.push(pk[0]); n = pk[1]; lg = st.n < T ? fast.step(st, pk[0]) : null;
      }
      let ai = P.unk; { let m = this.trie.root; for (const id of out) { m = m.k.get(id); if (!m) break; } if (m) ai = m.end >= 0 ? m.end : m.a[0]; }
      return done(tok.decode(out), ai, out.length, nb ? Math.exp(lp / nb) : null);
    }
  }

  Retr.pow = 2; Bot.cfg = { a: 20, b: .5, ut: .4, uv: .8 }; // a: 검색 가중치, b: 신경망 가중치, ut/uv: 검색이 약할 때 ‘모름’에 주는 점수

  /* ---------- 학습: 질문(증강) → 답의 앞부분을 맞히게 한다. 브라우저(tf.js)와 Node 모두에서 동작 ---------- */
  function setup(brain, o = {}) {
    const { merges = 200 } = o, P = prep(brain), pool = [...P.pats, ...OOD.map(q => [q, P.unk])];
    const text = [...pool.map(([q, i]) => `Q:${q}\nA:${P.ans[i]}`), PRE.join(" ") + " " + SUF.join(" ") + " " + SYL].join("\n");
    const tok = new Tok(text, merges), A = P.ans.map(a => tok.encode(a)), K = Math.min(48, trie(A).bd + 3), encQ = q => tok.encode("Q:" + q + "\nA:");
    let T = 0; for (const [q] of pool) T = Math.max(T, encQ("혹시 " + q + " 알려줘").length);
    T = Math.max(T, 40) + K + 8;
    const sample = () => { const [q, ai] = Math.random() < .12 ? [gib(), P.unk] : pick(pool), pr = encQ(aug(q)).slice(0, T - K - 2), ids = [...pr, ...A[ai].slice(0, K)]; if (A[ai].length <= K) ids.push(0); return { ids, pr: pr.length }; };
    return { P, tok, K, T, sample };
  }
  async function fit(brain, o = {}, prog) {
    const { steps = 1500, B = 16, D = 64, H = 4, L = 2 } = o, S = setup(brain, o), { P, tok, K, T } = S;
    if (tf.disposeVariables) tf.disposeVariables();
    const m = new LM(tok.size, { D, H, L, T }); let avg = null;
    for (let s = 1; s <= steps; s++) {
      const xs = [], ys = [], ws = [];
      for (let b = 0; b < B; b++) { const { ids, pr } = S.sample(); for (let t = 0; t < T; t++) { xs.push(ids[t] ?? 0); ys.push(ids[t + 1] ?? 0); ws.push(t + 1 >= ids.length ? 0 : t + 1 < pr ? .15 : 1); } }
      m.opt.learningRate = 3e-3 * (1 - .9 * s / steps);
      const l = m.step(xs, ys, B, T, ws); avg = avg === null ? l : avg * .95 + l * .05;
      if (s % 5 === 0 || s === steps) { if (prog) prog(s, steps, avg); await new Promise(r => setTimeout(r)); }
    }
    return { tok: tok.toJSON(), lm: m.toJSON(), loss: avg, steps, K, sig: sig(P), params: m.nParams };
  }

  /* ---------- 도구: 정확해야 하는 일(계산·시간 등)은 신경망 대신 도구가 처리 ---------- */
  const mem = {};
  /* ---------- 계산기: 안전한 재귀 하강 파서 (eval 사용 안 함) ---------- */
function evalExpr(src) {
  const t = src.match(/\d+\.?\d*|[-+*/^%()]/g) || []; let i = 0;
  const peek = () => t[i], next = () => t[i++];
  function prim() { const x = next(); if (x === "(") { const v = add(); if (next() !== ")") throw 0; return v; } if (x === "-") return -pow(); if (x === "+") return pow(); if (x === undefined || !/\d/.test(x[0])) throw 0; return parseFloat(x); }
  function pow() { const b = prim(); if (peek() === "^") { next(); return Math.pow(b, pow()); } return b; }
  function mul() { let v = pow(); while (peek() === "*" || peek() === "/" || peek() === "%") { const o = next(), r = pow(); v = o === "*" ? v * r : o === "/" ? v / r : v % r; } return v; }
  function add() { let v = mul(); while (peek() === "+" || peek() === "-") { const o = next(), r = mul(); v = o === "+" ? v + r : v - r; } return v; }
  const v = add(); if (i < t.length) throw 0; return v;
}

/* ---------- 추론: 사실·규칙·크기 비교를 기억하고, 이어 붙여서 결론을 낸다 ---------- */
const stripP = s => s.trim().replace(/[.!?。？！\s]+$/, "");
const END = /(이?었다|이에요|예요|입니다|합니다|한다|는다|ㄴ다|일까요?|을까요?|ㄹ까요?|할까요?|인가요?|이야|이다|해요|어요|해|어|야|다|까)$/;
const STMT = /(다|야|이에요|예요|입니다)$/;
const stem = p => stripP(p).replace(END, "").trim();
const key = s => { s = s.replace(/\s+/g, ""); return s.length > 2 ? s.replace(/이$/, "") : s; }; // 고양이/고양 같은 받침 없는 ‘이’ 차이를 흡수
const isQ = (raw, s) => /[?？]\s*$/.test(raw.trim()) || /(까요?|냐|나요|가요|뭐야|누구야)$/.test(s);

const ASUF = "(?:다|요|어|아|니다|습니다|지|죠|까|냐|니|워|워요|가|은가|은|을|을까|을까요|ㄹ까|해|해요|하다|아요|어요|지요)?";
const DIMS = [
  ["크기", "크|커|큰|클|큽", "작"], ["속도", "빠르|빨라|빠른|빠를|빠릅", "느리|느려|느린|느릴|느립"],
  ["무게", "무겁|무거|무거운|무거울", "가볍|가벼|가벼운|가벼울"], ["높이", "높", "낮"], ["길이", "길|긴", "짧"],
  ["양", "많", "적"], ["힘", "세|센|셀|강하|강해|강한|강할", "약하|약해|약한|약할"], ["가격", "비싸|비싼|비쌀|비쌉", "싸|싼|쌀|쌉"]
].map(([n, a, b]) => [n, new RegExp(`^(?:${a})${ASUF}$`), new RegExp(`^(?:${b})${ASUF}$`)]);
const adj = t => { t = stripP(t); for (let d = 0; d < DIMS.length; d++) { if (DIMS[d][1].test(t)) return { d, s: 1 }; if (DIMS[d][2].test(t)) return { d, s: -1 }; } return null; };

function logicSent(raw) { // “모든 X는 Y다 / A는 X이다 / A는 Y일까?” 해석
  const s = stripP(raw).replace(/\s(?:모두|전부)\s/, " "), q = isQ(raw, s);
  const m = s.match(/^모든\s*(\S+?)(?:은|는|이|가)?\s+(.+)$/), n = m ? null : s.match(/^(\S+?)(은|는|이|가)\s+(.+)$/);
  if (!m && !n) return null;
  const A = m ? m[1] : n[1], pred = m ? m[2] : n[3], R = m ? m[1] + "는" : n[1] + n[2], t = key(stem(pred));
  if (!t || /보다/.test(pred)) return null;
  if (q) return { type: "q", f: key(A), t, A: R };
  if (!STMT.test(pred) || /[\d,]/.test(pred)) return null;
  return { type: "s", e: { f: key(A), t, p: pred, s } };
}
function cmpSent(raw) { // “A는 B보다 크다 / A가 B보다 클까? / 누가 제일 커?” 해석
  const s = stripP(raw), q = isQ(raw, s);
  let m = s.match(/^(\S+?)(?:은|는|이|가)\s+(\S+?)\s*보다\s*(?:더\s*)?(\S+)$/);
  if (m) {
    const a = adj(m[3]); if (!a) return null; const x = key(m[1]), y = key(m[2]);
    return { type: q ? "cq" : "cs", d: a.d, hi: a.s > 0 ? x : y, lo: a.s > 0 ? y : x, H: a.s > 0 ? m[1] : m[2], L: a.s > 0 ? m[2] : m[1], a };
  }
  m = s.match(/(?:제일|가장|젤)\s*(?:더\s*)?(\S+)/);
  if (m && /누|뭐|무엇|어느|어떤/.test(s)) { const a = adj(m[1]); if (a) return { type: "top", d: a.d, a }; }
  return null;
}
function chain(kb, f, t) { // f에서 t까지 문장들을 이어 붙인 경로(BFS)
  const prev = new Map([[f, null]]), Q = [f];
  while (Q.length) { const x = Q.shift(); for (const e of kb) if (e.f === x && !prev.has(e.t)) { prev.set(e.t, e); Q.push(e.t); } }
  if (f === t || !prev.has(t)) return null;
  const p = []; for (let x = t; prev.get(x); x = prev.get(x).f) p.unshift(prev.get(x)); return p;
}
function known(kb, f) {
  const seen = new Set([f]), Q = [f], out = [];
  while (Q.length) { const x = Q.shift(); for (const e of kb) if (e.f === x) { out.push(e); if (!seen.has(e.t)) { seen.add(e.t); Q.push(e.t); } } }
  return out;
}
function pathOf(E, from, to) {
  const prev = new Map([[from, null]]), Q = [from];
  while (Q.length) { const x = Q.shift(); for (const e of E) if (e.hi === x && !prev.has(e.lo)) { prev.set(e.lo, x); Q.push(e.lo); } }
  if (from === to || !prev.has(to)) return null;
  const p = [to]; for (let x = to; prev.get(x) !== null; x = prev.get(x)) p.unshift(prev.get(x)); return p;
}
function farthest(E, from) {
  const seen = new Set([from]), Q = [from]; let last = from;
  while (Q.length) { const x = Q.shift(); last = x; for (const e of E) if (e.hi === x && !seen.has(e.lo)) { seen.add(e.lo); Q.push(e.lo); } }
  return last;
}
function reason(q) {
  const ss = q.trim().split(/(?<=[.!?。？])\s+/).filter(Boolean), kb = mem.kb || [], cmp = mem.cmp || [], learn = [], ans = [];
  let dirty = false;
  const add = (arr, o, same) => { if (arr.length < 200 && !arr.some(same)) { arr.push(o); dirty = true; } };
  for (const raw of ss) {
    const c = cmpSent(raw);
    if (c) {
      if (c.type === "cs") { add(cmp, { d: c.d, hi: c.hi, lo: c.lo, H: c.H, L: c.L }, e => e.d === c.d && e.hi === c.hi && e.lo === c.lo); learn.push(stripP(raw)); continue; }
      const E = cmp.filter(e => e.d === c.d), nm = new Map(); E.forEach(e => { nm.set(e.hi, e.H); nm.set(e.lo, e.L); });
      if (c.H) { nm.set(c.hi, nm.get(c.hi) || c.H); nm.set(c.lo, nm.get(c.lo) || c.L); }
      const N = x => nm.get(x) || x;
      if (c.type === "cq") {
        const fw = pathOf(E, c.hi, c.lo), bw = pathOf(E, c.lo, c.hi);
        if (fw) { ans.push(`응, 맞아! ${fw.map(N).join(" > ")} (${DIMS[c.d][0]})`); continue; }
        if (bw) { ans.push(`아니야, 오히려 ${bw.map(N).join(" > ")} 이야.`); continue; }
        if (E.some(e => [e.hi, e.lo].some(x => x === c.hi || x === c.lo))) { ans.push("알려준 비교만으로는 알 수 없어."); continue; }
      } else if (E.length) { // top: 가장 ~한 것
        const beat = new Set(E.map(e => c.a.s > 0 ? e.lo : e.hi)), cand = [...new Set(E.flatMap(e => [e.hi, e.lo]))].filter(x => !beat.has(x));
        if (cand.length === 1) {
          const R = c.a.s > 0 ? E : E.map(e => ({ hi: e.lo, lo: e.hi })), far = farthest(R, cand[0]), p = pathOf(R, cand[0], far);
          ans.push(`${N(cand[0])}! (${(c.a.s > 0 ? p : p.reverse()).map(N).join(" > ")})`);
        } else ans.push(`정보가 부족해서 단정할 수 없어. 후보: ${cand.map(N).join(", ")}`);
        continue;
      }
    }
    const l = logicSent(raw); if (!l) continue;
    if (l.type === "s") { add(kb, l.e, e => e.f === l.e.f && e.t === l.e.t); learn.push(l.e.s); continue; }
    if (/^(누구|뭐|무엇)$/.test(l.t)) { const u = known(kb, l.f); if (u.length) ans.push("알고 있는 건:\n" + u.map(e => "• " + e.s).join("\n")); continue; }
    const p = chain(kb, l.f, l.t);
    if (p) ans.push(p.length === 1 ? `응, 알려준 대로야.\n• ${p[0].s}` : `응, 이어서 생각하면 그래!\n${p.map(e => "• " + e.s).join("\n")}\n⇒ ${l.A} ${p[p.length - 1].p}`);
    else if (kb.some(e => e.f === l.f)) ans.push("내가 아는 것만으로는 알 수 없어.");
  }
  if (dirty) { mem.kb = kb; mem.cmp = cmp; root.LMKit.onMem && root.LMKit.onMem(mem); }
  if (ans.length) return ans.join("\n\n");
  if (learn.length) return "알겠어! 기억해둘게:\n" + learn.map(x => "• " + x).join("\n") + "\n이제 이걸로 물어봐!";
}

/* ---------- 간단한 문장제: 숫자와 행동 단서(먹었다·합치면…)를 읽어 식을 세운다 ---------- */
function wordProblem(q0) {
  const q = q0.replace(/(\d),(?=\d{3}\b)/g, "$1");
  if (!/몇|얼마/.test(q)) return;
  const nums = q.match(/\d+(?:\.\d+)?/g); if (!nums || nums.length < 2) return;
  const n = nums.map(Number), fmt = v => (+v.toPrecision(12)).toLocaleString("ko-KR", { maximumFractionDigits: 6 });
  let u = (q.match(/\d+(?:\.\d+)?\s*(개|명|마리|권|장|원|살|번|대|병|송이|자루|시간|분|초|cm|km|kg|잔|그루|켤레|봉지|상자)/) || [])[1] || "", expr, v;
  if (/몇\s*배/.test(q) && n.length === 2) { expr = `${n[0]} ÷ ${n[1]}`; v = n[0] / n[1]; u = "배"; }
  else if (/차이|더\s*많|더\s*적|얼마나\s*(많|적|크|작)/.test(q) && n.length === 2) { expr = `|${n[0]} − ${n[1]}|`; v = Math.abs(n[0] - n[1]); }
  else if (/나누|똑같이|등분/.test(q) && n.length === 2 && n[1] !== 0) { expr = `${n[0]} ÷ ${n[1]}`; v = n[0] / n[1]; }
  else if (/\d\s*배|곱|씩/.test(q)) { expr = n.join(" × "); v = n.reduce((a, b) => a * b, 1); }
  else {
    const MINUS = /먹|남|빼|뺀|줬|주었|주면|주고|잃|썼|쓰|쓴|팔|덜|버렸|떨어|가져가|꺼냈|나가|갔|보냈/, t = [];
    for (const cl of q.split(/[,，.。?？!]\s*|(?:는데|인데|지만|고|며|서)\s+/)) {
      const ns = cl.match(/\d+(?:\.\d+)?/g); if (ns) for (const x of ns) t.push((MINUS.test(cl) ? -1 : 1) * +x);
    }
    if (t.length < 2) return; t[0] = Math.abs(t[0]);
    expr = t.map((x, i) => i ? (x < 0 ? `− ${-x}` : `+ ${x}`) : `${x}`).join(" "); v = t.reduce((a, b) => a + b, 0);
  }
  return `풀이: ${expr} = ${fmt(v)}${u}\n(문장 속 숫자 ${n.length}개와 행동 단서를 읽어서 식을 세웠어)`;
}

/* ---------- 스킬: brain보다 먼저 시도하는 정확한 규칙들 ---------- */
const SK = [
  ["기억", q => {
    if (/내\s*이름.*(뭐|기억|알아)|my name\?|who am i/i.test(q)) return mem.name ? `네 이름은 ${mem.name}이지!` : "아직 이름을 못 들었어. ‘내 이름은 OO’ 라고 말해줘.";
    const m = q.match(/(?:내\s*이름은|my name is)\s*([^\s.!?,]+)/i); if (!m) return;
    mem.name = m[1].replace(/(이야|이에요|입니다|예요|야)$/, ""); root.LMKit.onMem && root.LMKit.onMem(mem); return `반가워, ${mem.name}! 기억해둘게.`;
  }],
  ["구구단", q => {
    const m = q.match(/(\d+)\s*단|구구단\s*(\d+)/); if (!m) return; const n = +(m[1] || m[2]); if (n > 99) return;
    return `${n}단\n` + [...Array(9)].map((_, i) => `${n} × ${i + 1} = ${n * (i + 1)}`).join("\n");
  }],
  ["제곱근", q => {
    const m = q.match(/(?:루트|√|sqrt)\s*\(?\s*(\d+\.?\d*)|(\d+\.?\d*)\s*의?\s*제곱근/i); if (!m) return;
    const n = +(m[1] || m[2]), r = Math.sqrt(n); return `√${n} = ${+r.toPrecision(10)}${Number.isInteger(r) ? " (딱 떨어져!)" : " (근삿값)"}`;
  }],
  ["소수 판별", q => {
    const m = q.match(/\d+/); if (!m || !/소수|prime/i.test(q) || /소수점/.test(q)) return;
    const n = +m[0]; if (n > 1e12) return "너무 커서 오래 걸려. 1조 이하로 물어봐!";
    if (n < 2) return `${n}은(는) 소수가 아니야. 소수는 2 이상의 수야.`;
    const f = []; let x = n; for (let p = 2; p * p <= x; p++) while (x % p === 0) { f.push(p); x /= p; } if (x > 1) f.push(x);
    return f.length === 1 ? `${n}은(는) 소수야! 1과 ${n}으로만 나누어떨어져.` : `${n}은(는) 소수가 아니야. ${n} = ${f.join(" × ")}`;
  }],
  ["계산", q => {
    const t = q.toLowerCase().replace(/\*\*/g, "^").replace(/(\d+\.?\d*)\s*(?:의\s*)?제곱(?!근)/g, "($1)^2")
      .replace(/더하기|플러스|\bplus\b/g, "+").replace(/빼기|마이너스|\bminus\b/g, "-").replace(/곱하기|×|\btimes\b/g, "*")
      .replace(/나누기|÷|\bdivided by\b/g, "/").replace(/(\d)\s*x\s*(?=[\d(])/g, "$1*");
    const m = t.match(/[\d(][\d\s.+\-*/^%()]*/); if (!m || !/\d/.test(m[0]) || !/[+\-*/^%]/.test(m[0])) return;
    let v; try { v = evalExpr(m[0]); } catch { return; }
    if (!isFinite(v)) return "0으로는 나눌 수 없어!";
    return `${m[0].trim().replace(/\*/g, "×").replace(/\//g, "÷")} = ${(+v.toPrecision(12)).toLocaleString("ko-KR", { maximumFractionDigits: 10 })}`;
  }],
  ["시계", q => {
    const d = new Date();
    if (/몇\s*시|지금\s*시간|현재\s*시간|what time/i.test(q)) return `지금은 ${d.toLocaleTimeString("ko-KR", { hour: "numeric", minute: "2-digit" })}이야.`;
    if (/(오늘|지금).*(날짜|며칠|요일)|무슨\s*요일|what day/i.test(q)) return `오늘은 ${d.toLocaleDateString("ko-KR", { year: "numeric", month: "long", day: "numeric", weekday: "long" })}이야.`;
  }],
  ["문장제", q => wordProblem(q)],
  ["추론", q => reason(q)]
];


  const tools = q => { for (const [n, fn] of SK) { const a = fn(q); if (a) return { a, src: n }; } return null; };
  const setTF = t => { tf = t; };

  root.LMKit = { Tok, LM, Fast, Retr, Bot, fit, setup, prep, sig, tools, mem, setTF, UNKA };
  if (typeof module !== "undefined") module.exports = root.LMKit;
})(typeof window !== "undefined" ? window : globalThis);
