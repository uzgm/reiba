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
      return done(out.length ? P.ans[ai] : tok.decode(out), ai, out.length, nb ? Math.exp(lp / nb) : null);
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
const adj = t => { t = stripP(t); for (let d = 0; d < DIMS.length; d++) { if (DIMS[d][1].test(t)) return { d, s: 1 }; if (DIMS[d][2].test(t)) return { d, s: -1 }; } const L = (mem.lex || {})[vstem(t)]; return L && L.d != null ? { d: L.d, s: L.s } : null; }; // 사전에 없으면 mem.lex(되물어 배운 말)
/* ---------- 추론 v2 ----------
   정규식으로 문장 전체를 맞추지 않고 ① 조사로 어절의 역할을 나누고 ② 서술어 어간을 사전에서 찾아
   ③ (주어, 관계, 대상, 부정) 조각으로 조립한 뒤 ④ 규칙(전이·상속·모순·조건)으로 추론한다. */
const PART = [["에게", "to"], ["한테", "to"], ["보다", "than"], ["은", "top"], ["는", "top"], ["이", "sub"], ["가", "sub"], ["을", "obj"], ["를", "obj"], ["도", "also"]];
const ADN = /(있는|없는|하는|되는|먹는|가진|같은|좋은|작은|높은|낮은|많은|적은|넓은|좁은|깊은|얕은)$/; // 명사를 꾸미는 말은 조사가 아니다
const roles = s => stripP(s).split(/\s+/).filter(Boolean).map(raw => {
  if (!ADN.test(raw)) for (const [p, r] of PART) if (raw.length > p.length && raw.endsWith(p)) return { w: raw.slice(0, -p.length), r, raw };
  return { w: raw, r: "bare", raw };
});
const WH = /^(누구|누|뭐|뭘|무엇|무얼|어디|어느|어떤)$/;
const DEIC = /^(나|너|저|내|네|우리|당신|이것|그것|이거|그거|이건|그건|이게|그게)$/;
const REL = { 먹: "eats", 마시: "drinks", 있: "has", 가지: "has", 좋아하: "likes", 싫어하: "dislikes" }; // 서술어 사전. 모르는 어간은 어간 자체가 관계 이름이 된다
const CH = { isa: 1, prop: 1 }; // 이어 붙일 수 있는(전이되는) 관계
const dn = d => typeof d === "number" ? DIMS[d][0] : d.slice(1) + "다 기준";
const fold = w => { // 활용형 받침 ㄴ/ㄹ을 떼서 기본형 쪽으로 (온→오, 갈→가)
  const c = w.charCodeAt(w.length - 1) - 0xAC00; if (!(c >= 0 && c < 11172)) return w;
  const f = c % 28; return f === 4 || f === 8 ? w.slice(0, -1) + String.fromCharCode(0xAC00 + c - f) : w;
};
const vstem = raw => fold(raw.replace(/(?:었|았|였|겠)?(?:는다|습니다|니다|는|은|을까요|을까|ㄹ까요|면|다|요|어|아|지|죠|까|냐)+$/, "").replace(/해$/, "하").replace(/[셔져쳐]$/, m => ({ 셔: "시", 져: "지", 쳐: "치" })[m]));
const same = (a, b) => a.f === b.f && a.t === b.t && (a.r === b.r || (CH[a.r] && CH[b.r])); // isa/prop은 같은 줄기로 본다
const miss = s => { const a = mem.miss || (mem.miss = []); if (!a.includes(s) && a.length < 30) a.push(s); };

/* 문장 하나를 조각으로: 주어·대상·비교 기준·서술어·부정·최상급 플래그 */
function parse(raw) {
  const s = stripP(raw), F = { s, q: isQ(raw, s), neg: false, sup: false };
  const ws = roles(s).filter(x => {
    if (/^(모든|모두|전부)$/.test(x.raw)) return false;
    if (/^(제일|가장|젤)$/.test(x.raw)) { F.sup = true; return false; }
    if (/^(안|못)$/.test(x.raw)) { F.neg = true; return false; }
    return !/^(더|훨씬|조금|좀|매우|아주|정말)$/.test(x.raw);
  });
  if (ws.length < 2) return null;
  const rest = ws.slice(0, -1); let verb = ws[ws.length - 1];
  if (/^(않|못하|못한)/.test(verb.raw) && rest.length > 1) { F.neg = true; verb = rest.pop(); } // “먹지 않는다”: 앞 어절이 서술어
  const tops = rest.filter(x => x.r === "top"), subs = rest.filter(x => x.r === "sub" || x.r === "also");
  const to = rest.find(x => x.r === "to"), subj = tops[0] || (to ? to : subs[0]);
  Object.assign(F, { subj, verb, vraw: verb.raw, than: rest.find(x => x.r === "than"), obj: rest.find(x => x.r === "obj") || (tops[0] || to ? subs[0] : subs[1]) });
  F.cop = /(이다|이야|이에요|예요|에요|입니다|인가요|인가|이냐|일까요|일까|이죠|야)$/.test(verb.raw);
  F.whS = !!subj && WH.test(subj.w);
  F.whO = rest.some(x => x !== subj && (WH.test(x.w) || WH.test(x.raw)));
  F.whV = WH.test(verb.raw.replace(/(이?야|이에요|예요|인가요|일까요?|이냐|이죠)$/, ""));
  if (!subj || F.than) return F;
  const words = s.split(/\s+/), i = words.indexOf(subj.raw), v = vstem(verb.raw), no = /^없/.test(v);
  const tail = rest.slice(rest.indexOf(subj) + 1).filter(x => x !== F.obj).map(x => x.raw);
  let r, t;
  if (/^아[니닙냐]/.test(verb.raw)) { F.neg = true; if (F.obj) { r = "isa"; t = key(F.obj.w); } }
  else if (F.obj && !F.cop) { if (no) F.neg = true; const vs = no ? "있" : v; r = REL[vs] || vs; t = key(F.obj.w); }
  else if (F.cop) { r = "isa"; t = key([...tail, stem(verb.raw)].join("")); }
  else { if (no) F.neg = true; r = "prop"; t = key([...tail, no ? "있" : v].join("")); }
  if (r && t) F.e = { f: key(subj.w), r, t, n: F.neg, p: words.slice(i + 1).join(" "), s, A: subj.raw };
  return F;
}
/* “A면 B다” → 앞절/뒷절 문장 */
function splitCond(s) {
  const w = s.replace(/^(만약|만일|혹시)\s+/, "").split(/\s+/), i = w.findIndex((x, j) => j > 0 && j < w.length - 1 && x.length > 1 && /면$/.test(x));
  return i < 0 ? null : { a: [...w.slice(0, i), w[i].replace(/으?면$/, "다")].join(" "), b: w.slice(i + 1).join(" ") };
}

/* ---------- 추론 규칙 ---------- */
const up = (E, f) => { const m = new Map([[f, []]]); for (const x of new Set(E.map(e => e.t))) { const p = chain(E, f, x); if (p) m.set(x, p); } return m; }; // f에서 이어 올라갈 수 있는 곳(상위 개념)과 그 경로
function lookup(E, f, r, t) { // 참/거짓/모순/모름 + 근거 문장들. 더 가까운(구체적인) 문장이 우선
  const CE = E.filter(e => !e.n && CH[e.r]), c = [];
  if (CH[r]) { const p = chain(CE, f, t); if (p) c.push({ n: false, d: p.length, why: p }); }
  for (const [g, pa] of up(CE, f)) for (const e of E)
    if (e.f === g && e.t === t && (CH[r] ? CH[e.r] && e.n : e.r === r)) c.push({ n: e.n, d: pa.length + 1, why: [...pa, e] });
  if (!c.length) return { v: null };
  const m = Math.min(...c.map(x => x.d)), b = c.filter(x => x.d === m), pos = b.find(x => !x.n), neg = b.find(x => x.n);
  return pos && neg ? { v: "conflict", pos, neg } : { v: !b[0].n, why: b[0].why, d: m };
}
function derive(kb, rules) { // 조건 규칙(modus ponens): 앞절이 사실이면 뒷절을 꺼낸다
  const D = [];
  for (let k = 0; k < 4; k++) {
    let grew = false;
    for (const ru of rules) {
      const E = [...kb, ...D], b = ru.b;
      if (E.some(e => same(e, b))) continue;
      const L = lookup(E, ru.a.f, ru.a.r, ru.a.t);
      if (L.v === !ru.a.n) { D.push({ ...b, d: { rule: ru, why: L.why } }); grew = true; }
    }
    if (!grew) break;
  }
  return D;
}
const lines = why => why.flatMap(e => e.d ? [...lines(e.d.why), "• " + e.d.rule.s] : ["• " + e.s]);

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

const adjOf = raw => adj(stripP(raw));
function reason(q) {
  const ss = q.trim().split(/(?<=[.!?。？])\s+/).filter(Boolean), kb = mem.kb || (mem.kb = []), cmp = mem.cmp || (mem.cmp = []), rules = mem.rules || (mem.rules = []);
  kb.forEach(e => { e.r = e.r || "isa"; e.n = !!e.n; });
  const kk = e => [e.f, e.r, e.t, e.n].join("|"), before = new Set(derive(kb, rules).map(kk)), learn = [], ans = [], chips = [];
  let dirty = false;
  const pack = t => chips.length ? { a: t, chips } : t;
  for (const raw of ss) {
    const s0 = stripP(raw), E = [...kb, ...derive(kb, rules)];
    /* 조건문: 앞절 → 뒷절 규칙으로 저장 */
    const cd = !/\d/.test(raw) && !isQ(raw, s0) && splitCond(s0);
    if (cd) {
      const A = parse(cd.a), B = parse(cd.b);
      if (A && B && A.e && B.e && !DEIC.test(A.subj.w)) {
        const ru = { a: { f: A.e.f, r: A.e.r, t: A.e.t, n: A.e.n }, b: B.e, s: s0 };
        if (rules.length < 100 && !rules.some(x => x.s === ru.s)) { rules.push(ru); dirty = true; }
        learn.push(s0);
      } else { miss(s0); if (A && A.e && !(B && B.e)) ans.push("조건문은 ‘비가 오면 땅이 젖는다’처럼 뒷절에도 주어가 있어야 이해할 수 있어. ‘이해 못한 문장’에 적어뒀어."); }
      continue;
    }
    const F = parse(raw); if (!F || !F.subj) continue;
    /* 크기·속도 같은 비교 */
    if (F.than || F.sup) {
      const a = adjOf(F.vraw), sv = vstem(F.vraw);
      if (!a) {
        if (!F.than || chips.length) continue;
        if (mem.lex && mem.lex[sv]) { miss(F.s); continue; }
        mem.pending = q; dirty = true; chips.push(`'${sv}다'는 크다류`, `'${sv}다'는 작다류`, `'${sv}다'는 기타`);
        ans.push(`'${sv}다'가 어떤 말인지 아직 몰라. 알려줄래?\n• 크다류: 앞에 온 쪽이 더 그래 (철수는 영희보다 ${sv}다 → 철수 > 영희)\n• 작다류: 앞에 온 쪽이 덜 그래 (철수 < 영희)`); continue;
      }
      if (F.neg && F.than) { ans.push("‘~지 않다’는 ‘반대’와 달라서(같을 수도 있어) 비교로는 기억하지 않았어."); continue; }
      const C = cmp.filter(e => e.d === a.d), nm = new Map(); C.forEach(e => { nm.set(e.hi, e.H); nm.set(e.lo, e.L); });
      if (F.than && !F.whS) {
        const x = key(F.subj.w), y = key(F.than.w), hi = a.s > 0 ? x : y, lo = a.s > 0 ? y : x, H = a.s > 0 ? F.subj.w : F.than.w, L = a.s > 0 ? F.than.w : F.subj.w;
        nm.set(hi, nm.get(hi) || H); nm.set(lo, nm.get(lo) || L);
        const N = z => nm.get(z) || z;
        if (!F.q) {
          const rv = pathOf(C, lo, hi);
          if (rv) { ans.push(`앞에서 알려준 ${rv.map(N).join(" > ")}(${dn(a.d)})와 반대라서 모순이야. 이 문장은 기억하지 않았어.`); continue; }
          if (cmp.length < 200 && !C.some(e => e.hi === hi && e.lo === lo)) { cmp.push({ d: a.d, hi, lo, H, L }); dirty = true; }
          learn.push(F.s); continue;
        }
        const fw = pathOf(C, hi, lo), bw = pathOf(C, lo, hi);
        if (fw) ans.push(`응, 맞아! ${fw.map(N).join(" > ")} (${dn(a.d)})`);
        else if (bw) ans.push(`아니야, 오히려 ${bw.map(N).join(" > ")} 이야.`);
        else if (C.some(e => [e.hi, e.lo].some(z => z === hi || z === lo))) ans.push("알려준 비교만으로는 알 수 없어.");
        continue;
      }
      if (F.than && F.whS && F.q) { // 누가 철수보다 커?
        const y = key(F.than.w), N = z => nm.get(z) || z, all = [...nm.keys()].filter(z => z !== y && (a.s > 0 ? pathOf(C, z, y) : pathOf(C, y, z)));
        if (all.length) ans.push(`${all.map(N).join(", ")} (${dn(a.d)})`); else if (nm.has(y)) ans.push("알려준 비교만으로는 알 수 없어.");
        continue;
      }
      if (F.sup && F.q && F.whS && C.length) { // 누가 제일 커?
        const beat = new Set(C.map(e => a.s > 0 ? e.lo : e.hi)), cand = [...new Set(C.flatMap(e => [e.hi, e.lo]))].filter(z => !beat.has(z)), N = z => nm.get(z) || z;
        if (cand.length === 1) {
          const R = a.s > 0 ? C : C.map(e => ({ hi: e.lo, lo: e.hi })), far = farthest(R, cand[0]), p = pathOf(R, cand[0], far);
          ans.push(`${N(cand[0])}! (${(a.s > 0 ? p : p.reverse()).map(N).join(" > ")})`);
        } else ans.push(`정보가 부족해서 단정할 수 없어. 후보: ${cand.map(N).join(", ")}`);
      }
      continue;
    }
    /* 사실·관계 */
    const f = key(F.subj.w);
    if (F.q) {
      if (F.whV) { const u = known(E, f); if (u.length) ans.push("알고 있는 건:\n" + u.map(e => "• " + e.s).join("\n")); continue; }
      if (F.whS && F.e) { const h = E.filter(e => e.r === F.e.r && e.t === F.e.t && !e.n); if (h.length) ans.push(h.map(e => "• " + e.s).join("\n")); continue; }
      if (F.whO) { // 고양이는 뭘 먹어?
        const v = vstem(F.vraw), r = REL[v] || v, U = up(E.filter(e => !e.n && CH[e.r]), f), h = E.filter(e => e.r === r && !e.n && U.has(e.f));
        if (h.length) ans.push("알고 있는 건:\n" + h.map(e => `• ${F.subj.raw} ${e.p}`).join("\n")); else if (E.some(e => e.f === f)) ans.push("내가 아는 것만으로는 알 수 없어.");
        continue;
      }
      if (!F.e) continue;
      const L = lookup(E, f, F.e.r, F.e.t);
      if (L.v === "conflict") ans.push(`모순이야! 서로 반대되는 문장이 있어:\n• ${L.pos.why.at(-1).s}\n• ${L.neg.why.at(-1).s}`);
      else if (L.v === null) { if (E.some(e => e.f === f)) ans.push("내가 아는 것만으로는 알 수 없어."); }
      else {
        const ok = L.v === !F.e.n, far = L.why.length > 1 || L.why[0].d;
        ans.push(`${ok ? "응," : "아니야,"} ${far ? "이어서 생각하면 " + (ok ? "그래!" : "반대야!") : ok ? "알려준 대로야." : "알려준 건 반대야."}\n${lines(L.why).join("\n")}${far ? `\n⇒ ${F.e.A} ${L.why.at(-1).p}` : ""}`);
      }
      continue;
    }
    /* 문장 기억하기 */
    if (!F.e || DEIC.test(F.subj.w) || /[\d,]/.test(F.s)) continue;
    const endw = s0.split(/\s+/).pop();
    if (!(STMT.test(endw) || (F.obj && /(어|아|해)$/.test(endw)))) continue;
    const e = F.e, dup = kb.find(x => same(x, e));
    if (dup && dup.n !== e.n) { mem.fix = e; dirty = true; ans.push(`앞에서 “${dup.s}”라고 했는데 이번 말은 반대야. 어느 쪽이 맞아?`); chips.push("새 문장으로 바꿔줘"); continue; }
    if (!dup && kb.length < 200) { kb.push(e); dirty = true; }
    learn.push(e.s);
  }
  if (dirty) { mem.kb = kb; mem.cmp = cmp; mem.rules = rules; root.LMKit.onMem && root.LMKit.onMem(mem); }
  if (ans.length) return pack(ans.join("\n\n"));
  if (learn.length) {
    const now = derive(kb, rules).filter(e => !before.has(kk(e)));
    return pack("알겠어! 기억해둘게:\n" + learn.map(x => "• " + x).join("\n") + (now.length ? "\n그러면 이것도 알 수 있어:\n" + now.map(e => "• " + e.s).join("\n") : "") + "\n이제 이걸로 물어봐!");
  }
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
  ["배우기", q => { // 되묻기 칩의 답: 어간을 mem.lex에 저장하고, 못 풀던 문장을 다시 푼다
    const m = q.trim().match(/^'(.+?)'\s*[은는]\s*(크다류|작다류|기타)$/); if (!m) return;
    (mem.lex || (mem.lex = {}))[m[1].replace(/다$/, "")] = m[2] === "기타" ? { x: 1 } : { d: "~" + m[1].replace(/다$/, ""), s: m[2] === "크다류" ? 1 : -1 };
    const p = mem.pending; delete mem.pending; root.LMKit.onMem && root.LMKit.onMem(mem);
    const r = p && reason(p), t = r && (r.a || r);
    return `알겠어! '${m[1]}'를 ${m[2]}로 기억할게.` + (t ? "\n\n" + t : "");
  }],
  ["정정", q => { // 모순 문장을 만났을 때 ‘새 문장으로 바꿔줘’
    if (q.trim() !== "새 문장으로 바꿔줘" || !mem.fix) return;
    const e = mem.fix, i = (mem.kb || []).findIndex(x => same(x, e)); if (i < 0) return;
    mem.kb[i] = e; delete mem.fix; root.LMKit.onMem && root.LMKit.onMem(mem); return `바꿨어! 이제 “${e.s}”로 기억해.`;
  }],
  ["이해 못한 문장", q => {
    if (!/이해\s*못|못\s*알아들|못\s*푼\s*문장/.test(q)) return;
    return (mem.miss || []).length ? "아직 이해 못한 문장이야:\n" + mem.miss.map(x => "• " + x).join("\n") : "이해 못한 문장은 아직 없어.";
  }],
  ["추론", q => reason(q)]
];


  const tools = q => { for (const [n, fn] of SK) { const a = fn(q); if (a) return typeof a === "string" ? { a, src: n } : { a: a.a, src: n, chips: a.chips }; } return null; };
  const setTF = t => { tf = t; };

  root.LMKit = { Tok, LM, Fast, Retr, Bot, fit, setup, prep, sig, tools, mem, setTF, UNKA };
  if (typeof module !== "undefined") module.exports = root.LMKit;
})(typeof window !== "undefined" ? window : globalThis);
