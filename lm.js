/* 진짜 작은 LLM: BPE 토크나이저 + 디코더 전용 트랜스포머 (tf.js로 브라우저에서 직접 학습이다) */
(function (root) {
  const tf = root.tf || require("@tensorflow/tfjs");
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
      let x = tf.oneHot(ids.flatten(), this.V).cast("float32").matMul(this.wte).reshape([B, T, D]).add(this.wpe.slice([0, 0], [T, D]));
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
    step(xs, ys, B, T) { // 다음 토큰 예측 손실을 줄이는 방향으로 가중치 업데이트
      return tf.tidy(() => {
        const cost = this.opt.minimize(() => {
          const logits = this.forward(tf.tensor2d(xs, [B, T], "int32"));
          return tf.losses.softmaxCrossEntropy(tf.oneHot(tf.tensor1d(ys, "int32"), this.V), logits);
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

  /* ---------- 학습 데이터: brain JSON -> "Q:질문 \n A:답" ---------- */
  function corpus(brain) {
    const examples = [];
    for (const d of Object.values(brain))
      for (const e of d.entries || [])
        for (const q of e.patterns || []) examples.push(`Q:${q}\nA:${e.answer}`);
    return { examples, text: examples.join("\n") };
  }
  function batch(ex, B, T) { // 예시들을 이어 붙여 길이 T+1 시퀀스 B개 (항상 Q:로 시작)
    const xs = [], ys = [];
    for (let b = 0; b < B; b++) {
      let seq = [];
      while (seq.length < T + 1) seq.push(...ex[Math.floor(Math.random() * ex.length)]);
      xs.push(...seq.slice(0, T)); ys.push(...seq.slice(1, T + 1));
    }
    return { xs, ys };
  }

  root.LMKit = { Tok, LM, corpus, batch };
  if (typeof module !== "undefined") module.exports = root.LMKit;
})(typeof window !== "undefined" ? window : globalThis);
