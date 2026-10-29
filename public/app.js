/* ========== 1) 서버(brain.json)에서 brain 불러오기 ========== */
const WELCOME="안녕! 나는 brain 폴더의 JSON을 읽고 대답하는 SLM이야. “소수가 뭐야”, “12*(3+4)”, “what is a language model”처럼 물어봐.";
async function loadServerBrain(first){
  try{
    const r=await fetch("brain.json",{cache:"no-store"});
    if(!r.ok) throw new Error(r.status);
    BRAIN=await r.json(); build();
    bubble("b", first?WELCOME:"서버 brain으로 돌아왔어.");
  }catch(e){
    BRAIN={}; build();
    bubble("b","brain.json을 불러오지 못했어. 배포된 주소에서 열거나, 위의 ‘brain 폴더 불러오기’로 직접 불러와줘.");
  }
}

/* ========== 2) 토크나이저 (단어 + 서브워드 bigram) ========== */
const PARTICLES = ["에서는","으로는","이라고","입니다","에게","에서","으로","까지","부터","이란","은","는","이","가","을","를","의","에","도","로","와","과","야","요"];
const hangul = s => /[가-힣]/.test(s);
function stem(w){ return w.length>4 ? w.replace(/(ing|ed|es|s)$/,"") : w; }
function tokenize(text){
  const words = text.toLowerCase().match(/[\p{L}\p{N}]+/gu) || [];
  const out = [];
  for (let w of words){
    if (hangul(w)){
      let s = w;
      for (const p of PARTICLES){ if (s.length>p.length && s.endsWith(p)){ s = s.slice(0,-p.length); break; } }
      out.push(s);
      for (let i=0;i<s.length-1;i++) out.push("#"+s.slice(i,i+2));
    } else out.push(stem(w));
  }
  return out;
}
const tokW = t => t[0]==="#" ? 0.5 : 1;

/* ========== 3) brain 인덱스 (TF-IDF) ========== */
let BRAIN = {}, DOCS = [], IDF = {};
function build(){
  DOCS = [];
  for (const [path,data] of Object.entries(BRAIN)){
    for (const e of (data.entries||[])){
      const tf = {};
      const add = (txt,w)=>tokenize(txt).forEach(t=>{ tf[t]=(tf[t]||0)+w*tokW(t); });
      (e.patterns||[]).forEach(p=>add(p,2));
      add((e.tags||[]).join(" "),2);
      add(e.answer||"",0.5);
      DOCS.push({path,topic:data.topic||"",id:e.id,answer:e.answer||"",tf});
    }
  }
  const df = {};
  DOCS.forEach(d=>Object.keys(d.tf).forEach(t=>df[t]=(df[t]||0)+1));
  IDF = {};
  for (const t in df) IDF[t] = Math.log(1 + DOCS.length/(1+df[t]));
  DOCS.forEach(d=>{
    let n=0;
    for (const t in d.tf){ d.tf[t]*=IDF[t]; n+=d.tf[t]**2; }
    d.norm = Math.sqrt(n)||1;
  });
  renderTree();
}
function search(query){
  const toks = tokenize(query), q = {};
  toks.forEach(t=>{ if(IDF[t]) q[t]=(q[t]||0)+tokW(t)*IDF[t]; });
  let qn = Math.sqrt(Object.values(q).reduce((a,v)=>a+v*v,0))||1;
  const ranked = DOCS.map(d=>{
    let dot=0; for (const t in q) if (d.tf[t]) dot+=q[t]*d.tf[t];
    return {d, score: dot/(qn*d.norm)};
  }).sort((a,b)=>b.score-a.score);
  return {toks, ranked};
}

/* ========== 4) 계산기 (math 폴더용 보조 능력, eval 없이 안전하게) ========== */
function tryMath(text){
  const m = text.replace(/×/g,"*").replace(/÷/g,"/").match(/[\d.()]+(?:\s*[+\-*/^%]\s*[\d.()]+)+/);
  if (!m) return null;
  const expr = m[0];
  if (!/^[\d\s+\-*/().^%]+$/.test(expr)) return null;
  try {
    const v = Function('"use strict";return ('+expr.replace(/\^/g,"**")+')')();
    return Number.isFinite(v) ? {expr, v:+v.toFixed(10)} : null;
  } catch { return null; }
}

/* ========== 5) 답변 생성 ========== */
const THRESHOLD = 0.14;
function respond(query){
  const calc = tryMath(query);
  const {toks, ranked} = search(query);
  const top = ranked[0];
  if (calc) return {text:`${calc.expr.trim()} = ${calc.v}`, meta:"계산기 · brain/math 연산 능력", toks};
  if (!top || top.score < THRESHOLD){
    const text = hangul(query)
      ? "아직 brain에 없는 내용이야. brain 폴더에 관련 JSON을 추가해서 불러오면 바로 배울 수 있어!"
      : "I don’t have that in my brain yet. Add a JSON file with patterns and answers, then load it.";
    return {text, meta:`최고 유사도 ${top?top.score.toFixed(2):"0.00"} (기준 ${THRESHOLD})`, toks};
  }
  const alt = ranked.slice(1,3).filter(r=>r.score>0.05).map(r=>`${r.d.id} ${r.score.toFixed(2)}`).join(", ");
  return {text:top.d.answer, meta:`${top.d.path} · ${top.d.id} · 유사도 ${top.score.toFixed(2)}${alt?" · 후보 "+alt:""}`, toks};
}

/* ========== 6) UI ========== */
const chat = document.getElementById("chat");
function bubble(cls,text){
  const el = document.createElement("div"); el.className="m "+cls; el.textContent=text;
  chat.appendChild(el); chat.scrollTop = chat.scrollHeight; return el;
}
function typeOut(el, text, meta, toks){
  const parts = text.split(/(\s+)/); let i=0; el.textContent="";
  (function step(){
    el.textContent += parts[i++] || "";
    chat.scrollTop = chat.scrollHeight;
    if (i<parts.length) setTimeout(step, 28);
    else {
      const m = document.createElement("span"); m.className="meta";
      m.textContent = `${meta}\n토큰: ${toks.filter(t=>t[0]!=="#").join(" · ")||"-"}`;
      m.style.whiteSpace="pre-line"; el.appendChild(m); chat.scrollTop = chat.scrollHeight;
    }
  })();
}
document.getElementById("f").addEventListener("submit",e=>{
  e.preventDefault();
  const inp = document.getElementById("q"), q = inp.value.trim();
  if (!q) return;
  inp.value=""; bubble("u",q);
  const r = respond(q), el = bubble("b","");
  typeOut(el, r.text, r.meta, r.toks);
});

function renderTree(){
  const dirs = {};
  Object.keys(BRAIN).sort().forEach(p=>{
    const parts = p.split("/"), file = parts.pop(), dir = parts.join("/");
    (dirs[dir] = dirs[dir]||[]).push(file);
  });
  document.getElementById("tree").innerHTML = Object.entries(dirs).map(([d,fs])=>
    `<li class="d">${d}/</li>`+fs.map(f=>`<li class="f">${f}</li>`).join("")).join("");
  document.getElementById("stat").textContent =
    `파일 ${Object.keys(BRAIN).length}개 · 지식 ${DOCS.length}개 학습 완료`;
}

/* 폴더/파일 불러오기 */
async function loadFiles(list, replace){
  const next = replace ? {} : {...BRAIN};
  let ok=0, bad=0;
  for (const f of list){
    if (!/\.json$/i.test(f.name)) continue;
    try {
      const data = JSON.parse(await f.text());
      if (!Array.isArray(data.entries)) throw 0;
      next[f.webkitRelativePath || ("brain/"+f.name)] = data; ok++;
    } catch { bad++; }
  }
  if (ok){ BRAIN = next; build(); }
  bubble("b", ok ? `JSON ${ok}개를 새로 읽었어.${bad?` (형식 오류 ${bad}개는 건너뜀)`:""} 지식 ${DOCS.length}개를 알고 있어.` : "읽을 수 있는 JSON이 없어. entries 배열이 있는 파일인지 확인해줘.");
}
const dirIn = document.getElementById("dirIn"), fileIn = document.getElementById("fileIn");
document.getElementById("pickDir").onclick = ()=>dirIn.click();
document.getElementById("pickFiles").onclick = ()=>fileIn.click();
dirIn.onchange = ()=>{ loadFiles([...dirIn.files], true); dirIn.value=""; };
fileIn.onchange = ()=>{ loadFiles([...fileIn.files], false); fileIn.value=""; };
document.getElementById("reset").onclick = ()=>loadServerBrain(false);

loadServerBrain(true);
