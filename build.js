// brain/ 폴더의 모든 JSON을 읽어 public/brain.json 하나로 묶는다. (Vercel 빌드 때 실행)
const fs = require("fs");
const path = require("path");

const ROOT = path.join(__dirname, "brain");
const OUT = path.join(__dirname, "public", "brain.json");
const bundle = {};
let bad = 0;

function walk(dir) {
  for (const name of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, name.name);
    if (name.isDirectory()) { walk(full); continue; }
    if (!name.name.toLowerCase().endsWith(".json")) continue;
    const rel = "brain/" + path.relative(ROOT, full).split(path.sep).join("/");
    try {
      const data = JSON.parse(fs.readFileSync(full, "utf-8"));
      if (!Array.isArray(data.entries)) throw new Error("entries 배열이 없음");
      bundle[rel] = data;
    } catch (e) {
      bad++;
      console.error(`✗ ${rel}: ${e.message}`);
    }
  }
}

walk(ROOT);
fs.writeFileSync(OUT, JSON.stringify(bundle));
const n = Object.values(bundle).reduce((a, d) => a + d.entries.length, 0);
console.log(`brain.json 생성: 파일 ${Object.keys(bundle).length}개, 지식 ${n}개`);
if (bad) process.exit(1);
