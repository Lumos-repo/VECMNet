const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");
const sharp = require("sharp");

const outDir = __dirname;
const pptxPath = path.join(outDir, "VECM_说明书附图_可编辑版.pptx");
const pngPath = path.join(outDir, "VECM_说明书附图_预览.png");
const svgPath = path.join(outDir, "VECM_说明书附图_预览.svg");

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Codex";
pptx.subject = "文本到图像行人重识别说明书附图";
pptx.title = "VECM 说明书附图";
pptx.lang = "zh-CN";
pptx.theme = {
  headFontFace: "Microsoft YaHei",
  bodyFontFace: "Microsoft YaHei",
  lang: "zh-CN",
};

const slide = pptx.addSlide();
slide.background = { color: "FFFFFF" };

const S = 120; // SVG pixels per PowerPoint inch.
const C = "111111";
const LW = 1.05;
let svgParts = [];

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" }[c]));
}

function addSvgText(x, y, w, h, label, fontSize, bold = false) {
  const lines = String(label).split("\n");
  const px = fontSize * 1.9;
  const lh = px * 1.15;
  const start = y * S + h * S / 2 - ((lines.length - 1) * lh) / 2;
  lines.forEach((line, i) => {
    svgParts.push(`<text x="${(x + w / 2) * S}" y="${start + i * lh}" font-size="${px}" font-weight="${bold ? 700 : 500}">${esc(line)}</text>`);
  });
}

function rect(x, y, w, h, label, fontSize = 7.2, bold = false) {
  slide.addShape(pptx.ShapeType.rect, {
    x, y, w, h,
    fill: { color: "FFFFFF", transparency: 100 },
    line: { color: C, width: LW },
  });
  if (label) {
    slide.addText(label, {
      x: x + 0.04,
      y: y + 0.035,
      w: w - 0.08,
      h: h - 0.07,
      fontFace: "Microsoft YaHei",
      fontSize,
      bold,
      color: C,
      margin: 0,
      fit: "shrink",
      align: "center",
      valign: "mid",
    });
  }
  svgParts.push(`<rect x="${x * S}" y="${y * S}" width="${w * S}" height="${h * S}" class="box"/>`);
  if (label) addSvgText(x, y, w, h, label, fontSize, bold);
}

function label(x, y, w, h, value, fontSize = 8, bold = false) {
  slide.addText(value, {
    x, y, w, h,
    fontFace: "Microsoft YaHei",
    fontSize,
    bold,
    color: C,
    margin: 0,
    fit: "shrink",
    align: "center",
    valign: "mid",
  });
  addSvgText(x, y, w, h, value, fontSize, bold);
}

function line(x1, y1, x2, y2, arrow = true) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color: C, width: LW, beginArrowType: "none", endArrowType: arrow ? "triangle" : "none" },
  });
  svgParts.push(`<path d="M${x1 * S} ${y1 * S} L${x2 * S} ${y2 * S}" class="${arrow ? "line" : "plain"}"/>`);
}

function poly(points, arrow = true) {
  for (let i = 0; i < points.length - 1; i++) {
    line(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1], arrow && i === points.length - 2);
  }
}

function group(x, y, w, h, title, inner) {
  rect(x, y, w, h, "", 7.2, false);
  label(x + 0.12, y + 0.08, w - 0.24, 0.34, title, 8.4, true);
  inner.forEach(item => rect(...item));
}

// Title and training loop.
label(5.75, 0.10, 3.25, 0.26, "更新模型，下一轮训练", 9.5, true);
poly([[12.43, 3.35], [12.43, 0.50], [5.18, 0.50], [5.18, 1.30]], true);

// Input and preprocessing.
rect(0.15, 1.18, 0.46, 5.04, "文\n本\n到\n图\n像\n行\n人\n重\n识\n别\n数\n据", 10.5, true);
rect(1.10, 3.06, 1.22, 0.78, "数据预处理\n图像缩放、归一化\n文本分词、随机掩码", 6.4, true);
line(0.61, 3.70, 1.10, 3.45);

// Branching.
rect(2.72, 2.18, 0.50, 0.60, "图像\n输入", 7.0, true);
rect(2.72, 5.02, 0.50, 0.60, "文本\n输入", 7.0, true);
poly([[2.32, 3.45], [2.52, 3.45], [2.52, 2.48], [2.72, 2.48]], true);
poly([[2.32, 3.45], [2.52, 3.45], [2.52, 5.32], [2.72, 5.32]], true);

// Encoders.
group(3.55, 1.30, 2.75, 1.28, "CLIP图像编码器\nViT-B/16", [
  [3.82, 1.88, 0.58, 0.46, "Patch\nEmbedding", 5.9, true],
  [4.55, 1.88, 0.72, 0.46, "Transformer", 6.2, true],
  [5.43, 1.88, 0.56, 0.46, "投影层", 6.6, true],
]);
line(3.22, 2.48, 3.55, 2.06);
line(4.40, 2.11, 4.55, 2.11);
line(5.27, 2.11, 5.43, 2.11);

group(3.55, 4.75, 2.75, 1.28, "CLIP文本编码器", [
  [3.82, 5.33, 0.58, 0.46, "Token\nEmbedding", 5.9, true],
  [4.55, 5.33, 0.72, 0.46, "Transformer", 6.2, true],
  [5.43, 5.33, 0.56, 0.46, "投影层", 6.6, true],
]);
line(3.22, 5.32, 3.55, 5.39);
line(4.40, 5.56, 4.55, 5.56);
line(5.27, 5.56, 5.43, 5.56);

// Feature boxes and encoder outputs.
rect(6.84, 2.90, 0.98, 0.50, "全局图文特征\n512维", 6.5, true);
rect(6.84, 4.16, 0.98, 0.50, "局部增强特征\n1024维", 6.3, true);
poly([[6.30, 2.04], [6.55, 2.04], [6.55, 3.15], [6.84, 3.15]], true);
poly([[6.30, 5.39], [6.55, 5.39], [6.55, 4.41], [6.84, 4.41]], true);

// Main modules.
rect(8.45, 1.38, 1.72, 0.92, "VNM 变分噪声建模\n双尺度损失分布建模\n干净/噪声样本划分", 6.2, true);
rect(8.45, 3.22, 1.72, 1.12, "CGR 一致性引导细化\n置信度融合\n伪标签优化\n输出 L_KL、L_PNCL", 6.1, true);
rect(8.45, 5.22, 1.72, 0.86, "CMM 跨模态掩码建模\n图像辅助文本重建", 6.2, true);

line(7.82, 3.15, 8.45, 1.84);
line(7.82, 4.41, 8.45, 1.84);
line(7.82, 3.15, 8.45, 3.75);
line(7.82, 4.41, 8.45, 3.75);
line(7.82, 3.15, 8.45, 5.65);
line(7.82, 4.41, 8.45, 5.65);
line(9.31, 2.30, 9.31, 3.22);
label(9.40, 2.64, 0.62, 0.20, "初始伪标签", 5.0, false);

// Losses and final loss.
rect(10.78, 1.56, 0.80, 0.38, "L_match", 7.2, true);
rect(10.78, 3.42, 0.80, 0.38, "L_KL", 7.2, true);
rect(10.78, 3.98, 0.80, 0.38, "L_PNCL", 7.2, true);
rect(10.78, 5.44, 0.80, 0.38, "L_CMM", 7.2, true);
rect(12.05, 3.40, 0.84, 0.58, "总损失 L", 8.8, true);

line(10.17, 1.84, 10.78, 1.75);
line(10.17, 3.66, 10.78, 3.61);
line(10.17, 3.95, 10.78, 4.17);
line(10.17, 5.65, 10.78, 5.63);
line(11.58, 1.75, 12.05, 3.52);
line(11.58, 3.61, 12.05, 3.64);
line(11.58, 4.17, 12.05, 3.76);
line(11.58, 5.63, 12.05, 3.90);

// Retrieval output.
rect(10.44, 6.36, 1.72, 0.64, "文本查询图像库\nTop-K检索结果", 7.4, true);
poly([[7.32, 4.66], [7.32, 6.68], [10.44, 6.68]], true);

// SVG render.
const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">
<defs>
<marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth">
<path d="M0,0 L10,4 L0,8 Z" fill="#111"/>
</marker>
<style>
text{font-family:"Microsoft YaHei","SimHei",Arial,sans-serif;fill:#111;text-anchor:middle;dominant-baseline:middle}
.box{fill:#fff;stroke:#111;stroke-width:2}
.line{stroke:#111;stroke-width:2;fill:none;marker-end:url(#arrow)}
.plain{stroke:#111;stroke-width:2;fill:none}
</style>
</defs>
<rect width="1600" height="900" fill="#fff"/>
${svgParts.join("\n")}
</svg>`;

fs.writeFileSync(svgPath, svg, "utf8");

(async () => {
  await pptx.writeFile({ fileName: pptxPath });
  await sharp(Buffer.from(svg)).png().toFile(pngPath);
  console.log(JSON.stringify({ pptxPath, pngPath, svgPath }, null, 2));
})();
