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
pptx.company = "VECMNet";
pptx.lang = "zh-CN";
pptx.theme = {
  headFontFace: "Microsoft YaHei",
  bodyFontFace: "Microsoft YaHei",
  lang: "zh-CN",
};
pptx.margin = 0;

const slide = pptx.addSlide();
slide.background = { color: "FFFFFF" };

const W = 13.333;
const H = 7.5;
const C = "111111";
const LW = 1.15;

function rect(x, y, w, h, text, opts = {}) {
  slide.addShape(pptx.ShapeType.rect, {
    x, y, w, h,
    fill: { color: "FFFFFF", transparency: 100 },
    line: { color: C, width: opts.lineWidth || LW },
  });
  if (text) {
    slide.addText(text, {
      x: x + 0.04,
      y: y + 0.04,
      w: w - 0.08,
      h: h - 0.08,
      fontFace: "Microsoft YaHei",
      fontSize: opts.fontSize || 8.5,
      bold: !!opts.bold,
      color: C,
      margin: 0,
      breakLine: false,
      fit: "shrink",
      align: opts.align || "center",
      valign: "mid",
    });
  }
}

function text(x, y, w, h, value, opts = {}) {
  slide.addText(value, {
    x, y, w, h,
    fontFace: "Microsoft YaHei",
    fontSize: opts.fontSize || 8,
    bold: !!opts.bold,
    color: C,
    margin: 0,
    fit: "shrink",
    align: opts.align || "center",
    valign: opts.valign || "mid",
  });
}

function line(x1, y1, x2, y2, arrow = true, width = LW) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1,
    y: y1,
    w: x2 - x1,
    h: y2 - y1,
    line: {
      color: C,
      width,
      beginArrowType: "none",
      endArrowType: arrow ? "triangle" : "none",
    },
  });
}

function poly(points, arrow = true) {
  for (let i = 0; i < points.length - 1; i++) {
    line(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1], arrow && i === points.length - 2);
  }
}

function groupBox(x, y, w, h, title) {
  rect(x, y, w, h, "", { lineWidth: LW });
  text(x + 0.08, y + 0.08, w - 0.16, 0.28, title, { fontSize: 9, bold: true });
}

// Title and outer training loop.
text(6.45, 0.12, 2.7, 0.3, "更新模型，下一轮训练", { fontSize: 10, bold: true });
poly([[6.0, 0.36], [6.0, 0.75], [4.32, 0.75], [4.32, 1.08]], true);
poly([[12.4, 0.36], [12.4, 6.65], [11.82, 6.65]], true);
line(6.0, 0.36, 12.4, 0.36, false);

// Left input and preprocessing.
rect(0.18, 1.24, 0.48, 4.6, "文\n本\n到\n图\n像\n行\n人\n重\n识\n别\n数\n据", { fontSize: 12, bold: true });
rect(1.25, 2.72, 1.25, 0.88, "数据预处理\n图像缩放、归一化\n文本分词、随机掩码", { fontSize: 7.5, bold: true });
line(0.66, 3.52, 1.25, 3.16);

// Branch labels.
rect(3.05, 1.92, 0.5, 0.62, "图像\n输入", { fontSize: 8, bold: true });
rect(3.05, 4.72, 0.5, 0.62, "文本\n输入", { fontSize: 8, bold: true });
poly([[2.5, 3.16], [2.78, 3.16], [2.78, 2.23], [3.05, 2.23]], true);
poly([[2.5, 3.16], [2.78, 3.16], [2.78, 5.03], [3.05, 5.03]], true);

// CLIP image encoder.
groupBox(3.82, 1.25, 2.7, 1.38, "CLIP图像编码器\nViT-B/16");
rect(4.05, 1.85, 0.58, 0.48, "Patch\nEmbedding", { fontSize: 6.2 });
rect(4.76, 1.85, 0.68, 0.48, "Transformer", { fontSize: 6.4 });
rect(5.58, 1.85, 0.55, 0.48, "投影层", { fontSize: 7 });
line(4.63, 2.09, 4.76, 2.09);
line(5.44, 2.09, 5.58, 2.09);
line(3.55, 2.23, 3.82, 2.02);

// CLIP text encoder.
groupBox(3.82, 4.32, 2.7, 1.38, "CLIP文本编码器");
rect(4.05, 4.92, 0.58, 0.48, "Token\nEmbedding", { fontSize: 6.2 });
rect(4.76, 4.92, 0.68, 0.48, "Transformer", { fontSize: 6.4 });
rect(5.58, 4.92, 0.55, 0.48, "投影层", { fontSize: 7 });
line(4.63, 5.16, 4.76, 5.16);
line(5.44, 5.16, 5.58, 5.16);
line(3.55, 5.03, 3.82, 5.01);

// Feature boxes.
rect(7.02, 2.78, 0.92, 0.52, "全局图文特征\n512维", { fontSize: 7.3, bold: true });
rect(7.02, 4.00, 0.92, 0.52, "局部增强特征\n1024维", { fontSize: 7.1, bold: true });
poly([[6.52, 1.94], [6.77, 1.94], [6.77, 3.04], [7.02, 3.04]], true);
poly([[6.52, 5.03], [6.77, 5.03], [6.77, 4.26], [7.02, 4.26]], true);

// Modules.
rect(8.42, 1.38, 1.48, 1.03, "VNM 变分噪声建模\n双尺度损失分布建模\n干净/噪声样本划分", { fontSize: 7.0, bold: true });
rect(8.42, 3.28, 1.48, 0.98, "CGR 一致性引导细化\n置信度融合\n伪标签优化", { fontSize: 7.0, bold: true });
rect(8.42, 5.10, 1.48, 0.88, "CMM 跨模态掩码建模\n图像辅助文本重建", { fontSize: 7.0, bold: true });

// Feature arrows into modules.
line(7.94, 3.04, 8.42, 1.90);
line(7.94, 4.26, 8.42, 1.90);
line(7.94, 3.04, 8.42, 3.70);
line(7.94, 4.26, 8.42, 3.70);
line(7.94, 3.04, 8.42, 5.54);
line(7.94, 4.26, 8.42, 5.54);
line(9.16, 2.41, 9.16, 3.28); // pseudo label from VNM into CGR
text(9.22, 2.62, 0.6, 0.28, "初始伪标签", { fontSize: 5.5, align: "left" });

// Loss boxes.
rect(10.28, 1.46, 0.66, 0.38, "L_match", { fontSize: 8, bold: true });
rect(10.28, 2.08, 0.66, 0.38, "L_KL", { fontSize: 8, bold: true });
rect(10.28, 2.70, 0.66, 0.38, "L_PNCL", { fontSize: 8, bold: true });
rect(10.28, 5.24, 0.66, 0.38, "L_CMM", { fontSize: 8, bold: true });
rect(11.85, 3.28, 0.74, 0.56, "总损失 L", { fontSize: 9.5, bold: true });

// Module to loss and total loss.
line(9.90, 1.78, 10.28, 1.65);
line(9.90, 3.55, 10.28, 2.27);
line(9.90, 3.88, 10.28, 2.89);
line(9.90, 5.54, 10.28, 5.43);
line(10.94, 1.65, 11.85, 3.38);
line(10.94, 2.27, 11.85, 3.48);
line(10.94, 2.89, 11.85, 3.58);
line(10.94, 5.43, 11.85, 3.75);

// Inference output.
rect(10.02, 6.20, 1.62, 0.62, "文本查询图像库\nTop-K检索结果", { fontSize: 8.4, bold: true });
poly([[7.48, 4.52], [7.48, 6.51], [10.02, 6.51]], true);
poly([[11.64, 6.51], [12.28, 6.51], [12.28, 3.84], [12.59, 3.84]], true);

// Small note that CGR owns KL/PNCL.
rect(8.62, 4.30, 1.08, 0.42, "输出 L_KL、L_PNCL", { fontSize: 6.6 });

// Build a matching SVG preview.
const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">
<defs>
<marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth">
<path d="M0,0 L10,4 L0,8 Z" fill="#111"/>
</marker>
<style>
text{font-family:"Microsoft YaHei","SimHei",Arial,sans-serif;fill:#111;text-anchor:middle;dominant-baseline:middle}
.box{fill:white;stroke:#111;stroke-width:2}
.line{stroke:#111;stroke-width:2;fill:none;marker-end:url(#arrow)}
.plain{stroke:#111;stroke-width:2;fill:none}
.small{font-size:24px}.mid{font-size:28px}.tiny{font-size:20px;font-weight:600}.bold{font-weight:700}
</style>
</defs>
<rect width="1600" height="900" fill="#fff"/>
<text x="940" y="38" class="mid bold">更新模型，下一轮训练</text>
<path d="M720 43 L720 90 L520 90 L520 130" class="line"/>
<path d="M720 43 L1488 43 L1488 798 L1420 798" class="line"/>
${svgRect(22,148,58,552,"文\n本\n到\n图\n像\n行\n人\n重\n识\n别\n数\n据","mid bold")}
${svgRect(150,326,150,106,"数据预处理\n图像缩放、归一化\n文本分词、随机掩码","tiny")}
<path d="M80 422 L150 379" class="line"/>
${svgRect(366,230,60,74,"图像\n输入","tiny")}
${svgRect(366,566,60,74,"文本\n输入","tiny")}
<path d="M300 379 L334 379 L334 267 L366 267" class="line"/>
<path d="M300 379 L334 379 L334 603 L366 603" class="line"/>
${svgGroup(458,150,324,166,"CLIP图像编码器\nViT-B/16",[
  [486,222,70,58,"Patch\nEmbedding"],[571,222,82,58,"Transformer"],[670,222,66,58,"投影层"]
])}
<path d="M426 267 L458 242" class="line"/><path d="M556 251 L571 251" class="line"/><path d="M653 251 L670 251" class="line"/>
${svgGroup(458,518,324,166,"CLIP文本编码器",[
  [486,590,70,58,"Token\nEmbedding"],[571,590,82,58,"Transformer"],[670,590,66,58,"投影层"]
])}
<path d="M426 603 L458 601" class="line"/><path d="M556 619 L571 619" class="line"/><path d="M653 619 L670 619" class="line"/>
${svgRect(842,334,110,62,"全局图文特征\n512维","tiny bold")}
${svgRect(842,480,110,62,"局部增强特征\n1024维","tiny bold")}
<path d="M782 233 L812 233 L812 365 L842 365" class="line"/>
<path d="M782 604 L812 604 L812 511 L842 511" class="line"/>
${svgRect(1010,166,178,124,"VNM 变分噪声建模\n双尺度损失分布建模\n干净/噪声样本划分","tiny bold")}
${svgRect(1010,394,178,118,"CGR 一致性引导细化\n置信度融合\n伪标签优化","tiny bold")}
${svgRect(1010,612,178,106,"CMM 跨模态掩码建模\n图像辅助文本重建","tiny bold")}
<path d="M952 365 L1010 228" class="line"/><path d="M952 511 L1010 228" class="line"/>
<path d="M952 365 L1010 444" class="line"/><path d="M952 511 L1010 444" class="line"/>
<path d="M952 365 L1010 665" class="line"/><path d="M952 511 L1010 665" class="line"/>
<path d="M1100 290 L1100 394" class="line"/><text x="1140" y="330" class="tiny">初始伪标签</text>
${svgRect(1234,175,80,46,"L_match","tiny bold")}
${svgRect(1234,250,80,46,"L_KL","tiny bold")}
${svgRect(1234,324,80,46,"L_PNCL","tiny bold")}
${svgRect(1234,629,80,46,"L_CMM","tiny bold")}
${svgRect(1422,394,88,68,"总损失 L","mid bold")}
<path d="M1188 214 L1234 198" class="line"/><path d="M1188 426 L1234 273" class="line"/>
<path d="M1188 466 L1234 347" class="line"/><path d="M1188 665 L1234 652" class="line"/>
<path d="M1314 198 L1422 406" class="line"/><path d="M1314 273 L1422 418" class="line"/>
<path d="M1314 347 L1422 430" class="line"/><path d="M1314 652 L1422 450" class="line"/>
${svgRect(1034,516,130,50,"输出 L_KL、L_PNCL","tiny")}
${svgRect(1202,744,194,74,"文本查询图像库\nTop-K检索结果","tiny bold")}
<path d="M898 542 L898 781 L1202 781" class="line"/>
<path d="M1396 781 L1474 781 L1474 462 L1510 462" class="line"/>
</svg>`;

function svgRect(x, y, w, h, label, cls) {
  const lines = label.split("\n");
  const lineH = cls.includes("mid") ? 32 : 24;
  const start = y + h / 2 - ((lines.length - 1) * lineH) / 2;
  return `<rect class="box" x="${x}" y="${y}" width="${w}" height="${h}"/>` +
    lines.map((l, i) => `<text x="${x + w / 2}" y="${start + i * lineH}" class="${cls}">${escapeXml(l)}</text>`).join("");
}

function svgGroup(x, y, w, h, title, inner) {
  const titleLines = title.split("\n");
  const titleSvg = titleLines.map((l, i) => `<text x="${x + w / 2}" y="${y + 28 + i * 26}" class="small bold">${escapeXml(l)}</text>`).join("");
  return `<rect class="box" x="${x}" y="${y}" width="${w}" height="${h}"/>${titleSvg}` +
    inner.map(([ix, iy, iw, ih, t]) => svgRect(ix, iy, iw, ih, t, "tiny")).join("");
}

function escapeXml(s) {
  return s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" }[c]));
}

fs.writeFileSync(svgPath, svg, "utf8");

(async () => {
  await pptx.writeFile({ fileName: pptxPath });
  await sharp(Buffer.from(svg)).png().toFile(pngPath);
  console.log(JSON.stringify({ pptxPath, pngPath, svgPath }, null, 2));
})();
