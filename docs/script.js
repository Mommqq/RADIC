"use strict";

const paperData = JSON.parse(document.querySelector("#paper-data").textContent);
const conditionLabels = { paired: "Paired SI", non_paired: "Non-paired SI", cross_domain: "Cross-domain SI" };
const methodLabels = { mdic: "MDIC", ndsc: "NDSC", resulic: "ResULIC", radic: "RADIC", gt: "Ground truth" };
const tabs = [...document.querySelectorAll(".tab")];

function renderCondition(condition) {
  const methods = paperData.conditions[condition];
  document.querySelector("#si-label").textContent = conditionLabels[condition];
  document.querySelector("#si-image").src = paperData.siImages[condition];
  document.querySelector("#robustness-panel").setAttribute("aria-label", conditionLabels[condition]);
  Object.entries(methods).forEach(([method, record]) => {
    const image = document.querySelector(`#${method}-image`);
    image.src = record.image;
    image.alt = `${method.toUpperCase()} reconstruction with ${conditionLabels[condition]}`;
    document.querySelector(`#${method}-metric`).textContent = `${record.bpp} bpp | ${record.psnr} dB | LPIPS ${record.lpips}`;
  });
  tabs.forEach((tab) => {
    const selected = tab.dataset.condition === condition;
    tab.classList.toggle("active", selected);
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
}

tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => renderCondition(tab.dataset.condition));
  tab.addEventListener("keydown", (event) => {
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = tabs.length - 1;
    if (next === undefined) return;
    event.preventDefault();
    renderCondition(tabs[next].dataset.condition);
    tabs[next].focus();
  });
});
renderCondition("paired");

function renderMetricLabel(label, method, record) {
  label.replaceChildren();
  const name = document.createElement("strong");
  name.textContent = methodLabels[method];
  label.append(name);
  if (!record.bpp) return;
  const values = document.createElement("span");
  values.className = "metric-values";
  [`${record.bpp} bpp`, `${record.psnr} dB`, `LPIPS ${record.lpips}`].forEach((text) => {
    const value = document.createElement("span");
    value.textContent = text;
    values.append(value);
  });
  label.append(values);
}

document.querySelectorAll("[data-comparison-card]").forEach((card) => {
  const scene = paperData.scenes[card.dataset.scene];
  const stage = card.querySelector(".compare-stage");
  const range = card.querySelector(".compare-range");
  const baselineImage = card.querySelector(".compare-baseline-image");
  const baselineLabel = card.querySelector(".compare-baseline-label");
  stage.style.setProperty("--position", `${range.value}%`);
  range.addEventListener("input", () => {
    stage.style.setProperty("--position", `${range.value}%`);
    range.setAttribute("aria-valuetext", `${range.value}% baseline, ${100 - Number(range.value)}% RADIC`);
  });
  renderMetricLabel(card.querySelector(".compare-label.right"), "radic", scene.methods.radic);
  renderMetricLabel(baselineLabel, "mdic", scene.methods.mdic);
  card.querySelectorAll(".method-button").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.classList.contains("active")));
    button.addEventListener("click", () => {
      card.querySelectorAll(".method-button").forEach((other) => {
        other.classList.toggle("active", other === button);
        other.setAttribute("aria-pressed", String(other === button));
      });
      const method = button.dataset.method;
      const record = scene.methods[method];
      baselineImage.src = record.image;
      baselineImage.alt = `${methodLabels[method]} for ${scene.name}`;
      renderMetricLabel(baselineLabel, method, record);
    });
  });
});

document.querySelector("#copy-bib").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  const text = document.querySelector("#bibtex").textContent;
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.append(textarea);
      textarea.select();
      const copied = document.execCommand("copy");
      textarea.remove();
      if (!copied) throw new Error("Copy unavailable");
    }
    button.textContent = "Copied";
  } catch {
    button.textContent = "Select BibTeX to copy";
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.querySelector("#bibtex"));
    selection.removeAllRanges();
    selection.addRange(range);
  }
  setTimeout(() => { button.textContent = "Copy BibTeX"; }, 1800);
});
