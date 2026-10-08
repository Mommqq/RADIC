const conditions = {
  paired: { label: "Paired SI", suffix: "paired", metrics: { ndsc: "0.0625 bpp · LPIPS 0.3377", mdic: "0.0539 bpp · LPIPS 0.2386", radic: "0.0522 bpp · LPIPS 0.1381" } },
  non_paired: { label: "Non-paired SI", suffix: "non_paired", metrics: { ndsc: "weak SI", mdic: "weak SI", radic: "" } },
  cross_domain: { label: "Cross-domain SI", suffix: "cross_domain", metrics: { ndsc: "cross-domain", mdic: "cross-domain", radic: "" } }
};

function renderCondition(condition) {
  const state = conditions[condition];
  document.querySelector("#si-label").textContent = state.label;
  document.querySelector("#si-image").src = `assets/si_${state.suffix}.png`;
  ["ndsc", "mdic", "radic"].forEach((method) => {
    document.querySelector(`#${method}-image`).src = `assets/${method}_${state.suffix}.png`;
    document.querySelector(`#${method}-metric`).textContent = state.metrics[method];
  });
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("active"));
    button.classList.add("active");
    renderCondition(button.dataset.condition);
  });
});

renderCondition("paired");

const comparisons = {
  mdic: { image: "assets/qual_mdic.png", label: "MDIC \u00b7 0.0491 bpp \u00b7 LPIPS 0.2435" },
  ndsc: { image: "assets/qual_ndsc.png", label: "NDSC \u00b7 0.0625 bpp \u00b7 LPIPS 0.5194" },
  resulic: { image: "assets/qual_resulic.png", label: "ResULIC \u00b7 0.0806 bpp \u00b7 LPIPS 0.2706" },
  camsic: { image: "assets/qual_camsic.png", label: "CAMSIC \u00b7 0.0980 bpp \u00b7 LPIPS 0.5953" }
};

function setDivider(value) {
  document.querySelector("#compare-stage").style.setProperty("--position", `${value}%`);
}

document.querySelector("#compare-range").addEventListener("input", (event) => setDivider(event.target.value));
document.querySelectorAll(".method-button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".method-button").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    const comparison = comparisons[button.dataset.method];
    document.querySelector("#compare-baseline-image").src = comparison.image;
    document.querySelector("#compare-baseline-image").alt = `${button.textContent} reconstruction`;
    document.querySelector("#compare-baseline-label").textContent = comparison.label;
  });
});

setDivider(50);

document.querySelector("#copy-bib").addEventListener("click", async (event) => {
  await navigator.clipboard.writeText(document.querySelector("#bibtex").innerText);
  event.currentTarget.textContent = "Copied";
  setTimeout(() => { event.currentTarget.textContent = "Copy BibTeX"; }, 1600);
});
