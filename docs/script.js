const conditions = {
  paired: { label: "Paired SI", suffix: "paired", metrics: { ndsc: "0.0625 bpp · LPIPS 0.3377", mdic: "0.0539 bpp · LPIPS 0.2386", radic: "0.0522 bpp · LPIPS 0.1381" } },
  non_paired: { label: "Non-paired SI", suffix: "non_paired", metrics: { ndsc: "weak SI", mdic: "weak SI", radic: "retrieved SI" } },
  cross_domain: { label: "Cross-domain SI", suffix: "cross_domain", metrics: { ndsc: "cross-domain", mdic: "cross-domain", radic: "retrieved SI" } }
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

document.querySelector("#copy-bib").addEventListener("click", async (event) => {
  await navigator.clipboard.writeText(document.querySelector("#bibtex").innerText);
  event.currentTarget.textContent = "Copied";
  setTimeout(() => { event.currentTarget.textContent = "Copy BibTeX"; }, 1600);
});
