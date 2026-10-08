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

const comparisonScenes = {
  general3: {
    name: "KITTI General sample 3",
    methods: {
      mdic: ["MDIC", "0.0539", "0.2386"],
      ndsc: ["NDSC", "0.0625", "0.3414"],
      resulic: ["ResULIC", "0.0611", "0.2591"],
      camsic: ["CAMSIC", "0.0518", "0.4201"]
    }
  },
  instereo8: {
    name: "InStereo2K sample 8",
    methods: {
      mdic: ["MDIC", "0.0518", "0.1779"],
      ndsc: ["NDSC", "0.0625", "0.3548"],
      resulic: ["ResULIC", "0.0673", "0.1987"],
      camsic: ["CAMSIC", "0.0480", "0.4272"]
    }
  },
  instereo47: {
    name: "InStereo2K sample 47",
    methods: {
      mdic: ["MDIC", "0.0491", "0.2435"],
      ndsc: ["NDSC", "0.0625", "0.5194"],
      resulic: ["ResULIC", "0.0806", "0.2706"],
      camsic: ["CAMSIC", "0.0980", "0.5953"]
    }
  }
};

document.querySelectorAll("[data-comparison-card]").forEach((card) => {
  const sceneKey = card.dataset.scene;
  const scene = comparisonScenes[sceneKey];
  const stage = card.querySelector(".compare-stage");
  const range = card.querySelector(".compare-range");
  const baselineImage = card.querySelector(".compare-baseline-image");
  const baselineLabel = card.querySelector(".compare-baseline-label");

  range.addEventListener("input", (event) => {
    stage.style.setProperty("--position", `${event.target.value}%`);
  });

  card.querySelectorAll(".method-button").forEach((button) => {
    button.addEventListener("click", () => {
      card.querySelectorAll(".method-button").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      const [method, bpp, lpips] = scene.methods[button.dataset.method];
      baselineImage.src = `assets/compare_${sceneKey}_${button.dataset.method}.png`;
      baselineImage.alt = `${method} reconstruction for ${scene.name}`;
      baselineLabel.innerHTML = `${method}<br><span>${bpp} bpp &middot; LPIPS ${lpips}</span>`;
    });
  });
});

document.querySelector("#copy-bib").addEventListener("click", async (event) => {
  await navigator.clipboard.writeText(document.querySelector("#bibtex").innerText);
  event.currentTarget.textContent = "Copied";
  setTimeout(() => { event.currentTarget.textContent = "Copy BibTeX"; }, 1600);
});
