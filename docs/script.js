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
      mdic: ["MDIC", "0.0329", "0.2403"],
      ndsc: ["NDSC", "0.0625", "0.3414"],
      resulic: ["ResULIC", "0.0732", "0.2675"],
      gt: ["Ground truth", null, null, "assets/qual_general3_gt.png"]
    }
  },
  city55: {
    name: "Cityscapes sample 55",
    methods: {
      mdic: ["MDIC", "0.0344", "0.1939", "assets/qual_city55_mdic.png"],
      ndsc: ["NDSC", "0.0625", "0.3058", "assets/qual_city55_ndsc.png"],
      resulic: ["ResULIC", "0.0738", "0.2358", "assets/qual_city55_resulic.png"],
      gt: ["Ground truth", null, null, "assets/qual_city55_gt.png"]
    }
  },
  instereo47: {
    name: "InStereo2K sample 47",
    methods: {
      mdic: ["MDIC", "0.0376", "0.2129"],
      ndsc: ["NDSC", "0.0625", "0.5194"],
      resulic: ["ResULIC", "0.0806", "0.2706"],
      gt: ["Ground truth", null, null, "assets/qual_gt.png"]
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
      const [method, bpp, lpips, image] = scene.methods[button.dataset.method];
      baselineImage.src = image || `assets/compare_${sceneKey}_${button.dataset.method}.png`;
      baselineImage.alt = button.dataset.method === "gt" ? `${method} for ${scene.name}` : `${method} reconstruction for ${scene.name}`;
      baselineLabel.innerHTML = bpp ? `${method}<br><span>${bpp} bpp &middot; LPIPS ${lpips}</span>` : method;
    });
  });
});

document.querySelector("#copy-bib").addEventListener("click", async (event) => {
  await navigator.clipboard.writeText(document.querySelector("#bibtex").innerText);
  event.currentTarget.textContent = "Copied";
  setTimeout(() => { event.currentTarget.textContent = "Copy BibTeX"; }, 1600);
});
