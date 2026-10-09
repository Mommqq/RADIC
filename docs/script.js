const conditions = {
  paired: { label: "Paired SI", suffix: "paired", metrics: { ndsc: "0.0625 bpp | 25.57 dB | LPIPS 0.361", mdic: "0.0561 bpp | 20.35 dB | LPIPS 0.364", radic: "0.0522 bpp | 23.63 dB | LPIPS 0.229" } },
  non_paired: { label: "Non-paired SI", suffix: "non_paired", metrics: { ndsc: "0.0625 bpp | 22.85 dB | LPIPS 0.498", mdic: "0.0561 bpp | 18.92 dB | LPIPS 0.435", radic: "0.0522 bpp | 22.45 dB | LPIPS 0.297" } },
  cross_domain: { label: "Cross-domain SI", suffix: "cross_domain", metrics: { ndsc: "0.0625 bpp | 22.29 dB | LPIPS 0.538", mdic: "0.0561 bpp | 18.73 dB | LPIPS 0.453", radic: "0.0522 bpp | 22.33 dB | LPIPS 0.299" } }
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
      mdic: ["MDIC", "0.0506", "20.65", "0.2403"],
      ndsc: ["NDSC", "0.0625", "23.66", "0.3414"],
      resulic: ["ResULIC", "0.0732", "19.74", "0.2675"],
      gt: ["Ground truth", null, null, null, "assets/qual_general3_gt.png"]
    }
  },
  stereo2: {
    name: "KITTI Stereo sample 2",
    methods: {
      mdic: ["MDIC", "0.0586", "22.74", "0.2371", "assets/qual_stereo2_mdic.png"],
      ndsc: ["NDSC", "0.0625", "25.89", "0.3947", "assets/qual_stereo2_ndsc.png"],
      resulic: ["ResULIC", "0.0755", "21.88", "0.2342", "assets/qual_stereo2_resulic.png"],
      gt: ["Ground truth", null, null, null, "assets/qual_stereo2_gt.png"]
    }
  },
  instereo47: {
    name: "InStereo2K sample 47",
    methods: {
      mdic: ["MDIC", "0.0572", "22.61", "0.2129"],
      ndsc: ["NDSC", "0.0625", "23.36", "0.5194"],
      resulic: ["ResULIC", "0.0806", "20.87", "0.2706"],
      gt: ["Ground truth", null, null, null, "assets/qual_gt.png"]
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
      const [method, bpp, psnr, lpips, image] = scene.methods[button.dataset.method];
      baselineImage.src = image || `assets/compare_${sceneKey}_${button.dataset.method}.png`;
      baselineImage.alt = button.dataset.method === "gt" ? `${method} for ${scene.name}` : `${method} reconstruction for ${scene.name}`;
      baselineLabel.innerHTML = bpp ? `${method}<br><span>${bpp} bpp | ${psnr} dB | LPIPS ${lpips}</span>` : method;
    });
  });
});

document.querySelector("#copy-bib").addEventListener("click", async (event) => {
  await navigator.clipboard.writeText(document.querySelector("#bibtex").innerText);
  event.currentTarget.textContent = "Copied";
  setTimeout(() => { event.currentTarget.textContent = "Copy BibTeX"; }, 1600);
});
