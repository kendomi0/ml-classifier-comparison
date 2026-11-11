let scatterPlot = document.getElementById("scatter-plot");
let togglePlotBtn = document.getElementById("toggle-plot-btn");
let showOrHide = document.querySelector(".show-or-hide");
let loadingOrRetry = document.getElementById("loading-or-retry");
let defaultText = document.querySelector(".default-text");
let combinations = document.querySelector(".combinations");
let firstClick = true;

let costTierIcon = document.getElementById("cost-tier-icon");
let runtimeIcon = document.getElementById("runtime-icon");
let rankingIcon = document.getElementById("ranking-icon");

let costTierQna = document.getElementById("cost-tier-qna");
let rankingQna = document.getElementById("ranking-qna");
let runtimeQna = document.getElementById("runtime-qna");

// Scatter plot button

function toggleBtn() {
    scatterPlot.classList.toggle("block");
    scatterPlot.classList.toggle("hidden");
    toggleCombinations();
    let isHidden = scatterPlot.classList.contains("hidden");
    showOrHide.textContent = isHidden ? "Show": "Hide";
}

function showPlot() {
    let key = scatterPlot.dataset.key;
    scatterPlot.src = "/plot/scatter?key=" + key + "&t=" + Date.now();
}

function enableBtn() {
    togglePlotBtn.disabled = false;
}

function showDefaultText() {
    defaultText.classList.remove("hidden");
    loadingOrRetry.classList.add("hidden");
}

function hideDefaultText() {
    defaultText.classList.add("hidden");
    loadingOrRetry.classList.remove("hidden");
}

function handleLoadError() {
    enableBtn();
    scatterPlot.removeEventListener("load", toggleAndEnableBtn);
    hideDefaultText();
    loadingOrRetry.textContent = "Retry Scatter Plot Generation";
}

function toggleCombinations() {
    combinations.classList.toggle("hidden");
}

function toggleAndEnableBtn() {
    toggleBtn();
    showDefaultText();
    enableBtn();
    firstClick = false;
    scatterPlot.removeEventListener("error", handleLoadError);
}

function runButtonFns() {
    if (firstClick) {
        hideDefaultText();
        loadingOrRetry.textContent = "Loading Scatter Plot...";
        togglePlotBtn.disabled = true;
        scatterPlot.addEventListener("load", toggleAndEnableBtn, { once: true });
        scatterPlot.addEventListener("error", handleLoadError, { once: true });
        showPlot();
    }
    else {
        toggleBtn();
    }
}

togglePlotBtn.addEventListener("click", runButtonFns);

// Header icons

iconsAndQnas = new Map([
    [costTierIcon, costTierQna],
    [runtimeIcon, runtimeQna],
    [rankingIcon, rankingQna]
])

function scrollAndOpenQna(qna) {
    qna.scrollIntoView({
        behavior: "smooth", 
        block: "start",
    })
    qna.open = true;
}

iconsAndQnas.forEach((qna, icon) => {
    console.log(typeof icon)
    icon.addEventListener("click", () => {
        scrollAndOpenQna(qna);
        }
    )
});


