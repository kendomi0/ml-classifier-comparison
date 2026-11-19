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
    if (icon) {
        icon.addEventListener("click", () => {
            scrollAndOpenQna(qna);
            }
        )
    }
});


