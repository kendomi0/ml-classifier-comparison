let scatterPlot = document.getElementById("scatter-plot");
let togglePlotBtn = document.getElementById("toggle-plot-btn");
let showOrHide = document.querySelector(".show-or-hide");
let clickCount = 0;

function showPlot() {
    let key = scatterPlot.dataset.key;
    scatterPlot.src = "/plot/scatter?key=" + key + "&t=" + Date.now();
    scatterPlot.style.display = "block";
    showOrHide.textContent = "Hide";
}

function toggleBtn() {
    scatterPlot.classList.toggle("display-none");
    let isHidden = scatterPlot.classList.contains("display-none");
    showOrHide.textContent = isHidden ? "Show": "Hide";
}


function runButtonFns() {
    if (clickCount === 0) {
        showPlot();
    }
    else {
        toggleBtn();
    }
    clickCount += 1;
}

togglePlotBtn.addEventListener("click", runButtonFns);
