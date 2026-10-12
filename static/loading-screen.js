export class LoadingScreen {
    constructor(container, pageTitle, loadingTitle) {
        this.form = container.querySelector("form");
        this.loadingScreen = container.querySelector(".loading-screen");
        this.loadingScreenTagline = container.querySelector(".loading-screen-tagline");
        this.pageTitle = pageTitle;
        this.loadingTitle = loadingTitle;
        this.websiteNameElement = container.querySelector(".website-name");
        this.websiteName = this.websiteNameElement.dataset.websiteName;
    }

    setLoadingScreenTitle() {
        this.pageTitle.textContent = `${this.loadingTitle} | ${this.websiteName}`
    }

    showLoadingScreen() {
        if (!this.loadingScreen) {
            console.warn("No element with loading-screen class");
        }
        else {
            this.loadingScreen.classList.remove("hidden");
            this.loadingScreen.classList.add("flex");
            this.loadingScreen.classList.add("flex-col");
            this.loadingScreen.classList.add("justify-center");
            this.setLoadingScreenTitle();
        }

        if (!this.loadingScreenTagline) {
            console.warn("No element with loading-screen-tagline class");
        }
        else {
            this.loadingScreenTagline.classList.add("self-center");
        }

        if (!this.form) {
            console.warn("No form element exists");
        }
        else {
            this.form.classList.add("hidden");
        }
    }
}