export class LoadingScreen {
    constructor(container) {
        this.form = container.querySelector("form");
        this.loadingScreen = container.querySelector(".loading-screen");
        this.loadingScreenTagline = container.querySelector(".loading-screen-tagline");
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