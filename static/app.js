const input = document.getElementById("video");

const label = document.getElementById("file-label");

const form = document.getElementById("upload-form");

const button = document.getElementById("submit-button");


if (input) {
    input.addEventListener("change", () => {

        if (input.files.length > 0) {
            label.textContent = input.files[0].name;
        } else {
            label.textContent = "Choose a video to upload";
        }

    });
}


if (form) {
    form.addEventListener("submit", () => {

        if (!input.files.length) {
            return;
        }

        button.disabled = true;

        button.textContent = "Analyzing video...";

    });
}
