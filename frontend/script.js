const form = document.getElementById("predictionForm");
const featureInputs = document.querySelectorAll(".feature-input");

const result = document.getElementById("result");
const emptyState = document.getElementById("emptyState");
const loadingState = document.getElementById("loadingState");

const analyzeButton = document.getElementById("analyzeButton");
const sampleButton = document.getElementById("sampleButton");
const manualToggle = document.getElementById("manualToggle");
const manualSection = document.getElementById("manualSection");

const csvFile = document.getElementById("csvFile");
const csvStatus = document.getElementById("csvStatus");

const inputStatus = document.getElementById("inputStatus");
const featureCount = document.getElementById("featureCount");

const predictionText = document.getElementById("predictionText");
const probabilityValue = document.getElementById("probabilityValue");
const probabilityBar = document.getElementById("probabilityBar");
const riskLevel = document.getElementById("riskLevel");
const thresholdValue = document.getElementById("thresholdValue");
const riskBadge = document.getElementById("riskBadge");


const sampleFeatures = [
    17.99,
    10.38,
    122.8,
    1001.0,
    0.1184,
    0.2776,
    0.3001,
    0.1471,
    0.2419,
    0.07871,
    1.095,
    0.9053,
    8.589,
    153.4,
    0.006399,
    0.04904,
    0.05373,
    0.01587,
    0.03003,
    0.006193,
    25.38,
    17.33,
    184.6,
    2019.0,
    0.1622,
    0.6656,
    0.7119,
    0.2654,
    0.4601,
    0.1189
];


function updateFeatureStatus() {

    const values = Array.from(featureInputs)
        .map(input => input.value.trim())
        .filter(value => value !== "");

    featureCount.textContent = `${values.length} / 30`;

    if (values.length === 30) {
        inputStatus.textContent = "Tumor measurements ready for analysis";
    } else if (values.length > 0) {
        inputStatus.textContent = "Tumor measurements partially loaded";
    } else {
        inputStatus.textContent = "Waiting for tumor measurements";
    }
}


function populateFeatures(values) {

    if (values.length !== 30) {
        throw new Error(
            `Expected exactly 30 features, but received ${values.length}.`
        );
    }

    featureInputs.forEach((input, index) => {
        input.value = values[index];
    });

    updateFeatureStatus();
}


sampleButton.addEventListener("click", () => {

    populateFeatures(sampleFeatures);

    inputStatus.textContent = "Sample tumor case loaded";

});


manualToggle.addEventListener("click", () => {

    manualSection.classList.toggle("open");

    if (manualSection.classList.contains("open")) {
        manualToggle.textContent = "Close Manual Form";
    } else {
        manualToggle.textContent = "Open Manual Form";
    }

});


featureInputs.forEach(input => {

    input.addEventListener("input", updateFeatureStatus);

});


csvFile.addEventListener("change", event => {

    const file = event.target.files[0];

    if (!file) {
        csvStatus.textContent = "No file selected";
        return;
    }

    if (!file.name.toLowerCase().endsWith(".csv")) {
        csvStatus.textContent = "Please select a CSV file.";
        csvFile.value = "";
        return;
    }

    csvStatus.textContent = `Reading ${file.name}...`;

    const reader = new FileReader();

    reader.onload = function(loadEvent) {

        try {

            const text = loadEvent.target.result.trim();

            if (!text) {
                throw new Error("CSV file is empty.");
            }

            const lines = text
                .split(/\r?\n/)
                .filter(line => line.trim() !== "");

            if (lines.length === 0) {
                throw new Error("No data found in CSV.");
            }

            let values = lines[0]
                .split(",")
                .map(value => value.trim());

            /*
             * If first row contains column names,
             * use the second row as the feature values.
             */
            const firstRowIsHeader = values.some(
                value => Number.isNaN(Number(value))
            );

            if (firstRowIsHeader) {

                if (lines.length < 2) {
                    throw new Error(
                        "CSV contains a header but no data row."
                    );
                }

                values = lines[1]
                    .split(",")
                    .map(value => value.trim());
            }

            const numericValues = values.map(value => Number(value));

            if (numericValues.length !== 30) {
                throw new Error(
                    `CSV must contain exactly 30 feature values. ` +
                    `Found ${numericValues.length}.`
                );
            }

            if (numericValues.some(value => Number.isNaN(value))) {
                throw new Error(
                    "CSV contains one or more non-numeric feature values."
                );
            }

            populateFeatures(numericValues);

            csvStatus.textContent =
                `${file.name} loaded successfully`;

            inputStatus.textContent =
                "CSV tumor measurements ready for analysis";

        } catch (error) {

            console.error(error);

            csvStatus.textContent = error.message;

            alert(error.message);

        }

    };

    reader.onerror = function() {

        csvStatus.textContent = "Unable to read CSV file.";

        alert("Unable to read the selected CSV file.");

    };

    reader.readAsText(file);

});


form.addEventListener("submit", async event => {

    event.preventDefault();

    const features = Array.from(featureInputs)
        .map(input => Number(input.value));

    if (
        features.length !== 30 ||
        features.some(value => Number.isNaN(value))
    ) {

        alert(
            "Please upload, load, or manually enter all 30 tumor features."
        );

        return;
    }


    emptyState.style.display = "none";
    result.style.display = "none";
    loadingState.style.display = "flex";

    analyzeButton.disabled = true;

    const buttonText =
        analyzeButton.querySelector("span");

    buttonText.textContent = "Analyzing...";


    try {

        const response = await fetch(
            "https://tumor-decision-support.onrender.com/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    features: features
                })
            }
        );


        if (!response.ok) {

            let errorMessage =
                `Prediction request failed with status ${response.status}`;

            try {

                const errorData = await response.json();

                if (errorData.detail) {
                    errorMessage = JSON.stringify(errorData.detail);
                }

            } catch (_) {
                // Keep original error message.
            }

            throw new Error(errorMessage);
        }


        const data = await response.json();

        loadingState.style.display = "none";
        result.style.display = "block";


        const probability =
            Number(data.malignant_probability || 0);


        predictionText.textContent =
            data.prediction || "Unavailable";


        probabilityValue.textContent =
            `${probability.toFixed(2)}%`;


        probabilityBar.style.width =
            `${Math.min(Math.max(probability, 0), 100)}%`;


        riskLevel.textContent =
            data.risk_level || "Unknown";


        thresholdValue.textContent =
            data.decision_threshold ?? "N/A";


        riskBadge.className = "risk-badge";
        riskLevel.className = "";


        const risk =
            String(data.risk_level || "").toLowerCase();


        if (risk === "high") {

            riskBadge.textContent = "HIGH RISK";

            riskBadge.classList.add("high-risk");
            riskLevel.classList.add("high-risk");

            probabilityBar.style.background = "#d64545";

        } else if (risk === "moderate") {

            riskBadge.textContent = "MODERATE RISK";

            riskBadge.classList.add("moderate-risk");
            riskLevel.classList.add("moderate-risk");

            probabilityBar.style.background = "#d9822b";

        } else {

            riskBadge.textContent = "LOW RISK";

            riskBadge.classList.add("low-risk");
            riskLevel.classList.add("low-risk");

            probabilityBar.style.background = "#15966a";

        }


        result.scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });


    } catch (error) {

        console.error(error);

        loadingState.style.display = "none";
        emptyState.style.display = "flex";


        alert(
            "Prediction failed.\n\n" +
            error.message +
            "\n\nPlease confirm that the FastAPI service is running."
        );


    } finally {

        analyzeButton.disabled = false;

        buttonText.textContent =
            "Analyze Tumor Risk";

    }

});


updateFeatureStatus();
