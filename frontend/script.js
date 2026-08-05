const form = document.getElementById("predictionForm");
const resultDiv = document.getElementById("result");

form.addEventListener("submit", async function (e) {

    e.preventDefault();

    const inputs = document.querySelectorAll("input");

    const features = [];

    inputs.forEach(input => {
        features.push(parseFloat(input.value));
    });

    try {

        const response = await fetch("http://127.0.0.1:8000/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                features: features
            })

        });

        const data = await response.json();

        let riskClass = "low";

        if (data.risk_level === "High")
            riskClass = "high";

        else if (data.risk_level === "Moderate")
            riskClass = "moderate";

        resultDiv.style.display = "block";

        resultDiv.innerHTML = `

            <div class="result-title">
                Prediction Result
            </div>

            <div class="result-item">
                <b>Status:</b> ${data.status}
            </div>

            <div class="result-item">
                <b>Prediction:</b> ${data.prediction}
            </div>

            <div class="result-item">
                <b>Malignant Probability:</b>
                ${data.malignant_probability}%
            </div>

            <div class="result-item ${riskClass}">
                <b>Risk Level:</b>
                ${data.risk_level}
            </div>

        `;

    }

    catch (error) {

        resultDiv.style.display = "block";

        resultDiv.innerHTML =

        `<h3 style="color:red;">
            Unable to connect to FastAPI Server.
        </h3>`;

    }

});