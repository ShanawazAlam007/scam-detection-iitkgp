# Scam Message Detection API - IIT KGP Hackathon

This project is a backend API developed for the IIT KGP Hackathon to detect fraudulent and scam messages in real-time. It uses a hybrid approach combining rule-based filtering with a machine learning model to achieve high accuracy and robustness.

## Problem Statement

Online financial fraud, particularly through deceptive messages (e.g., fake KYC updates, OTP requests, and urgent payment demands), is a growing threat. Scammers continuously evolve their tactics, making it difficult for users to distinguish legitimate messages from malicious ones. This project aims to provide an automated, reliable solution to identify and flag such scam attempts before users fall victim.

## Our Approach: Hybrid Detection

We use a two-stage hybrid model for robust and efficient scam detection:

1.  **Rule-Based Filtering (First Line of Defense):** A set of highly specific rules immediately flags messages containing obvious scam indicators (e.g., urgent requests for OTPs, threatening language) or clear safe patterns (e.g., standard corporate communication). This is fast and catches the most blatant cases.

2.  **Machine Learning Model (Deep Analysis):** If a message is not caught by the rules, it is passed to a trained Deep Learning model (using TensorFlow/Keras). The model analyzes the text semantics and associated metadata (e.g., presence of URLs, severity) to make a nuanced prediction.

This hybrid architecture ensures both speed and accuracy, leveraging the strengths of both deterministic rules and probabilistic models.

### Architecture Overview

```
[Input Message] -> [Rule-Based Engine] --(Rule Match?)--> [SAFE/SCAM] 
                      |
              (No Rule Match)
                      |
                      v
                  [ML Model] -> [SAFE/SCAM]
```

## How to Run Locally

This project is self-contained and can be run locally for evaluation. The required model files are included in the repository.

**Prerequisites:**
- Python 3.9+
- `venv` for virtual environment management

**Step-by-step setup:**

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Flask backend server:**
    ```bash
    python app.py
    ```
    The server will start on `http://127.0.0.1:5000`. You will see a "Model loaded successfully" message.

## API Usage

The API provides a simple interface for scam detection.

### Health Check

-   **Endpoint:** `GET /health`
-   **Description:** Confirms that the API is running.
-   **Response:** `{"status": "ok"}`

### Prediction

-   **Endpoint:** `POST /predict`
-   **Description:** Analyzes input text and metadata to predict if it's a scam.

**Sample `curl` Request:**

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "text": "Sir aapka KYC pending hai, OTP abhi bhejo warna account freeze ho jayega",
        "metadata": {
          "has_otp": 1,
          "has_urgency": 1,
          "has_threat": 1
        }
      }'
```

**Output Format:**

The API returns a simple JSON object with a single key:

-   **Success (Scam):**
    ```json
    {
      "prediction": "scam"
    }
    ```
-   **Success (Safe):**
    ```json
    {
      "prediction": "safe"
    }
    ```

## Privacy & Ethics

User privacy is paramount. This API is designed with the following principles:
- **No Data Storage:** Input messages and metadata are processed in-memory and are **never stored** on disk or in any database.
- **Ephemeral Inference:** All analysis is done per-request and discarded immediately after a response is sent.

## Deployment Note

For the purpose of this hackathon, the application is designed for local demonstration. The self-contained Flask server is sufficient for evaluation. However, the architecture is cloud-ready and can be deployed to platforms like Render, Heroku, or AWS with standard containerization (e.g., Docker) and a production-grade WSGI server like Gunicorn.
