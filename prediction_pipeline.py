import pickle
import time
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from rules import apply_rules, detect_hard_scam_override, detect_hard_safe_override, contains_any_keyword, GENERAL_SCAM_KEYWORDS
from unicodedata import normalize
import re

class ScamDetector:
    def __init__(self, model_path, tokenizer_path, scaler_path):
        print("Loading self-contained model and artifacts...")
        self.model = load_model(model_path, compile=False) # Load without compiling
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # Recompile
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        # Assuming input_shape[0][1] is max_len for text input
        # This will need to be robust for models with varying input shapes
        self.max_len = self.model.input[0].shape[1] if isinstance(self.model.input, list) else self.model.input[0].shape[1]
        print("Artifacts loaded successfully. No fastText dependency.")

    def preprocess_text(self, text):
        text = normalize('NFKC', text)
        text = text.lower()
        text = text.replace('\n', ' ')
        fillers = ['uh', 'um', 'ah', 'er', 'hmm', 'sir', 'ma\'am', 'ji']
        for filler in fillers:
            text = re.sub(r'\b' + filler + r'\b', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def predict(self, text, metadata):
        """
        Predicts if a given text is a scam or not using a self-contained model.
        Returns a dictionary with prediction details, including the reason.
        """
        start_time = time.time()
        clean_text = self.preprocess_text(text)
        prediction_source = "MODEL"
        prediction_reason = "Model Prediction"
        final_prediction = 0 # Default to safe
        confidence = 0.0

        # --- 1. Hard SAFE Override (Highest Priority) ---
        hard_safe_triggered, hard_safe_reason = detect_hard_safe_override(clean_text, metadata)
        if hard_safe_triggered:
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            print(f"Inference latency (HARD SAFE RULE): {latency:.2f} ms")
            return {
                "prediction": 0,
                "source": "RULE",
                "confidence": 1.0,
                "reason": hard_safe_reason,
                "latency_ms": latency
            }

        # --- 2. Hard SCAM Override ---
        hard_scam_triggered, hard_scam_reason = detect_hard_scam_override(clean_text)
        if hard_scam_triggered:
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            print(f"Inference latency (HARD SCAM RULE): {latency:.2f} ms")
            return {
                "prediction": 1,
                "source": "RULE",
                "confidence": 1.0,
                "reason": hard_scam_reason,
                "latency_ms": latency
            }

        # --- 3. General Rule-Based Overrides ---
        rule_pred, rule_reason = apply_rules(clean_text, metadata)
        if rule_pred is not None:
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            print(f"Inference latency (GENERAL RULE): {latency:.2f} ms")
            return {
                "prediction": rule_pred,
                "source": "RULE",
                "confidence": 1.0 if rule_pred == 1 else 0.0, # Rules are deterministic
                "reason": rule_reason,
                "latency_ms": latency
            }

        # --- 4. ML Model Prediction ---
        # Tokenize and pad text
        sequence = self.tokenizer.texts_to_sequences([clean_text])
        padded_sequence = pad_sequences(
            sequence, maxlen=self.max_len, padding="post", truncating="post"
        )

        # Create a full metadata dictionary with defaults to avoid KeyErrors
        metadata_cols = [
            "severity", "has_url", "has_upi", "has_otp", "has_qr",
            "has_phone", "has_threat", "has_urgency",
        ]
        complete_metadata = {col: metadata.get(col, 0) for col in metadata_cols}
        
        # Scale metadata
        metadata_df = pd.DataFrame([complete_metadata])
        metadata_features = metadata_df[metadata_cols].values
        
        # The scaler was likely trained only on the 'severity' column.
        # It's safer to transform only that column to avoid shape mismatches.
        metadata_features[:, 0] = self.scaler.transform(
            metadata_features[:, 0].reshape(-1, 1)
        ).flatten()

        prediction_prob = self.model.predict([padded_sequence, metadata_features], verbose=0)[0][0]
        confidence = float(prediction_prob)
        final_prediction = 1 if prediction_prob > 0.5 else 0

        # --- 5. Low-Confidence + Risk Safety Net (Secondary Protection) ---
        if final_prediction == 0 and confidence < 0.3 and contains_any_keyword(clean_text, GENERAL_SCAM_KEYWORDS):
            final_prediction = 1
            prediction_source = "MODEL (Low Confidence Safety Net)"
            prediction_reason = "Model low confidence (<30%) and general scam keywords present"
        elif final_prediction == 1:
            prediction_source = "MODEL"
            prediction_reason = "Model Prediction (SCAM)"
        else:
            prediction_source = "MODEL"
            prediction_reason = "Model Prediction (SAFE)"


        end_time = time.time()
        latency = (end_time - start_time) * 1000
        print(f"Inference latency ({prediction_source.split('(')[0].strip()}): {latency:.2f} ms")

        return {
            "prediction": final_prediction,
            "source": prediction_source,
            "confidence": confidence,
            "reason": prediction_reason,
            "latency_ms": latency
        }

if __name__ == '__main__':
    detector = ScamDetector('trained_model.h5', 'tokenizer.pkl', 'metadata_scaler.pkl')

    print("\n--- Mandatory Tests (with Final Balancing) ---")

    # Test 1: Original SAFE
    text1 = "Hi how are you, hope you are doing well."
    meta1 = {'severity': 0.0, 'has_url': 0, 'has_upi': 0, 'has_otp': 0, 'has_qr': 0, 'has_phone': 0, 'has_threat': 0, 'has_urgency': 0}
    result1 = detector.predict(text1, meta1)
    print(f"Text: '{text1}'")
    print(f"Prediction: {'SCAM' if result1['prediction'] == 1 else 'SAFE'}, Source: {result1['source']}, Reason: {result1['reason']}, Confidence: {result1['confidence']:.2%}\n")
    assert result1['prediction'] == 0

    # Test 2: Original SCAM (Model)
    text2 = "Congratulations! You have been selected for a special loan offer of 500000 rupees. No documentation required. This is a limited time offer. To avail this offer, please click on the link below and pay a small processing fee of 500 rupees. The link will expire in 2 hours. Click here: http://example.com/scam-loan"
    meta2 = {'severity': 0.8, 'has_url': 1, 'has_upi': 0, 'has_otp': 0, 'has_qr': 0, 'has_phone': 0, 'has_threat': 0, 'has_urgency': 1}
    result2 = detector.predict(text2, meta2)
    print(f"Text: '{text2}'")
    print(f"Prediction: {'SCAM' if result2['prediction'] == 1 else 'SAFE'}, Source: {result2['source']}, Reason: {result2['reason']}, Confidence: {result2['confidence']:.2%}\n")
    assert result2['prediction'] == 1

    # Test 3: Original SCAM (Rule)
    text3 = "Bhai, urgent help chahiye. Mere dost ka accident ho gaya hai. Please 10000 rupees is UPI id pe bhej do: scammer@upi. Bahut emergency hai."
    meta3 = {'severity': 0.9, 'has_url': 0, 'has_upi': 1, 'has_otp': 0, 'has_qr': 0, 'has_phone': 0, 'has_threat': 0, 'has_urgency': 1}
    result3 = detector.predict(text3, meta3)
    print(f"Text: '{text3}'")
    print(f"Prediction: {'SCAM' if result3['prediction'] == 1 else 'SAFE'}, Source: {result3['source']}, Reason: {result3['reason']}, Confidence: {result3['confidence']:.2%}\n")
    assert result3['prediction'] == 1

    # Mandatory Test (from previous step): Hard Scam Override
    critical_scam_text = "Sir aapka KYC pending hai. Agar aaj verify nahi kiya toh account freeze ho jayega. OTP abhi bhejo warna service band ho jayegi."
    critical_scam_meta = {'severity': 0.9, 'has_url': 0, 'has_upi': 0, 'has_otp': 1, 'has_qr': 0, 'has_phone': 0, 'has_threat': 1, 'has_urgency': 1}
    result_critical = detector.predict(critical_scam_text, critical_scam_meta)
    print(f"Text: '{critical_scam_text}'")
    print(f"Prediction: {'SCAM' if result_critical['prediction'] == 1 else 'SAFE'}, Source: {result_critical['source']}, Reason: {result_critical['reason']}, Confidence: {result_critical['confidence']:.2%}\n")
    assert result_critical['prediction'] == 1
    assert "Hard Scam Override" in result_critical['reason']

    # NEW MANDATORY TEST (Step 5): Hard Safe Override
    workplace_text = "Please update the spreadsheet before EOD. Make sure the numbers are correct and reviewed once. No rush, just complete it today."
    workplace_meta = {'severity': 0.1, 'has_url': 0, 'has_upi': 0, 'has_otp': 0, 'has_qr': 0, 'has_phone': 0, 'has_threat': 0, 'has_urgency': 0}
    result_workplace = detector.predict(workplace_text, workplace_meta)
    print(f"Text: '{workplace_text}'")
    print(f"Prediction: {'SCAM' if result_workplace['prediction'] == 1 else 'SAFE'}, Source: {result_workplace['source']}, Reason: {result_workplace['reason']}, Confidence: {result_workplace['confidence']:.2%}\n")
    assert result_workplace['prediction'] == 0
    assert "Hard Safe Override" in result_workplace['reason']

    print("\nAll tests passed!")