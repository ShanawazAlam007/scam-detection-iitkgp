import re

# --- HIGH-RISK SCAM PATTERNS ---
# These keywords are designed to be multi-language (English, Hindi, Hinglish)
# and cover critical scam vectors.

OTP_KEYWORDS = [
    r"\botp\b",
    r"\bone time password\b",
    r"share otp",
    "otp bhejo",
    "otp dena",
    "otp send",
]

KYC_KEYWORDS = [
    r"\bkyc\b",
    "kyc pending",
    "verify now",
    "verification required",
    "aadhaar link",
    "pan link",
]

ACCOUNT_THREAT_KEYWORDS = [
    "account freeze",
    "account block",
    "service band",
    "number deactivate",
    "account suspend",
    "block ho jayega",
    "freeze ho jayega",
    "band ho jayega",
    "deactivate ho jayega",
]

URGENCY_KEYWORDS = [
    r"\babhi\b", # now
    r"\baaj\b", # today
    r"\bturant\b", # immediately
    r"\bjaldi\b", # quickly
    r"\bimmediately\b",
    r"\bnow\b",
    r"\bwarn\ba", # otherwise
    r"\botherwise\b",
    r"\bfinal warning\b",
    "last chance",
]

# --- WORKPLACE / NORMAL HUMAN COMMUNICATION PATTERNS ---
WORKPLACE_KEYWORDS = [
    "update", "review", "spreadsheet", "project", "meeting", "document", "eod",
    "complete today", "no rush", "team", "colleague", "task", "report", "deadline",
    "client", "office", "call", "email", "discussion", "plan", "schedule", "work"
]

# --- GENERAL SCAM KEYWORDS (for low confidence safety net) ---
GENERAL_SCAM_KEYWORDS = [
    "prize", "lottery", "winner", "bank details", "credit card", "personal info",
    "transfer", "click link", "due", "fee", "blocked", "suspended", "deactivated",
    "send money", "rupees", "rs", "payment", "loan offer", "guarantee",
    "congratulations", "claim now", "urgent action", "install app", "remote access"
]

def contains_any_keyword(text, keywords):
    """Checks if text contains any of the given keywords/phrases (case-insensitive and whole word)."""
    for keyword in keywords:
        # If the keyword is already a regex pattern with \b, use it as is
        if r'\b' in keyword or r'\B' in keyword or r'^' in keyword or r'$' in keyword:
            pattern = keyword
        else:
            # Otherwise, add word boundaries to ensure whole word match
            pattern = r'\b' + re.escape(keyword) + r'\b' # re.escape handles special regex chars
        
        if re.search(pattern, text, re.IGNORECASE):
            # print(f"DEBUG:     Matched keyword: '{keyword}' with pattern '{pattern}' in text: '{text}'") # Debug print removed
            return True
    return False

def detect_hard_scam_override(text: str) -> (bool, str):
    """
    Detects high-risk scam patterns that *must* override any model prediction.
    Returns (True, reason_string) if a hard override is triggered, else (False, None).
    """
    text = text.lower() # Already lowercased in preprocessing, but good for robustness

    # KYC + OTP + Account Threat + Urgency (CRITICAL failure scenario)
    kyc_match = contains_any_keyword(text, KYC_KEYWORDS)
    otp_match = contains_any_keyword(text, OTP_KEYWORDS)
    account_threat_match = contains_any_keyword(text, ACCOUNT_THREAT_KEYWORDS)
    urgency_match = contains_any_keyword(text, URGENCY_KEYWORDS)

    if kyc_match and otp_match and account_threat_match and urgency_match:
        return True, "Hard Scam Override: KYC + OTP + Account Threat + Urgency"
    
    # OTP + Urgency + Action Demand (implicit in OTP context)
    # A simplified version of action demand for this specific rule
    action_demand_keywords_for_otp = ["send", "share", "de do", "bhejo", "click", "install", "dabaao"]
    if (
        otp_match and
        urgency_match and
        contains_any_keyword(text, action_demand_keywords_for_otp)):
        return True, "Hard Scam Override: OTP + Urgency + Action Demand (OTP related)"

    # KYC + Account Threat + Urgency
    if kyc_match and account_threat_match and urgency_match:
        return True, "Hard Scam Override: KYC + Account Threat + Urgency"
    
    # General highly suspicious phrases combinations (can be expanded)
    if (otp_match and contains_any_keyword(text, ["link", "click"])):
        return True, "Hard Scam Override: OTP + Click Link"

    return False, None

def detect_hard_safe_override(text: str, metadata: dict) -> (bool, str):
    """
    Detects normal human/workplace messages that *must* override any scam prediction,
    unless very strong scam indicators are present.
    Returns (True, reason_string) if a hard safe override is triggered, else (False, None).
    """
    text = text.lower()
    
    print(f"DEBUG: detect_hard_safe_override - Text: {text}")

    # Condition 1: Presence of workplace/task-related keywords
    workplace_keywords_present = contains_any_keyword(text, WORKPLACE_KEYWORDS)
    print(f"DEBUG: Condition 1 (workplace_keywords_present): {workplace_keywords_present}")
    if not workplace_keywords_present:
        return False, None

    # Condition 2: Absence of strong scam indicators
    # These override even the hard safe rule
    otp_present = contains_any_keyword(text, OTP_KEYWORDS)
    print(f"DEBUG:   OTP_KEYWORDS present: {otp_present}")
    kyc_present = contains_any_keyword(text, KYC_KEYWORDS)
    print(f"DEBUG:   KYC_KEYWORDS present: {kyc_present}")
    account_threat_present = contains_any_keyword(text, ACCOUNT_THREAT_KEYWORDS)
    print(f"DEBUG:   ACCOUNT_THREAT_KEYWORDS present: {account_threat_present}")
    financial_keywords_present_in_text = contains_any_keyword(text, ["financial request", "bank details", "credit card", "loan", "rupees", "rs"])
    print(f"DEBUG:   Explicit Financial keywords in text present: {financial_keywords_present_in_text}")
    has_upi_meta = metadata.get('has_upi')
    print(f"DEBUG:   metadata['has_upi']: {has_upi_meta}")
    has_threat_meta = metadata.get('has_threat')
    print(f"DEBUG:   metadata['has_threat']: {has_threat_meta}")
    severity_high = metadata.get('severity', 0) > 0.7
    print(f"DEBUG:   metadata['severity'] > 0.7: {severity_high}")

    strong_scam_indicators_present = \
       otp_present or \
       kyc_present or \
       account_threat_present or \
       financial_keywords_present_in_text or \
       has_upi_meta or \
       has_threat_meta or \
       severity_high
       
    print(f"DEBUG: Condition 2 (strong_scam_indicators_present): {strong_scam_indicators_present}")
    if strong_scam_indicators_present:
       return False, None # If any strong scam indicator is present, do NOT trigger hard safe rule

    # Condition 3: Neutral/polite tone (implicitly checked by absence of urgency/threat from scam rules)
    # This is a heuristic, assuming if no strong scam indicators and workplace keywords, it's neutral.
    
    return True, "Hard Safe Override: Workplace/Neutral Communication (no risk indicators)"


# --- Original Rule-based functions (adapted for new structure if needed) ---

def has_action_demand(text, metadata):
    action_keywords = ["pay", "send", "install", "download", "share", "give", "provide", "click", "verify", "update"]
    if any(keyword in text for keyword in action_keywords):
        return True
    if metadata.get('has_url') or metadata.get('has_upi'):
        return True
    return False

def has_financial_request(text, metadata):
    financial_keywords = ["rs", "rupees", "inr", "$", "usd", "fee", "charge", "payment", "loan"]
    if any(keyword in text for keyword in financial_keywords):
        return True
    if metadata.get('has_upi'):
        return True
    return False
    
def is_greeting(text):
    greetings = ["hi", "hello", "how are you"]
    # Check for exact match of short greetings
    if text.strip() in greetings:
        return True
    # Check for presence of greetings in longer text, but be careful not to be too broad
    if any(f" {greeting} " in text for greeting in greetings):
        return True
    return False

def is_normal_conversation(text, metadata):
    # This is a heuristic. A more sophisticated approach would be needed for a real system.
    # For now, we'll define "normal" as something that doesn't have any of the scam-indicating features.
    if not has_action_demand(text, metadata) and \
       not has_financial_request(text, metadata) and \
       not metadata.get('has_threat') and \
       not metadata.get('has_urgency') and \
       metadata.get('severity', 0) < 0.2:
           # and if the text is short and simple
           if len(text.split()) < 15:
               return True
    return False


def apply_rules(text, metadata):
    """
    Applies a set of deterministic rules to classify a text as SAFE or SCAM.

    Args:
        text (str): The input text.
        metadata (dict): A dictionary of metadata features.

    Returns:
        tuple: (prediction: int, reason: str) or (None, None) if no rule matched.
    """
    text = text.lower()

    # SCAM overrides (less critical than hard overrides, but still strong)
    if (metadata.get('has_otp') or metadata.get('has_upi')) and metadata.get('has_urgency'):
        return 1, "Scam Rule: OTP/UPI + Urgency"
    if metadata.get('has_threat') and has_action_demand(text, metadata):
        return 1, "Scam Rule: Threat + Action Demand"
    if has_financial_request(text, metadata) and metadata.get('severity', 0) > 0.8: # High severity
        return 1, "Scam Rule: Financial Request + High Severity"

    # SAFE overrides
    if is_greeting(text):
        if not has_action_demand(text, metadata) and not has_financial_request(text, metadata):
            return 0, "Safe Rule: Greeting (no action/financial demand)"
    
    if is_normal_conversation(text, metadata):
        return 0, "Safe Rule: Normal Conversation"

    if metadata.get('severity', 0) < 0.1 and not has_action_demand(text, metadata):
        return 0, "Safe Rule: Low Severity + No Action Demand"

    return None, None