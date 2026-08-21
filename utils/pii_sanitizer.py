"""
TriSeva PII (Personally Identifiable Information) Redaction & Data Protection Module.

Implements high-accuracy regex pattern matchers and name anonymization rules to mask
Indian national identifiers (Aadhaar, PAN), phone numbers, email addresses, bank accounts/IFSC,
and patient/landholder personal names in accordance with the Digital Personal Data Protection (DPDP) Act 2023.
"""

import os
import re

# ── Compiled Regex Patterns ───────────────────────────────────────────────────

# 1. Aadhaar Number (12 digits, optional spaces/dashes)
AADHAAR_REGEX = re.compile(r"\b([2-9]\d{3})[\s-]?(\d{4})[\s-]?(\d{4})\b")

# 2. PAN Card (10 alphanumeric characters: 5 letters, 4 digits, 1 letter)
PAN_REGEX = re.compile(r"\b([A-Z]{5})([0-9]{4})([A-Z]{1})\b")

# 3. Indian Mobile / Landline Phone Numbers
PHONE_REGEX = re.compile(r"(?:(?:\+91[\s-]?)|\b)([6-9]\d{4})[\s-]?(\d{5})\b")

# 4. Email Addresses
EMAIL_REGEX = re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")

# 5. Bank IFSC Code (4 letters, 0, 6 alphanumeric)
IFSC_REGEX = re.compile(r"\b([A-Z]{4})0([A-Z0-9]{6})\b")

# 6. Bank Account Numbers (9 to 18 digits preceding/following bank context)
BANK_ACC_REGEX = re.compile(r"\b(?:Account|Acc|A/C|No|Num)?[\s.:#]*([0-9]{9,18})\b", re.IGNORECASE)


# ── PII Masking Functions ─────────────────────────────────────────────────────

def mask_aadhaar(match: re.Match) -> str:
    """Masks 12-digit Aadhaar number keeping only the last 4 digits visible."""
    last4 = match.group(3)
    return f"[AADHAAR: XXXX-XXXX-{last4}]"


def mask_pan(match: re.Match) -> str:
    """Masks 10-char PAN card keeping only last 4 digits + letter visible."""
    digits = match.group(2)
    last_char = match.group(3)
    return f"[PAN: XXXXX{digits[1:]}{last_char}]"


def mask_phone(match: re.Match) -> str:
    """Masks phone number keeping only the last 4 digits visible."""
    last4 = match.group(2)[1:]
    return f"[PHONE: +91-XXXXX-{last4}]"


def mask_email(match: re.Match) -> str:
    """Masks email address keeping first letter and domain suffix."""
    local = match.group(1)
    domain = match.group(2)
    masked_local = local[0] + "***" + local[-1] if len(local) > 2 else "***"
    return f"[EMAIL: {masked_local}@{domain}]"


def mask_ifsc(match: re.Match) -> str:
    """Masks IFSC code preserving bank code prefix."""
    bank = match.group(1)
    return f"[IFSC: {bank}XXXX001]"


def mask_bank_acc(match: re.Match) -> str:
    """Masks bank account number keeping last 4 digits."""
    acc = match.group(1)
    if len(acc) < 9:
        return match.group(0)
    return f"[BANK ACC: XXXXXXXXXX{acc[-4:]}]"


def sanitize_names(text: str) -> str:
    """Anonymizes patient and landholder names in structured medical and land documents."""
    if not text:
        return text

    # Anonymize Patient Name fields
    text = re.sub(
        r"\b(Patient Name|Patient's Name|Name of Insured|Name of Patient)[\s:]+([A-Za-z\s\.]{2,30})(?=\n|,|\||\b)",
        r"\1: [PATIENT_NAME]",
        text,
        flags=re.IGNORECASE
    )

    # Anonymize Landholder / Farmer Name fields
    text = re.sub(
        r"\b(Farmer Name|Landholder Name|Landowner Name|Kisan Name|किसान का नाम)[\s:]+([^\n|,|\|]{2,30})",
        r"\1: [LANDHOLDER_NAME]",
        text,
        flags=re.IGNORECASE
    )

    # Anonymize Father's / Husband's Name fields
    text = re.sub(
        r"\b(Father's Name|Husband's Name|S/o|D/o|W/o|पिता का नाम)[\s:]+([^\n|,|\|]{2,30})",
        r"\1: [GUARDIAN_NAME]",
        text,
        flags=re.IGNORECASE
    )

    return text


def sanitize_pii(text: str, anonymize_names: bool = True) -> str:
    """Applies comprehensive PII redaction and masking to input text.

    Args:
        text (str): Input document text or prompt string.
        anonymize_names (bool, optional): Whether to anonymize patient/landholder names. Defaults to True.

    Returns:
        str: Redacted text with PII replaced by secure token masks.
    """
    if not text or not isinstance(text, str):
        return text

    redacted = text

    # 1. Mask Aadhaar numbers
    redacted = AADHAAR_REGEX.sub(mask_aadhaar, redacted)

    # 2. Mask PAN card numbers
    redacted = PAN_REGEX.sub(mask_pan, redacted)

    # 3. Mask Email addresses
    redacted = EMAIL_REGEX.sub(mask_email, redacted)

    # 4. Mask Phone numbers
    redacted = PHONE_REGEX.sub(mask_phone, redacted)

    # 5. Mask IFSC Codes
    redacted = IFSC_REGEX.sub(mask_ifsc, redacted)

    # 6. Mask Bank Account numbers
    redacted = BANK_ACC_REGEX.sub(mask_bank_acc, redacted)

    # 7. Anonymize Names if enabled
    if anonymize_names:
        redacted = sanitize_names(redacted)

    return redacted


def is_pii_redaction_enabled() -> bool:
    """Checks if PII redaction is enabled in environment settings."""
    return os.getenv("ENABLE_PII_REDACTION", "true").lower() == "true"
