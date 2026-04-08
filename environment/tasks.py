"""Task definitions: emails + ground truth labels for each task."""
from __future__ import annotations
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# TASK 1 – Spam Classification (Easy)
# ---------------------------------------------------------------------------
TASK1_EMAILS = [
    {
        "id": "t1_e1",
        "subject": "Congratulations! You've won $1,000,000",
        "body": (
            "Dear Friend, You have been selected as today's lucky winner. "
            "Click here NOW to claim your prize. No purchase necessary. "
            "Send your bank details to claim@lottery-winner.biz immediately!"
        ),
        "sender": "noreply@lottery-winner.biz",
        "timestamp": "2024-01-10 08:00:00",
    },
    {
        "id": "t1_e2",
        "subject": "Q3 Engineering Sprint Planning",
        "body": (
            "Hi team, please join the sprint planning session on Thursday at 10 AM. "
            "Agenda: review backlog, estimate stories, and assign owners. "
            "Zoom link in the calendar invite."
        ),
        "sender": "pm@company.com",
        "timestamp": "2024-01-10 09:15:00",
    },
    {
        "id": "t1_e3",
        "subject": "URGENT: Verify your account or lose access!",
        "body": (
            "Your account will be suspended in 24 hours unless you verify your information. "
            "Click the link below and enter your password and credit card number. "
            "http://secure-login-verify.xyz/confirm"
        ),
        "sender": "security@secure-login-verify.xyz",
        "timestamp": "2024-01-10 10:30:00",
    },
    {
        "id": "t1_e4",
        "subject": "Code review request: PR #482 - Auth refactor",
        "body": (
            "Hey, I've opened PR #482 for the auth service refactor we discussed. "
            "Could you take a look before EOD? I've added tests for the edge cases we found last week. "
            "GitHub link is in the PR description."
        ),
        "sender": "dev@company.com",
        "timestamp": "2024-01-10 11:00:00",
    },
    {
        "id": "t1_e5",
        "subject": "Exclusive offer: Free Rolex watches – limited time!",
        "body": (
            "Buy one get ten free! Luxury replica watches shipped worldwide. "
            "No customs fees guaranteed. Order in the next 2 hours to get 90% off. "
            "Visit: cheap-luxury-watches.net/offer?id=99283"
        ),
        "sender": "deals@cheap-luxury-watches.net",
        "timestamp": "2024-01-10 12:00:00",
    },
]

TASK1_LABELS: List[str] = [
    "spam",
    "not_spam",
    "spam",
    "not_spam",
    "spam",
]

TASK1_META: Dict[str, Any] = {
    "id": "spam_classification",
    "name": "Email Spam Classification",
    "difficulty": "easy",
    "description": (
        "Classify each email as 'spam' or 'not_spam'. "
        "Spam includes phishing, scam, and unsolicited promotional emails."
    ),
    "instructions": (
        "For each email, set spam_label to 'spam' or 'not_spam'. "
        "Optionally include your reasoning in the reasoning field."
    ),
}

# ---------------------------------------------------------------------------
# TASK 2 – Email Prioritization (Medium)
# ---------------------------------------------------------------------------
TASK2_EMAILS = [
    {
        "id": "t2_e1",
        "subject": "PRODUCTION DOWN – payment service returning 500s",
        "body": (
            "All payment transactions are failing. Error rate is 100% for the last 15 minutes. "
            "Revenue impact ~$50k/min. Need immediate attention. "
            "@oncall pinging you now."
        ),
        "sender": "alerts@monitoring.company.com",
        "timestamp": "2024-01-15 14:00:00",
    },
    {
        "id": "t2_e2",
        "subject": "Weekly newsletter – Tech digest",
        "body": (
            "This week in tech: AI breakthroughs, new programming languages, and more. "
            "Read our curated digest of the top 10 articles from the past week."
        ),
        "sender": "digest@technewsletter.io",
        "timestamp": "2024-01-15 08:00:00",
    },
    {
        "id": "t2_e3",
        "subject": "Invoice #INV-2024-089 due in 3 days",
        "body": (
            "Dear Client, this is a reminder that Invoice #INV-2024-089 for $4,200 "
            "is due on January 18, 2024. Please process payment to avoid late fees. "
            "Contact billing@vendor.com with questions."
        ),
        "sender": "billing@vendor.com",
        "timestamp": "2024-01-15 09:30:00",
    },
    {
        "id": "t2_e4",
        "subject": "Performance review scheduled for next Tuesday",
        "body": (
            "Your annual performance review has been scheduled for Tuesday Jan 23 at 2 PM "
            "with your manager Sarah. Please complete your self-assessment form by Friday."
        ),
        "sender": "hr@company.com",
        "timestamp": "2024-01-15 10:00:00",
    },
    {
        "id": "t2_e5",
        "subject": "Lunch menu for this week",
        "body": (
            "Hi everyone! The cafeteria lunch menu for this week is now posted. "
            "Monday: pasta, Tuesday: stir fry, Wednesday: tacos... Enjoy!"
        ),
        "sender": "cafeteria@company.com",
        "timestamp": "2024-01-15 07:45:00",
    },
]

TASK2_LABELS: List[Dict[str, str]] = [
    {"priority": "high",   "category": "support"},
    {"priority": "low",    "category": "promotion"},
    {"priority": "high",   "category": "finance"},
    {"priority": "medium", "category": "work"},
    {"priority": "low",    "category": "other"},
]

TASK2_META: Dict[str, Any] = {
    "id": "email_prioritization",
    "name": "Email Prioritization & Categorization",
    "difficulty": "medium",
    "description": (
        "Assign a priority (high/medium/low) and category "
        "(work/personal/finance/promotion/support/other) to each email."
    ),
    "instructions": (
        "For each email, set 'priority' and 'category' in your action. "
        "high=requires action today, medium=this week, low=informational."
    ),
}

# ---------------------------------------------------------------------------
# TASK 3 – Email Response Generation (Hard)
# ---------------------------------------------------------------------------
TASK3_EMAILS = [
    {
        "id": "t3_e1",
        "subject": "Extremely disappointed with your service – demand refund",
        "body": (
            "To whom it may concern,\n\n"
            "I placed order #ORD-9921 three weeks ago and still haven't received it. "
            "Every time I contact support I get a different story. "
            "I was promised delivery within 5 business days. "
            "I demand a full refund of $189.99 and an explanation for this disaster. "
            "If this isn't resolved in 48 hours I'm filing a chargeback and leaving "
            "a public review on every platform I can find.\n\n"
            "Furiously yours,\nMarco Vitelli"
        ),
        "sender": "marco.vitelli@email.com",
        "timestamp": "2024-01-20 09:00:00",
    },
    {
        "id": "t3_e2",
        "subject": "Question about enterprise pricing and custom integrations",
        "body": (
            "Hi,\n\n"
            "We are a mid-size logistics company (~500 employees) evaluating your platform. "
            "Could you share: (1) enterprise pricing tiers, (2) whether you support custom "
            "webhooks and SSO/SAML, and (3) your SLA guarantees for uptime?\n\n"
            "We'd also like to schedule a demo with your sales team.\n\n"
            "Best,\nLi Wei\nCTO, FastShip Ltd."
        ),
        "sender": "li.wei@fastship.com",
        "timestamp": "2024-01-20 10:30:00",
    },
    {
        "id": "t3_e3",
        "subject": "Possible security vulnerability in your API",
        "body": (
            "Hello Security Team,\n\n"
            "I am a security researcher and I believe I've identified a potential "
            "IDOR vulnerability in your /api/v2/orders/{id} endpoint that could allow "
            "authenticated users to access other users' order data by iterating IDs. "
            "I've attached a PoC in the interest of responsible disclosure. "
            "I have not shared this publicly and am giving you 14 days to respond "
            "before considering publication.\n\n"
            "Best regards,\nAisha Okonkwo"
        ),
        "sender": "aisha.okonkwo@secresearch.io",
        "timestamp": "2024-01-20 14:00:00",
    },
]

# Grading criteria per email – deterministic keyword/structure checks
TASK3_CRITERIA: List[Dict[str, Any]] = [
    {
        # t3_e1: angry customer wanting refund
        "required_keywords": ["refund", "sorry", "apologize", "order", "resolve"],
        "required_elements": {
            "greeting": ["dear", "hello", "hi", "mr.", "ms.", "marco"],
            "closing": ["sincerely", "regards", "best", "thank you", "yours"],
        },
        "min_words": 60,
        "forbidden": ["no refund", "your fault", "can't help"],
        "weight_keywords": 0.35,
        "weight_structure": 0.25,
        "weight_length": 0.20,
        "weight_tone": 0.20,
    },
    {
        # t3_e2: enterprise sales inquiry
        "required_keywords": ["pricing", "enterprise", "demo", "sso", "webhook", "sla", "integration"],
        "required_elements": {
            "greeting": ["dear", "hello", "hi", "li", "ms.", "mr."],
            "closing": ["sincerely", "regards", "best", "thank you"],
        },
        "min_words": 80,
        "forbidden": ["we don't support", "no sla", "can't schedule"],
        "weight_keywords": 0.40,
        "weight_structure": 0.20,
        "weight_length": 0.15,
        "weight_tone": 0.25,
    },
    {
        # t3_e3: security vulnerability report
        "required_keywords": ["thank", "security", "vulnerability", "investigate", "team", "report", "responsible"],
        "required_elements": {
            "greeting": ["dear", "hello", "hi", "aisha", "researcher"],
            "closing": ["sincerely", "regards", "best", "thank you"],
        },
        "min_words": 70,
        "forbidden": ["ignore", "not a bug", "won't fix"],
        "weight_keywords": 0.40,
        "weight_structure": 0.20,
        "weight_length": 0.15,
        "weight_tone": 0.25,
    },
]

TASK3_META: Dict[str, Any] = {
    "id": "email_response",
    "name": "Professional Email Response Generation",
    "difficulty": "hard",
    "description": (
        "Write a professional, empathetic, and complete response to each email. "
        "Responses are graded on keyword coverage, structure, length, and appropriate tone."
    ),
    "instructions": (
        "Set 'response_text' to your full email response. "
        "Include a greeting, address all points raised, and end with a professional closing."
    ),
}

ALL_TASKS = {
    "spam_classification": {
        "meta": TASK1_META,
        "emails": TASK1_EMAILS,
        "labels": TASK1_LABELS,
    },
    "email_prioritization": {
        "meta": TASK2_META,
        "emails": TASK2_EMAILS,
        "labels": TASK2_LABELS,
    },
    "email_response": {
        "meta": TASK3_META,
        "emails": TASK3_EMAILS,
        "criteria": TASK3_CRITERIA,
    },
}
