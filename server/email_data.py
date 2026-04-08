"""
Synthetic email corpus for the MailSort environment.

All emails are pre-defined with deterministic ground truth labels.
No randomness in the dataset — reproducibility is guaranteed.
"""

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Email schema fields:
#   id          : unique identifier used as action key
#   subject     : email subject line
#   sender      : sender display name
#   sender_email: sender address
#   body        : email body text
#   metadata    : dict with extra signals (x_priority, reply_to, etc.)
#
# Ground truth fields (not shown to agent):
#   category    : spam | urgent | routine | billing | hr | security
#   priority    : critical | high | medium | low
#   is_phishing : bool (only relevant for Task 3)
#   routing_dept: support | hr | finance | security | engineering | management | None
# ---------------------------------------------------------------------------

# ============================================================
# TASK 1 — email_classify  (easy, 1 step)
# Single email with clear, unambiguous signals.
# Default episode uses e1_01 (seed=0 → index 0).
# ============================================================

TASK1_EMAILS: List[Dict[str, Any]] = [
    {
        "id": "e1_01",
        "subject": "URGENT: Production server is DOWN — all services unreachable",
        "sender": "NOC Alerts",
        "sender_email": "noc-alerts@internal.company.com",
        "body": (
            "CRITICAL ALERT — our primary production server (prod-us-east-1) has been "
            "unreachable for the past 8 minutes. All customer-facing services are returning "
            "503 errors. Revenue impact is accumulating at approximately $12,000/minute. "
            "The on-call engineer has been paged but has not acknowledged. Immediate action "
            "required — please escalate to the VP of Engineering if no response in 5 minutes."
        ),
        "metadata": {"x_priority": "1", "auto_reply": False, "recipient_count": 12},
        "ground_truth": {
            "category": "urgent",
            "priority": "critical",
            "is_phishing": False,
            "routing_dept": "engineering",
        },
    },
    {
        "id": "e1_02",
        "subject": "Congratulations! You've been selected to receive $1,000,000!",
        "sender": "International Lottery Commission",
        "sender_email": "winner@intl-lottery-prize.net",
        "body": (
            "Dear Lucky Winner, You have been RANDOMLY SELECTED from millions of email "
            "addresses worldwide to receive ONE MILLION DOLLARS (USD $1,000,000)! "
            "To claim your prize, please reply with your full name, bank account number, "
            "routing number, date of birth, and social security number. Act within 48 hours "
            "or forfeit your prize. Do NOT tell anyone about this offer."
        ),
        "metadata": {"x_priority": "5", "auto_reply": False, "recipient_count": 1},
        "ground_truth": {
            "category": "spam",
            "priority": "low",
            "is_phishing": True,
            "routing_dept": None,
        },
    },
    {
        "id": "e1_03",
        "subject": "Meeting minutes from Q4 planning session — Oct 15",
        "sender": "Sarah Chen",
        "sender_email": "s.chen@company.com",
        "body": (
            "Hi team, please find attached the meeting minutes from yesterday's Q4 planning "
            "session. Key decisions: (1) Budget freeze until Nov 1, (2) roadmap review "
            "scheduled for Oct 22, (3) new hire onboarding pushed to Q1. No action items "
            "assigned yet — those will come out of the follow-up thread. Let me know if "
            "anything was missed. Thanks, Sarah"
        ),
        "metadata": {"x_priority": "3", "auto_reply": False, "recipient_count": 18},
        "ground_truth": {
            "category": "routine",
            "priority": "low",
            "is_phishing": False,
            "routing_dept": None,
        },
    },
    {
        "id": "e1_04",
        "subject": "Invoice #INV-2024-0891 — Payment now 30 days overdue",
        "sender": "Acme Cloud Services — Billing",
        "sender_email": "billing@acme-cloud.com",
        "body": (
            "This is a formal notice that Invoice #INV-2024-0891 for $47,250.00 is now "
            "30 days past due. Per our service agreement, a late fee of 1.5% per month "
            "has been applied. Outstanding balance including fees: $47,958.75. "
            "Failure to remit payment within 10 business days may result in suspension "
            "of your cloud services. Please remit payment via ACH or wire transfer to the "
            "account on file, or contact billing@acme-cloud.com immediately."
        ),
        "metadata": {"x_priority": "2", "auto_reply": False, "recipient_count": 2},
        "ground_truth": {
            "category": "billing",
            "priority": "high",
            "is_phishing": False,
            "routing_dept": "finance",
        },
    },
    {
        "id": "e1_05",
        "subject": "Action required: Benefits enrollment deadline this Friday",
        "sender": "HR — Benefits Team",
        "sender_email": "benefits@company.com",
        "body": (
            "Reminder: Open enrollment for 2025 benefits closes THIS FRIDAY at 5:00 PM PT. "
            "If you do not make your selections by the deadline, you will be automatically "
            "enrolled in the default plan. To review and update your elections — health, "
            "dental, vision, FSA, and 401(k) — log in to the benefits portal at "
            "benefits.company.com. Contact hr@company.com with any questions."
        ),
        "metadata": {"x_priority": "2", "auto_reply": False, "recipient_count": 450},
        "ground_truth": {
            "category": "hr",
            "priority": "medium",
            "is_phishing": False,
            "routing_dept": "hr",
        },
    },
    {
        "id": "e1_06",
        "subject": "Security Alert: Unrecognized login attempt from IP 185.220.101.47",
        "sender": "Security Operations Center",
        "sender_email": "soc@company.com",
        "body": (
            "We detected a login attempt to the admin console from IP address 185.220.101.47 "
            "(Tor exit node, geo: Unknown) at 03:47 UTC. The attempt used valid credentials "
            "but was blocked by our anomaly detection system due to unusual access patterns. "
            "If this was you, please acknowledge. If not, your credentials may be compromised "
            "— change your password immediately and contact the SOC at ext. 5555. "
            "We have temporarily restricted admin console access pending your response."
        ),
        "metadata": {"x_priority": "1", "auto_reply": False, "recipient_count": 3},
        "ground_truth": {
            "category": "security",
            "priority": "high",
            "is_phishing": False,
            "routing_dept": "security",
        },
    },
    {
        "id": "e1_07",
        "subject": "Team building: Pizza lunch next Thursday — RSVP by Wednesday",
        "sender": "Office Manager",
        "sender_email": "office@company.com",
        "body": (
            "Hey everyone! We're organizing a casual pizza lunch next Thursday, October 24, "
            "at 12:30 PM in the 3rd floor breakout space. We'll have a variety of options "
            "including gluten-free and vegan choices. Please RSVP by Wednesday EOD so we "
            "can get the right amount. Just reply to this email with your dietary preference "
            "if you have one. Hope to see you there!"
        ),
        "metadata": {"x_priority": "5", "auto_reply": False, "recipient_count": 85},
        "ground_truth": {
            "category": "routine",
            "priority": "low",
            "is_phishing": False,
            "routing_dept": None,
        },
    },
    {
        "id": "e1_08",
        "subject": "URGENT: Acme Corp master service agreement expires tomorrow",
        "sender": "Legal — Contracts",
        "sender_email": "contracts@company.com",
        "body": (
            "The master service agreement (MSA) with Acme Corp (contract #C-2021-0047, "
            "ARR: $2.1M) expires TOMORROW at midnight. Auto-renewal was not enabled. "
            "We need a signed renewal or extension letter by 5 PM today to avoid a lapse "
            "in coverage. The client has agreed to renew at current pricing — they are "
            "waiting for our countersignature. Please review the attached document and "
            "return signed via DocuSign immediately."
        ),
        "metadata": {"x_priority": "1", "auto_reply": False, "recipient_count": 4},
        "ground_truth": {
            "category": "urgent",
            "priority": "high",
            "is_phishing": False,
            "routing_dept": "management",
        },
    },
    {
        "id": "e1_09",
        "subject": "Payroll discrepancy detected — please verify your direct deposit details",
        "sender": "Payroll Department",
        "sender_email": "payroll@company.com",
        "body": (
            "We have detected a discrepancy in your payroll record for the period ending "
            "October 15. Your last direct deposit may have been sent to an outdated account. "
            "Please log in to the HR portal (hr.company.com) using your SSO credentials "
            "and verify your banking details under 'Payment Settings' by end of business "
            "today. If you have not received your paycheck, contact payroll@company.com or "
            "call ext. 2244 immediately."
        ),
        "metadata": {"x_priority": "2", "auto_reply": False, "recipient_count": 1},
        "ground_truth": {
            "category": "hr",
            "priority": "high",
            "is_phishing": False,
            "routing_dept": "hr",
        },
    },
    {
        "id": "e1_10",
        "subject": "October company newsletter — product updates & team spotlights",
        "sender": "Internal Communications",
        "sender_email": "comms@company.com",
        "body": (
            "Welcome to the October edition of our company newsletter! This month: "
            "(1) We shipped v3.2 of our flagship product with 47 improvements — read the "
            "full release notes on the engineering blog. (2) Spotlight on the Customer "
            "Success team's record NPS score of 72. (3) New office hours for the SF HQ. "
            "(4) Upcoming all-hands on November 5 — save the date. Full details inside. "
            "As always, reply with story ideas for next month's edition."
        ),
        "metadata": {"x_priority": "5", "auto_reply": False, "recipient_count": 500},
        "ground_truth": {
            "category": "routine",
            "priority": "low",
            "is_phishing": False,
            "routing_dept": None,
        },
    },
]

# Lookup by id
TASK1_BY_ID: Dict[str, Dict[str, Any]] = {e["id"]: e for e in TASK1_EMAILS}

# ============================================================
# TASK 2 — email_rank  (medium, 1 step)
# Five emails presented in shuffled order (deterministic seed=42).
# Agent must rank by priority AND classify each one.
# ============================================================

TASK2_EMAILS: List[Dict[str, Any]] = [
    {
        "id": "e2_01",
        "subject": "CRITICAL: Primary database cluster experiencing data corruption",
        "sender": "Database Ops",
        "sender_email": "dbops@internal.company.com",
        "body": (
            "Our primary PostgreSQL cluster (prod-db-cluster-1) is reporting checksum "
            "errors on multiple tables. We suspect partial corruption following last "
            "night's failed migration. Reads are succeeding but write operations are "
            "failing for approximately 30% of transactions. Customer data may be at risk. "
            "DBA team is investigating but needs immediate escalation. Backup from 02:00 "
            "UTC is available but rollback would lose 6 hours of transactions."
        ),
        "metadata": {"x_priority": "1", "auto_reply": False, "recipient_count": 8},
        "ground_truth": {
            "category": "urgent",
            "priority": "critical",
            "true_rank": 1,
            "is_phishing": False,
            "routing_dept": "engineering",
        },
    },
    {
        "id": "e2_02",
        "subject": "Security incident: Ransomware indicators detected on workstation fleet",
        "sender": "Security Operations Center",
        "sender_email": "soc@company.com",
        "body": (
            "Our EDR system has flagged ransomware-like behavior on 3 workstations in the "
            "Chicago office. Lateral movement activity detected. We have isolated the "
            "affected machines but the infection vector is unknown. IT security is "
            "conducting a forensic investigation. All Chicago employees should NOT open "
            "any attachments or click any links until further notice. "
            "SOC bridge line: 1-800-555-0199 PIN: 4892."
        ),
        "metadata": {"x_priority": "1", "auto_reply": False, "recipient_count": 25},
        "ground_truth": {
            "category": "security",
            "priority": "high",
            "true_rank": 2,
            "is_phishing": False,
            "routing_dept": "security",
        },
    },
    {
        "id": "e2_03",
        "subject": "Vendor dispute: GlobalTech contesting invoice #GT-8821 ($89,400)",
        "sender": "Accounts Payable",
        "sender_email": "ap@company.com",
        "body": (
            "GlobalTech is disputing invoice #GT-8821 for $89,400, claiming the services "
            "billed do not match the agreed scope of work in SOW v2.1. They are requesting "
            "a detailed breakdown and have threatened to pause our SLA until resolved. "
            "Our finance team needs guidance on whether to escalate to legal or negotiate "
            "directly. The disputed amount represents 15% of our Q4 vendor budget. "
            "Please review the attached correspondence and advise."
        ),
        "metadata": {"x_priority": "2", "auto_reply": False, "recipient_count": 3},
        "ground_truth": {
            "category": "billing",
            "priority": "medium",
            "true_rank": 3,
            "is_phishing": False,
            "routing_dept": "finance",
        },
    },
    {
        "id": "e2_04",
        "subject": "Annual performance review cycle starts November 1 — manager guide enclosed",
        "sender": "HR — People Operations",
        "sender_email": "peopleops@company.com",
        "body": (
            "The annual performance review cycle begins November 1. As a manager, you will "
            "need to: (1) Complete peer review nominations by Oct 28, (2) Submit self-assessment "
            "reminders to your team, (3) Schedule 1:1 calibration meetings with your skip-level. "
            "The full timeline and manager guide are attached. The HR business partner for your "
            "team is available for questions at ext. 3301. Reviews must be finalized by Nov 30."
        ),
        "metadata": {"x_priority": "3", "auto_reply": False, "recipient_count": 62},
        "ground_truth": {
            "category": "hr",
            "priority": "medium",
            "true_rank": 4,
            "is_phishing": False,
            "routing_dept": "hr",
        },
    },
    {
        "id": "e2_05",
        "subject": "Office supplies restock — please submit requests by EOW",
        "sender": "Facilities Management",
        "sender_email": "facilities@company.com",
        "body": (
            "Hi all, we're placing our monthly office supply order this Friday. "
            "If your team needs anything (paper, pens, whiteboards, printer cartridges, "
            "etc.), please submit requests through the facilities portal by Thursday EOD. "
            "Orders placed after the deadline will be processed in next month's cycle. "
            "The portal link is in the company intranet under 'Facilities.' Thanks!"
        ),
        "metadata": {"x_priority": "5", "auto_reply": False, "recipient_count": 200},
        "ground_truth": {
            "category": "routine",
            "priority": "low",
            "true_rank": 5,
            "is_phishing": False,
            "routing_dept": None,
        },
    },
]

# True priority ranking (most → least urgent)
TASK2_TRUE_RANKING: List[str] = ["e2_01", "e2_02", "e2_03", "e2_04", "e2_05"]

# Deterministic presentation order (seed=42 shuffle)
TASK2_PRESENTATION_ORDER: List[str] = ["e2_03", "e2_05", "e2_01", "e2_04", "e2_02"]

TASK2_BY_ID: Dict[str, Dict[str, Any]] = {e["id"]: e for e in TASK2_EMAILS}

# ============================================================
# TASK 3 — email_triage  (hard, 3 steps)
# Three complex emails with mixed/ambiguous signals.
# Step 1: classify + phishing detection
# Step 2: route to department
# Step 3: draft acknowledgment for top-priority non-spam email
# ============================================================

TASK3_EMAILS: List[Dict[str, Any]] = [
    {
        "id": "e3_01",
        "subject": "IT Security: Mandatory VPN certificate update — action required",
        "sender": "IT Helpdesk <it-helpdesk@company-secure.net>",
        "sender_email": "it-helpdesk@company-secure.net",
        "body": (
            "IMPORTANT: Your VPN certificate will expire in 24 hours. To avoid losing "
            "remote access, you must update your certificate immediately. "
            "Please click the link below and enter your Active Directory credentials: "
            "http://vpn-update.company-secure.net/cert-renewal "
            "Note: This link will expire in 2 hours. Do NOT share this link with anyone. "
            "If you have trouble, contact it-support@company-secure.net. "
            "— IT Helpdesk Team"
        ),
        "metadata": {
            "x_priority": "1",
            "auto_reply": False,
            "recipient_count": 1,
            "sender_domain": "company-secure.net",
            "note": "Domain is company-secure.net, NOT company.com — typosquatting indicator",
        },
        "ground_truth": {
            "category": "spam",
            "priority": "low",
            "is_phishing": True,
            "routing_dept": "security",
            "response_target": False,
        },
    },
    {
        "id": "e3_02",
        "subject": "Escalation: Nexus Dynamics contract at risk — $4.2M ARR on the line",
        "sender": "Marcus Webb — Enterprise Account Executive",
        "sender_email": "m.webb@company.com",
        "body": (
            "I need to escalate immediately. Nexus Dynamics ($4.2M ARR, our 3rd largest "
            "account) has been experiencing repeated API outages for the past 72 hours. "
            "Their CTO, Diana Reeves, called me personally this morning and stated they "
            "are 'actively evaluating alternatives.' I have a call with her and their "
            "VP of Engineering at 2 PM today. I need: (1) A root cause analysis from "
            "engineering — what caused the outages and when will they be fully resolved? "
            "(2) Executive sponsorship on the call — can the VP of Customer Success join? "
            "(3) A written remediation plan + SLA credit offer to send before the call. "
            "This is our highest-priority account situation right now."
        ),
        "metadata": {"x_priority": "1", "auto_reply": False, "recipient_count": 5},
        "ground_truth": {
            "category": "urgent",
            "priority": "critical",
            "is_phishing": False,
            "routing_dept": "support",
            "response_target": True,
        },
    },
    {
        "id": "e3_03",
        "subject": "Reminder: EEOC compliance training — complete by October 31",
        "sender": "HR Compliance",
        "sender_email": "hr-compliance@company.com",
        "body": (
            "This is a reminder that all employees are required to complete the annual "
            "EEOC compliance and workplace harassment prevention training by October 31. "
            "The training is available on the learning management system (LMS) at "
            "learn.company.com and takes approximately 45 minutes. Completion is tracked "
            "and managers will be notified of non-compliant team members on November 1. "
            "Contact hr-compliance@company.com with any questions."
        ),
        "metadata": {"x_priority": "3", "auto_reply": False, "recipient_count": 520},
        "ground_truth": {
            "category": "hr",
            "priority": "medium",
            "is_phishing": False,
            "routing_dept": "hr",
            "response_target": False,
        },
    },
]

# The non-spam email with highest priority (target for response draft)
TASK3_RESPONSE_TARGET_ID: str = "e3_02"

TASK3_BY_ID: Dict[str, Dict[str, Any]] = {e["id"]: e for e in TASK3_EMAILS}

# ============================================================
# Valid enumeration values (shared across all tasks)
# ============================================================

VALID_CATEGORIES = {"spam", "urgent", "routine", "billing", "hr", "security"}
VALID_PRIORITIES = {"critical", "high", "medium", "low"}
VALID_DEPARTMENTS = {"engineering", "support", "hr", "finance", "security", "management", "sales"}
