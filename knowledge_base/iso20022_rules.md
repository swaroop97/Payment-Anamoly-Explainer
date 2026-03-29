# ISO 20022 & Payment Compliance Rules (Knowledge Base)

## Rule 001: High-Value Transaction Reporting Threshold

Single or aggregate cash or wire transactions at or above USD 10,000 (or local equivalent) may trigger Currency Transaction Report (CTR) or equivalent reporting. Institutions must monitor amounts approaching the threshold within rolling windows to detect structuring.

## Rule 002: Off-Hours Transaction Monitoring

High-value or sensitive payment rails initiated outside normal business hours (e.g., late night or early morning local time) warrant enhanced review, as they correlate with fraud and money laundering typologies.

## Rule 003: Structuring and Threshold Avoidance

Multiple transactions deliberately kept below reporting thresholds (e.g., several wires just under USD 10,000) may indicate structuring. Monitor patterns across accounts, counterparties, and short time windows.

## Rule 004: Velocity and Burst Activity

A sudden spike in transaction count or volume compared to a customer’s baseline—especially across related BICs or accounts—may signal account takeover, mule activity, or layering.

## Rule 005: Sanctions and Restricted Counterparty Screening

Payments involving sanctioned jurisdictions, blocked entities, or close matches on SWIFT/BIC directories require screening holds and escalation per OFAC and internal policy.

## Rule 006: Round-Trip and Circular Flow Detection

Funds that move out and return through related parties or corridors without clear economic purpose may indicate layering or balance manipulation and should be investigated.

## Rule 007: Payment Rail Consistency

Mismatch between stated purpose, amount, and rail (e.g., urgent high-value RTP followed by international SWIFT splits) may indicate concealment or misuse of rails.

## Rule 008: Beneficiary and Originator Data Quality

Incomplete or inconsistent ISO 20022 party identifiers (BIC, LEI, name/address) reduce traceability and must be remediated before settlement when policy requires.

## Rule 009: Dormant Account Reactivation

Large outbound wires from previously dormant accounts are high risk and require stepped-up authentication and relationship review.

## Rule 010: Regulatory Record Retention

Evidence of monitoring decisions, alerts, and disposition must be retained for the period required by applicable law (often five years or longer) and be auditable.
