"""Canonical reason-code registry for deterministic risk floor (3B.1)."""

from __future__ import annotations

from enum import StrEnum


class RiskReasonCode(StrEnum):
  """Stable machine-readable codes tied to implemented business triggers."""

  # critical — registry order within severity band
  HEALTH_SYMPTOMS_AFTER_CONSUMPTION = "health_symptoms_after_consumption"
  DANGEROUS_FOREIGN_OBJECT = "dangerous_foreign_object"
  MASS_INCIDENT = "mass_incident"
  FRAUD_INDICATORS = "fraud_indicators"
  DIRECT_THREAT = "direct_threat"

  # high
  DELAY_OVER_120_MINUTES = "delay_over_120_minutes"
  NON_DELIVERY = "non_delivery"
  FALSE_DELIVERY_STATUS = "false_delivery_status"
  PACKAGE_TAMPERING = "package_tampering"
  FOOD_SPOILAGE = "food_spoilage"
  LEGAL_OR_REGULATORY_ESCALATION = "legal_or_regulatory_escalation"
  OFFICIAL_WRITTEN_RESPONSE = "official_written_response"
  PERSONAL_DATA_EXPOSURE = "personal_data_exposure"

  # medium
  DELAY_OVER_30_MINUTES = "delay_over_30_minutes"
  MISSING_ITEM = "missing_item"
  REFUND_REQUEST = "refund_request"
  WRONG_ITEM = "wrong_item"
