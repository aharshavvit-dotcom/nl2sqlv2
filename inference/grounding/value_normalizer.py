from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any


class ValueNormalizer:
    @staticmethod
    def normalize(raw_text: str, value_type: str, time_context: Any | None = None) -> Any:
        raw_clean = raw_text.strip()
        if value_type == "null":
            return None
        if value_type == "boolean":
            lower = raw_clean.lower()
            if lower in ("true", "1", "yes", "approved", "active", "on"):
                return True
            return False

        if value_type == "percentage":
            match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", raw_clean)
            if match:
                return float(match.group(1)) / 100.0
            cleaned = raw_clean.replace("%", "").strip()
            try:
                return float(cleaned) / 100.0
            except ValueError:
                return raw_clean

        if value_type == "currency":
            cleaned = re.sub(r"[^\d\.\s\w-]", "", raw_clean)
            multiplier = 1.0
            lower = cleaned.lower()
            if "lakh" in lower:
                multiplier = 100000.0
                cleaned = re.sub(r"lakh", "", lower).strip()
            elif "million" in lower:
                multiplier = 1000000.0
                cleaned = re.sub(r"million", "", lower).strip()
            elif "crore" in lower:
                multiplier = 10000000.0
                cleaned = re.sub(r"crore", "", lower).strip()
            
            match = re.search(r"(-?\d+(?:\.\d+)?)", cleaned)
            if match:
                try:
                    return float(match.group(1)) * multiplier
                except ValueError:
                    pass
            return raw_clean

        if value_type in ("integer", "decimal", "numeric_value"):
            cleaned = raw_clean.replace(",", "")
            try:
                if value_type == "integer":
                    return int(cleaned)
                return float(cleaned)
            except ValueError:
                match = re.search(r"(-?\d+(?:\.\d+)?)", cleaned)
                if match:
                    try:
                        return int(match.group(1)) if value_type == "integer" else float(match.group(1))
                    except ValueError:
                        pass
                return raw_clean

        if value_type == "list":
            items = re.split(r",\s*(?:or|and)?\s*|\s+or\s+|\s+and\s+", raw_clean)
            return [it.strip() for it in items if it.strip()]

        if value_type == "range":
            matches = re.findall(r"(-?\d+(?:\.\d+)?)", raw_clean)
            if len(matches) >= 2:
                try:
                    return [float(matches[0]), float(matches[1])]
                except ValueError:
                    pass
            return raw_clean

        if value_type in ("date", "datetime", "year", "quarter", "month", "relative date"):
            if time_context and hasattr(time_context, "current_datetime"):
                ref_time = time_context.current_datetime
            else:
                ref_time = datetime.now()

            lower = raw_clean.lower()
            if "yesterday" in lower:
                return (ref_time - timedelta(days=1)).strftime("%Y-%m-%d")
            if "today" in lower:
                return ref_time.strftime("%Y-%m-%d")
            if "last month" in lower:
                year = ref_time.year
                month = ref_time.month - 1
                if month == 0:
                    month = 12
                    year -= 1
                return f"{year}-{month:02d}"
            if "past 30 days" in lower:
                start = (ref_time - timedelta(days=30)).strftime("%Y-%m-%d")
                end = ref_time.strftime("%Y-%m-%d")
                return [start, end]

            match_q = re.search(r"\bq([1-4])\s+(\d{4})\b", lower)
            if match_q:
                q = int(match_q.group(1))
                y = int(match_q.group(2))
                return f"{y}-Q{q}"

            return raw_clean

        return raw_clean
