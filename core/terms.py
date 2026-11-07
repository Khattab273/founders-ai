"""Deal-breakers in code: currency conversion and the private compatibility check.

The intake model extracts numbers as written ("27k QAR a month"); Python converts them.
compatible() decides whether two people's terms can overlap without any model seeing
either side's private limits. Explanations never reveal the other person's private bound.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from core import storage
from core.schemas import Hire, Profile, Range, Terms, WorkMode

# Gulf currencies are pegged to the US dollar, so these rates are exact (units per 1 USD).
PEGS = {"USD": 1.0, "QAR": 3.64, "SAR": 3.75, "AED": 3.6725}

SEED_TERMS_FILE = storage.REPO_DATA / "seed_terms.json"


# ---------------- what the model extracts (numbers as written) ----------------

def _work_modes(v) -> list[str]:
    """'remote-friendly', 'On-site', 'in office' -> remote / onsite; unknown words are dropped."""
    out = []
    for item in v or []:
        w = str(item).lower()
        mode = ("hybrid" if "hybrid" in w else "remote" if "remote" in w
                else "onsite" if any(k in w for k in ("site", "office", "in person", "in-person")) else None)
        if mode and mode not in out:
            out.append(mode)
    return out


class RawRange(BaseModel):
    low: float | None = None
    high: float | None = None
    currency: str = "USD"
    period: Literal["year", "month"] | None = None     # salaries only
    low_private: bool = False
    high_private: bool = False

    @field_validator("low_private", "high_private", mode="before")
    @classmethod
    def _flag(cls, v):
        return bool(v)

    @field_validator("currency", mode="before")
    @classmethod
    def _cur(cls, v):
        v = str(v or "USD").strip().upper()
        aliases = {"$": "USD", "€": "EUR", "%": "USD", "QR": "QAR", "RIYAL": "QAR", "RIYALS": "QAR", "ريال": "QAR",
                   "SR": "SAR", "DIRHAM": "AED", "DIRHAMS": "AED", "درهم": "AED", "DOLLARS": "USD", "DOLLAR": "USD"}
        return aliases.get(v, v)

    @field_validator("period", mode="before")
    @classmethod
    def _period(cls, v):
        v = str(v or "").lower()
        return "month" if v.startswith("month") else "year" if v.startswith(("year", "annu")) else None


class RawHire(BaseModel):
    kind: Literal["engineer", "operator"] = "engineer"
    title: str = ""

    @model_validator(mode="before")
    @classmethod
    def _tolerate(cls, data):
        """Models write hires in different shapes: {"role": "head of ops", "work": "hybrid"}, a bare string, ..."""
        if isinstance(data, str):
            data = {"title": data}
        if isinstance(data, dict):
            data = dict(data)
            title = str(data.get("title") or data.get("role") or data.get("position") or "")
            data.setdefault("title", title)
            kind = str(data.get("kind") or "").lower()
            if kind not in ("engineer", "operator"):
                ops = ("operat", "ops", "coo", "growth", "sales", "partnership", "marketing", "finance", "head of")
                data["kind"] = "operator" if any(w in (kind + " " + title).lower() for w in ops) else "engineer"
            if isinstance(data.get("work"), str):
                data["work"] = [data["work"]]
        return data
    salary: RawRange | None = None
    equity: RawRange | None = None
    work: list[WorkMode] = Field(default_factory=list)

    _w = field_validator("work", mode="before")(lambda cls, v: _work_modes(v))


class RawTerms(BaseModel):
    valuation: RawRange | None = None
    hires: list[RawHire] = Field(default_factory=list)
    salary: RawRange | None = None
    equity: RawRange | None = None
    work: list[WorkMode] = Field(default_factory=list)
    other: list[str] = Field(default_factory=list)

    _w = field_validator("work", mode="before")(lambda cls, v: _work_modes([v] if isinstance(v, str) else v))

    @field_validator("hires", mode="before")
    @classmethod
    def _hires(cls, v):
        return [v] if isinstance(v, (dict, str)) else (v or [])

    @field_validator("other", mode="before")
    @classmethod
    def _other(cls, v):
        return [str(x) for x in (v or []) if x]


_MULT = {"k": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "thousand": 1e3, "b": 1e9, "bn": 1e9}


def _amount(token: str) -> float | None:
    """'$10M' -> 10000000, '€95k' -> 95000, '0.5%' -> 0.5, '$105,000' -> 105000."""
    import re
    m = re.match(r"[^\d]*([\d,]+(?:\.\d+)?)\s*([a-z%]*)", token.lower())
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    return value * _MULT.get(m.group(2), 1)


def _amount_in(text: str) -> bool:
    """True if the sentence holds a number this module checks in code (so it isn't a free-text must-have)."""
    from core.profiles import amounts_in
    low = text.lower()
    return bool(amounts_in(text)) and any(w in low for w in ("valuation", "cap", "salary", "pay", "equity", "%"))


def mark_private(raw: RawTerms, private_limits: list[str]) -> RawTerms:
    """A bound is private if its number appears in the person's private limits (code, not model judgement)."""
    from core.profiles import amounts_in
    private = {a for limit in private_limits for a in (_amount(t) for t in amounts_in(limit)) if a is not None}
    near = lambda v: v is not None and any(abs(v - x) <= 1e-6 * max(1, x) for x in private)
    ranges = [raw.valuation, raw.salary, raw.equity] + [r for h in raw.hires for r in (h.salary, h.equity)]
    for r in ranges:
        if r is not None:
            r.low_private = r.low_private or near(r.low)
            r.high_private = r.high_private or near(r.high)
    return raw


_RIYAL = ("riyal", "ريال", "qar", "qr", "sar")
_MONTH = ("month", "monthly", "/mo", "شهري", "شهر")


def _mentions(value: float, text: str) -> list[str]:
    """Windows of text around each place the member wrote this number (35000 as 35,000 / 35k / 35 ألف)."""
    import re
    forms = {f"{value:,.0f}", f"{value:.0f}"}
    if value >= 1000 and value % 1000 == 0:
        k = f"{value / 1000:g}"
        forms |= {f"{k}k", f"{k} k", f"{k} ألف", f"{k} الف", f"{k},000", f"{k} thousand"}
    out = []
    for form in forms:
        for m in re.finditer(r"(?<![\d.,])" + re.escape(form) + r"(?![\d])", text, re.I):
            out.append(text[max(0, m.start() - 30): m.end() + 40].lower())
    return out


def fix_units(raw: RawTerms, text: str, location: str = "") -> RawTerms:
    """Correct currency and period from the member's own words; models often default to USD per year."""
    def fix(r: RawRange | None):
        if r is None:
            return
        for v in (r.low, r.high):
            if v is None:
                continue
            for w in _mentions(v, text):
                if r.currency == "USD" and "$" not in w and "dollar" not in w and "usd" not in w:
                    if any(x in w for x in _RIYAL):
                        r.currency = "SAR" if ("sar" in w or "saudi" in location.lower()) else "QAR"
                    elif "dirham" in w or "درهم" in w or "aed" in w:
                        r.currency = "AED"
                    elif "€" in w or "eur" in w:
                        r.currency = "EUR"
                # Monthly only if their own words say so; otherwise a salary is per year (models guess "month" wrongly).
                r.period = "month" if any(x in w for x in _MONTH) else None
    for r in (raw.salary, *(h.salary for h in raw.hires)):
        fix(r)
    return raw


def to_range(raw: RawRange | None, money: bool = True) -> Range | None:
    """Convert pegged currencies to USD and monthly salaries to annual. Percentages pass through."""
    if raw is None or (raw.low is None and raw.high is None):
        return None
    cur = (raw.currency or "USD").upper().strip()
    factor = 1.0
    if money:
        if cur in PEGS:
            factor, cur = 1 / PEGS[cur], "USD"
        if raw.period == "month":
            factor *= 12
    else:
        cur = "%"
    conv = lambda v: None if v is None else round(v * factor, 2)
    return Range(low=conv(raw.low), high=conv(raw.high), currency=cur,
                 low_private=raw.low_private, high_private=raw.high_private)


def to_terms(raw: RawTerms) -> Terms:
    return Terms(
        valuation=to_range(raw.valuation),
        hires=[Hire(kind=h.kind, title=h.title, salary=to_range(h.salary), equity=to_range(h.equity, money=False),
                    work=h.work) for h in raw.hires],
        salary=to_range(raw.salary), equity=to_range(raw.equity, money=False),
        work=raw.work, other=[o.strip() for o in raw.other if o.strip()],
    )


# ---------------- seed terms ----------------

@lru_cache(maxsize=1)
def _seed() -> dict[str, dict]:
    if not SEED_TERMS_FILE.exists():
        return {}
    return json.loads(SEED_TERMS_FILE.read_text(encoding="utf-8"))


def terms_for(p: Profile) -> Terms | None:
    if p.terms is not None:
        return p.terms
    entry = _seed().get(p.id)
    return Terms.model_validate(entry["terms"]) if entry else None


def offers_for(p: Profile) -> str:
    return p.offers or _seed().get(p.id, {}).get("offers", "")


# ---------------- compatibility ----------------

@dataclass
class Conflict:
    field: Literal["valuation", "salary", "equity", "work"]
    theirs_private: bool     # the candidate's side of the clash is private: never explain it
    detail: str              # explanation for the viewer, safe to show only if not theirs_private


def _money(v: float, cur: str) -> str:
    sym = {"USD": "$", "EUR": "€"}.get(cur, cur + " ")
    if cur == "%":
        return f"{v:g}%"
    return f"{sym}{v / 1e6:g}M" if v >= 1e6 else f"{sym}{v / 1e3:g}k"


def _clash(need: Range | None, offer: Range | None) -> bool:
    """need.low (what one side requires at minimum) above offer.high (the most the other side gives)."""
    return bool(need and offer and need.low is not None and offer.high is not None
                and need.currency == offer.currency and need.low > offer.high)


def _work_clash(a: list[str], b: list[str]) -> bool:
    return bool(a and b and not set(a) & set(b))


def _modes(modes: list[str]) -> str:
    names = {"remote": "remote", "hybrid": "hybrid", "onsite": "on-site"}
    return " or ".join(names[m] for m in modes)


def _hire_conflicts(talent: Terms, hire: Hire, viewer_is_talent: bool, name: str) -> list[Conflict]:
    out = []
    for field, need, offer in (("salary", talent.salary, hire.salary), ("equity", talent.equity, hire.equity)):
        if _clash(need, offer):
            theirs_private = offer.high_private if viewer_is_talent else need.low_private
            detail = (f"{name}'s role offers up to {_money(offer.high, offer.currency)} {field}, below your minimum."
                      if viewer_is_talent else
                      f"{name} needs at least {_money(need.low, need.currency)} {field}, above your maximum.")
            out.append(Conflict(field, theirs_private, detail))
    if _work_clash(talent.work, hire.work):
        # Work mode is usually public for roles; a talent's "remote only" may be private.
        detail = (f"{name}'s role is {_modes(hire.work)}, and you want {_modes(talent.work)}." if viewer_is_talent
                  else f"{name} only wants {_modes(talent.work)} work, and your role is {_modes(hire.work)}.")
        out.append(Conflict("work", not viewer_is_talent, detail))
    return out


def conflicts(viewer: Profile, cand: Profile) -> list[Conflict]:
    """Why these two can't work, judged only on terms both sides set. Empty list means compatible."""
    a, b = terms_for(viewer), terms_for(cand)
    if a is None or b is None:
        return []
    name = cand.display_name.split()[0]
    roles = (viewer.role, cand.role)
    if set(roles) == {"founder", "investor"}:
        f, i = (a, b) if viewer.role == "founder" else (b, a)
        if _clash(f.valuation, i.valuation):
            viewer_is_founder = viewer.role == "founder"
            theirs_private = i.valuation.high_private if viewer_is_founder else f.valuation.low_private
            detail = (f"{name} invests up to {_money(i.valuation.high, i.valuation.currency)} valuation, "
                      f"below your floor." if viewer_is_founder else
                      f"{name} won't go below {_money(f.valuation.low, f.valuation.currency)} valuation, "
                      f"above your ceiling.")
            return [Conflict("valuation", theirs_private, detail)]
        return []
    if "founder" in roles and roles != ("founder", "founder"):
        viewer_is_talent = viewer.role != "founder"
        talent_p, founder_t, talent_t = (viewer, b, a) if viewer_is_talent else (cand, a, b)
        hires = [h for h in founder_t.hires if h.kind == talent_p.role]
        if not hires:
            return []
        per_hire = [_hire_conflicts(talent_t, h, viewer_is_talent, name) for h in hires]
        return [] if any(not c for c in per_hire) else per_hire[0]
    return []


PRIVATE_REASON = "Not a match based on their preferences."


def explain(c: Conflict) -> str:
    """What the viewer may be told about a removed candidate."""
    return PRIVATE_REASON if c.theirs_private else c.detail
