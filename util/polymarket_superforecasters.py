import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple


USER_KEYS = ("user_id", "user", "trader", "account", "wallet", "address")
TIME_KEYS = ("timestamp", "ts", "created_at", "action_time", "time")
EVENT_KEYS = ("event_id", "market_id", "question_id", "event")
SIDE_KEYS = ("side", "position", "bet_side", "outcome_side")
PROB_KEYS = ("entry_probability", "market_probability", "probability", "price", "implied_prob")
OUTCOME_KEYS = ("resolved_outcome", "outcome", "resolved", "winner")


@dataclass
class Action:
    user_id: str
    timestamp: datetime
    event_id: str
    probability_yes: float
    side_yes: bool
    outcome_yes: Optional[bool]


@dataclass
class UserScore:
    user_id: str
    actions: int
    active_weeks: int
    avg_actions_per_week: float
    max_actions_per_day: int
    median_gap_seconds: float
    avg_edge: float
    total_edge: float


def _pick(record: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[Any]:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _parse_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    text = str(value).strip().lower()
    if text in {"1", "yes", "y", "true", "t", "up"}:
        return True
    if text in {"0", "no", "n", "false", "f", "down"}:
        return False
    return None


def _parse_probability(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        p = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(p):
        return None
    if p > 1:
        p = p / 100.0
    return max(0.0, min(1.0, p))


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    for parser in (
        lambda t: datetime.fromisoformat(t),
        lambda t: datetime.strptime(t, "%Y-%m-%d %H:%M:%S"),
        lambda t: datetime.strptime(t, "%Y-%m-%d"),
    ):
        try:
            parsed = parser(text)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def normalize_action(record: Dict[str, Any]) -> Optional[Action]:
    user = _pick(record, USER_KEYS)
    ts = _parse_timestamp(_pick(record, TIME_KEYS))
    event = _pick(record, EVENT_KEYS)
    prob_yes = _parse_probability(_pick(record, PROB_KEYS))
    if user is None or ts is None or event is None or prob_yes is None:
        return None
    side = _parse_bool(_pick(record, SIDE_KEYS))
    if side is None:
        side = True
    outcome = _parse_bool(_pick(record, OUTCOME_KEYS))
    return Action(
        user_id=str(user),
        timestamp=ts,
        event_id=str(event),
        probability_yes=prob_yes,
        side_yes=side,
        outcome_yes=outcome,
    )


def _week_start(ts: datetime) -> datetime:
    return (ts - timedelta(days=ts.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)


def _coverage_weeks(actions: List[Action]) -> Tuple[int, int]:
    weeks = sorted({_week_start(a.timestamp) for a in actions})
    if not weeks:
        return 0, 0
    current = weeks[0]
    end = weeks[-1]
    expected = 0
    existing = set(weeks)
    covered = 0
    while current <= end:
        expected += 1
        if current in existing:
            covered += 1
        current += timedelta(days=7)
    return covered, expected


def _daily_counts(actions: List[Action]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for action in actions:
        counts[action.timestamp.strftime("%Y-%m-%d")] += 1
    return counts


def _weekly_counts(actions: List[Action]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for action in actions:
        year, week, _ = action.timestamp.isocalendar()
        counts[f"{year}-{week:02d}"] += 1
    return counts


def _median_gap_seconds(actions: List[Action]) -> float:
    if len(actions) <= 1:
        return float("inf")
    ordered = sorted(actions, key=lambda a: a.timestamp)
    gaps = []
    for i in range(1, len(ordered)):
        gaps.append((ordered[i].timestamp - ordered[i - 1].timestamp).total_seconds())
    return float(median(gaps))


def _edge(action: Action) -> Optional[float]:
    if action.outcome_yes is None:
        return None
    if action.side_yes:
        realized = 1.0 if action.outcome_yes else 0.0
        entry = action.probability_yes
    else:
        realized = 0.0 if action.outcome_yes else 1.0
        entry = 1.0 - action.probability_yes
    return realized - entry


def find_superforecasters(
    records: Iterable[Dict[str, Any]],
    min_expected_weeks: int = 4,
    max_actions_per_day: int = 40,
    max_actions_per_week: int = 150,
    max_avg_actions_per_week: float = 60.0,
    min_median_gap_seconds: float = 10.0,
    min_resolved_actions: int = 8,
    top_k: int = 25,
) -> List[UserScore]:
    by_user: Dict[str, List[Action]] = defaultdict(list)
    for record in records:
        action = normalize_action(record)
        if action is not None:
            by_user[action.user_id].append(action)

    scored: List[UserScore] = []
    for user_id, actions in by_user.items():
        covered, expected = _coverage_weeks(actions)
        if expected < min_expected_weeks or covered < expected:
            continue

        daily_counts = _daily_counts(actions)
        weekly_counts = _weekly_counts(actions)
        avg_weekly = len(actions) / max(1, expected)
        max_daily = max(daily_counts.values())
        max_weekly = max(weekly_counts.values())
        median_gap = _median_gap_seconds(actions)

        if max_daily > max_actions_per_day:
            continue
        if max_weekly > max_actions_per_week:
            continue
        if avg_weekly > max_avg_actions_per_week:
            continue
        if median_gap < min_median_gap_seconds:
            continue

        edges = [value for value in (_edge(a) for a in actions) if value is not None]
        if len(edges) < min_resolved_actions:
            continue
        avg_edge = sum(edges) / len(edges)
        total_edge = sum(edges)
        scored.append(
            UserScore(
                user_id=user_id,
                actions=len(actions),
                active_weeks=covered,
                avg_actions_per_week=avg_weekly,
                max_actions_per_day=max_daily,
                median_gap_seconds=median_gap,
                avg_edge=avg_edge,
                total_edge=total_edge,
            )
        )

    scored.sort(key=lambda s: (s.avg_edge, s.total_edge, -s.avg_actions_per_week), reverse=True)
    return scored[:top_k]


def _load_records(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            values = data.get("data")
            if isinstance(values, list):
                return values
        raise ValueError("Unsupported JSON structure; expected list or object with 'data' list.")
    if path.endswith(".csv"):
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    raise ValueError("Unsupported file type. Use .csv, .json, or .jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Find Polymarket superforecaster-like users.")
    parser.add_argument("--input", required=True, help="Path to .csv/.json/.jsonl actions file.")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--min-expected-weeks", type=int, default=4)
    parser.add_argument("--max-actions-per-day", type=int, default=40)
    parser.add_argument("--max-actions-per-week", type=int, default=150)
    parser.add_argument("--max-avg-actions-per-week", type=float, default=60.0)
    parser.add_argument("--min-median-gap-seconds", type=float, default=10.0)
    parser.add_argument("--min-resolved-actions", type=int, default=8)
    args = parser.parse_args()

    records = _load_records(args.input)
    winners = find_superforecasters(
        records=records,
        min_expected_weeks=args.min_expected_weeks,
        max_actions_per_day=args.max_actions_per_day,
        max_actions_per_week=args.max_actions_per_week,
        max_avg_actions_per_week=args.max_avg_actions_per_week,
        min_median_gap_seconds=args.min_median_gap_seconds,
        min_resolved_actions=args.min_resolved_actions,
        top_k=args.top_k,
    )

    print(json.dumps([score.__dict__ for score in winners], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
