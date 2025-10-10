# utils/date_utils.py
from datetime import date, timedelta
from typing import List, Dict

def generate_weeks(resumption_date: date, duration_weeks: int) -> List[Dict[str, date]]:
    """
    Generate week ranges starting from `resumption_date`.

    Returns a list of dicts:
        [{"start_date": date1, "end_date": date2}, ...] with length == duration_weeks

    Each week is Monday → Friday (5 school days). Adjust `resumption_date` to Monday if needed.
    """
    weeks = []

    # Adjust resumption date to the nearest Monday if it's not already
    start = resumption_date
    if start.weekday() != 0:  # 0 = Monday
        start -= timedelta(days=start.weekday())

    for _ in range(duration_weeks):
        week_start = start
        week_end = week_start + timedelta(days=4)  # Monday → Friday
        weeks.append({"start_date": week_start, "end_date": week_end})
        start += timedelta(days=7)  # move to next week

    return weeks
