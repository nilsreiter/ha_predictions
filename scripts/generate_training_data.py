#!/usr/bin/env python3
"""Generate realistic test states through the Home Assistant REST API."""

from __future__ import annotations

import argparse
import json
import random
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def call_service(base_url: str, token: str, domain: str, service: str, data: dict) -> None:
    """Call a Home Assistant service."""
    request = Request(
        f"{base_url}/api/services/{domain}/{service}",
        data=json.dumps(data).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=10) as response:  # noqa: S310
        if response.status >= 300:
            raise RuntimeError(f"Home Assistant returned HTTP {response.status}")


def set_boolean(base_url: str, token: str, entity_id: str, value: bool) -> None:
    """Set an input_boolean state."""
    call_service(
        base_url,
        token,
        "input_boolean",
        "turn_on" if value else "turn_off",
        {"entity_id": entity_id},
    )


def set_number(base_url: str, token: str, entity_id: str, value: float) -> None:
    """Set an input_number state."""
    call_service(
        base_url,
        token,
        "input_number",
        "set_value",
        {"entity_id": entity_id, "value": value},
    )


def generate(args: argparse.Namespace) -> None:
    """Generate a varied set of household states."""
    rng = random.Random(args.seed)
    base_url = args.url.rstrip("/")
    target_is_on = False
    set_boolean(base_url, args.token, args.target, False)

    for index in range(args.samples):
        # Walk through several simulated days instead of repeating one pattern.
        hour = (index * 3 + rng.randint(-1, 1)) % 24
        weekday = (index // 24) % 7
        sleeping = hour < 6 or hour >= 23
        at_home = not sleeping and rng.random() > (0.25 if weekday < 5 else 0.10)
        motion = at_home and rng.random() < (0.72 if 7 <= hour < 22 else 0.35)

        # Darkness is higher at night, with weather-like random variation.
        if sleeping:
            lux = rng.randint(0, 12)
        elif 8 <= hour < 18:
            lux = rng.randint(180, 850)
        else:
            lux = rng.randint(15, 220)

        # The target follows a useful pattern, but includes realistic exceptions.
        should_be_on = at_home and (motion or lux < 55) and not sleeping
        if rng.random() < 0.08:
            should_be_on = not should_be_on

        set_number(base_url, args.token, args.hour, hour)
        set_number(base_url, args.token, args.lux, lux)
        set_boolean(base_url, args.token, args.presence, at_home)
        set_boolean(base_url, args.token, args.motion, motion)
        if should_be_on != target_is_on:
            set_boolean(base_url, args.token, args.target, should_be_on)
            target_is_on = should_be_on

        if args.delay:
            time.sleep(args.delay)

        if (index + 1) % 20 == 0 or index + 1 == args.samples:
            print(f"Generated {index + 1}/{args.samples} samples")


def main() -> None:
    """Parse command-line arguments and generate data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", required=True, help="Home Assistant long-lived access token")
    parser.add_argument("--url", default="http://localhost:8123")
    parser.add_argument("--samples", type=int, default=180)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", default="input_boolean.test_prediction_target")
    parser.add_argument("--motion", default="input_boolean.test_motion")
    parser.add_argument("--presence", default="input_boolean.test_presence")
    parser.add_argument("--hour", default="input_number.test_hour")
    parser.add_argument("--lux", default="input_number.test_ambient_lux")
    args = parser.parse_args()

    try:
        generate(args)
    except (HTTPError, URLError) as error:
        raise SystemExit(f"Could not reach Home Assistant: {error}") from error


if __name__ == "__main__":
    main()
