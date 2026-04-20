"""Interactive student-oriented client for guided debug submissions."""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

import requests


STAT_ORDER = {"max": 0, "avg": 1, "rms": 2, "min": 3, "pp": 4}
COMMON_LAB_STATS = ("max", "avg", "rms", "min", "pp")
def parse_args() -> argparse.Namespace:
    """Parse CLI flags for the interactive student workflow."""
    p = argparse.ArgumentParser(
        description=(
            "Interactive student client for Circuit Debug API. "
            "Fetches required nodes for a circuit and prompts for measurements one at a time."
        )
    )
    p.add_argument("--base-url", default="http://127.0.0.1:8001")
    p.add_argument("--circuit", default=None, help="Optional exact circuit name to skip circuit selection.")
    p.add_argument(
        "--ask-source-currents",
        action="store_true",
        help="Also prompt for optional source currents one at a time.",
    )
    p.add_argument(
        "--show-golden",
        action="store_true",
        help="Show golden values in prompts (useful for instructor demo, not student use).",
    )
    p.add_argument(
        "--no-strict",
        action="store_false",
        dest="strict",
        default=True,
        help="Submit with strict=false (not recommended; allows missing nodes).",
    )
    p.add_argument(
        "--save-payload",
        default=None,
        help="Optional path to save the submitted payload JSON.",
    )
    p.add_argument(
        "--debug-timeout",
        type=int,
        default=900,
        help="Timeout in seconds for the POST /debug request. Increase this for first-load model warmup.",
    )
    return p.parse_args()


def pretty(obj: Any) -> str:
    """Pretty-print JSON payloads for terminal output."""
    return json.dumps(obj, indent=2)


def fetch_json(url: str, timeout: int = 30) -> dict[str, Any]:
    """Fetch a JSON object and validate that the response shape is a mapping."""
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object from {url}")
    return data


def choose_circuit(base: str, preselected: str | None) -> tuple[str, list[str]]:
    """Let the user choose a circuit, grouped by lab, unless one is preselected."""
    circuits_doc = fetch_json(f"{base}/circuits")
    circuits = circuits_doc.get("circuits", [])
    if not isinstance(circuits, list) or not circuits:
        raise RuntimeError("No circuits returned by API")

    if preselected:
        if preselected not in circuits:
            raise RuntimeError(f"Unknown circuit '{preselected}'. Use /circuits to list valid names.")
        return preselected, circuits

    def lab_key(name: str) -> str:
        """Group circuit names by their leading lab prefix for selection menus."""
        m = re.match(r"(?i)^(lab\d+)", name.strip())
        if m:
            return f"Lab{m.group(1)[3:]}"
        return "Other"

    def lab_sort_key(lab: str) -> tuple[int, str]:
        """Sort numbered labs first in numeric order, then any other buckets."""
        m = re.match(r"^Lab(\d+)$", lab)
        if m:
            return (0, f"{int(m.group(1)):04d}")
        return (1, lab.lower())

    by_lab: dict[str, list[str]] = {}
    for name in circuits:
        by_lab.setdefault(lab_key(str(name)), []).append(str(name))
    for lab in by_lab:
        by_lab[lab].sort(key=lambda s: s.lower())

    labs = sorted(by_lab.keys(), key=lab_sort_key)

    print("Available labs:")
    for i, lab in enumerate(labs, start=1):
        print(f"  {i:>3}. {lab} ({len(by_lab[lab])} circuits)")

    selected_lab: str | None = None
    while True:
        raw = input("\nChoose a lab by number or name (example: 4 or Lab4): ").strip()
        if not raw:
            continue
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(labs):
                selected_lab = labs[idx - 1]
                break
            # Also accept bare lab numbers like "4" -> "Lab4"
            lab_name = f"Lab{idx}"
            if lab_name in by_lab:
                selected_lab = lab_name
                break
            print(f"Invalid lab selection. Enter 1-{len(labs)} or a valid lab name.")
            continue
        normalized = raw.strip()
        if re.fullmatch(r"(?i)lab\d+", normalized):
            normalized = f"Lab{int(re.sub(r'(?i)^lab', '', normalized))}"
        elif re.fullmatch(r"\d+", normalized):
            normalized = f"Lab{int(normalized)}"
        if normalized in by_lab:
            selected_lab = normalized
            break
        print("Invalid lab name. Try again.")

    assert selected_lab is not None
    lab_circuits = by_lab[selected_lab]
    print(f"\nCircuits in {selected_lab}:")
    for i, name in enumerate(lab_circuits, start=1):
        print(f"  {i:>3}. {name}")

    while True:
        raw = input("\nChoose a circuit by number or exact name: ").strip()
        if not raw:
            continue
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(lab_circuits):
                return lab_circuits[idx - 1], circuits
            print(f"Invalid number. Enter 1-{len(lab_circuits)}.")
            continue
        if raw in lab_circuits:
            return raw, circuits
        print(f"Invalid circuit name for {selected_lab}. Try again.")


def _parse_float_input(raw: str) -> float:
    """Parse a single numeric terminal input."""
    return float(raw.strip())


def _sorted_stats(item: dict[str, Any]) -> list[str]:
    """Return the stats for one node/source in a stable prompt order."""
    stats = item.get("student_available_stats")
    if not isinstance(stats, list) or not stats:
        stats = item.get("available_stats", [])
    cleaned = [str(stat).strip().lower() for stat in stats if str(stat).strip()] if isinstance(stats, list) else []
    if not cleaned:
        cleaned = list(COMMON_LAB_STATS)
    return sorted(set(cleaned), key=lambda stat: (STAT_ORDER.get(stat, 99), stat))


def _pick_legacy_max_value(stat_map: dict[str, float]) -> float | None:
    """Return the legacy compatibility value only when an actual max stat exists."""
    if not isinstance(stat_map, dict):
        return None
    value = stat_map.get("max")
    return float(value) if isinstance(value, (int, float)) else None


def prompt_stat_measurements(
    label: str,
    items: list[dict[str, Any]],
    key_name: str,
    *,
    show_golden: bool,
    allow_empty_item: bool,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Prompt for flexible per-item stat subsets and return compatibility plus nested maps."""
    values_by_name: dict[str, dict[str, float]] = {}
    if not items:
        return {}, {}

    print(f"\nEnter {label}.")
    print(
        "Per stat: type a number to record it, press Enter to skip that stat, "
        "'done' to move to the next item, 'back' to revisit the previous stat, or 'quit' to exit."
    )
    if allow_empty_item:
        print("You may leave an entire item blank if you did not measure it, or type 'skip' to skip it immediately.")
    else:
        print("At least one stat is required for each listed node.")

    total_items = len(items)
    for item_index, item in enumerate(items, start=1):
        name = str(item.get(key_name))
        stats = _sorted_stats(item)
        golden_values = item.get("golden_values", {})
        if not isinstance(golden_values, dict):
            golden_values = {}

        print(f"\n[{item_index}/{total_items}] {name}")
        print(f"Available stats: {', '.join(stats)}")
        item_values = dict(values_by_name.get(name, {}))
        stat_index = 0
        while stat_index < len(stats):
            stat = stats[stat_index]
            prompt = f"  {name} {stat}"
            if show_golden and golden_values.get(stat) is not None:
                prompt += f" (golden {golden_values[stat]})"
            existing = item_values.get(stat)
            if existing is not None:
                prompt += f" [current {existing}]"
            prompt += ": "

            raw = input(prompt).strip()
            lower = raw.lower()
            if lower in {"quit", "exit", "q"}:
                raise KeyboardInterrupt()
            if lower == "back":
                if stat_index == 0:
                    print("Already at the first stat for this item.")
                    continue
                prev_stat = stats[stat_index - 1]
                item_values.pop(prev_stat, None)
                stat_index -= 1
                continue
            if lower in {"done", "next"}:
                break
            if lower == "skip":
                if allow_empty_item:
                    item_values = {}
                    break
                print("Skip is only allowed for optional items.")
                continue
            if raw == "":
                stat_index += 1
                continue
            try:
                item_values[stat] = _parse_float_input(raw)
                stat_index += 1
            except ValueError:
                print("Invalid number. Enter a numeric value in volts/amps (examples: 5, -0.23, 1.2e-3).")

        if item_values:
            values_by_name[name] = item_values
            continue
        if allow_empty_item:
            values_by_name.pop(name, None)
            continue
        print(f"At least one measurement is required for {name} while strict mode is enabled.")
        while not item_values:
            raw = input(f"Enter one value for {name} using '<stat> <value>' or type 'quit': ").strip()
            lower = raw.lower()
            if lower in {"quit", "exit", "q"}:
                raise KeyboardInterrupt()
            if " " not in raw:
                print(f"Please use one of: {', '.join(stats)} followed by a numeric value.")
                continue
            stat_name, raw_value = raw.split(None, 1)
            stat_name = stat_name.strip().lower()
            if stat_name not in stats:
                print(f"Unknown stat '{stat_name}'. Choose from: {', '.join(stats)}")
                continue
            try:
                item_values[stat_name] = _parse_float_input(raw_value)
            except ValueError:
                print("Invalid number. Enter a numeric value in volts/amps.")
        values_by_name[name] = item_values

    compatibility_values = {
        name: picked
        for name, stat_map in values_by_name.items()
        if (picked := _pick_legacy_max_value(stat_map)) is not None
    }
    return compatibility_values, values_by_name


def print_node_checklist(nodes_doc: dict[str, Any]) -> None:
    """Display the circuit's required node and optional source-current checklist."""
    circuit_name = nodes_doc.get("circuit_name", "<unknown>")
    print(f"\nSelected circuit: {circuit_name}")
    print(f"Required nodes: {nodes_doc.get('node_count', 0)}")
    print(f"Optional source currents: {nodes_doc.get('source_current_count', 0)}")
    print("You can enter any subset of the listed stats for each node/source.")

    nodes = nodes_doc.get("nodes", [])
    if nodes:
        print("\nNode list:")
        for item in nodes:
            stats = ", ".join(_sorted_stats(item))
            print(f"  - {item.get('node_name')} [{stats}]")

    srcs = nodes_doc.get("source_currents", [])
    if srcs:
        print("\nSource current list (optional):")
        for item in srcs:
            stats = ", ".join(_sorted_stats(item))
            print(f"  - {item.get('source_name')} [{stats}]")


def main() -> int:
    """Drive the end-to-end interactive student submission flow."""
    args = parse_args()
    base = args.base_url.rstrip("/")

    try:
        health = fetch_json(f"{base}/health")
        print("API health:")
        print(pretty(health))

        circuit_name, _ = choose_circuit(base, args.circuit)
        nodes_doc = fetch_json(f"{base}/circuits/{circuit_name}/nodes")
        print_node_checklist(nodes_doc)

        nodes = nodes_doc.get("nodes", [])
        srcs = nodes_doc.get("source_currents", [])

        node_voltages, node_measurements = prompt_stat_measurements(
            "node voltage stats (V, measured relative to ground)",
            nodes if isinstance(nodes, list) else [],
            "node_name",
            show_golden=args.show_golden,
            allow_empty_item=not bool(args.strict),
        )

        source_currents: dict[str, float] = {}
        source_current_measurements: dict[str, dict[str, float]] = {}
        if args.ask_source_currents and isinstance(srcs, list) and srcs:
            print("\nSource currents are optional. You can type 'skip' for any source you did not measure.")
            source_currents, source_current_measurements = prompt_stat_measurements(
                "source current stats (A)",
                srcs,
                "source_name",
                show_golden=args.show_golden,
                allow_empty_item=True,
            )
        elif isinstance(srcs, list) and srcs:
            yn = input("\nDo you want to enter source currents too? [y/N]: ").strip().lower()
            if yn in {"y", "yes"}:
                print("You can type 'skip' for any source you did not measure.")
                source_currents, source_current_measurements = prompt_stat_measurements(
                    "source current stats (A)",
                    srcs,
                    "source_name",
                    show_golden=args.show_golden,
                    allow_empty_item=True,
                )

        defaults = nodes_doc.get("golden_defaults", {}) if isinstance(nodes_doc.get("golden_defaults"), dict) else {}
        payload: dict[str, Any] = {
            "circuit_name": circuit_name,
            "node_voltages": node_voltages,
            "node_measurements": node_measurements,
            "source_currents": source_currents,
            "source_current_measurements": source_current_measurements,
            "temp": float(defaults.get("temp", 27.0)),
            "tnom": float(defaults.get("tnom", 27.0)),
            "strict": bool(args.strict),
        }

        print("\nPayload to submit:")
        print(pretty(payload))
        submit = input("\nSubmit to /debug now? [Y/n]: ").strip().lower()
        if submit in {"n", "no"}:
            print("Submission cancelled.")
            return 0

        if args.save_payload:
            with open(args.save_payload, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Saved payload: {args.save_payload}")

        r = requests.post(f"{base}/debug", json=payload, timeout=args.debug_timeout)
        r.raise_for_status()
        result = r.json()
        print("\nAPI response:")
        print(pretty(result))
        return 0

    except KeyboardInterrupt:
        print("\nCancelled by user.")
        return 130
    except requests.HTTPError as e:
        detail = None
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text if e.response is not None else str(e)
        print("\nAPI error:")
        print(pretty(detail) if isinstance(detail, dict) else str(detail))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
