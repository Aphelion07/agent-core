"""The tool set every benchmark task is run against.

Every task gets the full registry, including ``get_weather`` - which no task
ever actually needs. It exists purely as a distractor: calling it costs a
step and signals the strategy reached for a tool without checking whether it
was relevant.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from agent_core.example_tools import calculator_tool
from agent_core.tools import Tool, ToolRegistry

_CATALOG: dict[str, float] = {
    "widget-42": 19.99,
    "widget-7": 4.50,
    "gadget-1": 129.00,
}

_LENGTH_MASS_FACTORS: dict[tuple[str, str], float] = {
    ("km", "mi"): 0.621371,
    ("mi", "km"): 1.60934,
    ("kg", "lb"): 2.20462,
    ("lb", "kg"): 0.453592,
}


class CatalogLookupArgs(BaseModel):
    sku: str


def _catalog_lookup(args: CatalogLookupArgs) -> str:
    price = _CATALOG.get(args.sku)
    if price is None:
        return f"no such SKU in the catalog: {args.sku}"
    return f"{price:.2f} USD"


def catalog_tool() -> Tool[CatalogLookupArgs]:
    return Tool(
        name="catalog_lookup",
        description="Look up the USD price of a product by its SKU.",
        parameters=CatalogLookupArgs,
        func=_catalog_lookup,
    )


class UnitConvertArgs(BaseModel):
    value: float
    from_unit: str
    to_unit: str


def _unit_convert(args: UnitConvertArgs) -> str:
    if args.from_unit == args.to_unit:
        return f"{args.value:.2f}"
    if args.from_unit == "c" and args.to_unit == "f":
        return f"{args.value * 9 / 5 + 32:.2f}"
    if args.from_unit == "f" and args.to_unit == "c":
        return f"{(args.value - 32) * 5 / 9:.2f}"
    factor = _LENGTH_MASS_FACTORS.get((args.from_unit, args.to_unit))
    if factor is None:
        return f"unsupported conversion: {args.from_unit} -> {args.to_unit}"
    return f"{args.value * factor:.2f}"


def unit_convert_tool() -> Tool[UnitConvertArgs]:
    return Tool(
        name="unit_convert",
        description="Convert a numeric value between units. Supports km<->mi, kg<->lb, c<->f.",
        parameters=UnitConvertArgs,
        func=_unit_convert,
    )


class DateDiffArgs(BaseModel):
    start: str
    end: str


def _date_diff(args: DateDiffArgs) -> str:
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    return str((end - start).days)


def date_diff_tool() -> Tool[DateDiffArgs]:
    return Tool(
        name="date_diff",
        description="Number of days between two ISO dates (YYYY-MM-DD), end minus start.",
        parameters=DateDiffArgs,
        func=_date_diff,
    )


class WeatherArgs(BaseModel):
    city: str


def _weather(args: WeatherArgs) -> str:
    return f"{args.city}: 18C, partly cloudy"


def weather_tool() -> Tool[WeatherArgs]:
    return Tool(
        name="get_weather",
        description="Get the current weather for a city.",
        parameters=WeatherArgs,
        func=_weather,
    )


def build_registry() -> ToolRegistry:
    return ToolRegistry(
        [calculator_tool(), catalog_tool(), unit_convert_tool(), date_diff_tool(), weather_tool()]
    )
