import ast
import asyncio
import datetime
import json
import math
import operator
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

HTTP_TIMEOUT_SECONDS = 10


# ---- get_date_and_time ----

async def get_date_and_time(timezone=None):
    if timezone:
        try:
            now = datetime.datetime.now(ZoneInfo(timezone))
        except (ZoneInfoNotFoundError, ValueError):
            return {"error": f"Unknown timezone '{timezone}'. Use an IANA name like 'Asia/Tokyo'."}
    else:
        now = datetime.datetime.now().astimezone()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "dayOfWeek": now.strftime("%A"),
        "timezone": timezone or "local",
        "utcOffset": now.isoformat(timespec="minutes")[-6:],
    }


# ---- get_weather (Open-Meteo, no API key) ----

WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "violent showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm with hail",
}


def _http_get_json(url, params):
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(full_url, timeout=HTTP_TIMEOUT_SECONDS) as response:
        return json.load(response)


def _fetch_weather(location, units):
    geo = _http_get_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        {"name": location, "count": 1, "format": "json"},
    )
    if not geo.get("results"):
        return {"error": f"Could not find a place named '{location}'."}
    place = geo["results"][0]

    forecast = _http_get_json(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "temperature_unit": units,
            "wind_speed_unit": "kmh" if units == "celsius" else "mph",
            "timezone": "auto",
            "forecast_days": 3,
        },
    )
    current, daily = forecast["current"], forecast["daily"]
    temp_unit = "°C" if units == "celsius" else "°F"
    wind_unit = "km/h" if units == "celsius" else "mph"
    return {
        "location": ", ".join(p for p in (place.get("name"), place.get("admin1"), place.get("country")) if p),
        "current": {
            "condition": WMO_CODES.get(current["weather_code"], "unknown"),
            "temperature": f"{current['temperature_2m']}{temp_unit}",
            "feelsLike": f"{current['apparent_temperature']}{temp_unit}",
            "humidity": f"{current['relative_humidity_2m']}%",
            "wind": f"{current['wind_speed_10m']} {wind_unit}",
        },
        "forecast": [
            {
                "date": daily["time"][i],
                "condition": WMO_CODES.get(daily["weather_code"][i], "unknown"),
                "high": f"{daily['temperature_2m_max'][i]}{temp_unit}",
                "low": f"{daily['temperature_2m_min'][i]}{temp_unit}",
                "chanceOfRain": f"{daily['precipitation_probability_max'][i]}%",
            }
            for i in range(len(daily["time"]))
        ],
    }


async def get_weather(location, units="celsius"):
    if units not in ("celsius", "fahrenheit"):
        units = "celsius"
    try:
        return await asyncio.to_thread(_fetch_weather, location, units)
    except OSError as e:
        return {"error": f"Weather service unavailable: {e}"}


# ---- calculate (safe arithmetic, no eval) ----

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "log": math.log, "log10": math.log10,
    "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}
MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 100
MAX_RESULT_BITS = 4096


def _evaluate(node):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise ValueError(f"exponent larger than {MAX_EXPONENT} is not allowed")
            # Chained powers like (9**99)**99 stay under MAX_EXPONENT but explode in size.
            if abs(left) > 1 and right > 0 and right * math.log2(abs(left)) > MAX_RESULT_BITS:
                raise ValueError("result is too large")
        return _BIN_OPS[type(node.op)](left, right)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*(_evaluate(arg) for arg in node.args))
    raise ValueError("only numbers, + - * / // % **, parentheses, pi, e and basic math functions are allowed")


async def calculate(expression):
    if len(expression) > MAX_EXPRESSION_LENGTH:
        return {"error": "expression is too long"}
    try:
        result = _evaluate(ast.parse(expression.replace("^", "**"), mode="eval"))
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
        return {"expression": expression, "error": str(e) or type(e).__name__}
    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        result = int(result)
    elif isinstance(result, float):
        result = round(result, 10)
    return {"expression": expression, "result": result}


# ---- registry ----

TOOLS = {
    "get_date_and_time": {
        "handler": get_date_and_time,
        "description": (
            "Get the current date, time and day of the week. Use this whenever the user asks about "
            "the current time or date, or for anything relative to today."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone name such as 'Asia/Tokyo' or 'America/New_York'. "
                    "Omit for the user's local time.",
                }
            },
            "required": [],
        },
    },
    "get_weather": {
        "handler": get_weather,
        "description": (
            "Get the current weather and a 3-day forecast for a city. Use this when the user asks "
            "about weather, temperature, rain, or what to wear."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. 'Tokyo' or 'Paris'"},
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units (default celsius)",
                },
            },
            "required": ["location"],
        },
    },
    "calculate": {
        "handler": calculate,
        "description": (
            "Evaluate an arithmetic expression exactly. Use this for any math beyond trivial mental "
            "arithmetic, e.g. '1234 * 5678', '(15 / 100) * 2400', 'sqrt(2)'."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Expression using numbers, + - * / // % ** ( ), pi, e, "
                    "sqrt, abs, round, log, log10, exp, sin, cos, tan",
                }
            },
            "required": ["expression"],
        },
    },
}


def tool_specs():
    # Nova Sonic expects inputSchema.json as a JSON string, not an object.
    return [
        {
            "toolSpec": {
                "name": name,
                "description": tool["description"],
                "inputSchema": {"json": json.dumps(tool["schema"])},
            }
        }
        for name, tool in TOOLS.items()
    ]


async def run_tool(name, arguments):
    """Run a tool by name; always returns a JSON-serializable dict, with 'error' on failure."""
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"Unknown tool '{name}'"}
    allowed = tool["schema"]["properties"]
    missing = [p for p in tool["schema"]["required"] if not arguments.get(p)]
    if missing:
        return {"error": f"Missing required argument(s): {', '.join(missing)}"}
    kwargs = {k: v for k, v in arguments.items() if k in allowed}
    try:
        return await tool["handler"](**kwargs)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
