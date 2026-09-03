


import json
import sys


def get_weather_by_city(city: str) -> dict:
    """Return demo weather data for a supported city."""
    if city == "北京":
        return {
            "city": "北京",
            "weather": "小雨",
            "temperature": "18℃",
            "wind": "东北风",
            "humidity": "72%",
            "suggestion": "建议带伞，出门注意保暖。",
        }

    if city == "上海":
        return {
            "city": "上海",
            "weather": "多云",
            "temperature": "24℃",
            "wind": "东南风",
            "humidity": "60%",
            "suggestion": "适合出行，暂时不需要带伞。",
        }

    return {
        "city": city,
        "weather": "未知",
        "temperature": "未知",
        "wind": "未知",
        "humidity": "未知",
        "suggestion": "当前演示版只支持北京和上海。",
    }


def main() -> None:
    city = sys.argv[1] if len(sys.argv) > 1 else "北京"
    result = get_weather_by_city(city)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
