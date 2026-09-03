import operator
from typing import Annotated

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class PackageState(TypedDict):
    """State: the shared package information carried through the graph."""

    package_id: str
    origin: str
    destination: str
    priority: str
    status: str
    history: Annotated[list[str], operator.add]
    total_distance: int


def receive_package(state: PackageState) -> dict:
    """Node 1: receive the package."""
    return {
        "status": "已揽收",
        "history": [f"在{state['origin']}揽收包裹"],
    }


def sort_package(state: PackageState) -> dict:
    """Node 2: sort the package by destination."""
    destination = state["destination"]

    if "北京" in destination:
        next_station = "北京分拣中心"
    elif "上海" in destination:
        next_station = "上海分拣中心"
    else:
        next_station = "其他地区分拣中心"

    return {
        "status": "已分拣",
        "history": [f"根据目的地分拣至{next_station}"],
    }


def standard_delivery(state: PackageState) -> dict:
    """Node 3A: standard delivery route."""
    return {
        "status": "标准运输中",
        "history": ["选择标准陆运路线"],
        "total_distance": 500,
    }


def express_delivery(state: PackageState) -> dict:
    """Node 3B: express delivery route."""
    return {
        "status": "加急运输中",
        "history": ["选择空运加急路线"],
        "total_distance": 800,
    }


def final_delivery(state: PackageState) -> dict:
    """Node 4: deliver the package."""
    return {
        "status": "已签收",
        "history": [f"已送达{state['destination']}并签收"],
    }


def select_delivery_route(state: PackageState) -> str:
    """Conditional edge router: choose the next node from current state."""
    if state["priority"] == "加急":
        return "加急配送"
    return "标准配送"


def build_delivery_graph():
    """Build and compile the workflow graph."""
    builder = StateGraph(PackageState)

    builder.add_node("揽收站", receive_package)
    builder.add_node("分拣中心", sort_package)
    builder.add_node("标准配送", standard_delivery)
    builder.add_node("加急配送", express_delivery)
    builder.add_node("派送站", final_delivery)

    builder.add_edge(START, "揽收站")
    builder.add_edge("揽收站", "分拣中心")
    builder.add_conditional_edges(
        "分拣中心",
        select_delivery_route,
        ["标准配送", "加急配送"],
    )
    builder.add_edge("标准配送", "派送站")
    builder.add_edge("加急配送", "派送站")
    builder.add_edge("派送站", END)

    return builder.compile()


def print_result(result: PackageState) -> None:
    print("最终状态:", result["status"])
    print("总里程:", result["total_distance"])
    print("配送历史:")
    for index, item in enumerate(result["history"], start=1):
        print(f"  {index}. {item}")


def main() -> None:
    delivery_system = build_delivery_graph()

    test_packages = [
        {
            "package_id": "P001",
            "origin": "北京",
            "destination": "上海",
            "priority": "普通",
            "status": "待揽收",
            "history": [],
            "total_distance": 0,
        },
        {
            "package_id": "P002",
            "origin": "广州",
            "destination": "乌鲁木齐",
            "priority": "加急",
            "status": "待揽收",
            "history": [],
            "total_distance": 0,
        },
    ]

    for package in test_packages:
        print("\n" + "=" * 80)
        print(f"开始配送包裹: {package['package_id']} / 优先级: {package['priority']}")
        result = delivery_system.invoke(package)
        print_result(result)


if __name__ == "__main__":
    main()
