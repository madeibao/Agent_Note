

from langchain.tools import tool

@tool
def recommend_tutorial(
    language: str,
    level: str = "入门",
    count: int = 3,
) -> str:
    """根据编程语言和难度推荐菜鸟教程 RUNOOB 的课程。

    Args:
        language: 编程语言，如 Python、Java、HTML
        level: 难度级别，可选 "入门"、"进阶"、"高级"。默认 "入门"
        count: 推荐的课程数量，默认 3
    """
    tutorials = {
        "python": ["Python3 基础", "Python 面向对象", "Python 爬虫", "Python 数据分析"],
        "java": ["Java 基础", "Java 面向对象", "Java 集合框架", "Java 多线程"],
    }
    all_tutorials = tutorials.get(language.lower(), [f"{language} 基础教程"])

    selected = all_tutorials[:count]
    return f"推荐 {language} {level} 级别课程：{'、'.join(selected)}"


# 调用时只需要传必填参数
print(recommend_tutorial.invoke({"language": "Python"}))
# 输出：推荐 Python 入门 级别课程：Python3 基础、Python 面向对象、Python 爬虫

# 也可以覆盖默认参数
print(recommend_tutorial.invoke({
    "language": "Java",
    "level": "进阶",
    "count": 2,
}))
# 输出：推荐 Java 进阶 级别课程：Java 基础、Java 面向对象