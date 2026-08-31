
import streamlit as st
import pandas as pd

st.title("我的第一个数据应用")
st.write("这是一个简单的表格展示:")

data = pd.DataFrame({
    '第一列': [1, 2, 3, 4],
    '第二列': [10, 20, 30, 40]
})

st.dataframe(data)


# 运行代码： streamlit run app.py
# 会打开一个网页，然后显示标题和表格。