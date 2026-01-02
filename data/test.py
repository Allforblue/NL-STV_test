# import pandas as pd
# import os
#
# # 定义文件名
# file_path = 'yellow_tripdata_2025-01.parquet'
#
# # 检查文件是否存在
# if os.path.exists(file_path):
#     try:
#         # 读取parquet文件
#         # columns参数指定只读取 'fare_amount' 这一列，这样处理大文件更快且省内存
#         df = pd.read_parquet(file_path, columns=['fare_amount','trip_distance'])
#
#         # 筛选出 fare_amount 小于 0 的行
#         negative_fare_rows = df[df['fare_amount'] < 0]
#         negative_fare_rows = df[df['trip_distance'] == 0]
#
#         # 获取行数
#         count = len(negative_fare_rows)
#
#         print(f"文件读取成功。")
#         print(f"fare_amount 列为负数的行数共有: {count} 行")
#
#         # 如果你想看前几行具体的数据，可以取消下面这行的注释
#         # print(negative_fare_rows.head())
#
#     except Exception as e:
#         print(f"读取或处理文件时发生错误: {e}")
# else:
#     print(f"错误: 在当前目录下未找到文件 {file_path}")

# import pandas as pd
# import plotly.express as px
#
# # 1. 读取数据 (假设我们要分析所有数据)
# # 注意：如果文件很大，groupby 仍然可以运行，但内存消耗会增加。
# # 我们可以只读取需要的两列
# df = pd.read_parquet('yellow_tripdata_2025-01.parquet', columns=['VendorID', 'trip_distance'])
#
# # 2. 数据聚合处理
# # 计算每个 VendorID 的平均距离
# grouped_df = df.groupby('VendorID')['trip_distance'].mean().reset_index()
#
# # 3. 绘图
# # 建议将 VendorID 转为字符串，这样图表更像是分类对比而不是数值趋势
# grouped_df['VendorID'] = grouped_df['VendorID'].astype(str)
#
# fig = px.bar(grouped_df,
#              x='VendorID',
#              y='trip_distance',
#              title='Average Trip Distance by Vendor ID',
#              labels={'trip_distance': 'Avg Distance (miles)', 'VendorID': 'Vendor Provider'}) # 优化标签显示
#
# # 4. 显示图表
# fig.show()

import pandas as pd
import plotly.express as px

# 1. 读取数据
# 建议只读取 needed 列 ('trip_distance') 以节省内存和提高速度
df = pd.read_parquet('yellow_tripdata_2025-01.parquet')

# # 2. 绘制直方图
# # 这里完全保留原始数据，不截断也不过滤
# fig = px.histogram(df, x='trip_distance', title='Histogram of Trip Distance')
#
# # 3. 显示图表
# fig.show()
# 按“天”聚合，观察每日收入波动

# df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
# df_weekly = df.groupby(pd.Grouper(key='tpep_pickup_datetime', freq='W')).sum(numeric_only=True)
#
# fig = px.line(df_weekly, x=df_weekly.index, y='total_amount',
#               title='Weekly Trend of Total Amount Over Time')

fig = px.histogram(df, x='total_amount', title='Histogram of Total Amount')
fig.show()