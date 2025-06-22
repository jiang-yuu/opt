import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置参数
np.random.seed(42)
num_vehicles = 30
num_passengers = 100
city_size = 20  # km
vehicle_speed_kmh = 40  # 城市速度（km/h）
vehicle_speed_kmpm = vehicle_speed_kmh / 60  # km/min
min_distance = 3.0  # 最小出行距离 km
#gamma = 1.3  # 绕路容忍比例

# 城市中心
city_center = np.array([city_size / 2, city_size / 2])  # [15, 15]

# 将车辆初始位置限制在市中心的10x10 km区域内
center_region_min = city_center - 5  # [10, 10]
center_region_max = city_center + 5  # [20, 20]
vehicle_positions = np.random.uniform(low=center_region_min, high=center_region_max, size=(num_vehicles, 2))
# 初始化乘客数据列表
pickup_points = []
dropoff_points = []
valid_indices = []

# 循环直到收集到足够满足距离约束的乘客数据
while len(pickup_points) < num_passengers:
    p = np.random.normal(loc=city_size/2, scale=5, size=(1, 2))
    d = np.random.normal(loc=city_size/2, scale=6, size=(1, 2))
    p = np.clip(p, 0, city_size)
    d = np.clip(d, 0, city_size)
    dist = np.linalg.norm(d - p)
    if dist >= min_distance:
        pickup_points.append(p[0])
        dropoff_points.append(d[0])

pickup_points = np.array(pickup_points)
dropoff_points = np.array(dropoff_points)

# 请求时间（分钟）
request_times = np.sort(np.random.uniform(0, 20, size=num_passengers))

# 最短行驶时间（分钟）
distances = np.linalg.norm(dropoff_points - pickup_points, axis=1)
min_travel_times = distances / vehicle_speed_kmpm

# 绕路容忍时间
#delta_tolerance = gamma * min_travel_times

# 可接受上车时间窗（不早到，仅惩罚晚到）
preferred_start = request_times
preferred_end = np.minimum(request_times + 5, 25)  # 合理等待范围
late_accept = np.minimum(request_times + 10, 30)   # 最大可接受上车时间

# 构造乘客 DataFrame
passenger_df = pd.DataFrame({
    'passenger_id': np.arange(num_passengers),
    'pickup_x': pickup_points[:, 0],
    'pickup_y': pickup_points[:, 1],
    'dropoff_x': dropoff_points[:, 0],
    'dropoff_y': dropoff_points[:, 1],
    'request_time': request_times,
    'preferred_start': preferred_start,
    'preferred_end': preferred_end,
    'late_accept': late_accept,
    'min_travel_time': min_travel_times
    #'delta_tolerance': delta_tolerance
})

# 构造车辆 DataFrame
vehicle_df = pd.DataFrame({
    'vehicle_id': np.arange(num_vehicles),
    'start_x': vehicle_positions[:, 0],
    'start_y': vehicle_positions[:, 1]
})

# 可视化
city_center = [city_size / 2, city_size / 2]
plt.figure(figsize=(8, 8))
plt.scatter(pickup_points[:, 0], pickup_points[:, 1], c='blue', label='Pickup', alpha=0.6)
plt.scatter(dropoff_points[:, 0], dropoff_points[:, 1], c='green', label='Dropoff', alpha=0.6)
plt.scatter(vehicle_positions[:, 0], vehicle_positions[:, 1], c='red', marker='x', s=100, label='Vehicles')
plt.scatter(*city_center, c='black', marker='*', s=200, label='City Center')
plt.xlim(0, city_size)
plt.ylim(0, city_size)
plt.title("Ride-Sharing City Layout (20×20 km, 40 km/h)")
plt.legend()
plt.grid(True)
plt.show()

# 输出 CSV
vehicle_df.to_csv('vehicles5.csv', index=False)
passenger_df.to_csv('passengers5.csv', index=False)

print("✅ 数据生成完毕！已保存为 'vehicles.csv' 和 'passengers.csv'")