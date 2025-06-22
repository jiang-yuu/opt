import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib.patches as mpatches
import os
import math

# ==================== 参数定义 ====================
TIME_INTERVAL = 5
MAX_CAPACITY = 4
VEHICLE_SPEED = 0.67  # km/min
COST_PER_KM = 1.0
FIXED_COST = 5.0
ALPHA_2 = 2.0
ALPHA_3 = 5.0
BETA = 3.0
DELAT = 3.0

# 蚁群算法参数
NUM_ANTS = 20
ALPHA = 1.0  # 信息素重要程度因子
BETA_ACO = 2.0  # 启发式信息重要程度因子
RHO = 0.5  # 信息素挥发因子
Q = 100.0  # 信息素增加强度系数
MAX_ITER = 50
INITIAL_PHEROMONE = 0.1

# 指定要输出的车辆ID列表（修改此处选择需要的车辆）
TARGET_VEHICLES = [0,1,2,3]  

# ==================== 数据读取 ====================
vehicles_df = pd.read_csv('jingtai/data/vehicles5.csv')
passengers_df = pd.read_csv('jingtai/data/passengers5.csv')

# 按时间段划分请求
total_time = passengers_df['request_time'].max()
T = int(np.ceil(total_time / TIME_INTERVAL))
D_t_list = [[] for _ in range(T)]
for _, row in passengers_df.iterrows():
    t = int(row['request_time'] // TIME_INTERVAL)
    D_t_list[t].append(row.to_dict())

# ==================== 基本函数定义 ====================
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def travel_time(p1, p2):
    return euclidean(p1, p2) / VEHICLE_SPEED

def initialize_vehicles():
    state = {}
    for _, row in vehicles_df.iterrows():
        state[row['vehicle_id']] = {
            'location': (row['start_x'], row['start_y']),
            'route': [],
            'onboard': [],
            'served': set(),
            'time': 0,
            'history': []
        }
    return state

# ==================== 目标函数与惩罚计算 ====================
def compute_cost_and_penalty(route, passengers, vehicle_start):
    cost = 0
    penalty = 0
    onboard = set()
    time = 0
    loc = vehicle_start

    for pt in route:
        pid, x, y, typ = pt
        dist = euclidean(loc, (x, y))
        travel = dist / VEHICLE_SPEED
        time += travel
        cost += dist * COST_PER_KM
        passenger = passengers[pid]

        if typ == 'pickup':
            if time > passenger['preferred_start']:
                penalty += ALPHA_2 * max(time - passenger['preferred_start'], 0)
            if time > passenger['preferred_end']:
                penalty += ALPHA_3 * max(time - passenger['preferred_end'], 0)
            passenger['pickup_time'] = time
            onboard.add(pid)
        else:
            delta = time - passenger['pickup_time']
            shortest = passenger['min_travel_time']
            if delta < shortest * DELAT:
                penalty += BETA * (delta - shortest)
            onboard.discard(pid)

        loc = (x, y)

    if route:
        cost += FIXED_COST

    return cost + penalty

# ==================== 插入可行性判断 ====================
def is_feasible_insert(route, new_order, insert_pick, insert_drop, passengers, vehicle_start):
    new_route = route[:insert_pick] + [new_order['pickup']] + route[insert_pick:insert_drop] + [new_order['dropoff']] + route[insert_drop:]
    onboard = []
    time = 0
    loc = vehicle_start

    for pt in new_route:
        pid, x, y, typ = pt
        time += travel_time(loc, (x, y))
        if typ == 'pickup':
            onboard.append(pid)
            if time > passengers[pid]['preferred_end']:
                return False
        else:
            if pid not in onboard:
                return False
            pickup_time = passengers[pid].get('pickup_time', 0)
            delta = time - pickup_time
            if delta > passengers[pid]['min_travel_time'] * DELAT:
                return False
            onboard.remove(pid)
        if len(onboard) > MAX_CAPACITY:
            return False
        loc = (x, y)
    return True

# ==================== 蚁群算法核心 ====================
def ant_colony_optimization():
    best_solution = None
    best_cost = float('inf')
    best_metrics = None  # 记录最优解的指标
    
    passengers_dict = {row['passenger_id']: row for _, row in passengers_df.iterrows()}
    
    # 初始化信息素矩阵
    pheromone_matrix = {}  # 键为(vehicle_id, order_id, insert_pick, insert_drop)，值为信息素浓度
    
    for iter in range(MAX_ITER):
        # 每只蚂蚁构建一个解决方案
        all_ants_solutions = []
        all_ants_costs = []
        all_ants_metrics = []  # 记录每个解的指标
        
        for ant in range(NUM_ANTS):
            vehicles = initialize_vehicles()
            
            for t in range(T):
                new_orders = D_t_list[t]
                for order in new_orders:
                    pid = order['passenger_id']
                    order['pickup'] = (pid, order['pickup_x'], order['pickup_y'], 'pickup')
                    order['dropoff'] = (pid, order['dropoff_x'], order['dropoff_y'], 'dropoff')
                    
                    # 计算所有可能的插入位置及其概率
                    possible_insertions = []
                    
                    for vid, v in vehicles.items():
                        path = v['route']
                        for i in range(len(path) + 1):
                            for j in range(i + 1, len(path) + 2):
                                if is_feasible_insert(path, order, i, j, passengers_dict, v['location']):
                                    # 计算启发式信息
                                    heuristic = calculate_heuristic(path, order, i, j, passengers_dict, v['location'])
                                    
                                    # 获取信息素浓度
                                    pheromone_key = (vid, pid, i, j)
                                    pheromone = pheromone_matrix.get(pheromone_key, INITIAL_PHEROMONE)
                                    
                                    # 计算概率
                                    probability = (pheromone ** ALPHA) * (heuristic ** BETA_ACO)
                                    possible_insertions.append((vid, i, j, probability))
            
                    # 如果有可行的插入位置，选择一个
                    if possible_insertions:
                        total_prob = sum(p[3] for p in possible_insertions)
                        probabilities = [p[3]/total_prob for p in possible_insertions]
                        
                        # 轮盘赌选择
                        choice_idx = np.random.choice(len(possible_insertions), p=probabilities)
                        chosen_vid, i, j, _ = possible_insertions[choice_idx]
                        
                        # 执行插入
                        v = vehicles[chosen_vid]
                        v['route'] = v['route'][:i] + [order['pickup']] + v['route'][i:j-1] + [order['dropoff']] + v['route'][j-1:]
                        v['served'].add(pid)
            
            # 计算当前解的总代价
            total_cost = 0
            for vid, v in vehicles.items():
                total_cost += compute_cost_and_penalty(v['route'], passengers_dict, v['location'])
            
            # 计算额外指标
            metrics = evaluate_solution(vehicles, passengers_dict)
            
            all_ants_solutions.append(vehicles)
            all_ants_costs.append(total_cost)
            all_ants_metrics.append(metrics)
            
            # 更新最优解
            if total_cost < best_cost:
                best_cost = total_cost
                best_solution = {vid: v.copy() for vid, v in vehicles.items()}
                best_metrics = metrics
        
        # 更新信息素矩阵
        update_pheromone(pheromone_matrix, all_ants_solutions, all_ants_costs, RHO, Q)
        
        # 打印迭代进度
        if (iter + 1) % 10 == 0:
            print(f"迭代 {iter+1}/{MAX_ITER}, 最优成本: {best_cost:.2f}")
    
    # 输出最优解的指标
    print("\n最优解指标:")
    for key, value in best_metrics.items():
        print(f"  {key}: {value:.2f}")
    
    return best_solution

# 计算启发式信息 - 考虑就近送原则
def calculate_heuristic(route, new_order, insert_pick, insert_drop, passengers, vehicle_start):
    # 插入新订单后的新路径
    new_route = route[:insert_pick] + [new_order['pickup']] + route[insert_pick:insert_drop] + [new_order['dropoff']] + route[insert_drop:]
    
    # 计算新路径的总距离
    total_distance = 0
    current_loc = vehicle_start
    
    for pt in new_route:
        x, y = pt[1], pt[2]
        total_distance += euclidean(current_loc, (x, y))
        current_loc = (x, y)
    
    # 计算dropoff点之间的距离 - 就近送原则
    dropoff_distance = 0
    last_dropoff = None
    
    for pt in new_route:
        if pt[3] == 'dropoff':
            if last_dropoff is not None:
                dropoff_distance += euclidean(last_dropoff, (pt[1], pt[2]))
            last_dropoff = (pt[1], pt[2])
    
    # 综合评估：总距离越小越好，dropoff点之间的距离越小越好
    distance_weight = 1.0
    dropoff_weight = 2.0  # 提高dropoff距离的权重，更重视就近送
    
    score = 1.0 / (distance_weight * total_distance + dropoff_weight * dropoff_distance + 0.001)
    return score

# 评估解决方案的指标
    return total
def evaluate_total_cost(solution, passengers_dict):
    """计算一个完整解决方案的总成本（所有车辆成本之和）。"""
    total = 0
    for vid, v in solution.items():
        total += compute_cost_and_penalty(v['route'], passengers_dict, v['location'])
    return total
def evaluate_solution(vehicles, passengers):
    """
    评估一个完整解决方案（所有车辆的路径）的各项性能指标（KPIs）。
    [已修正版本]
    
    Args:
        vehicles (dict): 包含所有车辆状态和路径的解决方案。
        passengers (dict): 所有乘客信息的字典。
        
    Returns:
        dict: 包含各项性能指标的字典。
    """
    total_distance = 0
    total_waiting_time = 0
    total_travel_time = 0
    
    #使用集合来统计实际服务的乘客，更可靠。
    all_served_pids = set()
    
    total_capacity_used = 0.0
    total_capacity_time = 0.0
    
    used_vehicles = 0
    
    #  创建一个局部字典来存储上车时间，避免修改全局的 passengers 字典。
    pickup_times_local = {}

    # 遍历每辆车计算其贡献的指标
    for vid, v in vehicles.items():
        path = v['route']
        if not path:
            continue
        used_vehicles += 1
        
        current_loc = v['location']
        time = 0
        onboard = []
        
        # 遍历路径计算距离、等待时间、旅行时间等
        for pt in path:
            pid, x, y, typ = pt
            dist = euclidean(current_loc, (x, y))
            duration = dist / VEHICLE_SPEED
            
            # 
            # (每段路程的载客量 * 该路程的行驶时间)
            total_capacity_used += len(onboard) * duration
            # 总行驶时间
            total_capacity_time += duration
            
            time += duration
            current_loc = (x, y)
            total_distance += dist
            
            # 将遇到的乘客ID加入集合
            all_served_pids.add(pid)

            if typ == 'pickup':
                wait_time = max(time - passengers[pid]['request_time'], 0)
                total_waiting_time += wait_time
                
                # 将上车时间存入局部字典，而不是修改全局字典。
                pickup_times_local[pid] = time 
                onboard.append(pid)
            else: # typ == 'dropoff'
                #从局部字典安全地获取上车时间。
                if pid in pickup_times_local:
                    ride_time = time - pickup_times_local[pid]
                    total_travel_time += ride_time
                if pid in onboard:
                    onboard.remove(pid)
        
        # total_cost += compute_cost_and_penalty(path, passengers, v['location'])
    
    # 用集合的大小作为服务乘客数。
    served_passengers = len(all_served_pids)
    
    # 在所有路径信息都处理完毕后，一次性计算总成本。
    total_cost = evaluate_total_cost(vehicles, passengers)

    # 计算平均指标
    avg_waiting_time = total_waiting_time / served_passengers if served_passengers else 0
    avg_travel_time = total_travel_time / served_passengers if served_passengers else 0
    
    # 计算平均容量利用率，结果是一个0-1的比率。
    avg_capacity_util = (total_capacity_used / MAX_CAPACITY) / total_capacity_time if total_capacity_time > 0 else 0
    
    unit_cost = total_cost / served_passengers if served_passengers else 0
    vehicle_utilization = used_vehicles / len(vehicles) if vehicles else 0

    return {
        'served_passengers': served_passengers,
        'avg_waiting_time': avg_waiting_time,
        'avg_travel_time': avg_travel_time,
        'total_distance': total_distance,
        'total_cost': total_cost,
        'unit_cost': unit_cost,
        'vehicle_utilization': vehicle_utilization,
        'avg_capacity_util': avg_capacity_util
    }


# 更新信息素矩阵
def update_pheromone(pheromone_matrix, solutions, costs, rho, q):
    # 信息素挥发
    for key in list(pheromone_matrix.keys()):
        pheromone_matrix[key] *= (1 - rho)
        if pheromone_matrix[key] < 0.0001:
            del pheromone_matrix[key]
    
    # 信息素增强 - 使用成本的倒数
    for solution, cost in zip(solutions, costs):
        pheromone_delta = q / cost
        
        for t in range(T):
            new_orders = D_t_list[t]
            for order in new_orders:
                pid = order['passenger_id']
                
                # 查找该订单被分配给了哪辆车
                assigned_vid = None
                for vid, v in solution.items():
                    if pid in v['served']:
                        assigned_vid = vid
                        break
                
                if assigned_vid is not None:
                    v = solution[assigned_vid]
                    path = v['route']
                    
                    # 找到pickup和dropoff的位置
                    pickup_idx = next((i for i, pt in enumerate(path) if pt[0] == pid and pt[3] == 'pickup'), None)
                    dropoff_idx = next((i for i, pt in enumerate(path) if pt[0] == pid and pt[3] == 'dropoff'), None)
                    
                    if pickup_idx is not None and dropoff_idx is not None:
                        pheromone_key = (assigned_vid, pid, pickup_idx, dropoff_idx)
                        current_pheromone = pheromone_matrix.get(pheromone_key, INITIAL_PHEROMONE)
                        pheromone_matrix[pheromone_key] = current_pheromone + pheromone_delta

# ==================== 可视化函数 ====================
def save_vehicle_route_animation(vehicle_id, vehicle_path, passenger_points, capacity_states, mp4_path, gif_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    xs = [p[1] for p in vehicle_path]
    ys = [p[2] for p in vehicle_path]
    line, = ax.plot([], [], 'k-', lw=2)
    scatter_pickup = {}
    scatter_dropoff = {}
    texts = []
    legend_patches = []
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1, max(ys) + 1)
    ax.set_title(f"Vehicle {vehicle_id} Route Animation")

    pid_list = list(set([pt[0] for pt in vehicle_path]))
    color_map = {pid: plt.cm.tab10(i % 10) for i, pid in enumerate(pid_list)}

    # 生成图例
    for pid in pid_list:
        patch = mpatches.Patch(color=color_map[pid], label=f'P{pid}')
        legend_patches.append(patch)
    ax.legend(handles=legend_patches, title='Passenger Color', loc='upper right', fontsize='small')

    def update(frame):
        line.set_data(xs[:frame + 1], ys[:frame + 1])
        for txt in texts:
            txt.remove()
        texts.clear()

        pt = vehicle_path[frame]
        pid, x, y, typ = pt
        color = color_map[pid]

        if typ == 'pickup' and pid not in scatter_pickup:
            scatter_pickup[pid] = ax.scatter(x, y, c=[color], marker='o', s=100)
        elif typ == 'dropoff' and pid not in scatter_dropoff:
            scatter_dropoff[pid] = ax.scatter(x, y, c=[color], marker='v', s=100)  # 用三角形向下表示下车点
            ax.text(x, y, '✓', fontsize=12, color=color, ha='center', va='center')  # 用 ✓ 绘制在坐标上

        texts.append(ax.text(x, y, f"{capacity_states[frame]}", fontsize=8, ha='center', va='bottom', color='red'))
        return [line] + list(scatter_pickup.values()) + list(scatter_dropoff.values()) + texts

    ani = animation.FuncAnimation(fig, update, frames=len(vehicle_path), interval=800, blit=True)
    ani.save(gif_path, writer=PillowWriter(fps=1))
    plt.close(fig)
# ==================== 可视化函数 ====================
def save_vehicle_route_animation(vehicle_id, vehicle_path, capacity_states, folder_path):
    if not vehicle_path:
        print(f"Vehicle {vehicle_id} has no route, skipping animation.")
        return
    
    # 创建结果文件夹（如果不存在）
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    xs = [p[1] for p in vehicle_path]
    ys = [p[2] for p in vehicle_path]
    line, = ax.plot([], [], 'k-', lw=2)
    scatter_pickup = {}
    scatter_dropoff = {}
    texts = []
    legend_patches = []
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1, max(ys) + 1)
    ax.set_title(f"Vehicle {vehicle_id} Route Animation")

    # 为每个乘客分配一个唯一的颜色
    pid_list = list(set([pt[0] for pt in vehicle_path]))
    color_map = {pid: plt.cm.tab10(i % 10) for i, pid in enumerate(pid_list)}

    # 创建图例
    for pid in pid_list:
        patch = mpatches.Patch(color=color_map[pid], label=f'P{pid}')
        legend_patches.append(patch)
    ax.legend(handles=legend_patches, title='Passenger Color', loc='upper right', fontsize='small')

    # 动画更新函数，每一帧绘制路径的一部分
    def update(frame):
        line.set_data(xs[:frame + 1], ys[:frame + 1]) # 绘制已行驶的路径
        for txt in texts:
            txt.remove()
        texts.clear()

        pt = vehicle_path[frame]
        pid, x, y, typ = pt
        color = color_map[pid]

        # 标记上车点和下车点
        if typ == 'pickup' and pid not in scatter_pickup:
            scatter_pickup[pid] = ax.scatter(x, y, c=[color], marker='o', s=100)
        elif typ == 'dropoff' and pid not in scatter_dropoff:
            scatter_dropoff[pid] = ax.scatter(x, y, c=[color], marker='v', s=100)
            ax.text(x, y, '✓', fontsize=12, color=color, ha='center', va='center')

        # 显示当前车上的乘客数量
        texts.append(ax.text(x, y, f"{capacity_states[frame]}", fontsize=8, ha='center', va='bottom', color='red'))
        return [line] + list(scatter_pickup.values()) + list(scatter_dropoff.values()) + texts

    # 创建并保存动画
    ani = animation.FuncAnimation(fig, update, frames=len(vehicle_path), interval=800, blit=True)
    gif_path = os.path.join(folder_path, f'vehicle_{vehicle_id}.gif')
    try:
        ani.save(gif_path, writer=PillowWriter(fps=1))
    except Exception as e:
        print(f"Could not save animation for vehicle {vehicle_id}. Error: {e}.")
    plt.close(fig)
# ==================== 主程序入口 ====================
if __name__ == '__main__':
    # 运行蚁群算法
    best_solution = ant_colony_optimization()
    
    # 创建结果文件夹
    folder_name = "jingtai/results/ant_colony_optimization"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # 写入结果总结
    with open(os.path.join(folder_name, "summary.txt"), 'w') as f:
        # 写入全局优化指标
        metrics = evaluate_solution(best_solution, {row['passenger_id']: row for _, row in passengers_df.iterrows()})
        f.write("调度结果关键性能指标（KPI）:\n")
        f.write("=" * 50 + "\n")
        for key, value in metrics.items():
            if key == 'served_passengers':
                f.write(f"{key}: {int(value)}\n")
            else:
                f.write(f"{key}: {value:.2f}\n")
        f.write("=" * 50 + "\n\n")
        
        # 写入指定车辆的路线信息
        f.write("车辆路线详细信息:\n")
        f.write("=" * 50 + "\n")
        for vid in TARGET_VEHICLES:
            if vid not in best_solution or not best_solution[vid]['route']:
                f.write(f"车辆 {vid}: 未分配任何乘客\n\n")
                continue
                
            f.write(f"车辆 {vid} 路线:\n")
            v = best_solution[vid]
            path = v['route']
            onboard = set()
            capacity_states = []
            
            for pt in path:
                pid, x, y, typ = pt
                if typ == 'pickup':
                    onboard.add(pid)
                elif typ == 'dropoff':
                    onboard.discard(pid)
                capacity_states.append(len(onboard))
                f.write(f"  {typ.upper()} 乘客 P{pid} at ({x:.2f}, {y:.2f}), 车内乘客数: {len(onboard)}\n")
            
            f.write(f"  总行驶距离: {sum(euclidean(path[i][1:3], path[i+1][1:3]) for i in range(len(path)-1)) + euclidean(v['location'], path[0][1:3]):.2f} km\n")
            f.write(f"  服务乘客数: {len(v['served'])}\n\n")
            
            # 保存车辆路线动画
            save_vehicle_route_animation(
                vehicle_id=vid,
                vehicle_path=path,
                capacity_states=capacity_states,
                folder_path=folder_name
            )
            
            f.write(f"车辆 {vid} 的路线动画已保存为: {folder_name}/vehicle_{vid}.gif\n")
            f.write("-" * 50 + "\n")

    print(f"调度结果已保存至 {folder_name}/summary.txt")
    print(f"已为车辆 {TARGET_VEHICLES} 生成路线总结和可视化动画，保存在 {folder_name} 文件夹中")
