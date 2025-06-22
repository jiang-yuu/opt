# ==============================================================================
#                      代码功能：预约式拼车的车辆路径问题 (DVRP) 求解器
#
# 本代码实现了六种元启发式算法来解决一个带时间窗和容量限制的车辆路径问题。
# 乘客请求是预约式出现的，算法需要为多辆车规划路径，以最低的总成本服务尽可能多的乘客。
#
# 算法包括:
# 1. 模拟退火 (Simulated Annealing, SA)
# 2. 遗传算法 (Genetic Algorithm, GA)
# 3. 蚁群优化 (Ant Colony Optimization, ACO)
# 4. A*
# 5. Dijkstra
# 6. 粒子群
#
# 输出：
# - 为每种算法生成一个结果文件夹。
# - 文件夹内包含一个 summary.txt 文件，记录了调度结果的关键性能指标（KPI）。
# - 为指定的车辆生成行驶路径的GIF动画。
# ==============================================================================

# ====== 通用调度参数 ======
TIME_INTERVAL = 10          # 时间窗口长度，单位：分钟。用于将动态请求分批处理。
MAX_CAPACITY = 4            # 每辆车最大载客量
VEHICLE_SPEED = 0.67        # 车辆速度（km/min），约等于 40 km/h
COST_PER_KM = 1.0           # 每公里成本
FIXED_COST = 5.0            # 每辆车只要被使用，就会产生的固定启动成本
DELAT = 3.0                 # 乘客最短行程容忍比例（δ），乘客实际乘车时间不应超过其最短直达时间的 DELAT 倍

# 指定要输出的车辆ID列表（修改此处选择需要的车辆）
TARGET_VEHICLES = [0, 1, 2, 3]

# ====== 惩罚项参数（通用于所有算法） ======
# 这些参数用于量化对服务质量不佳的惩罚，计入总成本
ALPHA_1 = 2.0               # 超过乘客期望上车时间 `preferred_start` 的惩罚系数
ALPHA_2 = 5.0               # 超过乘客最晚上车时间 `preferred_end` 的惩罚系数
BETA = 3.0                  # 超出乘客最短行程容忍比例的惩罚系数

# --- 导入所需库 ---
import pandas as pd             # 用于数据处理和读取CSV文件
import numpy as np              # 用于数值计算，特别是向量和矩阵运算
import math                     # 用于数学计算
import random                   # 用于生成随机数，是启发式算法的核心
import matplotlib.pyplot as plt # 用于绘图
import matplotlib.animation as animation # 用于创建动画
from matplotlib.animation import FFMpegWriter # 用于保存mp4格式动画
import matplotlib.patches as mpatches   # 用于在图例中创建自定义颜色块
from matplotlib.animation import PillowWriter # 用于保存GIF动画
import os                       # 用于文件和目录操作，如创建结果文件夹
import copy                     # 用于创建对象的深拷贝，防止算法间数据污染
from tqdm import tqdm           # 用于显示进度条，方便跟踪算法运行进度

# --- 数据加载与预处理 ---
# 从CSV文件中加载车辆和乘客数据
vehicles_df = pd.read_csv('jingtai/data/vehicles5.csv')
passengers_df = pd.read_csv('jingtai/data/passengers5.csv')

# 将乘客数据转换为以 passenger_id 为键的字典，方便快速查找
passengers_dict = {row['passenger_id']: row for _, row in passengers_df.iterrows()}

# 计算总的时间窗口数量 T
T = int(np.ceil(passengers_df['request_time'].max() / TIME_INTERVAL))

# 创建一个列表 D_t_list，用于按时间窗口存储新出现的乘客请求
D_t_list = [[] for _ in range(T)]
for _, row in passengers_df.iterrows():
    # 根据请求时间，计算其所属的时间窗口索引 t
    t = int(row['request_time'] // TIME_INTERVAL)
    # 将该乘客的请求信息添加到对应时间窗口的列表中
    D_t_list[t].append(row.to_dict())


# --- 基础工具函数 ---
# 深拷贝解决方案的函数
def copy_solution(solution):
    """
    对车辆路径解决方案进行深拷贝，避免不同粒子之间的数据相互污染。
    
    Args:
        solution (dict): 包含所有车辆状态和路径的解决方案。
        
    Returns:
        dict: 深拷贝后的解决方案。
    """
    return copy.deepcopy(solution)

def euclidean(p1, p2):
    """计算两个坐标点 p1 和 p2 之间的欧氏距离。"""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def travel_time(p1, p2):
    """根据欧氏距离和车辆速度，计算两点之间的行驶时间。"""
    return euclidean(p1, p2) / VEHICLE_SPEED

def initialize_vehicles():
    """
    创建并返回一个字典，用于存储所有车辆的初始状态。
    每辆车的状态包括：
    - 'location': 初始位置 (x, y)
    - 'route': 初始为空的行驶路径
    - 'onboard': 初始为空的车上乘客列表
    - 'served': 初始为空的已服务乘客集合
    - 'time': 初始为0的车辆当前时间
    - 'history': 初始为空的历史记录
    """
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

# 路径可行性判断函数
def is_feasible_insert(route, new_order, insert_pick, insert_drop, passengers, vehicle_start):
    """
    检查将一个新订单（new_order）插入到现有路径（route）的指定位置后，新路径是否可行。
    
    Args:
        route (list): 车辆当前的路径，每个元素是一个代表 pickup 或 dropoff 的元组。
        new_order (dict): 包含 'pickup' 和 'dropoff' 节点信息的新订单。
        insert_pick (int): 'pickup' 节点的插入位置索引。
        insert_drop (int): 'dropoff' 节点在插入 'pickup' 后的新路径中的插入位置索引。
        passengers (dict): 所有乘客信息的字典。
        vehicle_start (tuple): 车辆的起始位置。
    
    Returns:
        bool: 如果插入后路径可行，返回 True；否则返回 False。
    
    可行性约束检查包括：
    1. 容量约束：车上乘客数不超过 MAX_CAPACITY。
    2. 时间窗约束：乘客的上车时间不晚于其 'preferred_end'。
    3. 行程顺序约束：必须先上车（pickup）后下车（dropoff）。
    4. 行程时长约束：乘客的实际乘车时间不能超过其最短直达时间的 DELAT 倍。
    """
    # 先构造包含 pickup 点的临时路径
    temp_route_with_pickup = route[:insert_pick] + [new_order['pickup']] + route[insert_pick:]
    # 再在临时路径上插入 dropoff 点，得到最终要检查的新路径
    new_route = temp_route_with_pickup[:insert_drop] + [new_order['dropoff']] + temp_route_with_pickup[insert_drop:]

    onboard = []  # 模拟车上的乘客列表
    time = 0      # 模拟当前时间
    loc = vehicle_start # 模拟当前位置
    # 使用局部字典来追踪本次模拟中的上车时间，避免污染全局状态
    pickup_times_local = {}

    # 遍历模拟的新路径，检查每一步的约束
    for pt in new_route:
        pid, x, y, typ = pt
        passenger_info = passengers[pid]
        
        # 更新时间和位置
        time += travel_time(loc, (x, y))
        loc = (x, y)
        
        if typ == 'pickup':
            # 记录上车时间到局部字典
            pickup_times_local[pid] = time
            onboard.append(pid)
            # 检查上车时间窗
            if time > passenger_info['late_accept']:
                return False
        else: # typ == 'dropoff'
            # 检查是否先上车
            if pid not in onboard:
                return False # 必须先上车才能下车
            
            # 从局部字典获取本次模拟的上车时间
            p_time = pickup_times_local.get(pid)
            # 如果找不到上车时间（例如路径顺序错误），则不可行
            if p_time is None:
                return False
                
            delta = time - p_time
             # 检查最大行程时长约束，依赖 passengers 字典中的 min_travel_time
            if delta > passenger_info['min_travel_time'] * DELAT:
                return False
            onboard.remove(pid)
            
        # 检查容量约束
        if len(onboard) > MAX_CAPACITY:
            return False
            
    return True # 如果所有检查都通过，则路径可行

# 成本与惩罚计算函数
def compute_cost_and_penalty(route, passengers, vehicle_start):
    """
    计算给定路径的总成本，包括行驶成本和各项惩罚。
    
    Args:
        route (list): 要评估的车辆路径。
        passengers (dict): 所有乘客信息的字典。
        vehicle_start (tuple): 车辆的起始位置。
    
    Returns:
        float: 该路径的总成本（行驶成本 + 惩罚）。
    """
    cost = 0      # 初始化行驶成本
    penalty = 0   # 初始化惩罚值
    time = 0      # 模拟当前时间
    loc = vehicle_start # 模拟当前位置
    # 使用局部字典来追踪本次计算中的上车时间
    pickup_times_local = {}

    # 遍历路径中的每个节点
    for pt in route:
        pid, x, y, typ = pt
        dist = euclidean(loc, (x, y))
        travel = dist / VEHICLE_SPEED
        
        # 更新时间和成本
        time += travel
        cost += dist * COST_PER_KM
        
        passenger = passengers[pid]

        if typ == 'pickup':
            # 计算上车延迟惩罚
            delay_from_start = max(0, time - passenger['preferred_start'])
            if time <= passenger['preferred_end']:
                penalty += ALPHA_1 * delay_from_start
            else: # 如果晚于最晚上车时间，惩罚加重
                penalty_in_window = ALPHA_1 * (passenger['preferred_end'] - passenger['preferred_start'])
                penalty_outside_window = ALPHA_2 * (time - passenger['preferred_end'])
                penalty += penalty_in_window + penalty_outside_window
            # 将上车时间存入局部字典，而不是修改外部传入的 passengers 字典
            pickup_times_local[pid] = time
        else: # typ == 'dropoff'
            # 从局部字典获取上车时间
            p_time = pickup_times_local.get(pid)
            if p_time is None:
                # 路径无效（先下车后上车），返回极大值
                return float('inf')
            
            # 计算超长行程惩罚
            delta = time - p_time
            shortest = passenger['min_travel_time']
            if delta < shortest * DELAT:
                penalty += BETA * (delta - shortest)
        
        loc = (x, y)

    # 如果车辆被使用（路径不为空），则增加固定成本
    if route:
        cost += FIXED_COST

    return cost + penalty # 返回总成本

# 评估解决方案的指标 
# 假设有一个计算总成本的辅助函数
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


def save_vehicle_route_animation(vehicle_id, vehicle_path, passenger_points, capacity_states, folder_path):
    """为单个车辆的路径生成并保存GIF动画。"""
    if not vehicle_path:
        print(f"Vehicle {vehicle_id} has no route, skipping animation.")
        return

    # 初始化绘图
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

### 模拟退火算法 (Simulated Annealing, SA) ###


# ====== 模拟退火（SA）专用参数 ======
INITIAL_TEMPERATURE = 500   # 初始温度
COOLING_RATE = 0.98         # 降温速率
MAX_ITER = 50               # 迭代次数


# NOTE: 我们需要一个启发式函数。由于第一段代码没有提供，
# # 我们创建一个简单的、符合其逻辑的启发式函数：成本越低，分数越高。
def calculate_heuristic_forsa(route, passengers_dict, vehicle_start):
    """一个简单的启发式函数，成本的倒数。"""
    cost = compute_cost_and_penalty(route, passengers_dict, vehicle_start)
    # 加1e-6防止除以零
    return 1.0 / (cost + 1e-6)

def initialize_solution(passengers_dict):
    vehicles = initialize_vehicles()

    for t in range(T):
        new_orders = D_t_list[t]
        for order in new_orders:
            pid = order['passenger_id']
            order['pickup'] = (pid, order['pickup_x'], order['pickup_y'], 'pickup')
            order['dropoff'] = (pid, order['dropoff_x'], order['dropoff_y'], 'dropoff')

            best_score = -1
            best_insert = None
            # best_vid 保留原样，虽然在此实现中没有被直接使用，但为了保留原结构而不删除
            best_vid = None

            for vid, v in vehicles.items():
                path = v['route']
                for i in range(len(path) + 1):
                    for j in range(i + 1, len(path) + 2):
                        # 检查插入是否可行
                        if is_feasible_insert(path, order, i, j, passengers_dict, v['location']):
                            # 预览新路径以计算得分
                            temp_route_for_score = path[:i] + [order['pickup']] + path[i:]
                            new_route_for_score = temp_route_for_score[:j] + [order['dropoff']] + temp_route_for_score[j:]
                            
                            score = calculate_heuristic_forsa(new_route_for_score, passengers_dict, v['location'])
                            if score > best_score:
                                best_score = score
                                best_insert = (vid, i, j)

            if best_insert:
                vid, i, j = best_insert
                v = vehicles[vid]
                # FIX 1: 修复了原代码中致命的路径更新错误。
                # 原代码: v['route'][:i:j-1] 会导致路径点丢失。
                temp_route = v['route'][:i] + [order['pickup']] + v['route'][i:]
                v['route'] = temp_route[:j] + [order['dropoff']] + temp_route[j:]
                v['served'].add(pid)

    return vehicles

def copy_solution(solution):
    # WARNING: 这是一个浅拷贝实现，保留自第一段代码。
    # 对于当前数据结构是安全的，但如果 history 包含复杂对象则有风险。
    return {vid: {
        'location': v['location'],
        'route': list(v['route']),
        'onboard': list(v['onboard']),
        'served': set(v['served']),
        'time': v['time'],
        'history': list(v['history'])
    } for vid, v in solution.items()}

def evaluate_total_cost(solution, passengers_dict):
    total = 0
    for vid, v in solution.items():
        total += compute_cost_and_penalty(v['route'], passengers_dict, v['location'])
    return total

def perturb_solution(solution, passengers_dict):
    vids = list(solution.keys())
    # 增加一个保护，防止车辆列表为空
    if not vids:
        return solution
        
    v1 = random.choice(vids)
    path = solution[v1]['route']

    if len(path) < 2: # 至少需要一个乘客（2个点）才能扰动
        return solution

    pids = list(set([pt[0] for pt in path]))
    if not pids:
        return solution
        
    random.shuffle(pids)

    for pid in pids:
        indices = [i for i, pt in enumerate(path) if pt[0] == pid]
        if len(indices) != 2:
            continue
        
        # 确保 pickup 在 dropoff 之前
        i1, i2 = sorted(indices)

        # 生成移除该乘客后的路径
        path_wo = path[:i1] + path[i1+1:i2] + path[i2+1:]
        
        insert_positions = list(range(len(path_wo) + 1))
        random.shuffle(insert_positions)

        for i in insert_positions:
            # Dropoff位置必须在Pickup之后
            for j in range(i + 1, len(path_wo) + 2):
                # FIX 2: 修复了原代码中“先构建再检查”的逻辑错误。
                # 现在是“先检查可行性，再构建路径”。
                temp_order_info = {'pickup': path[i1], 'dropoff': path[i2]}
                if is_feasible_insert(path_wo, temp_order_info, i, j, passengers_dict, solution[v1]['location']):
                    # 如果可行，则构建新路径并立即返回，保留“首次可行”的策略
                    temp_route = path_wo[:i] + [path[i1]] + path_wo[i:]
                    new_path = temp_route[:j] + [path[i2]] + temp_route[j:]
                    solution[v1]['route'] = new_path
                    return solution

    return solution

def simulated_annealing():
    # 乘客字典的创建方式保留原样
    passengers_dict = {row['passenger_id']: row for _, row in passengers_df.iterrows()}
    
    current_solution = initialize_solution(passengers_dict)
    best_solution = copy_solution(current_solution)
    # FIX 3: 引入 current_cost 并正确计算 delta，这是SA算法的核心修正
    current_cost = evaluate_total_cost(current_solution, passengers_dict)
    best_cost = current_cost

    T_now = INITIAL_TEMPERATURE
    for iter in range(MAX_ITER):
        new_solution = perturb_solution(copy_solution(current_solution), passengers_dict)
        new_cost = evaluate_total_cost(new_solution, passengers_dict)
        
        # FIX 3 (续): 使用 current_cost 而不是 best_cost 来计算 delta
        delta = new_cost - current_cost
        
        accepted = False # 保留原有的 accepted 标志
        if delta < 0 or random.random() < np.exp(-delta / T_now):
            current_solution = new_solution
            current_cost = new_cost # FIX 3 (续): 更新 current_cost
            accepted = True # 更新接受状态
            
            # 更新最优解的逻辑保留原样
            if new_cost < best_cost:
                best_solution = copy_solution(new_solution)
                best_cost = new_cost
        
        # 打印信息的格式和内容完全保留原样
        print(f"Iter {iter+1:03d}: Cost = {new_cost:.2f}, Best = {best_cost:.2f}, Temp = {T_now:.2f}, Accepted = {accepted}")
        T_now *= COOLING_RATE

    return best_solution

### 主函数更新 (添加新算法)
def main():
    """程序主入口函数。"""
    # 定义要运行的算法，包括新增的三种
    algorithms = {
        'Simulated Annealing': simulated_annealing
    }

    for pid, p in passengers_dict.items():
        p['min_travel_time'] = travel_time((p['pickup_x'], p['pickup_y']), (p['dropoff_x'], p['dropoff_y']))

    # 依次运行每种算法
    for name, algorithm_fn in algorithms.items():
        print(f"\n=== Running {name} ===")
        result = algorithm_fn()
        
        if not result:
            print(f"{name} did not return a valid solution.")
            continue

        # 评估最终结果
        metrics = evaluate_solution(result, passengers_dict)

        # 创建结果文件夹（已修改为 jingtai/results/）
        folder_name = f"jingtai/results/{name.lower().replace(' ', '_')}"
        os.makedirs(folder_name, exist_ok=True)

        # 将结果摘要写入文件
        summary_path = os.path.join(folder_name, 'summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"{name} Results Summary:\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")
            f.write("\n" + "=" * 50 + "\n\n")

            # 为指定的车辆生成路径详情和动画
            for vid in TARGET_VEHICLES:
                if vid not in result:
                    continue
                v = result[vid]
                path = v['route']
                
                if not path:
                    continue

                onboard_pids = []
                capacity_states = []

                f.write(f"Vehicle {vid} Route:\n")
                for pt in path:
                    pid, x, y, typ = pt
                    if typ == 'pickup':
                        onboard_pids.append(pid)
                    elif typ == 'dropoff':
                        if pid in onboard_pids:
                            onboard_pids.remove(pid)
                    capacity_states.append(len(onboard_pids))
                    f.write(f"  - {typ.capitalize()} P{pid} at ({x:.2f}, {y:.2f}), Onboard: {len(onboard_pids)}\n")
                f.write("\n")

                # 保存动画
                passenger_points_for_anim = {p_id: passengers_dict[p_id] for p_id in v['served']}
                save_vehicle_route_animation(
                    vehicle_id=vid,
                    vehicle_path=path,
                    passenger_points=passenger_points_for_anim,
                    capacity_states=capacity_states,
                    folder_path=folder_name
                )

                f.write(f"动画保存为: vehicle_{vid}.gif\n")
                f.write("\n" + "=" * 50 + "\n\n")

        print(f"{name} 已完成，结果保存至 {folder_name}/")

if __name__ == '__main__':
    main()
