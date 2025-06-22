import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib.patches as mpatches
import os
import math
import copy
from tqdm import tqdm


# ==================== 参数定义 ====================
# 问题参数
TIME_INTERVAL = 5
MAX_CAPACITY = 4
VEHICLE_SPEED = 0.67  # km/min
COST_PER_KM = 1.0
FIXED_COST = 5.0
ALPHA_1 = 2.0  # 乘客早到惩罚
ALPHA_2 = 5.0  # 乘客晚到惩罚
BETA = 5.0     # 乘客绕路惩罚
DELAT = 3.0    # 绕路时间容忍系数

# 遗传算法参数
POPULATION_SIZE = 30       # 种群大小
NUM_GENERATIONS = 50      # 迭代代数
CROSSOVER_RATE = 0.85      # 交叉概率
MUTATION_RATE = 0.1        # 变异概率
TOURNAMENT_SIZE = 5        # 锦标赛选择中竞争者的数量
ELITISM_SIZE = 2           # 精英主义，保留最优个体的数量

# 指定要输出的车辆ID列表（修改此处选择需要的车辆）
TARGET_VEHICLES = [0, 1, 2, 3]

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
            'served': set()
        }
    return state

# 预处理所有乘客数据，并创建 passengers_dict
passengers_dict = {row['passenger_id']: row.to_dict() for _, row in passengers_df.iterrows()}
for pid, p in passengers_dict.items():
    p['pickup'] = (pid, p['pickup_x'], p['pickup_y'], 'pickup')
    p['dropoff'] = (pid, p['dropoff_x'], p['dropoff_y'], 'dropoff')
    p['min_travel_time'] = travel_time((p['pickup_x'], p['pickup_y']), (p['dropoff_x'], p['dropoff_y']))

# 从处理好的 passengers_dict 中创建 all_orders 列表
all_orders = list(passengers_dict.values())


# ==================== 目标函数与惩罚计算 ====================
def compute_cost_and_penalty(route, passengers, vehicle_start):
    cost = 0
    penalty = 0
    onboard = set()
    time = 0
    loc = vehicle_start
    
    # 创建一个临时字典来存储乘客的实际上车时间
    temp_passengers_info = {pid: {} for pid in {pt[0] for pt in route}}

    for pt in route:
        pid, x, y, typ = pt
        dist = euclidean(loc, (x, y))
        travel = dist / VEHICLE_SPEED
        time += travel
        cost += dist * COST_PER_KM
        
        passenger = passengers[pid]
        
        if typ == 'pickup':
            if time > passenger['preferred_start']:
                penalty += ALPHA_1 * max(time - passenger['preferred_start'], 0)
            if time > passenger['preferred_end']:
                penalty += ALPHA_2 * max(time - passenger['preferred_end'], 0)
            temp_passengers_info[pid]['pickup_time'] = time
            onboard.add(pid)
        else: # dropoff
            # 检查乘客是否真的上车了
            if 'pickup_time' not in temp_passengers_info[pid]:
                # 这种情况在非法路径中可能发生，给予巨大惩罚
                penalty += 10000 
                continue
            
            delta = time - temp_passengers_info[pid]['pickup_time']
            shortest = passenger['min_travel_time']
            if delta <= shortest * DELAT:
                penalty += BETA * (delta - shortest)
            onboard.discard(pid)
        
        loc = (x, y)

    if route:
        cost += FIXED_COST

    return cost + penalty

def calculate_total_fitness(individual, passengers):
    total_cost = 0
    for vid, v_data in individual.items():
        total_cost += compute_cost_and_penalty(v_data['route'], passengers, v_data['location'])
    # 适应度是成本的倒数，成本越小，适应度越高
    return 1.0 / (total_cost + 1e-6)


# ==================== 插入可行性判断 ====================
def is_feasible_insert(route, new_order, insert_pick_idx, insert_drop_idx, passengers, vehicle_start):
    # 模拟插入
    temp_route = list(route)
    temp_route.insert(insert_pick_idx, new_order['pickup'])
    temp_route.insert(insert_drop_idx + 1, new_order['dropoff'])
    
    onboard = set()
    time = 0
    loc = vehicle_start
    
    temp_pickup_times = {}

    for pt in temp_route:
        pid, x, y, typ = pt
        
        # 检查载客量
        if typ == 'pickup':
            onboard.add(pid)
            if len(onboard) > MAX_CAPACITY:
                return False
        
        travel = travel_time(loc, (x, y))
        time += travel
        loc = (x, y)

        # 检查时间窗和绕路
        if typ == 'pickup':
            # 记录上车时间
            temp_pickup_times[pid] = time
            # 检查上车时间是否晚于乘客的最晚出发时间
            if time > passengers[pid]['late_accept']:
                return False
        else: # dropoff
            if pid not in temp_pickup_times: # Dropoff before pickup
                return False
            
            pickup_time = temp_pickup_times[pid]
            delta = time - pickup_time
            if delta > passengers[pid]['min_travel_time'] * DELAT:
                return False
            onboard.remove(pid)
            
    return True

# ==================== 遗传算法核心 ====================

def find_best_insertion(route, order, passengers, vehicle_start):
    """为单个订单找到在单个车辆路径中的最佳插入位置"""
    best_cost_increase = float('inf')
    best_insertion = None

    for i in range(len(route) + 1):
        for j in range(i, len(route) + 1):
            if is_feasible_insert(route, order, i, j, passengers, vehicle_start):
                temp_route = list(route)
                temp_route.insert(i, order['pickup'])
                temp_route.insert(j + 1, order['dropoff'])
                
                new_cost = compute_cost_and_penalty(temp_route, passengers, vehicle_start)
                original_cost = compute_cost_and_penalty(route, passengers, vehicle_start)
                cost_increase = new_cost - original_cost

                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_insertion = (i, j + 1) # 返回插入pickup和dropoff的位置
    return best_insertion, best_cost_increase


def create_initial_population(size, orders, passengers):
    """使用贪心策略创建初始种群"""
    population = []
    for _ in range(size):
        individual = initialize_vehicles()
        
        # 随机打乱订单顺序，增加初始种群多样性
        random.shuffle(orders)
        
        for order in orders:
            pid = order['passenger_id']
            best_vid = -1
            best_insertion_points = None
            min_cost_increase = float('inf')

            for vid, v_data in individual.items():
                insertion_points, cost_increase = find_best_insertion(v_data['route'], order, passengers, v_data['location'])
                if insertion_points and cost_increase < min_cost_increase:
                    min_cost_increase = cost_increase
                    best_vid = vid
                    best_insertion_points = insertion_points
            
            if best_vid != -1:
                v_data = individual[best_vid]
                i, j = best_insertion_points
                v_data['route'].insert(i, order['pickup'])
                v_data['route'].insert(j, order['dropoff'])
                v_data['served'].add(pid)
        
        population.append(individual)
    return population

def selection(population, fitnesses):
    """锦标赛选择"""
    selected = []
    for _ in range(2): # 选择两个父代
        tournament = random.sample(list(zip(population, fitnesses)), TOURNAMENT_SIZE)
        tournament.sort(key=lambda x: x[1], reverse=True) # 按适应度降序
        selected.append(tournament[0][0])
    return selected[0], selected[1]

def crossover(parent1, parent2, orders, passengers):
    """基于路线的交叉算子"""
    child = initialize_vehicles()
    
    # 随机选择一个交叉点（车辆ID）
    vehicle_ids = list(parent1.keys())
    crossover_point = random.randint(1, len(vehicle_ids) - 1)
    
    p1_vehicles = vehicle_ids[:crossover_point]
    p2_vehicles = vehicle_ids[crossover_point:]
    
    served_by_child = set()

    # 1. 从父代1继承部分车辆的路线
    for vid in p1_vehicles:
        child[vid] = copy.deepcopy(parent1[vid])
        served_by_child.update(parent1[vid]['served'])

    # 2. 从父代2获取未被服务的订单
    unserved_orders = []
    for order in orders:
        if order['passenger_id'] not in served_by_child:
            unserved_orders.append(order)
            
    # 3. 将这些未服务的订单贪心地插入到子代的路线中
    for order in unserved_orders:
        pid = order['passenger_id']
        best_vid = -1
        best_insertion_points = None
        min_cost_increase = float('inf')

        for vid, v_data in child.items():
            insertion_points, cost_increase = find_best_insertion(v_data['route'], order, passengers, v_data['location'])
            if insertion_points and cost_increase < min_cost_increase:
                min_cost_increase = cost_increase
                best_vid = vid
                best_insertion_points = insertion_points
        
        if best_vid != -1:
            v_data = child[best_vid]
            i, j = best_insertion_points
            v_data['route'].insert(i, order['pickup'])
            v_data['route'].insert(j, order['dropoff'])
            v_data['served'].add(pid)
            
    return child


def mutate(individual, orders, passengers):
    """变异算子：包含多种策略"""
    mutated_individual = copy.deepcopy(individual)
    
    # 随机选择一种变异策略
    mutation_type = random.choice(['reinsert_order', 'swap_vehicles'])

    if mutation_type == 'reinsert_order' and sum(len(v['served']) for v in mutated_individual.values()) > 0:
        # 策略1：重新插入一个订单
        # 随机选择一辆有任务的车和一个订单
        non_empty_vehicles = [vid for vid, v in mutated_individual.items() if v['served']]
        if not non_empty_vehicles: return mutated_individual
        
        vid = random.choice(non_empty_vehicles)
        pid = random.choice(list(mutated_individual[vid]['served']))
        order = passengers[pid]
        
        # 从路径中移除该订单
        mutated_individual[vid]['route'] = [pt for pt in mutated_individual[vid]['route'] if pt[0] != pid]
        mutated_individual[vid]['served'].remove(pid)
        
        # 贪心法重新插入该订单到任意车辆
        best_vid, best_insertion, min_cost = -1, None, float('inf')
        for new_vid, v_data in mutated_individual.items():
            insertion_points, cost_increase = find_best_insertion(v_data['route'], order, passengers, v_data['location'])
            if insertion_points and cost_increase < min_cost:
                min_cost = cost_increase
                best_vid = new_vid
                best_insertion = insertion_points
        
        if best_vid != -1:
            v_data = mutated_individual[best_vid]
            i, j = best_insertion
            v_data['route'].insert(i, order['pickup'])
            v_data['route'].insert(j, order['dropoff'])
            v_data['served'].add(pid)
            
    elif mutation_type == 'swap_vehicles' and sum(len(v['served']) for v in mutated_individual.values()) > 1:
        # 策略2：将一个订单从一辆车移动到另一辆
        vids_with_orders = [vid for vid, v in mutated_individual.items() if v['served']]
        if len(vids_with_orders) < 1: return mutated_individual
        
        vid1 = random.choice(vids_with_orders)
        pid = random.choice(list(mutated_individual[vid1]['served']))
        order = passengers[pid]

        # 从路径中移除
        mutated_individual[vid1]['route'] = [pt for pt in mutated_individual[vid1]['route'] if pt[0] != pid]
        mutated_individual[vid1]['served'].remove(pid)
        
        # 尝试插入到另一辆车
        other_vids = list(mutated_individual.keys())
        other_vids.remove(vid1)
        if not other_vids: return mutated_individual # 如果只有一辆车，无法交换
        
        vid2 = random.choice(other_vids)
        insertion, _ = find_best_insertion(mutated_individual[vid2]['route'], order, passengers, mutated_individual[vid2]['location'])
        if insertion:
            i, j = insertion
            mutated_individual[vid2]['route'].insert(i, order['pickup'])
            mutated_individual[vid2]['route'].insert(j, order['dropoff'])
            mutated_individual[vid2]['served'].add(pid)
        else: # 如果无法插入，则恢复原状
            return individual
            
    return mutated_individual



def genetic_algorithm():
    #1.初始化种群
    population = create_initial_population(POPULATION_SIZE, all_orders, passengers_dict)
    
    best_solution = None
    best_fitness = -1.0
    
    #  修改主循环以使用tqdm
    with tqdm(range(NUM_GENERATIONS), desc="遗传算法演进中") as pbar:
        for generation in pbar:
            #2.计算适应度
            fitnesses = [calculate_total_fitness(ind, passengers_dict) for ind in population]
            #记录当前代最优解
            current_best_idx = np.argmax(fitnesses)
            if fitnesses[current_best_idx] > best_fitness:
                best_fitness = fitnesses[current_best_idx]
                best_solution = copy.deepcopy(population[current_best_idx])
            #3.生成下一代
            new_population = []
            #精英主义
            elite_indices = np.argsort(fitnesses)[-ELITISM_SIZE:]
            for i in elite_indices:
                new_population.append(copy.deepcopy(population[i]))
            #填充剩余种群   
            while len(new_population) < POPULATION_SIZE:
                #选择
                parent1, parent2 = selection(population, fitnesses)
                #交叉
                if random.random() < CROSSOVER_RATE:
                    child = crossover(parent1, parent2, all_orders, passengers_dict)
                else:
                    child = copy.deepcopy(parent1)
                
                #变异
                if random.random() < MUTATION_RATE:
                    child = mutate(child, all_orders, passengers_dict)
                    
                new_population.append(child)
                
            population = new_population
            
            #  更新进度条的后缀信息，而不是打印
            best_cost = (1.0 / best_fitness) if best_fitness > 0 else float('inf')
            pbar.set_postfix(best_cost=f'{best_cost:.2f}')

    # 输出最终最优解的指标
    print("\n最优解指标:")
    best_metrics = evaluate_solution(best_solution, passengers_dict)
    # 确保 value 是数值类型
    for key, value in best_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
            
    return best_solution



# ==================== 评估与可视化 (与原版相同) ====================

# 新增的辅助函数，用于计算总成本
def evaluate_total_cost(solution, passengers_dict):
    """计算一个完整解决方案的总成本（所有车辆成本之和）。"""
    total = 0
    for vid, v in solution.items():
        total += compute_cost_and_penalty(v['route'], passengers_dict, v['location'])
    return total


def evaluate_solution(vehicles, passengers):
    """
    评估一个完整解决方案（所有车辆的路径）的各项性能指标（KPIs）。
    [这是来自代码2的增强版本]
    """
    total_distance = 0
    total_waiting_time = 0
    total_travel_time = 0
    
    # 使用集合来统计实际服务的乘客，更可靠。
    all_served_pids = set()
    
    # 新增：用于计算平均容量利用率的变量
    total_capacity_used = 0.0
    total_capacity_time = 0.0
    
    used_vehicles = 0
    
    # 创建一个局部字典来存储上车时间，避免修改全局的 passengers 字典。
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
                
                # 将上车时间存入局部字典
                pickup_times_local[pid] = time 
                onboard.append(pid)
            else: # typ == 'dropoff'
                # 从局部字典安全地获取上车时间
                if pid in pickup_times_local:
                    ride_time = time - pickup_times_local[pid]
                    total_travel_time += ride_time
                if pid in onboard:
                    onboard.remove(pid)
        
    # 用集合的大小作为服务乘客数
    served_passengers = len(all_served_pids)
    
    # 在所有路径信息都处理完毕后，一次性计算总成本
    total_cost_val = evaluate_total_cost(vehicles, passengers)

    # 计算平均指标
    avg_waiting_time = total_waiting_time / served_passengers if served_passengers else 0
    avg_travel_time = total_travel_time / served_passengers if served_passengers else 0
    
    # 新增：计算平均容量利用率，结果是一个0-1的比率
    avg_capacity_util = (total_capacity_used / MAX_CAPACITY) / total_capacity_time if total_capacity_time > 0 else 0
    
    unit_cost = total_cost_val / served_passengers if served_passengers else 0
    vehicle_utilization = used_vehicles / len(vehicles) if vehicles else 0

    return {
        'served_passengers': served_passengers,
        'avg_waiting_time': avg_waiting_time,
        'avg_travel_time': avg_travel_time,
        'total_distance': total_distance,
        'total_cost': total_cost_val,
        'unit_cost': unit_cost,
        'vehicle_utilization': vehicle_utilization,
        'avg_capacity_util': avg_capacity_util  # 新增的返回指标
    }

def save_vehicle_route_animation(vehicle_id, vehicle_path, passenger_points, capacity_states, gif_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    all_x = [p[1] for p in vehicle_path] + [p['pickup_x'] for p in passenger_points.values()] + [p['dropoff_x'] for p in passenger_points.values()]
    all_y = [p[2] for p in vehicle_path] + [p['pickup_y'] for p in passenger_points.values()] + [p['dropoff_y'] for p in passenger_points.values()]
    
    ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
    ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
    ax.set_title(f"Vehicle {vehicle_id} Route Animation")
    
    line, = ax.plot([], [], 'k-', lw=2, zorder=1)
    vehicle_dot, = ax.plot([], [], 'ro', markersize=10, zorder=3, label='Vehicle') # 车辆当前位置
    
    pid_list = sorted(list(set([pt[0] for pt in vehicle_path])))
    color_map = {pid: plt.cm.tab20(i % 20) for i, pid in enumerate(pid_list)}

    # 预先绘制所有乘客的起终点
    for pid, p in passenger_points.items():
        color = color_map.get(pid, 'gray')
        ax.scatter(p['pickup_x'], p['pickup_y'], c=[color], marker='o', s=80, edgecolors='k', label=f'P{pid} Pickup' if pid in pid_list else None)
        ax.scatter(p['dropoff_x'], p['dropoff_y'], c=[color], marker='v', s=80, edgecolors='k', label=f'P{pid} Dropoff' if pid in pid_list else None)
        ax.text(p['pickup_x'], p['pickup_y'], f'P{pid}', fontsize=9, ha='right')
        ax.text(p['dropoff_x'], p['dropoff_y'], f'P{pid}', fontsize=9, ha='right')
    
    # 创建图例
    legend_patches = [mpatches.Patch(color=color_map[pid], label=f'P{pid}') for pid in pid_list]
    ax.legend(handles=legend_patches, title="Passengers", loc='upper right', fontsize='small')
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    capacity_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    
    path_points = [vehicles_df.iloc[vehicle_id][['start_x', 'start_y']].values] + [(p[1], p[2]) for p in vehicle_path]
    times = [0.0]
    current_time = 0
    for i in range(len(path_points) - 1):
        current_time += travel_time(path_points[i], path_points[i+1])
        times.append(current_time)

    def update(frame):
        # 绘制已走过的路径
        line.set_data([p[0] for p in path_points[:frame+1]], [p[1] for p in path_points[:frame+1]])
        
        # 更新车辆位置
        vehicle_dot.set_data([path_points[frame][0]], [path_points[frame][1]])
        
        # 更新文本信息
        time_text.set_text(f'Time: {times[frame]:.2f} min')
        if frame > 0:
             capacity_text.set_text(f'Onboard: {capacity_states[frame-1]}/{MAX_CAPACITY}')
        else:
             capacity_text.set_text('Onboard: 0/4')
             
        return line, vehicle_dot, time_text, capacity_text

    ani = animation.FuncAnimation(fig, update, frames=len(path_points), interval=500, blit=True)
    ani.save(gif_path, writer=PillowWriter(fps=2))
    plt.close(fig)

# ==================== 主程序入口 ====================
# ==================== 主程序入口 ====================
if __name__ == '__main__':
    result = genetic_algorithm()

    if result:
        # 使用新的评估函数计算最终指标
        metrics = evaluate_solution(result, passengers_dict)

        # ==================== 路径修改 ====================
        output_dir = 'jingtai/results/genetic_algorithm'
        os.makedirs(output_dir, exist_ok=True)
        # =================================================

        summary_file_path = os.path.join(output_dir, 'result_summary_ga.txt')
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            f.write("全局优化指标 (遗传算法):\n")
            f.write(f"  服务乘客数: {metrics['served_passengers']}\n")
            f.write(f"  平均等待时间: {metrics['avg_waiting_time']:.2f} 分钟\n")
            f.write(f"  平均行程时间: {metrics['avg_travel_time']:.2f} 分钟\n")
            f.write(f"  总行驶距离: {metrics['total_distance']:.2f} 公里\n")
            f.write(f"  总调度成本: {metrics['total_cost']:.2f} 元\n")
            f.write(f"  单位服务成本: {metrics['unit_cost']:.2f} 元/人\n")
            f.write(f"  车辆使用率: {metrics['vehicle_utilization']:.2%}\n")
            f.write(f"  平均容量利用率: {metrics['avg_capacity_util']:.2%}\n")
            f.write("\n" + "="*50 + "\n\n")

            # 输出每辆目标车辆的路径与动画
            for vid in TARGET_VEHICLES:
                if vid not in result or not result[vid]['route']:
                    f.write(f"车辆 {vid} 不存在或未分配任何乘客\n\n")
                    continue

                v = result[vid]
                path = v['route']
                onboard = set()
                capacity = []

                f.write(f"Vehicle {vid} 路线总结:\n")
                for pt in path:
                    pid, x, y, typ = pt
                    if typ == 'pickup':
                        onboard.add(pid)
                    elif typ == 'dropoff' and pid in onboard:
                        onboard.remove(pid)
                    capacity.append(len(onboard))
                    f.write(f"  {typ.upper():<8} P{pid:<3} at ({x:5.2f}, {y:5.2f}), 车内乘客数: {len(onboard)}\n")
                f.write("\n")

                served_ids = set([pt[0] for pt in path])
                passenger_points_filtered = {pid: passengers_dict[pid] for pid in served_ids}

                gif_filename = f'vehicle_{vid}_ga.gif'
                gif_path = os.path.join(output_dir, gif_filename)
                save_vehicle_route_animation(
                    vehicle_id=vid,
                    vehicle_path=path,
                    passenger_points=passenger_points_filtered,
                    capacity_states=capacity,
                    gif_path=gif_path
                )

                f.write(f"车辆 {vid} 的路线动画已保存为: {gif_path}\n")
                f.write("\n" + "="*50 + "\n\n")
        
        print(f"调度结果已保存至 {summary_file_path}")
        print(f"已为车辆 {TARGET_VEHICLES} 生成路线总结和可视化图像。")

        print("\n最终评估指标:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    else:
        print("遗传算法未能找到有效解决方案。")
