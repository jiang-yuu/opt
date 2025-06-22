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
ALPHA_1 = 2.0  # 惩罚1
ALPHA_2 = 5.0  # 晚到惩罚
BETA = 5.0     # 绕路惩罚
DELAT = 3.0    # 绕路时间容忍系数

# 遗传算法重优化参数
GA_REOPT_ENABLED = True
GA_REOPT_POP_SIZE = 30
GA_REOPT_GENERATIONS = 50 
GA_REOPT_CROSSOVER_RATE = 0.85
GA_REOPT_MUTATION_RATE = 0.15 # 稍微增加变异率以探索更多可能性
GA_REOPT_TOURNAMENT_SIZE = 5
GA_REOPT_ELITISM_SIZE = 2

# 指定要输出的车辆ID列表
TARGET_VEHICLES = [0, 1, 2, 3]

# ==================== 数据读取 ====================
vehicles_df = pd.read_csv('dongtai/data/vehicles5.csv')
passengers_df = pd.read_csv('dongtai/data/passengers5.csv')

total_time = passengers_df['request_time'].max()
buffer_time_steps = int(np.ceil(max(200, total_time) / TIME_INTERVAL)) # 增加更长的收尾时间
T = int(np.ceil(total_time / TIME_INTERVAL)) + buffer_time_steps

D_t_list = [[] for _ in range(T)]
for _, row in passengers_df.iterrows():
    t_idx = int(row['request_time'] // TIME_INTERVAL)
    if t_idx < T: D_t_list[t_idx].append(row['passenger_id'])

# ==================== 核心函数定义 ====================

# --- 基本函数 ---
def euclidean(p1, p2): return np.linalg.norm(np.array(p1) - np.array(p2))
def travel_time(p1, p2): return euclidean(p1, p2) / VEHICLE_SPEED

# --- 全局乘客字典 ---
passengers_dict = {row['passenger_id']: row.to_dict() for _, row in passengers_df.iterrows()}
for pid, p in passengers_dict.items():
    p['pickup'] = (pid, p['pickup_x'], p['pickup_y'], 'pickup')
    p['dropoff'] = (pid, p['dropoff_x'], p['dropoff_y'], 'dropoff')
    p['min_travel_time'] = travel_time((p['pickup_x'], p['pickup_y']), (p['dropoff_x'], p['dropoff_y']))

# --- 成本计算函数 ---

def compute_cost_and_penalty_dynamic(route, passengers, vehicle_current_loc, vehicle_current_time, initial_onboard_pids):
    """
    计算一个路线的成本和惩罚。
    使用最终版、基于您最新解释的、最精确的分段线性惩罚逻辑。
    """
    cost, penalty, time, loc = 0.0, 0.0, vehicle_current_time, vehicle_current_loc
    
    # 初始化一个临时的上车时间字典，用于在本次“想象”中追踪
    temp_pickup_times = {}
    for pid in initial_onboard_pids:
        # 如果乘客是初始就在车上的，从全局信息中获取其真实上车时间
        if 'actual_pickup_time' in passengers[pid]:
            temp_pickup_times[pid] = passengers[pid]['actual_pickup_time']

    for pt in route:
        pid, x, y, typ = pt
        dist = euclidean(loc, (x, y))
        travel = dist / VEHICLE_SPEED
        time += travel
        cost += dist * COST_PER_KM
        loc = (x, y)
        passenger = passengers[pid]

        if typ == 'pickup':
            # --- 【最终版、最精确的惩罚逻辑】 ---
            # 基于：preferred_start 是下单时间，time 永远大于它。
            
            # 1. 计算偏离理想开始时间的延迟
            delay_from_start = time - passenger['preferred_start']

            # 2. 判断是否超过了期望的结束时间
            if time <= passenger['preferred_end']:
                # 场景A: 在理想窗口 [preferred_start, preferred_end] 内
                # 惩罚以 ALPHA_1 的速率随延迟增加
                penalty += ALPHA_1 * delay_from_start
            else:
                # 场景B: 超过了理想窗口 (time > preferred_end)
                # 这是一个更平滑、更有效的惩罚模型：
                # a) 首先，计算在理想窗口内产生的最大惩罚
                penalty_in_window = ALPHA_1 * (passenger['preferred_end'] - passenger['preferred_start'])
                # b) 然后，计算超出窗口部分的、更严厉的惩罚
                penalty_outside_window = ALPHA_2 * (time - passenger['preferred_end'])
                # c) 总惩罚是两部分之和
                penalty += penalty_in_window + penalty_outside_window
            
            temp_pickup_times[pid] = time
        
        else: # typ == 'dropoff'
            if pid not in temp_pickup_times:
                penalty += 10000; continue
            
            delta = time - temp_pickup_times[pid]
            shortest = passenger['min_travel_time']
            if delta <= shortest * DELAT:
                penalty += BETA * (delta - shortest)
            
    if route:
        cost += FIXED_COST
        
    return cost + penalty


# -----可行性------
def is_route_feasible_pure(route_to_check, passengers, start_loc, start_time, initial_onboard_pids):
    """
    检查一个给定的路线片段对于一个给定的起始状态是否自洽和可行。
    这是一个纯函数，所有依赖都通过参数传入。
    """
    onboard_pids = set(initial_onboard_pids)
    time = start_time
    loc = start_loc
    
    temp_pickup_times = {}

    for pt in route_to_check:
        pid, x, y, typ = pt
        travel = travel_time(loc, (x, y))
        time += travel
        loc = (x, y)
        passenger = passengers[pid]

        if typ == 'pickup':
            onboard_pids.add(pid)
            if len(onboard_pids) > MAX_CAPACITY: return False
            if time > passenger['late_accept']: return False
            temp_pickup_times[pid] = time
        else: # typ == 'dropoff'
            if pid not in onboard_pids: return False
            
            actual_pickup_time = -1
            if pid in temp_pickup_times:
                # 场景A: 乘客是在这个规划路线中被接上的
                actual_pickup_time = temp_pickup_times[pid]
            elif 'actual_pickup_time' in passenger:
                # 场景B: 乘客是初始就在车上的，从全局信息获取其真实上车时间
                actual_pickup_time = passenger['actual_pickup_time']

            # 如果能找到上车时间，就必须检查绕路约束
            if actual_pickup_time != -1:
                delta = time - actual_pickup_time
                if delta > passenger['min_travel_time'] * DELAT: 
                    return False
            else:
                # 如果连全局信息里都没有上车时间（理论上不应发生），判定为不可行
                return False

            onboard_pids.discard(pid)
            
    return True


# --- 为GA服务的插入函数 ---

def find_best_insertion_for_ga(route, order_id, passengers, start_loc, start_time, initial_onboard_pids):
    """
    为GA内部的规划寻找最佳插入位置，使用纯粹的可行性检查。
    """
    best_cost_increase, best_insertion = float('inf'), None
    new_order = passengers[order_id]
    pickup_point, dropoff_point = new_order['pickup'], new_order['dropoff']
    

    original_cost = compute_cost_and_penalty_dynamic(route, passengers, start_loc, start_time, initial_onboard_pids)

    for i in range(len(route) + 1):
        temp_route_after_pickup = route[:i] + [pickup_point] + route[i:]
        for j in range(i + 1, len(temp_route_after_pickup) + 1):
            candidate_route = temp_route_after_pickup[:j] + [dropoff_point] + temp_route_after_pickup[j:]
            
            if is_route_feasible_pure(candidate_route, passengers, start_loc, start_time, initial_onboard_pids):
                new_cost = compute_cost_and_penalty_dynamic(candidate_route, passengers, start_loc, start_time, initial_onboard_pids)
                cost_increase = new_cost - original_cost
                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_insertion = (i, j)

    return best_insertion, best_cost_increase


# --- 遗传算法核心算子 ---
def create_initial_individual(vehicles_state, passengers, orders_to_plan):
    """为GA创建一个单独的、可行的初始个体（解决方案）"""
    individual_routes = [[] for _ in vehicles_state]
    shuffled_orders = random.sample(orders_to_plan, len(orders_to_plan))
    
    for order_id in shuffled_orders:
        best_vid_idx, best_insertion, min_cost_increase = -1, None, float('inf')
        for vid_idx, route in enumerate(individual_routes):
            vid = sorted(vehicles_state.keys())[vid_idx]
            v_data = vehicles_state[vid]
            # 调用新的、为GA定制的插入函数，并传入正确的初始状态
            insertion, cost_increase = find_best_insertion_for_ga(
                route, order_id, passengers, v_data['current_location'], v_data['current_time'], v_data['onboard']
            )
            if insertion and cost_increase < min_cost_increase:
                min_cost_increase, best_vid_idx, best_insertion = cost_increase, vid_idx, insertion
        
        if best_vid_idx != -1:
            i, j = best_insertion
            route_to_modify = individual_routes[best_vid_idx]
            new_route = route_to_modify[:i] + [passengers[order_id]['pickup']] + route_to_modify[i:]
            new_route = new_route[:j] + [passengers[order_id]['dropoff']] + new_route[j:]
            individual_routes[best_vid_idx] = new_route
            
    return individual_routes

def crossover_dynamic(parent1, parent2, vehicles_state, passengers):
    """健壮的交叉算子，确保子代可行"""
    child_routes = [[] for _ in parent1]
    
    # 1. 随机选择交叉点，继承父代1的部分基因
    crossover_point = random.randint(1, len(parent1) - 1) if len(parent1) > 1 else 0
    child_routes[:crossover_point] = copy.deepcopy(parent1[:crossover_point])
    
    # 2. 收集所有已规划和未规划的订单
    planned_orders = {pt[0] for route in child_routes for pt in route}
    orders_from_parents = {pt[0] for route in parent1 for pt in route}
    orders_from_parents.update({pt[0] for route in parent2 for pt in route})
    unplanned_orders = list(orders_from_parents - planned_orders)
    
    # 3. 将未规划的订单贪心插入到子代中
    shuffled_unplanned = random.sample(unplanned_orders, len(unplanned_orders))
    for order_id in shuffled_unplanned:
        best_vid_idx, best_insertion, min_cost_increase = -1, None, float('inf')
        for vid_idx, route in enumerate(child_routes):
             vid = sorted(vehicles_state.keys())[vid_idx]
             v_data = vehicles_state[vid]
             insertion, cost_increase = find_best_insertion_for_ga(
                 route, order_id, passengers, v_data['current_location'], v_data['current_time'], v_data['onboard']
             )
             if insertion and cost_increase < min_cost_increase:
                min_cost_increase, best_vid_idx, best_insertion = cost_increase, vid_idx, insertion
        
        if best_vid_idx != -1:
            i, j = best_insertion
            route_to_modify = child_routes[best_vid_idx]
            new_route = route_to_modify[:i] + [passengers[order_id]['pickup']] + route_to_modify[i:]
            new_route = new_route[:j] + [passengers[order_id]['dropoff']] + new_route[j:]
            child_routes[best_vid_idx] = new_route

    return child_routes

def mutate_dynamic(chromosome, vehicles_state, passengers):
    """健壮的变异算子，确保变异后仍然可行"""
    mutated_chromosome = copy.deepcopy(chromosome)
    
    # 1. 随机选择一个有路线的车辆，并从中随机选择一个订单
    eligible_indices = [i for i, r in enumerate(mutated_chromosome) if r]
    if not eligible_indices: return mutated_chromosome
    vid_idx_to_mutate = random.choice(eligible_indices)
    
    route_to_mutate = mutated_chromosome[vid_idx_to_mutate]
    pids_in_route = list({pt[0] for pt in route_to_mutate})
    if not pids_in_route: return mutated_chromosome
    pid_to_move = random.choice(pids_in_route)
    
    # 2. 从染色体中移除这个订单的所有相关停靠点
    for i in range(len(mutated_chromosome)):
        mutated_chromosome[i] = [pt for pt in mutated_chromosome[i] if pt[0] != pid_to_move]

    # 3. 将这个被移除的订单重新进行贪心插入
    best_vid_idx, best_insertion, min_cost_increase = -1, None, float('inf')
    for vid_idx, route in enumerate(mutated_chromosome):
        vid = sorted(vehicles_state.keys())[vid_idx]
        v_data = vehicles_state[vid]
        insertion, cost_increase = find_best_insertion_for_ga(
            route, pid_to_move, passengers, v_data['current_location'], v_data['current_time'], v_data['onboard']
        )
        if insertion and cost_increase < min_cost_increase:
            min_cost_increase, best_vid_idx, best_insertion = cost_increase, vid_idx, insertion
            
    if best_vid_idx != -1:
        i, j = best_insertion
        route_to_modify = mutated_chromosome[best_vid_idx]
        new_route = route_to_modify[:i] + [passengers[pid_to_move]['pickup']] + route_to_modify[i:]
        new_route = new_route[:j] + [passengers[pid_to_move]['dropoff']] + new_route[j:]
        mutated_chromosome[best_vid_idx] = new_route
    else:
        # 如果无法重新插入，则撤销变异 (返回原始染色体)
        return chromosome
        
    return mutated_chromosome

def calculate_total_fitness_dynamic(chromosome, vehicles_state, passengers):
    """计算整个解决方案的总适应度"""
    total_cost = 0
    vehicle_ids = sorted(list(vehicles_state.keys()))
    for vid_idx, route in enumerate(chromosome):
        vid = vehicle_ids[vid_idx]
        v_data = vehicles_state[vid]
        
        # 【已修复】在这里的调用中加入了 v_data['onboard'] 作为 initial_onboard_pids
        total_cost += compute_cost_and_penalty_dynamic(
            route, 
            passengers, 
            v_data['current_location'], 
            v_data['current_time'],
            v_data['onboard']  
        )
    return 1.0 / (total_cost + 1e-6)


# ---遗传算法重优化主函数 ---
def genetic_algorithm_reoptimizer(vehicles_state, passengers, new_order_ids):
    # 1. 确定所有需要重新规划的订单
    pids_in_future_routes = {pt[0] for v_data in vehicles_state.values() for pt in v_data['route']}
    orders_to_plan = list(pids_in_future_routes.union(set(new_order_ids)))
    if not orders_to_plan:
        return [v['route'] for v in vehicles_state.values()]

    # 2. 生成初始种群 (调用修复后的函数)
    population = [create_initial_individual(vehicles_state, passengers, orders_to_plan) for _ in range(GA_REOPT_POP_SIZE)]
    
    # 3. 演化过程
    best_chromosome_overall = None
    best_fitness_overall = -1.0

    for _ in range(GA_REOPT_GENERATIONS):
        fitnesses = [calculate_total_fitness_dynamic(ind, vehicles_state, passengers) for ind in population]
        
        # 更新全局最优解
        current_best_idx = np.argmax(fitnesses)
        if fitnesses[current_best_idx] > best_fitness_overall:
            best_fitness_overall = fitnesses[current_best_idx]
            best_chromosome_overall = copy.deepcopy(population[current_best_idx])
        
        # 选择、交叉、变异
        new_population = []
        # 精英主义参数
        elite_indices = np.argsort(fitnesses)[-GA_REOPT_ELITISM_SIZE:]
        for i in elite_indices:
            new_population.append(copy.deepcopy(population[i]))
        
        while len(new_population) < GA_REOPT_POP_SIZE:
            # 锦标赛选择
            tournament_indices = random.sample(range(len(population)), GA_REOPT_TOURNAMENT_SIZE)
            p1_idx = max(tournament_indices, key=lambda i: fitnesses[i])
            tournament_indices = random.sample(range(len(population)), GA_REOPT_TOURNAMENT_SIZE)
            p2_idx = max(tournament_indices, key=lambda i: fitnesses[i])
            parent1, parent2 = population[p1_idx], population[p2_idx]
            
            if random.random() < GA_REOPT_CROSSOVER_RATE:
                # 交叉函数
                child = crossover_dynamic(parent1, parent2, vehicles_state, passengers)
            else:
                child = copy.deepcopy(parent1)
            
            if random.random() < GA_REOPT_MUTATION_RATE:
                # 变异函数
                child = mutate_dynamic(child, vehicles_state, passengers)
            
            new_population.append(child)
            
        population = new_population
        
    return best_chromosome_overall if best_chromosome_overall is not None else [v['route'] for v in vehicles_state.values()]

# ==================== 动态模拟器 ====================
def initialize_vehicles_dynamic():
    state = {}
    for _, row in vehicles_df.iterrows():
        vid = row['vehicle_id']
        state[vid] = {
            'start_location': (row['start_x'], row['start_y']), 'current_location': (row['start_x'], row['start_y']),
            'current_time': 0.0, 'route': [], 'onboard': set(), 'served_pids': set(),
            'full_path_history': [((row['start_x'], row['start_y']), 0.0)], 'served_stops_history': []
        }
    return state

def update_simulation_state(vehicles_state, passengers, time_step):
    for vid, v in vehicles_state.items():
        if not v['route']:
            v['current_time'] += time_step
            continue
        time_left_in_step = time_step
        while time_left_in_step > 0 and v['route']:
            next_stop_info = v['route'][0]
            pid, x, y, typ = next_stop_info
            time_to_next_stop = travel_time(v['current_location'], (x, y))
            if time_left_in_step >= time_to_next_stop:
                completed_stop = v['route'].pop(0)
                # 防止重复记录服务历史 (驱除幽灵订单)
                if completed_stop not in v['served_stops_history']:
                    v['served_stops_history'].append(completed_stop)

                time_left_in_step -= time_to_next_stop
                v['current_time'] += time_to_next_stop
                v['current_location'] = (x, y)
                v['full_path_history'].append(((x, y), v['current_time']))

                if typ == 'pickup':
                    v['onboard'].add(pid)
                    passengers[pid]['actual_pickup_time'] = v['current_time']
                else:
                    v['onboard'].discard(pid)
                    v['served_pids'].add(pid)
                    passengers[pid]['actual_dropoff_time'] = v['current_time']
            else:
                ratio = time_left_in_step / time_to_next_stop
                new_x = v['current_location'][0] + ratio * (x - v['current_location'][0])
                new_y = v['current_location'][1] + ratio * (y - v['current_location'][1])
                v['current_location'] = (new_x, new_y)
                v['current_time'] += time_left_in_step
                v['full_path_history'].append(((new_x, new_y), v['current_time']))
                time_left_in_step = 0
    return vehicles_state

# ==================== 评估与可视化  ====================
# ==================== 评估与可视化 (修改后版本) ====================

def generate_final_report_and_metrics_unified(vehicles_state, passengers):
    """
    统一的最终报告与指标生成函数。
    [该版本已修改 "服务人数" 的定义为 "所有上过车的乘客"]
    """
    # 初始化所有指标
    total_distance = 0.0
    total_waiting_time = 0.0
    total_travel_time = 0.0
    total_cost_val = 0.0
    total_capacity_used = 0.0
    total_capacity_time = 0.0

    # --- 修改点 1: 核心定义变更 ---
    # 旧定义: fully_served_pids = {pid for pid, p_info in passengers.items() if 'actual_dropoff_time' in p_info}
    # 新定义: pids_with_pickup 成为我们衡量 "服务人数" 的标准
    pids_with_pickup = {pid for pid, p_info in passengers.items() if 'actual_pickup_time' in p_info}
    fully_served_pids = {pid for pid, p_info in passengers.items() if 'actual_dropoff_time' in p_info}
    
    # 新的服务人数定义
    num_served_new_def = len(pids_with_pickup)
    # 保持旧的定义用于计算必须完成行程才能计算的指标
    num_fully_served_for_calc = len(fully_served_pids)
    
    # --- 1. 遍历每辆车的历史记录来计算核心数据 (这部分不变) ---
    for vid, v_data in vehicles_state.items():
        full_path = v_data['full_path_history']
        if len(full_path) <= 1:
            continue

        for i in range(len(full_path) - 1):
            pos1, time1 = full_path[i]
            pos2, time2 = full_path[i+1]
            segment_dist = euclidean(pos1, pos2)
            segment_duration = time2 - time1
            
            if segment_duration <= 0: continue

            total_distance += segment_dist
            total_capacity_time += segment_duration

            onboard_pids_in_segment = set()
            onboard_pids_in_segment.update(v_data['onboard'] if v_data['current_time'] <= time1 else set())
            for stop in v_data['served_stops_history']:
                stop_pid, _, _, stop_type = stop
                event_time = passengers[stop_pid].get('actual_pickup_time' if stop_type == 'pickup' else 'actual_dropoff_time', float('inf'))
                if event_time <= time1:
                    if stop_type == 'pickup':
                        onboard_pids_in_segment.add(stop_pid)
                    else:
                        onboard_pids_in_segment.discard(stop_pid)
            
            total_capacity_used += len(onboard_pids_in_segment) * segment_duration

    # --- 2. 基于乘客的最终状态计算时间相关指标 ---
    # 修改点 2: 调整总等待时间和总行程时间的计算逻辑
    if num_served_new_def > 0:
        # 等待时间对所有 "上过车" 的乘客计算
        for pid in pids_with_pickup:
            p_info = passengers[pid]
            total_waiting_time += p_info['actual_pickup_time'] - p_info['request_time']
            
    if num_fully_served_for_calc > 0:
        # 行程时间只对 "已送达" 的乘客计算
        for pid in fully_served_pids:
            p_info = passengers[pid]
            total_travel_time += p_info['actual_dropoff_time'] - p_info['actual_pickup_time']
            
    # --- 3. 重新计算总成本 (这部分不变) ---
    total_fixed_cost = 0.0
    total_penalty = 0.0
    used_vehicles = 0
    for vid, v_data in vehicles_state.items():
        if v_data['served_stops_history']:
            used_vehicles += 1
            total_fixed_cost += FIXED_COST
            
    for pid in pids_with_pickup:
        p_info = passengers[pid]
        actual_pickup_time = p_info['actual_pickup_time']
        delay_from_start = actual_pickup_time - p_info['preferred_start']
        if actual_pickup_time <= p_info['preferred_end']:
            total_penalty += ALPHA_1 * delay_from_start
        else:
            penalty_in_window = ALPHA_1 * (p_info['preferred_end'] - p_info['preferred_start'])
            penalty_outside_window = ALPHA_2 * (actual_pickup_time - p_info['preferred_end'])
            total_penalty += penalty_in_window + penalty_outside_window
        if pid in fully_served_pids:
            delta = p_info['actual_dropoff_time'] - actual_pickup_time
            shortest = p_info['min_travel_time']
            if delta > shortest * DELAT: # 注意：绕路惩罚依然只能对已完成行程的乘客计算
                penalty += BETA * (delta - shortest)
    
    total_distance_cost = total_distance * COST_PER_KM
    total_cost_val = total_distance_cost + total_fixed_cost + total_penalty
            
    # --- 4. 计算最终的平均指标 (修改点 3: 使用新的分母) ---
    avg_waiting_time = total_waiting_time / num_served_new_def if num_served_new_def > 0 else 0
    # 注意: avg_travel_time 的分母保持为 num_fully_served_for_calc，因为只有他们有 travel_time
    avg_travel_time = total_travel_time / num_fully_served_for_calc if num_fully_served_for_calc > 0 else 0
    unit_cost = total_cost_val / num_served_new_def if num_served_new_def > 0 else 0
    vehicle_utilization = used_vehicles / len(vehicles_state) if vehicles_state else 0
    avg_capacity_util = (total_capacity_used / MAX_CAPACITY) / total_capacity_time if total_capacity_time > 0 else 0
    # 修改点 4: 更新未服务乘客数的计算
    unserved_passengers = len(passengers) - num_served_new_def

    # --- 5. 返回结构化的字典 (修改点 5: 使用新的服务人数) ---
    return {
        'served_passengers': num_served_new_def,  # 使用新的定义
        'avg_waiting_time': avg_waiting_time,
        # 建议在报告中注明 avg_travel_time 的计算口径
        'avg_travel_time (of completed trips only)': avg_travel_time, 
        'total_distance': total_distance,
        'total_cost': total_cost_val,
        'unit_cost': unit_cost,
        'vehicle_utilization': vehicle_utilization,
        'avg_capacity_util': avg_capacity_util,
        'unserved_passengers': unserved_passengers,
        'total_penalty_cost': total_penalty,
        # 您也可以选择性地报告已完成行程的乘客数以供对比
        'fully_completed_passengers': num_fully_served_for_calc,
    }


def save_dynamic_animation(vehicle_id, full_path_data, passengers, served_stops_history, gif_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    pids_on_this_route = sorted(list({pt[0] for pt in served_stops_history}))
    all_xs = [p[0] for p, t in full_path_data] + [p_info['pickup_x'] for pid in pids_on_this_route for p_info in [passengers[pid]]] + [p_info['dropoff_x'] for pid in pids_on_this_route for p_info in [passengers[pid]]]
    all_ys = [p[1] for p, t in full_path_data] + [p_info['pickup_y'] for pid in pids_on_this_route for p_info in [passengers[pid]]] + [p_info['dropoff_y'] for pid in pids_on_this_route for p_info in [passengers[pid]]]
    
    if not all_xs or not all_ys: plt.close(fig); return
    ax.set_xlim(min(all_xs)-1, max(all_xs)+1); ax.set_ylim(min(all_ys)-1, max(all_ys)+1)
    ax.set_xlabel("X-Coordinate"); ax.set_ylabel("Y-Coordinate"); ax.grid(True, linestyle='--', alpha=0.6)

    color_map = {pid: plt.cm.tab20(i % 20) for i, pid in enumerate(pids_on_this_route)}
    pickup_points, dropoff_points = {}, {}
    for pid in pids_on_this_route:
        p_info, color = passengers[pid], color_map[pid]
        pickup_points[pid] = ax.scatter(p_info['pickup_x'], p_info['pickup_y'], color=color, facecolors='none', marker='o', s=150, linewidths=2, label=f'P{pid} Pickup')
        dropoff_points[pid] = ax.scatter(p_info['dropoff_x'], p_info['dropoff_y'], color=color, marker='x', s=150, linewidths=2, label=f'P{pid} Dropoff')

    vehicle_marker, = ax.plot([], [], 's', c='black', ms=12, label='Vehicle')
    path_line, = ax.plot([], [], '-', c='black', lw=2, alpha=0.7)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, ha='left', va='top', fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    onboard_text = ax.text(0.98, 0.98, '', transform=ax.transAxes, ha='right', va='top', fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.5))

    events = {}
    for stop in served_stops_history:
        pid, _, _, typ = stop
        event_time = passengers[pid].get('actual_pickup_time' if typ == 'pickup' else 'actual_dropoff_time')
        if event_time is not None:
            if event_time not in events: events[event_time] = []
            events[event_time].append((pid, typ))
    sorted_event_times = sorted(events.keys())

    def update(frame):
        current_pos, current_time = full_path_data[frame]
        path_points = [pos for pos, time in full_path_data[:frame+1]]
        if path_points: path_line.set_data(*zip(*path_points))
        vehicle_marker.set_data([current_pos[0]], [current_pos[1]])
        time_text.set_text(f"Time: {current_time:.2f} min")
        
        onboard_pids = set()
        for event_time in sorted_event_times:
            if current_time >= event_time:
                for pid, typ in events[event_time]:
                    if typ == 'pickup':
                        onboard_pids.add(pid)
                        pickup_points[pid].set_facecolors(color_map[pid])
                    else:
                        onboard_pids.discard(pid)
                        # 为了让 'x' 看起来更实心，增加大小
                        dropoff_points[pid].set_linewidths(3)
            else: break
        
        onboard_str = "Onboard: " + (', '.join(f'P{int(p)}' for p in sorted(list(onboard_pids))) or "None")
        onboard_text.set_text(onboard_str)
        ax.set_title(f"Vehicle {int(vehicle_id)} Dynamic Route Simulation", fontsize=16)
        return [path_line, vehicle_marker, time_text, onboard_text] + list(pickup_points.values()) + list(dropoff_points.values())

    print(f"  -> Generating and saving animation for Vehicle {int(vehicle_id)} to {gif_path}...")
    ani = animation.FuncAnimation(fig, update, frames=len(full_path_data), interval=50, blit=True)
    ani.save(gif_path, writer=PillowWriter(fps=20))
    plt.close(fig)

# ==================== 主程序入口 (动态模拟循环) ====================
if __name__ == '__main__':
    # --- 1. 定义并创建输出目录 ---
    # 这是新增的部分，确保结果文件夹存在，如果不存在会自动创建
    output_dir = 'dongtai/results/dynamic_GA'
    os.makedirs(output_dir, exist_ok=True)

    vehicles_state = initialize_vehicles_dynamic()
    
    print("--- 开始动态车辆调度模拟 (结合最终修复版遗传算法进行重优化) ---")
    with tqdm(range(T), desc="模拟时间推进") as pbar:
        for t in pbar:
            vehicles_state = update_simulation_state(vehicles_state, passengers_dict, TIME_INTERVAL)
            new_request_ids = D_t_list[t] if t < len(D_t_list) else []
            
            if new_request_ids and GA_REOPT_ENABLED:
                reoptimized_routes = genetic_algorithm_reoptimizer(vehicles_state, passengers_dict, new_request_ids)
                if reoptimized_routes:
                    for i, vid in enumerate(sorted(vehicles_state.keys())):
                        vehicles_state[vid]['route'] = reoptimized_routes[i]
                pbar.set_postfix_str(f"新请求:{len(new_request_ids)}, GA重优化启动")
            else:
                 pbar.set_postfix_str(f"无新请求")

    # --- 模拟结束，生成最终报告 ---
    print("\n--- 模拟结束, 开始生成最终报告 ---")
    # 调用统一评估函数 (这部分来自我上一个回答，保持不变)
    metrics = generate_final_report_and_metrics_unified(vehicles_state, passengers_dict)
    
    # --- 2. 修改报告文件的输出路径 ---
    # 使用 os.path.join 来构建完整的文件路径
    output_filename = os.path.join(output_dir, 'result_summary_dynamic_ga.txt')

    # 使用你提供的代码片段进行文件写入
    # --- 这是修复后的代码 ---
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("="*20 + " 动态调度最终评估报告 (统一指标版) " + "="*20 + "\n\n")
        
        f.write("核心性能指标:\n")
        f.write(f"  服务乘客数 (至少被接上车): {metrics['served_passengers']}\n")
        f.write(f"  完全送达乘客数: {metrics['fully_completed_passengers']}\n")
        f.write(f"  平均等待时间: {metrics['avg_waiting_time']:.2f} 分钟\n")
        # 使用新的键名，并更新了文本描述使其更清晰
        f.write(f"  平均行程时间 (仅限已送达乘客): {metrics['avg_travel_time (of completed trips only)']:.2f} 分钟\n")
        f.write(f"  总行驶距离: {metrics['total_distance']:.2f} 公里\n")
        f.write(f"  总调度成本: {metrics['total_cost']:.2f} 元\n")
        f.write(f"    - 其中惩罚成本: {metrics['total_penalty_cost']:.2f} 元\n")
        f.write(f"  单位服务成本 (按接上车人数算): {metrics['unit_cost']:.2f} 元/人\n")
        f.write(f"  车辆使用率: {metrics['vehicle_utilization']:.2%}\n")
        f.write(f"  平均容量利用率: {metrics['avg_capacity_util']:.2%}\n")
        f.write(f"  未服务乘客数 (从未被接上车): {metrics['unserved_passengers']}\n")

        f.write("\n" + "="*70 + "\n\n")

        
        f.write("各车辆服务历史详情:\n\n")
        for vid in sorted(vehicles_state.keys()):
            if int(vid) not in TARGET_VEHICLES: continue
            
            v_data = vehicles_state[vid]
            f.write(f"--- 车辆 {vid} ---\n")
            
            if not v_data['served_stops_history']:
                f.write("  未服务任何乘客。\n\n")
                continue

            f.write(f"  服务乘客数 (已送达): {len(v_data['served_pids'])}\n")
            f.write("  服务历史 (按时间顺序):\n")
            for stop in v_data['served_stops_history']:
                pid, x, y, typ = stop
                stop_time = passengers_dict[pid].get('actual_pickup_time' if typ=='pickup' else 'actual_dropoff_time', 0)
                f.write(f"    - [{stop_time:6.2f} min] {typ.upper():<8} P{int(pid):<3} at ({x:5.2f}, {y:5.2f})\n")
            f.write("\n")
            
            full_path_data = v_data['full_path_history']
            if len(full_path_data) > 1:
                # --- 3. 修改动画文件的保存路径 ---
                gif_base_name = f'vehicle_{int(vid)}_dynamic_unified.gif'
                gif_full_path = os.path.join(output_dir, gif_base_name)
                
                save_dynamic_animation(vid, full_path_data, passengers_dict, v_data['served_stops_history'], gif_full_path)
                f.write(f"  动画已保存至: {gif_full_path}\n\n")

    print("\n报告和动画生成完毕。")
    print(f"调度结果已保存至 {output_filename}")
    print(f"已为车辆 {TARGET_VEHICLES} 生成路线总结和可视化图像。")
    print("\n最终评估指标:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and 'passengers' not in key:
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
