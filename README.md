# 拼车调度问题优化求解器

## 研究背景与问题描述

本项目旨在解决**拼车调度问题**，核心任务是在满足乘客出行偏好与服务约束的前提下，使用多辆车以尽可能低的总成本服务尽可能多的乘客。

每位乘客会提前发出拼车请求，包含上车时间偏好、上/下车地点、可容忍绕路程度等。系统需在规定的时间窗内，为多个车辆动态安排合理的服务路径，权衡运营成本与服务质量。

---

## 优化目标

我们希望最小化以下**目标函数**：

\[
\text{Total Cost} = \sum_{v \in V} \left( \text{Distance Cost} + \text{Fixed Cost} \right) + \sum_{p \in P} \left( \text{Time Delay Penalty} + \text{Overtravel Penalty} \right)
\]

其中：
- **Distance Cost**：车辆实际行驶距离 × 单位成本
- **Fixed Cost**：车辆被调度即产生固定成本
- **Time Delay Penalty**：
  - 超过 `preferred_start` 的轻度惩罚（系数 α₁）
  - 超过 `preferred_end` 的严重惩罚（系数 α₂）
- **Overtravel Penalty**：乘客绕路比例超过容忍度的惩罚（系数 β）

---

## 约束条件

调度方案需满足以下约束：

1. **容量约束**：每辆车同时最多搭载 `MAX_CAPACITY` 名乘客；
2. **时窗约束**：
   - 上车时间不得晚于乘客的 `late_accept`；
   - 优先满足其 `preferred_start ~ preferred_end` 范围；
3. **行程顺序约束**：每位乘客必须先上车再下车；
4. **行程时长约束**：乘车时间不得超过 `min_travel_time × δ`；
5. **路径可行性**：所有插入的路径必须满足上面所有约束。

---

## 支持算法

本项目实现了以下 **七种算法** 用于调度求解：

| 算法编号 | 名称                      | 英文缩写 | 说明 |
|----------|---------------------------|-----------|------|
| 1        | 模拟退火                  | SA        | 使用邻域扰动和温度控制进行解空间探索 |
| 2        | 遗传算法                  | GA        | 基于编码、交叉、变异演化优化解 |
| 3        | 蚁群优化算法              | ACO       | 使用信息素和启发式引导路径搜索 |
| 4        | 启发式 A* 搜索            | A*        | 使用估价函数从当前状态前向搜索 |
| 5        | Dijkstra 最短路径算法     | Dijkstra  | 基于最短时间构造调度路径 |
| 6        | 粒子群优化                | PSO       | 多粒子全局搜索调度解空间 |
| 7        | 模拟退火算法（静态版本）  | SA        | 针对静态拼车场景的预约调度优化 |

---

## 文件目录说明
project_root/
│
├── jingtai/
│ ├── data/
│ │ ├── passengers5.csv # 乘客请求数据（含时间窗、地点、期望时间等）
│ │ └── vehicles5.csv # 车辆初始位置数据
│ │
│ ├── results/
│ │ ├── simulated_annealing/ # 模拟退火结果（SA）
│ │ ├── genetic_algorithm/ # 遗传算法结果（GA）
│ │ ├── ant_colony_optimization/ # 蚁群算法结果（ACO）
│ │ ├── particle_swarm_optimization/ # 粒子群（PSO）
│ │ ├── a_star/ # A算法结果
│ │ ├── dijkstra/ # Dijkstra算法结果
│ │ └── simulated_annealing_static/ # 静态模拟退火版本
│ │
│ └── main_.py # 七个算法对应的主程序入口文件（详见下表）
│
└── README.md # 项目说明文档


---

## 🔎 主程序文件对应说明

| 文件名                          | 对应算法       | 输出目录                              |
|----------------------------------|----------------|----------------------------------------|
| `main_sa.py`                     | 模拟退火（SA） | `jingtai/results/simulated_annealing/` |
| `main_ga.py`                     | 遗传算法（GA） | `jingtai/results/genetic_algorithm/`   |
| `main_aco.py`                    | 蚁群优化（ACO）| `jingtai/results/ant_colony_optimization/` |
| `main_pso.py`                    | 粒子群（PSO）  | `jingtai/results/particle_swarm_optimization/` |
| `main_astar.py`                  | A* 启发式搜索  | `jingtai/results/a_star/`              |
| `main_dijkstra.py`              | Dijkstra       | `jingtai/results/dijkstra/`            |
| `main_sa_static.py`             | 静态模拟退火   | `jingtai/results/simulated_annealing_static/` |

---

## 评估指标（输出于 summary.txt）

每种算法运行结束后，会生成对应的 `summary.txt` 文件，记录以下关键绩效指标（KPI）：

- `served_passengers`: 成功服务的乘客数  
- `avg_waiting_time`: 平均等待时间（从请求到上车）  
- `avg_travel_time`: 平均乘车时间  
- `total_distance`: 所有车辆行驶总距离  
- `total_cost`: 总运营成本（含惩罚项）  
- `unit_cost`: 单位乘客服务成本  
- `vehicle_utilization`: 车辆使用率（使用车辆数 / 总车辆数）  
- `avg_capacity_util`: 平均容量利用率（加权时间利用）  

---

## 动画输出

为指定的车辆（默认 `[0, 1, 2, 3]`）生成服务路径的 **GIF 动画**，用于展示调度过程，包括：

- 圆点：上车点
- 勾号：下车点
- 路径颜色：统一车辆轨迹
- 数字：当前车上乘客人数
- 图例：颜色对应乘客 ID

动画保存在对应 `results/` 文件夹内，例如：
jingtai/results/genetic_algorithm/vehicle_0.gif



