import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


# --------------------------
# 1. 环境建模（三维栅格地图+障碍物）
# 作用：构建无人机飞行的三维环境，包含起点、终点和圆柱状障碍物，提供碰撞检测功能
# --------------------------
class Env3D:
    def __init__(self, start=[0, 0, 10], goal=[100, 100, 50],
                 obstacle_num=5, obstacle_radius_range=[5, 15]):
        """
        初始化三维环境参数
        参数说明：
            start: 无人机起点坐标（默认[0,0,10]，单位m）
            goal: 无人机终点坐标（默认[100,100,50]，单位m）
            obstacle_num: 障碍物数量（默认15个）
            obstacle_radius_range: 障碍物半径范围（默认[5,15]m，圆柱状障碍物）
        """
        self.start = np.array(start)  # 存储起点坐标（转换为numpy数组方便计算）
        self.goal = np.array(goal)    # 存储终点坐标
        self.obstacle_num = obstacle_num  # 存储障碍物数量
        self.obstacle_radius_range = obstacle_radius_range  # 存储障碍物半径范围

        # 生成随机圆柱障碍物（调用内部方法）
        self.obstacles = self._generate_obstacles()

    def _generate_obstacles(self):
        """生成随机圆柱障碍物（内部方法）：每个障碍物用中心点坐标和半径描述"""
        obstacles = []  # 存储所有障碍物的列表
        for _ in range(self.obstacle_num):
            # 随机生成障碍物中心坐标：在起点和终点的最大范围之内（x/y/z轴分别取最大值）
            center = np.random.uniform(
                [0, 0, 0],  # 坐标下界（从原点开始）
                [max(self.start[0], self.goal[0]),  # x轴上界（起点/终点的最大x）
                 max(self.start[1], self.goal[1]),  # y轴上界（起点/终点的最大y）
                 max(self.start[2], self.goal[2])]  # z轴上界（起点/终点的最大z）
            )
            # 随机生成障碍物半径：在指定范围内均匀分布
            radius = np.random.uniform(*self.obstacle_radius_range)
            # 将障碍物信息（中心+半径）存入列表
            obstacles.append({"center": center, "radius": radius})
        return obstacles

    def is_collision(self, point):
        """
        判断某个三维点是否与障碍物碰撞（圆柱障碍物仅检测xy平面距离）
        参数：point - 待检测的三维点坐标（numpy数组）
        返回：布尔值（True=碰撞，False=无碰撞）
        """
        for obs in self.obstacles:
            # 计算点与障碍物中心的xy平面距离（圆柱障碍物高度无限制，仅检测水平碰撞）
            dist = np.linalg.norm(point[:2] - obs["center"][:2])
            # 若距离小于障碍物半径，说明碰撞
            if dist < obs["radius"]:
                return True
        return False  # 所有障碍物都不碰撞则返回False


# --------------------------
# 2. 改进PSO的无人机路径规划算法
# 作用：基于粒子群优化（PSO）算法，搜索从起点到终点的最优无碰撞路径
# --------------------------
class UAVPathPlannerPSO:
    def __init__(self, env, waypoint_num=5, pop_size=30, max_iter=100,
                 w=0.7, c1=1.5, c2=1.5):
        """
        初始化PSO路径规划器参数
        参数说明：
            env: 三维环境实例（Env3D类对象）
            waypoint_num: 路径点数量（不含起点/终点，默认5个）
            pop_size: 粒子数量（种群大小，默认30个）
            max_iter: 最大迭代次数（默认100次）
            w: 惯性权重（平衡全局/局部搜索，默认0.7）
            c1: 个体学习因子（粒子向自身最优学习的权重，默认1.5）
            c2: 群体学习因子（粒子向全局最优学习的权重，默认1.5）
        """
        self.env = env  # 存储环境实例
        self.waypoint_num = waypoint_num  # 存储路径点数量
        self.pop_size = pop_size  # 存储粒子数量
        self.max_iter = max_iter  # 存储最大迭代次数
        self.w = w  # 存储惯性权重
        self.c1 = c1  # 存储个体学习因子
        self.c2 = c2  # 存储群体学习因子

        # 定义路径点的搜索范围（基于起点/终点的坐标范围）
        self.x_range = [0, max(env.start[0], env.goal[0])]  # x轴范围
        self.y_range = [0, max(env.start[1], env.goal[1])]  # y轴范围
        self.z_range = [10, 100]  # z轴高度范围（限制无人机飞行高度）

        # 初始化粒子群：每个粒子对应一组路径点（waypoint_num个三维坐标）
        self.particles = self._init_particles()
        # 初始化粒子速度：速度范围[-5,5]，形状与粒子群一致。速度决定粒子每代的移动步长
        self.velocities = np.random.uniform(-5, 5, self.particles.shape)

        # 初始化个体最优（pbest）：初始时等于粒子自身位置
        self.pbest = self.particles.copy()
        # 计算个体最优的代价（每个粒子对应路径的总成本）
        self.pbest_cost = np.array([self._path_cost(p) for p in self.particles])
        # 初始化全局最优（gbest）：个体最优中代价最小的粒子，从所有粒子的pbest中选出代价最小的粒子，作为整个种群的初始最优解
        self.gbest = self.pbest[np.argmin(self.pbest_cost)]
        self.gbest_cost = np.min(self.pbest_cost)  # 全局最优的代价

        # 记录迭代过程：存储每代的粒子群和最优路径（用于可视化）
        self.history = []

    def _init_particles(self):
        """初始化粒子群（内部方法）：每个粒子是waypoint_num个路径点的组合"""
        particles = []  # 存储所有粒子的列表
        for _ in range(self.pop_size): #循环生成pop_size个粒子，每个粒子包含waypoint_num个三维路径点
            waypoints = []  # 单个粒子对应的路径点列表
            for _ in range(self.waypoint_num):
                # 随机生成单个路径点（在搜索范围内均匀分布）
                x = np.random.uniform(*self.x_range)
                y = np.random.uniform(*self.y_range)
                z = np.random.uniform(*self.z_range)
                waypoints.append([x, y, z])  # 添加到路径点列表
            particles.append(np.array(waypoints))  # 将路径点转换为数组并加入粒子群
        return np.array(particles)  # 转换为二维数组返回

    def _path_cost(self, waypoints):
        """
        计算路径总成本（目标函数）：距离代价 + 碰撞惩罚
        参数：waypoints - 一组路径点（粒子对应的路径点）
        返回：路径总成本（值越小路径越优）
        """
        # 构建完整路径：起点 → 路径点 → 终点（垂直堆叠成完整路径数组）
        path = np.vstack([self.env.start, waypoints, self.env.goal])

        # 1. 距离代价：计算路径总长度（相邻点之间的欧式距离之和）
        dist_cost = 0
        for i in range(1, len(path)):
            dist_cost += np.linalg.norm(path[i] - path[i - 1])  # 累加相邻点距离

        # 2. 碰撞惩罚：路径点或路径段与障碍物碰撞则添加高额惩罚
        collision_penalty = 0
        # 检查每个路径点是否碰撞
        for point in path:
            if self.env.is_collision(point):
                collision_penalty += 1000  # 路径点碰撞惩罚1000
        # 检查路径段是否碰撞（在路径段上采样10个点检测）
        for i in range(1, len(path)):
            start_p = path[i - 1]  # 路径段起点
            end_p = path[i]        # 路径段终点
            # 采在相邻路径点之间采样 10 个中间点检测（线性插值start_p + t*(end_p-start_p)），
            # 若任意采样点碰撞，加 500 惩罚（避免路径 “穿障而过”）。
            for t in np.linspace(0, 1, 10):
                check_p = start_p + t * (end_p - start_p)  # 线性插值计算采样点
                if self.env.is_collision(check_p):
                    collision_penalty += 500  # 路径段碰撞惩罚500

        # 总成本 = 距离代价 + 碰撞惩罚
        return dist_cost + collision_penalty

    def optimize(self):
        """执行PSO优化过程：迭代更新粒子位置和速度，搜索最优路径"""
        for _ in range(self.max_iter):
            # 记录当前代的最优路径（用于可视化）：起点+全局最优路径点+终点
            best_path = np.vstack([self.env.start, self.gbest, self.env.goal])
            self.history.append((self.particles.copy(), best_path.copy(), self.gbest_cost))

            # 生成随机学习因子（r1/r2∈[0,1]，形状与粒子群一致）
            r1 = np.random.rand(*self.particles.shape)
            r2 = np.random.rand(*self.particles.shape)

            # 更新粒子速度：PSO经典速度更新公式
            # 速度 = 惯性项 + 个体学习项 + 群体学习项
            self.velocities = (self.w * self.velocities +
                               self.c1 * r1 * (self.pbest - self.particles) +  # 个体最优引导
                               self.c2 * r2 * (self.gbest - self.particles))   # 全局最优引导

            # 更新粒子位置：位置 = 当前位置 + 速度
            self.particles += self.velocities

            # 边界约束：确保路径点不超出预设的搜索范围
            self.particles[:, :, 0] = np.clip(self.particles[:, :, 0], *self.x_range)  # x轴约束
            self.particles[:, :, 1] = np.clip(self.particles[:, :, 1], *self.y_range)  # y轴约束
            self.particles[:, :, 2] = np.clip(self.particles[:, :, 2], *self.z_range)  # z轴约束

            # 更新个体最优（pbest）：若当前粒子代价更小，则更新
            current_cost = np.array([self._path_cost(p) for p in self.particles])  # 计算当前粒子代价
            update_idx = current_cost < self.pbest_cost  # 找到代价更小的粒子索引
            self.pbest_cost[update_idx] = current_cost[update_idx]  # 更新个体最优代价
            self.pbest[update_idx] = self.particles[update_idx]     # 更新个体最优位置

            # 更新全局最优（gbest）：若个体最优中出现更小代价，则更新
            current_gbest_idx = np.argmin(self.pbest_cost)  # 找到当前个体最优中代价最小的索引
            if self.pbest_cost[current_gbest_idx] < self.gbest_cost:
                self.gbest_cost = self.pbest_cost[current_gbest_idx]  # 更新全局最优代价
                self.gbest = self.pbest[current_gbest_idx].copy()     # 更新全局最优位置

        # 生成最终路径：起点+全局最优路径点+终点
        self.final_path = np.vstack([self.env.start, self.gbest, self.env.goal])
        return self.final_path, self.gbest_cost  # 返回最终路径和总成本


# --------------------------
# 3. 可视化无人机路径规划过程
# 作用：动态展示PSO算法的迭代过程，直观呈现粒子群演化和最优路径生成
# --------------------------
def visualize_uav_path_planning(env, planner):
    """
    可视化无人机路径规划过程
    参数：
        env: 三维环境实例（Env3D类对象）
        planner: PSO路径规划器实例（UAVPathPlannerPSO类对象）
    """
    fig = plt.figure(figsize=(12, 8))  # 创建画布（尺寸12×8英寸）
    ax = fig.add_subplot(111, projection='3d')  # 创建3D子图

    # 绘制起点和终点：绿色圆点表示起点，红色星号表示终点
    ax.scatter(env.start[0], env.start[1], env.start[2], c='green', s=100, label='Start', marker='o')
    ax.scatter(env.goal[0], env.goal[1], env.goal[2], c='red', s=100, label='Goal', marker='*')

    # 绘制圆柱障碍物：包括底面、顶面和侧面
    for obs in env.obstacles:
        theta = np.linspace(0, 2 * np.pi, 50)  # 生成圆周角度（50个点）
        # 绘制圆柱底面（z=0）
        x = obs["center"][0] + obs["radius"] * np.cos(theta)
        y = obs["center"][1] + obs["radius"] * np.sin(theta)
        z = np.zeros_like(x) + 0  # 底面z坐标为0
        ax.plot(x, y, z, c='gray', alpha=0.5)
        # 绘制圆柱顶面（z=100）
        z_top = np.zeros_like(x) + 100  # 顶面z坐标为100
        ax.plot(x, y, z_top, c='gray', alpha=0.5)
        # 绘制圆柱侧面（连接底面和顶面的竖线）
        for i in range(len(x) - 1):
            ax.plot([x[i], x[i]], [y[i], y[i]], [0, 100], c='gray', alpha=0.3)

    # 初始化绘图元素：绘制前10个粒子的路径（蓝色半透明）和最优路径（红色实线）
    particle_paths = []
    for _ in range(min(10, planner.pop_size)):  # 限制绘制粒子数量，避免画面混乱
        line, = ax.plot([], [], [], 'b-', alpha=0.3)
        particle_paths.append(line)
    best_path_line, = ax.plot([], [], [], 'r-', linewidth=2, label='Best Path')

    # 设置坐标轴标签和标题
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('UAV 3D Path Planning with PSO')
    ax.legend()  # 显示图例

    def update(frame):
        """动画更新函数：每帧对应PSO的一代，更新粒子路径和最优路径"""
        particles, best_path, _ = planner.history[frame]  # 获取当前代的历史数据

        # 更新粒子路径的显示
        for i, line in enumerate(particle_paths):
            if i < len(particles):
                # 构建粒子的完整路径：起点+粒子路径点+终点
                path = np.vstack([env.start, particles[i], env.goal])
                line.set_data(path[:, 0], path[:, 1])  # 设置x/y坐标
                line.set_3d_properties(path[:, 2])     # 设置z坐标

        # 更新最优路径的显示
        best_path_line.set_data(best_path[:, 0], best_path[:, 1])  # 设置最优路径的x/y坐标
        best_path_line.set_3d_properties(best_path[:, 2])         # 设置最优路径的z坐标

        return particle_paths + [best_path_line]  # 返回更新后的绘图元素

    # 创建动画：每200ms更新一帧，不重复播放
    ani = FuncAnimation(fig, update, frames=len(planner.history),
                        interval=200, blit=False, repeat=False)

    plt.tight_layout()  # 自动调整布局，避免标签重叠
    plt.show()  # 显示动画

    return ani


# --------------------------
# 4. 主程序运行
# 作用：整合环境建模、路径规划和可视化，执行完整的无人机路径规划流程
# --------------------------
if __name__ == "__main__":
    # 1. 创建三维环境：设置起点、终点和障碍物数量
    env = Env3D(start=[0, 0, 10], goal=[100, 100, 50], obstacle_num=5)

    # 2. 初始化PSO路径规划器：设置路径点数量、粒子数和迭代次数
    planner = UAVPathPlannerPSO(env, waypoint_num=5, pop_size=30, max_iter=50)

    # 3. 执行路径规划：获取最终路径和总成本
    final_path, final_cost = planner.optimize()

    # 4. 输出结果：打印最终路径点和总成本
    print("最终路径点坐标：")
    for i, point in enumerate(final_path):
        print(f"Point {i}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")  # 保留两位小数
    print(f"路径总成本（距离+惩罚）：{final_cost:.2f}")

    # 5. 可视化规划过程：动态展示PSO迭代寻优过程
    ani = visualize_uav_path_planning(env, planner)