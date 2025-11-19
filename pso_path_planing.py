import numpy as np
import matplotlib.pyplot as plt

# 起点 终点
START = np.array([0.0, 0.0])
GOAL  = np.array([10.0, 8.0])

# 障碍物(x,y,r)
OBSTACLES = [
    (3.0, 4.0, 1.2),
    (6.0, 6.0, 1.0),
    (7.5, 2.5, 0.9),
    (4.5, 1.5, 0.7),
]

# 地图边界
X_MIN, X_MAX = -1.0, 12.0
Y_MIN, Y_MAX = -1.0, 10.0

# 航路点表示
N_WAYPOINTS = 6


# ---------------------------
# Utility functions
# ---------------------------

'''
    计算路径长度
'''
def path_length(points):
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return seg_lengths.sum()

'''
    计算智能体与障碍物最小距离 d
    d > 0 未碰撞
    d < 0 发生碰撞
'''
def min_distance_to_obstacles(pt, obstacles):
    dists = []
    for (ox, oy, r) in obstacles:
        center = np.array([ox, oy])
        d = np.linalg.norm(pt - center) - r
        dists.append(d)
    return min(dists)

'''
    碰撞惩罚函数
'''
def collision_penalty_for_path(points, obstacles, safety_dist=0.2, penalty_factor=100.0):
    penalty = 0.0
    n_samples_per_seg = 6
    for i in range(len(points)-1):
        a = points[i]
        b = points[i+1]
        for t in np.linspace(0, 1, n_samples_per_seg):
            pt = a + (b - a) * t
            for (ox, oy, r) in obstacles:
                d = np.linalg.norm(pt - np.array([ox, oy]))
                allowed = r + safety_dist
                if d < allowed:
                    # quadratic penalty grows as it penetrates more
                    penetration = allowed - d
                    penalty += penalty_factor * (penetration**2)
    return penalty
'''
    平滑性惩罚（减少急转弯）
'''
def smoothness_penalty(points, weight=1.0):
    penalty = 0.0
    for i in range(1, len(points)-1):
        v1 = points[i] - points[i-1]
        v2 = points[i+1] - points[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        # 计算两个向量的夹角余弦（cosθ）
        cosang = np.clip(np.dot(v1, v2) / (n1*n2),a_min=-1.0, a_max=1.0)
        # 计算角度
        angle = np.arccos(cosang)  # radians
        penalty += weight * (angle**2)
    return penalty




'''
    
'''
class PSOPathPlanner:
    def __init__(self,
                 start,
                 goal,
                 obstacles,
                 n_waypoints=12,
                 pop_size=40,
                 max_iter=200,
                 bounds=((X_MIN, X_MAX), (Y_MIN, Y_MAX)),
                 w=0.7, c1=1.5, c2=1.5,
                 collision_penalty_factor=120.0,
                 safety_dist=0.2,
                 smoothness_weight=0.8):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.n_waypoints = n_waypoints
        self.pop_size = pop_size
        self.dim = 2 * n_waypoints  # each waypoint has x,y
        self.max_iter = max_iter
        self.bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.collision_penalty_factor = collision_penalty_factor
        self.safety_dist = safety_dist
        self.smoothness_weight = smoothness_weight

        # initialize particle positions (flattened vector of waypoints)
        low = np.array([bounds[0][0], bounds[1][0]] * n_waypoints)
        high = np.array([bounds[0][1], bounds[1][1]] * n_waypoints)
        self.x = np.random.uniform(low, high, (pop_size, self.dim))
        # small initial velocities
        self.v = np.random.uniform(-0.5, 0.5, (pop_size, self.dim))

        # personal bests
        self.pbest = self.x.copy()
        self.pbest_fitness = np.array([self.fitness(p) for p in self.pbest])

        best_idx = np.argmin(self.pbest_fitness)
        self.gbest = self.pbest[best_idx].copy()
        self.gbest_fitness = self.pbest_fitness[best_idx]

    '''
        将粒子位置转化为路径
    '''
    def waypoints_from_particle(self, particle):
        pts = particle.reshape((self.n_waypoints, 2))
        full = np.vstack([self.start, pts, self.goal])
        return full
    '''
        路径优化目标函数
    '''
    def fitness(self, particle):
        pts = self.waypoints_from_particle(particle)
        length = path_length(pts)
        # 碰撞惩罚
        coll = collision_penalty_for_path(pts, self.obstacles,
                                          safety_dist=self.safety_dist,
                                          penalty_factor=self.collision_penalty_factor)
        # 平滑惩罚
        smooth = smoothness_penalty(pts, weight=self.smoothness_weight)
        # 成本函数
        return length + coll + smooth


    def optimize(self, verbose=True):
        curve = []
        for it in range(self.max_iter):
            for i in range(self.pop_size):
                # update velocity
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                '''
                    PSO核心算法
                '''
                cognitive = self.c1 * r1 * (self.pbest[i] - self.x[i])

                social = self.c2 * r2 * (self.gbest - self.x[i])

                self.v[i] = self.w * self.v[i] + cognitive + social

                # optionally clip velocity (to avoid huge jumps)
                vmax = np.array([ (self.bounds[0][1] - self.bounds[0][0]) * 0.2,
                                  (self.bounds[1][1] - self.bounds[1][0]) * 0.2 ] * self.n_waypoints).flatten()
                self.v[i] = np.clip(self.v[i], -vmax, vmax)

                # update position
                self.x[i] = self.x[i] + self.v[i]

                # boundary handling (clip)
                for d in range(self.n_waypoints):
                    xi = 2*d
                    yi = xi + 1
                    self.x[i, xi] = np.clip(self.x[i, xi], self.bounds[0][0], self.bounds[0][1])
                    self.x[i, yi] = np.clip(self.x[i, yi], self.bounds[1][0], self.bounds[1][1])

                # evaluate fitness
                fit = self.fitness(self.x[i])

                # update personal best
                if fit < self.pbest_fitness[i]:
                    self.pbest[i] = self.x[i].copy()
                    self.pbest_fitness[i] = fit

                # update global best
                if fit < self.gbest_fitness:
                    self.gbest = self.x[i].copy()
                    self.gbest_fitness = fit

            curve.append(self.gbest_fitness)

            if verbose and (it % max(1, self.max_iter // 10) == 0 or it == self.max_iter-1):
                print(f"Iter {it+1}/{self.max_iter}  BestFitness = {self.gbest_fitness:.4f}")

        return self.gbest, self.gbest_fitness, curve


# ---------------------------
# Visualization functions
# ---------------------------

def plot_map(start, goal, obstacles, best_path_pts=None, show=True, filename=None):
    fig, ax = plt.subplots(figsize=(8,6))
    # plot obstacles
    for (ox, oy, r) in obstacles:
        circle = plt.Circle((ox, oy), r, color='gray', alpha=0.7)
        ax.add_patch(circle)
        # safety boundary
        safec = plt.Circle((ox, oy), r+0.2, color='gray', alpha=0.15, linestyle='--', fill=False)
        ax.add_patch(safec)

    # start & goal
    ax.scatter(start[0], start[1], marker='o', color='green', s=80, label='Start')
    ax.scatter(goal[0],  goal[1],  marker='X', color='red',   s=80, label='Goal')

    # best path
    if best_path_pts is not None:
        ax.plot(best_path_pts[:,0], best_path_pts[:,1], '-o', linewidth=2, markersize=6, label='Planned Path')
        # annotate waypoint indices
        for i, p in enumerate(best_path_pts):
            ax.text(p[0]+0.05, p[1]+0.05, str(i), fontsize=9)

    ax.set_xlim(X_MIN-0.5, X_MAX+0.5)
    ax.set_ylim(Y_MIN-0.5, Y_MAX+0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('PSO Path Planning')
    ax.grid(True)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    if filename:
        plt.savefig(filename, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_convergence(curve, show=True, filename=None):
    fig, ax = plt.subplots()
    ax.plot(curve, '-')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Fitness')
    ax.set_title('Convergence Curve')
    ax.grid(True)
    if filename:
        plt.savefig(filename, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)



def main():
    planner = PSOPathPlanner(
        start=START,
        goal=GOAL,
        obstacles=OBSTACLES,
        n_waypoints=N_WAYPOINTS,
        pop_size=60,
        max_iter=300,
        bounds=((X_MIN, X_MAX), (Y_MIN, Y_MAX)),
        w=0.72, c1=1.6, c2=1.6,
        collision_penalty_factor=200.0,  # increase to avoid collisions
        safety_dist=0.25,
        smoothness_weight=1.0
    )

    best_particle, best_f, curve = planner.optimize(verbose=True)

    best_pts = planner.waypoints_from_particle(best_particle)
    print("\nBest fitness:", best_f)
    print("Best path waypoints (including start & goal):")
    for i, p in enumerate(best_pts):
        print(f"  {i:2d}: ({p[0]:.3f}, {p[1]:.3f})")

    # visualize
    plot_map(START, GOAL, OBSTACLES, best_path_pts=best_pts)
    plot_convergence(curve)


if __name__ == "__main__":
    main()
