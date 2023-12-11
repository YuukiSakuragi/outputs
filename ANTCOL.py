from pyvirtualdisplay import Display
display = Display(visible=0, size=(1024, 768))
display.start()
import numpy as np
import random, copy

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

# Initialize Super Mario environment
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
env = JoypadSpace(env, [["right","B"], ["right","A"],["right","A","B"]])

# パラメータの設定
num_ants = 10
num_iterations = 100
evaporation_rate = 0.5
alpha = 1.0
beta = 2.0
Q = 1.0
MAX_TIME = 15000
ELITE_NUMB = 1

#マリオのゲームマップの初期化関数
def initialize_map():
    # マップの初期化ロジックを記述する
    # マップの情報を適切なデータ構造で表現し、初期値を設定する
    # 例えば、2次元配列やグラフなどを使用してマップを表現することができる
    
    # 以下は簡単な例として、3x3のマップを作成して初期化する
    map = np.zeros((3, 3))  # 3x3のゼロ行列で初期化
    
    # マップの各セルに適切な値を設定する
    map[0][1] = 1  # ゴール位置を1とする
    
    return map

#アリがスタートする位置を初期化する関数
def initialize_start_action():
    # スタート位置の初期化ロジックを記述する
    # スタート位置を適切に初期化する方法を実装する
    # 例えば、ランダムな位置を選択する場合は、適切な範囲内でランダムに位置を選択する
    
    # 以下は簡単な例として、3つの都市からランダムにスタート位置を選択する場合
    start_action = random.randint(0, 2)  # 0から2までのランダムな数を選択
    
    return start_action

def select_next_action(current_action, pheromone, map, alpha, beta):
    # フェロモン情報とヒューリスティック情報を考慮して次の位置を選択する関数
    # current_location: 現在の位置
    # pheromone: フェロモン情報の行列
    # map: マップ情報の行列
    # alpha: フェロモン重みパラメータ
    # beta: ヒューリスティック重みパラメータ
    
    num_actions = pheromone.shape[0]
    pheromone_values = pheromone[current_action]  # 現在の位置のフェロモン量
    heuristic_values = 1.0 / (map[current_action] + 1e-6)  # ヒューリスティック情報（例: 距離の逆数）
    
    probabilities = np.power(pheromone_values, alpha) * np.power(heuristic_values, beta)
    probabilities /= np.sum(probabilities)  # 確率分布の正規化
    
    next_action = np.random.choice(num_actions, p=probabilities)  # 確率に基づいて次の位置を選択
    
    return next_action

def update_pheromone(pheromone, ant_routes, evaporation_rate, Q):
    # フェロモン情報の更新関数
    # pheromone: フェロモン情報の行列
    # ant_routes: 各蟻の経路のリスト
    # evaporation_rate: フェロモン蒸発率
    # Q: フェロモンの放出量（定数）
    
    num_actions = pheromone.shape[0]
    
    # フェロモン情報の蒸発
    pheromone *= (1 - evaporation_rate)
    
    # フェロモン情報の更新
    for route in ant_routes:
        for i in range(len(route)-1):
            current_action = route[i]
            next_action = route[i+1]
            pheromone[current_action][next_action] += Q / len(route)  # フェロモンの放出量
    
    # フェロモン情報の制約（上限、下限）を設定する場合は、適切な処理を追加する


def evaporate_pheromone(pheromone, evaporation_rate):
    # フェロモン情報の蒸発関数
    # pheromone: フェロモン情報の行列
    # evaporation_rate: フェロモン蒸発率
    
    pheromone *= (1 - evaporation_rate)
    
    # フェロモン情報の制約（上限、下限）を設定する場合は、適切な処理を追加する


# マップの初期化
map = initialize_map()  # マリオのゲームマップの初期化

# フェロモン情報の初期化
pheromone = np.ones_like(map)  # フェロモン情報の初期値を1とする

# 最適経路とその評価値
best_route = None
best_score = float('-inf')


y = []
elite_ant_routes = [0]*ELITE_NUMB
# 反復の開始
for iteration in range(num_iterations):
    ant_routes = []  # 各蟻の経路を保持するリスト
    reward_arr = []  # 各蟻のtotal_rewardのリスト
    
    for ant in range(num_ants):
        current_action = initialize_start_action()  # ランダムにスタート位置を選択
        route = [current_action]  # 経路の初期化
        
        observation = env.reset()  # reset for each new trial
        done = False
        total_reward = 0
        total_time = 0

        # ゴールまで移動
        while not done and total_time < MAX_TIME:  # doneがtrueになるか時間切れになるまで繰り返し
            next_state, reward, done, info = env.step(current_action)
            total_reward += reward
            total_time += 1
            next_action = select_next_action(current_action, pheromone, map, alpha, beta)
            route.append(next_action)
            current_action = next_action
        
        ant_routes.append(route)
        reward_arr.append(total_reward)
    
    #保存されたエリートで蟻を上書き
    if iteration != 0:
        for i in range(ELITE_NUMB):
            ant_routes[i] = elite_ant_routes[i]
            while not done and total_time < MAX_TIME:
                action = ant_routes[i,total_time]
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                total_time += 1
            reward_arr[i] = total_reward

    print(reward_arr)

    
    # フェロモン情報の更新
    update_pheromone(pheromone, ant_routes, evaporation_rate, Q)
    evaporate_pheromone(pheromone, evaporation_rate)
    
    # 最適経路の更新
    k = 0
    for route in ant_routes:
        score = reward_arr[k]
        if score > best_score:
            best_route = route
            best_score = score
        k += 1
    y.append(best_score)
    print(best_score)

    #エリートの保存
    elite_ant_idx = np.argsort(reward_arr)[::-1][0:ELITE_NUMB]
    print(elite_ant_idx)
    for i in range(ELITE_NUMB):
        elite_ant_routes[i] = ant_routes[elite_ant_idx[i]]
    
print("Best Route:", best_route)
print("Best Score:", best_score)

observation = env.reset()  # reset for each new trial
done = False
total_reward = 0
total_time = 0
time = 0
frames=[]
while not done and total_time < MAX_TIME: # run for MAX_TIME timesteps or until done, whichever is first
    frames.append(copy.deepcopy(env.render(mode = 'rgb_array')))
    action = best_route[total_time]
    #print(action)
    #print("\n")
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    total_time += 1

import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
from IPython.display import HTML

x = np.arange(1,num_iterations+1,1)
plt.plot(x,y)
#plt.xticks( np.arange(1, EPISODE_NUMB+1, 1))	
plt.savefig("score.png")

plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
patch = plt.imshow(frames[0])
plt.axis('off')
animate = lambda i: patch.set_data(frames[i])
ani = matplotlib.animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval = 50)
#HTML(ani.to_jshtml())
ani.save("/workspace/mario.gif", writer="pillow")
