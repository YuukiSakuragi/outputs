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

# 試行回数
EPISODE_NUMB = 30

# 最大試行時間
MAX_TIME = 15000

#個体数
INDIVIDUALS_NUMB = 150
#遺伝子数
GEN_ELENGTH = MAX_TIME

#遺伝子配列
gen = np.random.randint(0,3,(INDIVIDUALS_NUMB, GEN_ELENGTH))

#エリート保存数
ELITE_NUMB = 3

#突然変異
INDIVIDUAL_MUTATION_RATE = 0.1
GEN_MUTATION_RATE = 0.1

#トーナメント選択の個体数
TORNAMENT_NUMB = 5

y = []
#x_posが増加しない場合に終了するまでのフレーム数
retire_frame = 300
for i in range(EPISODE_NUMB):
  reward_arr = []
  
  for j in range(INDIVIDUALS_NUMB):
    observation = env.reset()  # reset for each new trial
    done = False
    total_reward = 0
    total_time = 0
    x_pos = []
    while not done and total_time < MAX_TIME: # run for MAX_TIME timesteps or until done, whichever is first
        action = gen[j,total_time]
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        total_time += 1
        x_pos.append(info["x_pos"])
        if total_time > retire_frame:
          x_pos_retire_frame = x_pos[-retire_frame:]
          if x_pos_retire_frame.count(x_pos_retire_frame[0]) == len(x_pos_retire_frame):
            done = True
    reward_arr.append(total_reward)
  next_gen = np.zeros_like(gen)
  #エリート保存
  elite_idx = np.argsort(reward_arr)[::-1][0:ELITE_NUMB]
  elite_tmp =gen[elite_idx]
  print(sorted(reward_arr))
  print(elite_idx)
  print(elite_tmp)
  selected_parent_id = []
  for k in range(0, INDIVIDUALS_NUMB, 2):
    #ルーレット選択
    #親個体の選択
    #ratio_arr = reward_arr / np.sum(reward_arr)
    #parent1_idx = np.random.choice(INDIVIDUALS_NUMB,size=(1,),p=ratio_arr)
    #parent2_idx = np.random.choice(INDIVIDUALS_NUMB,size=(1,),p=ratio_arr)
    #print("parent1_idx=" + str(parent1_idx) + " parent2_idx=" + str(parent2_idx))
    #while parent1_idx == parent2_idx:
    #  parent2_idx = np.random.choice(INDIVIDUALS_NUMB,size=(1,),p=ratio_arr)
    #parent1 = gen[parent1_idx]
    #parent2 = gen[parent2_idx]
    #トーナメント選択
    parent1_idx = parent2_idx = 0
    while parent1_idx == parent2_idx:
      parents1_idx = random.sample(list(range(INDIVIDUALS_NUMB)),TORNAMENT_NUMB)
      parents2_idx = random.sample(list(range(INDIVIDUALS_NUMB)),TORNAMENT_NUMB)
      parents1_idx = np.array(parents1_idx)
      parents2_idx = np.array(parents2_idx)
      reward_arr = np.array(reward_arr)
      parents1_reward_arr = reward_arr[parents1_idx]
      parents2_reward_arr = reward_arr[parents2_idx]
      parent1_idx = np.argmax(parents1_reward_arr)
      parent2_idx = np.argmax(parents2_reward_arr)
    parent1 = gen[parent1_idx]
    parent2 = gen[parent2_idx]
    selected_parent_id.append(parent1_idx)
    selected_parent_id.append(parent2_idx)
    #二点交差の交差点の決定
    cross1 = random.randint(1,INDIVIDUALS_NUMB-3)
    cross2 = random.randint(1,INDIVIDUALS_NUMB-1)
    while cross1>=cross2:
      cross2 = random.randint(1,INDIVIDUALS_NUMB-1)
    #子供の作成
    child1 = np.concatenate([parent1[:cross1],parent2[cross1:cross2],parent1[cross2:]])
    child2 = np.concatenate([parent2[:cross1],parent1[cross1:cross2],parent2[cross2:]])
    next_gen[k] = child1
    next_gen[k+1] = child2
  print(len(np.unique(selected_parent_id)))
  #突然変異
  tmp = 0
  for l in range(INDIVIDUALS_NUMB):
    rnd = np.random.rand()
    if rnd<=INDIVIDUAL_MUTATION_RATE:
      for m in range(MAX_TIME):
        rnd = np.random.rand()
        if rnd<=GEN_MUTATION_RATE:
          next_gen[l][m] = tmp
          mutate_gen = random.randint(0,2)
          while mutate_gen != tmp:
             mutate_gen = random.randint(0,2)
          next_gen[l][m] = mutate_gen
  #保存したエリートを子供にする
  for n in range(ELITE_NUMB):
      next_gen[n] = elite_tmp[n]
  gen = next_gen
  print(np.max(reward_arr))
  y.append(np.max(reward_arr))

optimal_gen_idx = np.argmax(total_reward)
optimal_gen = gen[optimal_gen_idx] 

observation = env.reset()  # reset for each new trial
done = False
total_reward = 0
total_time = 0
time = 0
frames=[]
x_pos = []
while not done and total_time < MAX_TIME: # run for MAX_TIME timesteps or until done, whichever is first
    frames.append(copy.deepcopy(env.render(mode = 'rgb_array')))
    action = optimal_gen[total_time]
    #print(action)
    #print("\n")
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    total_time += 1
    x_pos.append(info["x_pos"])
    if total_time > retire_frame:
        x_pos_retire_frame = x_pos[-retire_frame:]
        if x_pos_retire_frame.count(x_pos_retire_frame[0]) == len(x_pos_retire_frame):
           done = True

import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
from IPython.display import HTML

x = np.arange(1,EPISODE_NUMB+1,1)
plt.plot(x,y)
#plt.xticks( np.arange(1, EPISODE_NUMB+1, 1))	
plt.savefig("score.png")

plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
patch = plt.imshow(frames[0])
plt.axis('off')
animate = lambda i: patch.set_data(frames[i])
ani = matplotlib.animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval = 50)
#HTML(ani.to_jshtml())
ani.save("/workspace/mario_gen.gif", writer="pillow")

