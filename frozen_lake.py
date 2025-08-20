import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def choose_action(Qstate, eps):
  rand_x = np.random.random()
  #exploration vs explotation
  if rand_x < eps and Qstate.sum() > 0:
    prob = Qstate / Qstate.sum()
    action = np.random.choice(np.arange(Qstate.shape[0])) #p=prob
  elif Qstate.sum() == 0:
    action = np.random.choice(np.arange(Qstate.shape[0]))
  else: #choose action with the highest return value
    action = Qstate.argmax()
  return action


def update_epsilon_lr(eps, lr_const, i, epochs):
  if i < 0.2*epochs:
    eps = 1.0
    lr = 0.2*lr_const
  elif i >= 0.2*epochs and i < 0.4*epochs:
    eps = 0.8
    lr = 0.1*lr_const
  elif i >= 0.4*epochs and i < 0.7*epochs:
    eps = 0.6
    lr = 0.05*lr_const
  elif i >= 0.7*epochs:
    eps = 0.4
    lr = 0.02*lr_const

  return eps, lr 


def validation_q(env, Q_table, n_trials=100):
  directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
  success = 0
  trajectory_length = 0
  for trial in range(n_trials):
    episode_over = False
    env.reset()
    current_state = 0
    trajectory_list = []
    while not episode_over:
        action = Q_table[current_state,:].argmax()
        observation, reward, terminated, truncated, info = env.step(action)
        trajectory_list.append((current_state, action, observation, reward))
        current_state = observation
        episode_over = terminated or truncated
    if reward == 1:
       success = success + 1

    trajectory_length = trajectory_length + len(trajectory_list)

  success_rate = 100 * success / n_trials
  trajectory_length = trajectory_length / n_trials
  print(f"successfull trials: {success_rate:.1f}%, trajectory length: {trajectory_length:.1f}")


def train_q(env, Q_table, gamma, lr, eps, epochs):
   for i in range(epochs):
        episode_over = False
        env.reset()
        current_state = 0
        #eps - exploration rate
        eps, lr = update_epsilon_lr(eps, lr_const, i, epochs)

        while not episode_over:
            action = choose_action(Q_table[current_state, :], eps)
            #new step on the ice
            observation, reward, terminated, truncated, info = env.step(action)
            old_value = Q_table[current_state, action]
            #new iteration weights
            new_value = reward + gamma * Q_table[observation,:].max()
            #update - some kind of SGD
            Q_table[current_state, action] = old_value + lr*(new_value - old_value)
            current_state = observation
            episode_over = terminated or truncated


def qtable_directions_map(qtable, map_size):
    """Get the best learned action & map it to arrows."""
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.arange(map_size*map_size)
    qtable_directions = qtable_directions.astype("<U3")
    eps = np.finfo(float).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            # Assign an arrow only if a minimal Q-value has been learned as best action
            # otherwise since 0 is a direction, it also gets mapped on the tiles where
            # it didn't actually learn anything
            qtable_directions[idx] = qtable_directions[idx] + directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions


def plot_q_values_map(qtable, env, map_size):
    """Plot the last frame of the simulation and the policy learned."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    # Plot the policy

    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt='',
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "large"}, #xx-large
    ).set(title="Learned Q-values\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    img_title = f"frozenlake_q_values_{map_size}x{map_size}.png"
    fig.savefig("arro_map.jpg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    #'FrozenLake-v1'
    #gym.pprint_registry()
    env = gym.make(id='FrozenLake8x8-v1', desc=None, map_name="8x8", \
                    is_slippery=True, render_mode="rgb_array", p=0.98) #render_mode ansi
    n_space = env.observation_space.n
    n_action = env.action_space.n
    Q_table = np.zeros([n_space, n_action])
    gamma = 0.995
    eps = 0.99
    epochs = 7000
    lr_const = 2.0

    train_q(env, Q_table, gamma, lr_const, eps, epochs)

    # the most impressive in results is the case when the model chose ->  
    # instead of goes up if slippery is available. though it is false direction
    # it helps to avoid a trap.
    validation_q(env, Q_table, n_trials=100)

    lake_render = env.render()
    #print(lake_render)

    Q = Q_table.max(axis=1)
    Q = np.reshape(Q, [8,8])
    print(np.round(Q, decimals=2))

    plot_q_values_map(Q_table, env, 8)