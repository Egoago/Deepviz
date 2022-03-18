import logging
import os
import time

from tqdm import tqdm
from matplotlib import pyplot as plt
from environment import Environment
from agents import DQN
from environment import UI


def test(agent, env, name, trial_num=100):
    dir_path = f'../media/{name}/'
    os.makedirs(dir_path, exist_ok=True)
    #env.toggle_show(False)
    max_len = 0
    max_index = 0
    lengths = []
    for i in tqdm(range(trial_num), desc=name, unit='trial'):
        state = env.reset()
        env.view.renderer.render_to(f'{dir_path}trial-{i}')
        length = 0
        while not env.done:
            action = agent.move(state)
            state = env.step(env.actions[action])[0]
            length += 1
        env.view.renderer.video.release()
        lengths.append(length)
        if length > max_len:
            max_len = length
            if i > 0:
                os.remove(f'{dir_path}trial-{max_index}.mp4')
            max_index = i
        else:
            os.remove(f'{dir_path}trial-{i}.mp4')
    logging.info(f"Best run survived {max_len} in steps and {max_len / env.fps:.0f} in seconds")
    bx_plot = plt.boxplot(lengths, patch_artist=True)
    plt.title(name)
    plt.ylabel('Survived steps')
    plt.setp(bx_plot['boxes'], facecolor='lightgreen')
    plt.savefig(f'{dir_path}trial_statistics.png')
    plt.show()


def demo(agent, env):
    target_dt = 1/env.fps
    state = env.reset()
    start = time.time()
    while not env.done:
        action = agent.move(state)
        state = env.step(env.actions[action])[0]
        end = time.time()
        dt = end-start
        if dt < target_dt:
            time.sleep(target_dt-dt)
        start = end


def main():
    logging.basicConfig(level=logging.INFO)
    env = Environment(True)
    agent = DQN(env.state_shape(), len(env.actions))
    #UI.register_key('^', agent.save)
    #UI.register_key('&', agent.plot_training)
    #UI.register_key('%', env.toggle_show)
    test(agent, env, 'before_training')
    #agent.train()
    agent.load('../saves/1647563137.2862642.h5')
    test(agent, env, 'after_training')


if __name__ == '__main__':
    main()
