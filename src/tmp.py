import time

from tqdm import tqdm

num_episodes = 10000

if __name__ == '__main__':
    # for i in range(10):
    #     with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
    #         for episode in range(1000):
    #             time.sleep(0.01)
    #             pbar.update(1)
    #             pbar.set_postfix({
    #                 'episode': episode,
    #                 "iteration": i,
    #             })

    is_over = [1 if terminated or truncated else 0 for terminated, truncated in
               zip([True,True, False], [False,True, False])]
    print(is_over)