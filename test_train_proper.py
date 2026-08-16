import sys, time
import rl_training.train as train

if __name__ == '__main__':
    train.CYCLES = 2
    train.STEPS_PER_CYCLE = 256
    train.N_ENVS = 2
    sys.argv = ['train.py', '--fast-eval', '--cycles', '2', '--steps-per-cycle', '256', '--n-envs', '2']
    t0 = time.time()
    train.main()
    print(f"DONE in {time.time()-t0:.1f}s")
