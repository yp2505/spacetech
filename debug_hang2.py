import sys, traceback, faulthandler
import rl_training.train as train

if __name__ == '__main__':
    faulthandler.dump_traceback_later(10, repeat=False, file=sys.stderr)
    train.CYCLES = 1
    train.STEPS_PER_CYCLE = 256
    train.N_ENVS = 2
    sys.argv = ['train.py', '--fast-eval', '--cycles', '1', '--steps-per-cycle', '256', '--n-envs', '2']
    train.main()
