import sys
import rl_training.train as train


def main() -> None:
    """Optional short integration run; safe to import during unittest discovery."""
    train.CYCLES = 2
    train.STEPS_PER_CYCLE = 2048
    train.main()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
