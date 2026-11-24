from .components.config import arguments
from .components.runner import run

if __name__ == '__main__':
    directory, failures = run(arguments())
    print(f'Reports: {directory}')
    if failures:
        raise SystemExit(f'{failures} sample(s) failed; inspect samples/*.json.')
