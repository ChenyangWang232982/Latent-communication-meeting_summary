"""Compatibility entry point: prepare only the three QMSum domains."""
from .prepare_all import DOMAINS, main


if __name__ == '__main__':
    main(default_domains=list(DOMAINS))
