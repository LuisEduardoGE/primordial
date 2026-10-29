"""Head-to-head: do evolved brains beat random ones, or each other?

    python scripts/compete.py run.pt                 # evolved vs random brains
    python scripts/compete.py new.pt old.pt          # two evolved populations

Thin wrapper around `primordial.tournament` (which also powers `primordial experiment`).
"""

from primordial.tournament import main

if __name__ == "__main__":
    main()
