"""Entry point of the interactive-sphere marker demo: starts the viewer."""

import os
import sys

# All algorithm modules live in Functions/ next to this entry script.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "Functions"))


def main():
    sys.argv = ["viewer_qt.py"] + sys.argv[1:]
    import viewer_qt
    viewer_qt.main()


if __name__ == "__main__":
    main()
