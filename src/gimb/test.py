"""Compatibility launcher; run python -m src.gimb for normal use."""
import runpy

if __name__ == "__main__":
    runpy.run_module("src.gimb", run_name="__main__")
