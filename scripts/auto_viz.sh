#!/bin/bash

while inotifywait -e modify results/5x5/*.csv; do
    pixi run python viz.py
done
