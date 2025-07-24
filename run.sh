#!/usr/bin/env bash

# run.sh - A script to execute the Connect 4 GPU game.
#
# This script assumes the executable 'connect4_gpu' has already been
# built using the Makefile (e.g., by running 'make').
# It runs the game with default settings.

# Check if the executable exists before trying to run it
if [ ! -f ./connect4_gpu ]; then
    echo "Error: Executable 'connect4_gpu' not found."
    echo "Please build the project first by running 'make'."
    exit 1
fi

# Run the game
./connect4_gpu
