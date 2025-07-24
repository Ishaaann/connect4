# Connect 4 with GPU-Accelerated AI

This project is a C++/CUDA implementation of the classic game Connect 4, featuring two AI players who use the power of a GPU to decide their next move. The primary goal is to demonstrate a practical application of parallel computing for game AI.

## Overview

The program simulates a full game of Connect 4 between two AI opponents, "GPU 1" (Player X) and "GPU 2" (Player O). Instead of complex algorithms like Minimax, this project uses a simpler, parallelized heuristic approach. The "thinking" for each turn is offloaded to the GPU, where multiple possible moves are evaluated simultaneously.

The state of the game board is printed to the console after each move, allowing you to watch the game unfold turn by turn.

## How It Works

The project uses a host-device architecture common in CUDA programming:

* **Host (CPU):** The main C++ code running on the CPU manages the overall game loop, controls player turns, prints the board, and checks for win/draw conditions.
* **Device (GPU):** A CUDA kernel is launched for each AI player's turn.

The AI's decision-making process is parallelized as follows:
1.  A CUDA kernel is launched with 7 threads, one for each column of the Connect 4 board.
2.  Each thread independently evaluates the move of dropping a piece into its assigned column.
3.  The evaluation is based on a simple heuristic:
    * **+100 points:** If the move results in an immediate win.
    * **+50 points:** If the move blocks an opponent's immediate win.
    * **A small random value:** Otherwise, to ensure a valid move is always chosen.
4.  The scores for all 7 possible moves are returned to the host.
5.  The host code selects the move with the highest score and updates the game state.

## Prerequisites

To build and run this project, you will need:
* An NVIDIA GPU that supports CUDA.
* The **NVIDIA CUDA Toolkit** (version 10.x or newer recommended) installed on your system. This includes the NVIDIA CUDA Compiler (`nvcc`).

## Building the Project

A `Makefile` is provided for easy compilation. It contains the necessary commands and flags to build the executable.

To build the project, simply run the `make` command in your terminal:
```bash
make

This will invoke nvcc to compile the connect4_cuda_ai.cu file and create an executable named connect4_gpu.

Running the Game
A shell script, run.sh, is included to run the game with default settings.

./run.sh

You can also run the executable directly from the command line.

./connect4_gpu [options]

Command-Line Arguments
The program accepts the following optional command-line argument:

-d <milliseconds>: Sets the delay in milliseconds between each turn. This is useful for slowing down the game to watch it more easily. The default is 1000ms (1 second).

Examples:

Run with a 2-second delay between turns:

./connect4_gpu -d 2000

Run with no delay for a fast simulation:

./connect4_gpu -d 0

Code Style
This project aims to adhere to the Google C++ Style Guide. This includes conventions for naming, formatting, comments, and overall structure to ensure the code is clean, readable, and maintainable.

File Structure
.
├── connect4_cuda_ai.cu   # The main C++/CUDA source code.
├── Makefile              # The build script for compiling the project.
├── run.sh                # A simple script to execute the game.
└── README.md             # This file.
