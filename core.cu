#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <ctime>

// Platform-specific headers for sleep functionality.
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Constants for the game board.
constexpr int kRows = 6;
constexpr int kCols = 7;
constexpr int kEmpty = 0;
constexpr int kPlayerX = 1;  // Represents GPU 1
constexpr int kPlayerO = 2;  // Represents GPU 2

// ===================================================================
// Host (CPU) Functions
// ===================================================================

// Prints the game board to the console.
void PrintBoard(int* board) {
  std::cout << "\n  1   2   3   4   5   6   7\n";
  std::cout << "-----------------------------\n";
  for (int i = 0; i < kRows; ++i) {
    std::cout << "|";
    for (int j = 0; j < kCols; ++j) {
      char piece = ' ';
      if (board[i * kCols + j] == kPlayerX) piece = 'X';
      if (board[i * kCols + j] == kPlayerO) piece = 'O';
      std::cout << " " << piece << " |";
    }
    std::cout << std::endl;
  }
  std::cout << "-----------------------------\n" << std::flush;
}

// Checks if a player has won. This is run on the host after each move.
bool CheckWin(int* board, int player) {
  // Check horizontal.
  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c < kCols - 3; ++c) {
      if (board[r * kCols + c] == player &&
          board[r * kCols + c + 1] == player &&
          board[r * kCols + c + 2] == player &&
          board[r * kCols + c + 3] == player) {
        return true;
      }
    }
  }
  // Check vertical.
  for (int r = 0; r < kRows - 3; ++r) {
    for (int c = 0; c < kCols; ++c) {
      if (board[r * kCols + c] == player &&
          board[(r + 1) * kCols + c] == player &&
          board[(r + 2) * kCols + c] == player &&
          board[(r + 3) * kCols + c] == player) {
        return true;
      }
    }
  }
  // Check positive diagonal.
  for (int r = 0; r < kRows - 3; ++r) {
    for (int c = 0; c < kCols - 3; ++c) {
      if (board[r * kCols + c] == player &&
          board[(r + 1) * kCols + c + 1] == player &&
          board[(r + 2) * kCols + c + 2] == player &&
          board[(r + 3) * kCols + c + 3] == player) {
        return true;
      }
    }
  }
  // Check negative diagonal.
  for (int r = 3; r < kRows; ++r) {
    for (int c = 0; c < kCols - 3; ++c) {
      if (board[r * kCols + c] == player &&
          board[(r - 1) * kCols + c + 1] == player &&
          board[(r - 2) * kCols + c + 2] == player &&
          board[(r - 3) * kCols + c + 3] == player) {
        return true;
      }
    }
  }
  return false;
}

// ===================================================================
// Device (GPU) Functions & Kernels
// ===================================================================

// A helper __device__ function that runs on the GPU to check for a win.
__device__ bool CheckWinDevice(int* board, int player) {
  // Horizontal
  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c < kCols - 3; ++c) {
      if (board[r * kCols + c] == player && board[r * kCols + c + 1] == player &&
          board[r * kCols + c + 2] == player && board[r * kCols + c + 3] == player) return true;
    }
  }
  // Vertical
  for (int r = 0; r < kRows - 3; ++r) {
    for (int c = 0; c < kCols; ++c) {
      if (board[r * kCols + c] == player && board[(r + 1) * kCols + c] == player &&
          board[(r + 2) * kCols + c] == player && board[(r + 3) * kCols + c] == player) return true;
    }
  }
  // Positive Diagonal
  for (int r = 0; r < kRows - 3; ++r) {
    for (int c = 0; c < kCols - 3; ++c) {
      if (board[r * kCols + c] == player && board[(r + 1) * kCols + c + 1] == player &&
          board[(r + 2) * kCols + c + 2] == player && board[(r + 3) * kCols + c + 3] == player) return true;
    }
  }
  // Negative Diagonal
  for (int r = 3; r < kRows; ++r) {
    for (int c = 0; c < kCols - 3; ++c) {
      if (board[r * kCols + c] == player && board[(r - 1) * kCols + c + 1] == player &&
          board[(r - 2) * kCols + c + 2] == player && board[(r - 3) * kCols + c + 3] == player) return true;
    }
  }
  return false;
}

// Kernel to initialize the cuRAND states for each GPU thread.
__global__ void SetupKernel(curandState *state, unsigned long long seed) {
  int id = threadIdx.x;
  curand_init(seed, id, 0, &state[id]);
}

// The main CUDA kernel. Each thread evaluates one possible move (one column).
__global__ void EvaluateMoves(int* board, int player, int* scores, curandState* states) {
  int col = threadIdx.x;
  if (col >= kCols) return;

  // Each thread gets its own copy of the board to simulate a move.
  int temp_board[kRows * kCols];
  for(int i = 0; i < kRows * kCols; ++i) {
    temp_board[i] = board[i];
  }

  // Find the first empty row in the column assigned to this thread.
  int row = -1;
  for (int r = kRows - 1; r >= 0; --r) {
    if (temp_board[r * kCols + col] == kEmpty) {
      row = r;
      break;
    }
  }

  if (row != -1) {
    // 1. Check for an immediate win.
    temp_board[row * kCols + col] = player;
    if (CheckWinDevice(temp_board, player)) {
      scores[col] = 100;  // High score for a winning move.
      return;
    }

    // 2. Check if the opponent can win on the next turn (blocking move).
    int opponent = (player == kPlayerX) ? kPlayerO : kPlayerX;
    for (int next_col = 0; next_col < kCols; ++next_col) {
      int next_row = -1;
      for (int r_opp = kRows - 1; r_opp >= 0; --r_opp) {
        if (temp_board[r_opp * kCols + next_col] == kEmpty) {
          next_row = r_opp;
          break;
        }
      }
      if (next_row != -1) {
        temp_board[next_row * kCols + next_col] = opponent;
        if (CheckWinDevice(temp_board, opponent)) {
          scores[col] = 50;  // Good score for a blocking move.
          temp_board[next_row * kCols + next_col] = kEmpty;
          return;
        }
        temp_board[next_row * kCols + next_col] = kEmpty;
      }
    }
    
    // 3. If no critical move, assign a small random score.
    curandState local_state = states[col];
    scores[col] = curand_uniform(&local_state) * 10;
    states[col] = local_state;

  } else {
    scores[col] = -1;  // Invalid move (column is full).
  }
}

// ===================================================================
// Main Game Loop
// ===================================================================
int main(int argc, char** argv) {
  int delay_ms = 1000; // Default delay of 1 second.
  // Command-line argument parsing for custom delay.
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-d" && i + 1 < argc) {
      try {
        delay_ms = std::stoi(argv[i + 1]);
      } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid value for delay. Using default." << std::endl;
      }
    }
  }

  // --- Host Memory Allocation ---
  int* h_board = new int[kRows * kCols];
  int* h_scores = new int[kCols];
  std::fill(h_board, h_board + kRows * kCols, kEmpty);

  // --- Device Memory Allocation ---
  int* d_board;
  int* d_scores;
  curandState* d_states;
  cudaMalloc(&d_board, kRows * kCols * sizeof(int));
  cudaMalloc(&d_scores, kCols * sizeof(int));
  cudaMalloc(&d_states, kCols * sizeof(curandState));
  
  // --- Initialize CUDA Random Number Generator States ---
  SetupKernel<<<1, kCols>>>(d_states, time(NULL));
  cudaDeviceSynchronize();

  int current_player = kPlayerX;
  int turn = 0;
  const int max_turns = kRows * kCols;

  std::cout << "Connect 4: GPU 1 (X) vs. GPU 2 (O)" << std::endl;
  PrintBoard(h_board);

  while (turn < max_turns) {
    // --- AI Makes a Move ---
    cudaMemcpy(d_board, h_board, kRows * kCols * sizeof(int), cudaMemcpyHostToDevice);
    EvaluateMoves<<<1, kCols>>>(d_board, current_player, d_scores, d_states);
    cudaDeviceSynchronize();
    cudaMemcpy(h_scores, d_scores, kCols * sizeof(int), cudaMemcpyDeviceToHost);

    int best_move = -1;
    int max_score = -2;
    for (int i = 0; i < kCols; ++i) {
      if (h_scores[i] > max_score) {
        max_score = h_scores[i];
        best_move = i;
      }
    }
    
    if (max_score < 0) {
      std::cout << "\nNo valid moves left. It's a draw." << std::endl;
      break;
    }

    for (int r = kRows - 1; r >= 0; --r) {
      if (h_board[r * kCols + best_move] == kEmpty) {
        h_board[r * kCols + best_move] = current_player;
        break;
      }
    }

    // --- Post-Move ---
    std::cout << "\nPlayer " << (current_player == kPlayerX ? "'X'" : "'O'") 
              << " (GPU " << current_player << ") drops piece in column " << best_move + 1 << std::endl;
    PrintBoard(h_board);

    if (CheckWin(h_board, current_player)) {
      std::cout << "\nPlayer " << (current_player == kPlayerX ? "'X'" : "'O'") << " wins!" << std::endl;
      break;
    }

    current_player = (current_player == kPlayerX) ? kPlayerO : kPlayerX;
    turn++;

    if (turn == max_turns) {
      std::cout << "\nIt's a draw!" << std::endl;
    }
    
    // Delay between turns.
    #ifdef _WIN32
    Sleep(delay_ms);
    #else
    usleep(delay_ms * 1000);
    #endif
  }

  // --- Cleanup ---
  delete[] h_board;
  delete[] h_scores;
  cudaFree(d_board);
  cudaFree(d_scores);
  cudaFree(d_states);

  return 0;
}

