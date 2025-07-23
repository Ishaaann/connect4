#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <curand_kernel.h>

// Constants for the game board
#define ROWS 6
#define COLS 7
#define EMPTY 0
#define PLAYER_X 1 // Represents GPU 1
#define PLAYER_O 2 // Represents GPU 2

// ===================================================================
// Host (CPU) Functions
// ===================================================================

// Function to print the game board to the console
void printBoard(int* board) {
    std::cout << "\n  1   2   3   4   5   6   7\n";
    std::cout << "-----------------------------\n";
    for (int i = 0; i < ROWS; ++i) {
        std::cout << "|";
        for (int j = 0; j < COLS; ++j) {
            char piece = ' ';
            if (board[i * COLS + j] == PLAYER_X) piece = 'X';
            if (board[i * COLS + j] == PLAYER_O) piece = 'O';
            std::cout << " " << piece << " |";
        }
        std::cout << std::endl;
    }
    std::cout << "-----------------------------\n";
}

// Function to check if a player has won
// This function is simple enough to run on the host after each move.
bool checkWin(int* board, int player) {
    // Check horizontal
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS - 3; ++c) {
            if (board[r * COLS + c] == player &&
                board[r * COLS + c + 1] == player &&
                board[r * COLS + c + 2] == player &&
                board[r * COLS + c + 3] == player) {
                return true;
            }
        }
    }
    // Check vertical
    for (int r = 0; r < ROWS - 3; ++r) {
        for (int c = 0; c < COLS; ++c) {
            if (board[r * COLS + c] == player &&
                board[(r + 1) * COLS + c] == player &&
                board[(r + 2) * COLS + c] == player &&
                board[(r + 3) * COLS + c] == player) {
                return true;
            }
        }
    }
    // Check positive diagonal
    for (int r = 0; r < ROWS - 3; ++r) {
        for (int c = 0; c < COLS - 3; ++c) {
            if (board[r * COLS + c] == player &&
                board[(r + 1) * COLS + c + 1] == player &&
                board[(r + 2) * COLS + c + 2] == player &&
                board[(r + 3) * COLS + c + 3] == player) {
                return true;
            }
        }
    }
    // Check negative diagonal
    for (int r = 3; r < ROWS; ++r) {
        for (int c = 0; c < COLS - 3; ++c) {
            if (board[r * COLS + c] == player &&
                board[(r - 1) * COLS + c + 1] == player &&
                board[(r - 2) * COLS + c + 2] == player &&
                board[(r - 3) * COLS + c + 3] == player) {
                return true;
            }
        }
    }
    return false;
}

// ===================================================================
// Device (GPU) Functions & Kernel
// ===================================================================

// __device__ function: A helper function that runs on the GPU.
// Checks for a win condition on a given board state.
__device__ bool checkWin_device(int* board, int player) {
    // Horizontal
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS - 3; ++c) {
            if (board[r * COLS + c] == player && board[r * COLS + c + 1] == player &&
                board[r * COLS + c + 2] == player && board[r * COLS + c + 3] == player) return true;
        }
    }
    // Vertical
    for (int r = 0; r < ROWS - 3; ++r) {
        for (int c = 0; c < COLS; ++c) {
            if (board[r * COLS + c] == player && board[(r + 1) * COLS + c] == player &&
                board[(r + 2) * COLS + c] == player && board[(r + 3) * COLS + c] == player) return true;
        }
    }
    // Positive Diagonal
    for (int r = 0; r < ROWS - 3; ++r) {
        for (int c = 0; c < COLS - 3; ++c) {
            if (board[r * COLS + c] == player && board[(r + 1) * COLS + c + 1] == player &&
                board[(r + 2) * COLS + c + 2] == player && board[(r + 3) * COLS + c + 3] == player) return true;
        }
    }
    // Negative Diagonal
    for (int r = 3; r < ROWS; ++r) {
        for (int c = 0; c < COLS - 3; ++c) {
            if (board[r * COLS + c] == player && board[(r - 1) * COLS + c + 1] == player &&
                board[(r - 2) * COLS + c + 2] == player && board[(r - 3) * COLS + c + 3] == player) return true;
        }
    }
    return false;
}

// __global__ function: The CUDA kernel that runs on the GPU.
// Each thread evaluates one possible move.
__global__ void evaluateMoves(int* board, int player, int* scores, curandState* states) {
    int col = threadIdx.x; // Each thread takes one column (0-6)
    if (col >= COLS) return;

    // Create a temporary board in shared memory for this thread's simulation
    __shared__ int temp_board[ROWS * COLS];
    
    // Each thread copies the global board to its local temp board
    // This is a simplification; for larger boards, use thread-local memory.
    for(int i = 0; i < ROWS * COLS; ++i) {
        temp_board[i] = board[i];
    }
    __syncthreads(); // Wait for all threads to finish copying

    // Find the first empty row in the assigned column
    int row = -1;
    for (int r = ROWS - 1; r >= 0; --r) {
        if (temp_board[r * COLS + col] == EMPTY) {
            row = r;
            break;
        }
    }

    // If the column is not full, evaluate the move
    if (row != -1) {
        // 1. Check for an immediate win
        temp_board[row * COLS + col] = player;
        if (checkWin_device(temp_board, player)) {
            scores[col] = 100; // High score for a winning move
            return;
        }

        // 2. Check if the opponent can win on the next turn (blocking move)
        int opponent = (player == PLAYER_X) ? PLAYER_O : PLAYER_X;
        // Check all possible opponent responses to our move
        for (int next_col = 0; next_col < COLS; ++next_col) {
            // Find empty spot in opponent's column choice
            int next_row = -1;
            for (int r_opp = ROWS - 1; r_opp >= 0; --r_opp) {
                if (temp_board[r_opp * COLS + next_col] == EMPTY) {
                    next_row = r_opp;
                    break;
                }
            }
            if (next_row != -1) {
                temp_board[next_row * COLS + next_col] = opponent;
                if (checkWin_device(temp_board, opponent)) {
                    scores[col] = 50; // Good score for a blocking move
                    // Undo opponent's hypothetical move
                    temp_board[next_row * COLS + next_col] = EMPTY; 
                    return;
                }
                // Undo opponent's hypothetical move
                temp_board[next_row * COLS + next_col] = EMPTY;
            }
        }
        
        // 3. If no immediate win or block, assign a random small score
        // This makes the AI's choice non-deterministic when no critical moves exist.
        curandState localState = states[col];
        scores[col] = curand_uniform(&localState) * 10; // Score between 0 and 10
        states[col] = localState;

    } else {
        scores[col] = -1; // Invalid move (column is full)
    }
}


// ===================================================================
// Main Game Loop
// ===================================================================
int main() {
    // --- Host Memory Allocation ---
    int* h_board = new int[ROWS * COLS];
    int* h_scores = new int[COLS];
    std::fill(h_board, h_board + ROWS * COLS, EMPTY);

    // --- Device Memory Allocation ---
    int* d_board;
    int* d_scores;
    curandState* d_states;
    cudaMalloc(&d_board, ROWS * COLS * sizeof(int));
    cudaMalloc(&d_scores, COLS * sizeof(int));
    cudaMalloc(&d_states, COLS * sizeof(curandState));
    
    // --- Initialize CUDA Random Number Generator ---
    curandCreateGenerator(&d_states, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(d_states, time(NULL));


    int currentPlayer = PLAYER_X;
    int turn = 0;
    const int max_turns = ROWS * COLS;

    std::cout << "Connect 4: GPU 1 (X) vs. GPU 2 (O)" << std::endl;
    printBoard(h_board);

    while (turn < max_turns) {
        // --- AI Makes a Move ---
        
        // 1. Copy current board state from Host to Device
        cudaMemcpy(d_board, h_board, ROWS * COLS * sizeof(int), cudaMemcpyHostToDevice);

        // 2. Launch the Kernel to evaluate moves in parallel
        // We launch 1 block with 7 threads (one for each column)
        evaluateMoves<<<1, COLS>>>(d_board, currentPlayer, d_scores, d_states);
        cudaDeviceSynchronize(); // Wait for the kernel to finish

        // 3. Copy the resulting scores from Device back to Host
        cudaMemcpy(h_scores, d_scores, COLS * sizeof(int), cudaMemcpyDeviceToHost);

        // 4. Host chooses the best move based on scores
        int bestMove = -1;
        int maxScore = -2;
        for (int i = 0; i < COLS; ++i) {
            if (h_scores[i] > maxScore) {
                maxScore = h_scores[i];
                bestMove = i;
            }
        }
        
        // If no valid move was found, something is wrong, but we'll break.
        if (bestMove == -1) {
            std::cout << "No valid moves found!" << std::endl;
            break;
        }

        // 5. Make the chosen move on the host board
        int row = -1;
        for (int r = ROWS - 1; r >= 0; --r) {
            if (h_board[r * COLS + bestMove] == EMPTY) {
                h_board[r * COLS + bestMove] = currentPlayer;
                row = r;
                break;
            }
        }

        // --- Post-Move ---
        std::cout << "\nPlayer " << (currentPlayer == PLAYER_X ? "'X'" : "'O'") 
                  << " (GPU " << currentPlayer << ") drops piece in column " << bestMove + 1 << std::endl;
        printBoard(h_board);

        // Check for win
        if (checkWin(h_board, currentPlayer)) {
            std::cout << "\nPlayer " << (currentPlayer == PLAYER_X ? "'X'" : "'O'") << " wins!" << std::endl;
            break;
        }

        // Switch player
        currentPlayer = (currentPlayer == PLAYER_X) ? PLAYER_O : PLAYER_X;
        turn++;

        if (turn == max_turns) {
            std::cout << "\nIt's a draw!" << std::endl;
        }
        
        // Add a small delay to make the game watchable
        #ifdef _WIN32
        #include <windows.h>
        Sleep(1000); // 1 second delay
        #else
        #include <unistd.h>
        usleep(1000000); // 1 second delay
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
