use ::rand::prelude::*;

pub const BOARD_WIDTH: usize = 10;
pub const BOARD_HEIGHT: usize = 20;
pub const HIDDEN_BOARDS_ROWS: u32 = 2;

pub struct Board {
    pub grid: [[u8; BOARD_HEIGHT]; BOARD_WIDTH],
}

impl Board {
    pub fn new() -> Self{
        let mut grid = [[0 as u8; BOARD_HEIGHT]; BOARD_WIDTH];
        for x in 0..BOARD_WIDTH {
            for y in 0..BOARD_HEIGHT {
                grid[x][y] = 0;
            }
        }
        Board{grid: grid}
    }
    pub fn spawn_random_board(&mut self) {
        let mut rng = rand::thread_rng();
        for x in 0..BOARD_WIDTH {
            for y in 0..BOARD_HEIGHT {
                if rng.gen_bool(0.5) {
                    self.grid[x][y] = 1;
                }
                else
                {
                    self.grid[x][y] = 0;
                }
            }
        }
    }
}



