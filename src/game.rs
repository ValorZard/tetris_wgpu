use ::rand::prelude::*;

pub const BOARD_WIDTH: usize = 10;
pub const BOARD_HEIGHT: usize = 20;
pub const HIDDEN_BOARDS_ROWS: u32 = 2;

pub struct Board {
    pub grid: Vec<Vec<i8>>,
}

impl Board {
    pub fn new() -> Self {
        let mut grid = vec![vec![0; BOARD_HEIGHT]; BOARD_WIDTH];
        Board { grid: grid }
    }
    pub fn spawn_random_board(&mut self) {
        let mut rng = rand::thread_rng();
        for x in 0..BOARD_WIDTH {
            for y in 0..BOARD_HEIGHT {
                if rng.gen_bool(0.5) {
                    self.grid[x][y] = 1;
                } else {
                    self.grid[x][y] = 0;
                }
            }
        }
    }

    // function to compute the next generation
    // reference taken from https://dev.to/dineshgdk/game-of-life-in-rust-4mfc
    /* 
    pub fn update_board(&mut self) {
        // get the number of rows
        let n = self.grid.len();

        // get the number of columns
        let m = self.grid[0].len();

        // create an empty grid to compute the future generation
        let mut future: Vec<Vec<i8>> = vec![vec![0; n]; m];

        // iterate through each and every cell
        for i in 0..n {
            for j in 0..m {
                // the current state of the cell (alive / dead)
                let cell_state = self.grid[i][j];

                // variable to track the number of alive neighbors
                let mut live_neighbors = 0;

                // iterate through every neighbors including the current cell
                for x in -1i8..=1 {
                    for y in -1i8..=1 {
                        // position of one of the neighbors (new_x, new_y)
                        let new_x = (i as i8) + x;
                        let new_y = (j as i8) + y;

                        // make sure the position is within the bounds of the grid
                        if new_x > 0 && new_y > 0 && new_x < n as i8 && new_y < m as i8 {
                            live_neighbors += self.grid[new_x as usize][new_y as usize];
                        }
                    }
                }

                // substract the state of the current cell to get the number of alive neighbors
                live_neighbors -= cell_state;

                println!("i : {}, j: {}", i, j);
                // applying the rules of game of life to get the future generation
                if cell_state == 1 && live_neighbors < 2 {
                    future[i][j] = 0;
                } else if cell_state == 1 && live_neighbors > 3 {
                    future[i][j] = 0;
                } else if cell_state == 0 && live_neighbors == 3 {
                    future[i][j] = 1;
                } else {
                    future[i][j] = cell_state;
                }
            }
        }

        // return the future generation
        self.grid = future;
    }
    */

    /*
    fn count_live_cells_around_cell(&self, x: i16, y: i16) -> u8{
        let mut cell_count = 8; // we are actually going to subtract from this instead of adding
        // 1 2 3
        // 4 0 5 -> zero is the cell around it.
        // 6 7 8

        // 1 cell
        let x1 = x - 1;
        let y1 = y + 1;

        if x1 < 0 || y1 > BOARD_HEIGHT as i16 - 1{
            cell_count -= 1;
        }
        else if self.grid[x1 as usize][y1 as usize] == 0
        {
            cell_count -= 1;
        }

        // 2 cell
        let x2 = x;
        let y2 = y + 1;
        if y2 > BOARD_HEIGHT as i16 - 1{
            cell_count -= 1;
        }
        else if self.grid[x2 as usize][y2 as usize] == 0
        {
            cell_count -= 1;
        }

        // 3 cell
        let x3 = x + 1;
        let y3 = y + 1;
        if y3 > BOARD_HEIGHT as i16 - 1 || x3 > BOARD_WIDTH as i16 - 1{
            cell_count -= 1;
        }
        else if self.grid[x3 as usize][y3 as usize] == 0
        {
            cell_count -= 1;
        }

        // 4 cell
        let x4 = x - 1;
        let y4 = y;
        if x4 < 0 {
            cell_count -= 1;
        }
        else if self.grid[x4 as usize][y4 as usize] == 0
        {
            cell_count -= 1;
        }

        // 5 cell
        let x5 = x + 1;
        let y5 = y;
        if x5 > BOARD_WIDTH as i16 - 1{
            cell_count -= 1;
        }
        else if self.grid[x5 as usize][y5 as usize] == 0
        {
            cell_count -= 1;
        }

        // 6 cell
        let x6 = x - 1;
        let y6 = y - 1;
        if x6 < 0 || y6 < 0 {
            cell_count -= 1;
        }
        else if self.grid[x6 as usize][y6 as usize] == 0
        {
            cell_count -= 1;
        }

        // 7 cell
        let x7 = x;
        let y7 = y - 1;
        if y2 < 0 {
            cell_count -= 1;
        }
        else if self.grid[x7 as usize][y7 as usize] == 0
        {
            cell_count -= 1;
        }

        // 8 cell
        let x8 = x + 1;
        let y8 = y - 1;
        if y8 < 0 || x8 > BOARD_WIDTH as i16 - 1{
            cell_count -= 1;
        }
        else if self.grid[x8 as usize][y8 as usize] == 0
        {
            cell_count -= 1;
        }

        cell_count
    }

    pub fn update(&mut self) {
        // Any live cell with two or three live neighbours survives.
        // Any dead cell with three live neighbours becomes a live cell.
        // All other live cells die in the next generation. Similarly, all other dead cells stay dead.
        for x in 0..BOARD_WIDTH {
            for y in 0..BOARD_HEIGHT {
                if self.count_live_cells_around_cell(x as i16, y as i16) == 3 {
                    if self.grid[x][y] == 0 {
                        self.grid[x][y] = 1;
                    }
                }
                else if self.count_live_cells_around_cell(x as i16, y as i16) != 2 {
                    if self.grid[x][y] == 1 {
                        self.grid[x][y] = 0;
                    }
                }
            }
        }
    }
    */
}
