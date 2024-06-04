#include <iostream>
#include <vector>

#include <trajopt/terrain/terrain_grid.hpp>

static constexpr size_t Rows = 3;
static constexpr size_t Cols = 3;

int main()
{
    std::vector<double> grid;
    grid.resize(Rows * Cols);
    // grid[0] = 1.;
    // grid[1] = 1.;
    // grid[2] = 1.;
    // grid[3] = 1.;
    // grid[4] = 1.;
    // grid[5] = 1.;
    // grid[6] = 1.;
    // grid[7] = 1.;
    // grid[8] = 1.;

    trajopt::TerrainGrid terrain(Rows, Cols, 0.7, -100, -100, 100, 100);
    // terrain.set_grid(grid);

    std::cout << terrain.n(0., 0.) << std::endl;

    return 0;
}
