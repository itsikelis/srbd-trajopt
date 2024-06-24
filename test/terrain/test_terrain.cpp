#include <iostream>

#include <trajopt/terrain/terrain_grid.hpp>

static constexpr size_t Rows = 3;
static constexpr size_t Cols = 3;

int main()
{

    trajopt::TerrainGrid terrain(Rows, Cols, 0.7, -100, -100, 100, 100);
    terrain.SetZero();
    // terrain.set_grid(grid);

    std::cout << terrain.GetHeight(0., 0.) << std::endl;

    return 0;
}
