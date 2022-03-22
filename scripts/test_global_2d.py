from PLR_Hilbert_Maps.data import GridManager
from PLR_Hilbert_Maps.visualizer import VisualizePredictions


def main():
    exp_name = "global_occ_1000_free_1000"

    # load GridManager log
    gm = GridManager.gm_load(exp_name)

    # Visualizer
    x_min, x_max = 0, 10
    y_min, y_max = 0, 10
    resolution = 0.1
    visualizer = VisualizePredictions(gm, x_min, x_max, y_min, y_max, resolution)

    # visualize
    visualizer.visualize()


if __name__ == "__main__":
    main()
