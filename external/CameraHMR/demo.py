import argparse
from mesh_estimator import HumanMeshEstimator


def make_parser():
    parser = argparse.ArgumentParser(description="CameraHMR Regressor")
    parser.add_argument(
        "--image_folder", "--image_folder", type=str, help="Path to input image folder."
    )
    parser.add_argument(
        "--output_folder",
        "--output_folder",
        type=str,
        help="Path to folder output folder.",
    )
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    estimator = HumanMeshEstimator()
    estimator.run_on_images(args.image_folder, args.output_folder)


if __name__ == "__main__":
    main()
