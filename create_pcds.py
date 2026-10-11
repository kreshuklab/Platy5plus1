import z5py
import numpy as np
from pathlib import Path
from ome_zarr import writer, scale
from zarr.codecs import BloscCodec
from ome_zarr.io import parse_url
import zarr
from ome_zarr.reader import Reader
from skimage.transform import resize
from skimage.measure import regionprops_table
import pandas as pd
from platy5p1.data import read_sample_images


def main():
    data_dir = Path("data")
    reg_gt = pd.read_csv("data/match_gt.csv")
    reg_gt_reflected = pd.read_csv("data/match_gt_reflected.csv")

    sample_names = [
        "stmhc_pl1",
        "stmhc_pl2",
        "stmhc_pl3",
        "stmhc_pl3",
        "stmhc_pl4",
        "stmhc_pl5",
        "em",
    ]

    for sample_name in sample_names:
        print(sample_name)
        sample_images, resolution = read_sample_images(data_dir, sample_name)

        if sample_name == "em":
            segm = sample_images["nuclei_segm"]
        else:
            segm = sample_images["cell_segm"]
        props = pd.DataFrame(
            regionprops_table(
                segm,
                intensity_image=sample_images["muscle"],
                properties=("label", "centroid", "intensity_mean"),
            )
        )
        props = props.rename(
            {
                "centroid-0": "z",
                "centroid-1": "y",
                "centroid-2": "x",
                "intensity_mean": "mean_stmhc",
            },
            axis=1,
        )

        # Assign GT
        gt_dict = pd.Series(reg_gt.idx.values, index=reg_gt[sample_name]).to_dict()
        props["reg_gt"] = [
            gt_dict[idx] if idx in gt_dict.keys() else -1 for idx in props["label"]
        ]

        # Assign reflected GT
        gt_dict_reflected = pd.Series(
            reg_gt_reflected.idx.values, index=reg_gt_reflected[sample_name]
        ).to_dict()
        props["reg_gt_reflected"] = [
            gt_dict_reflected[idx] if idx in gt_dict_reflected.keys() else -1
            for idx in props["label"]
        ]

        props.to_csv(data_dir / f"{sample_name}.csv", index=False)


if __name__ == "__main__":
    main()
