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


def main():
    for i in range(1, 6):
        sample_name = f"Pl{i}"
        print(sample_name)
        output_dir = Path("/g/kreshuk/buglakova/data/platy_reg_benchmark/")
        output_path = output_dir / f"stmhc_{sample_name}.ome.zarr"

        reader = Reader(parse_url(output_path))
        nodes = list(reader())
        for idx, node in enumerate(nodes):
            print(f"The node at the level {idx} is {node.metadata}")

        image_node = nodes[0]

        gene_channel = image_node.data[0][1, ...].compute()
        nuclei_segm_node = nodes[2]
        resolution = nuclei_segm_node.metadata["coordinateTransformations"][0][0][
            "scale"
        ]
        print("resolution", resolution)
        nuclei_segm = nuclei_segm_node.data[0].compute()

        cell_segm_node = nodes[3]
        cell_segm = cell_segm_node.data[0].compute()
        print("Gene staining shape", gene_channel.shape)
        print("Nuclei segmentation shape", nuclei_segm.shape)
        print("Cell segmentation shape", cell_segm.shape)

        reg_gt = pd.read_csv("match_gt.csv")
        gt_column = f"stmhc_pl{i}"
        gt_dict = pd.Series(reg_gt.em_id.values, index=reg_gt[gt_column]).to_dict()
        print(reg_gt)
        props = pd.DataFrame(
            regionprops_table(
                cell_segm,
                intensity_image=gene_channel,
                properties=("label", "centroid", "intensity_mean"),
            )
        )
        print(props)
        props = props.rename(
            {
                "centroid-0": "z",
                "centroid-1": "y",
                "centroid-2": "x",
                "intensity_mean": "mean_stmhc",
            },
            axis=1,
        )
        props["reg_gt"] = [
            gt_dict[idx] if idx in gt_dict.keys() else -1 for idx in props["label"]
        ]
        print(props)
        props.to_csv(output_dir / f"stmhc_{sample_name}.csv", index=False)
        print(output_dir / f"stmhc_{sample_name}.csv")

        gt_segm = np.zeros_like(nuclei_segm)
        for idx, gt_idx in gt_dict.items():
            gt_segm[nuclei_segm == idx] = gt_idx

        store = parse_url(output_path, mode="w").store

        axes = [
            dict(name="z", type="space", unit="micrometer"),
            dict(name="y", type="space", unit="micrometer"),
            dict(name="x", type="space", unit="micrometer"),
        ]
        coordinate_transforms = [
            [{"scale": resolution, "type": "scale"}],
        ]
        storage_options = dict(
            chunks=(256, 256, 256),  # Output chunk size
            compressors=BloscCodec(
                cname="blosclz"
            ),  # Compressor to be used, choose one: ['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']
            overwrite=True,  # Overwrite existing data if any
        )

        label_name = "reg_gt"
        labels_grp = zarr.open_group(output_path)["labels"]
        # label_grp = labels_grp["reg_gt"]
        label_grp = labels_grp.create_group(label_name)
        writer.write_image(
            gt_segm,
            label_grp,
            axes=axes,
            scale_factors=[],
            coordinate_transformations=coordinate_transforms,
            name=label_name,
            storage_options=storage_options,
            zarr_format=3,
        )
        # labels_list = ["foreground"]
        # labels_grp.attrs["labels"] = labels_list
        writer.write_label_metadata(labels_grp, "reg_gt")


if __name__ == "__main__":
    main()
