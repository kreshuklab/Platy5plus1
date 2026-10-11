import z5py
import numpy as np
from pathlib import Path
from ome_zarr import writer, scale
from zarr.codecs import BloscCodec
from ome_zarr.io import parse_url
import zarr
from ome_zarr.reader import Reader
from skimage.transform import resize
from scipy.ndimage import affine_transform
from skimage.transform import rescale
import matplotlib.pyplot as plt
from typing import List
import pandas as pd
from platy5p1.data import read_sample_images

def bbox_3D(img, margin=10):
        if isinstance(margin, int):
                margin = [margin] * img.ndim
        r = np.any(img, axis=(1, 2))
        c = np.any(img, axis=(0, 2))
        z = np.any(img, axis=(0, 1))

        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        return max(rmin - margin[0], 0), min(rmax + margin[0], img.shape[0]), max(cmin - margin[1], 0), min(cmax + margin[1], img.shape[1]), max(zmin - margin[2], 0), min(zmax + margin[2], img.shape[2])




def transform_volume_affine(moving_img: np.array, output_shape: List, affine_mat: np.array, order=0):
    registered_img = affine_transform(moving_img, affine_mat, order=order, output_shape=output_shape, prefilter=False)
    return registered_img


def prealign(vol: np.array, order=0, cur_res=[0.1, 0.079, 0.079]):
    final_res = [0.129, 0.129, 0.129]
    em_res = [0.32, 0.32, 0.32]
    final_shape = [1736, 2232, 2232]

    affine_mat = np.loadtxt("affine.txt")
    rescaling_factors = [c / f for f, c in zip(cur_res, em_res)]
    S_1 = np.zeros_like(affine_mat)
    S_1[0, 0] = rescaling_factors[0]
    S_1[1, 1] = rescaling_factors[1]
    S_1[2, 2] = rescaling_factors[2]
    S_1[3, 3] = 1

    rescaling_factors = [c / f for f, c in zip(em_res, final_res)]
    S_2 = np.zeros_like(affine_mat)
    S_2[0, 0] = rescaling_factors[0]
    S_2[1, 1] = rescaling_factors[1]
    S_2[2, 2] = rescaling_factors[2]
    S_2[3, 3] = 1

    affine_mat = S_1 @ affine_mat @ S_2
    prealigned_vol = transform_volume_affine(vol, final_shape, affine_mat, order=order)
    print(prealigned_vol.shape)


    return prealigned_vol


def main():
    sample_name = f"EM"
    output_dir = Path("/g/kreshuk/buglakova/data/platy_reg_benchmark/")
    output_path = output_dir / f"em.ome.zarr"

    raw_n5_path = f"/scratch/buglakova/data/platy5plus1/prospr-6dpf-1-whole-raw.n5/"
    raw_n5_key = "setup0/timepoint0/s3/"

    nuclei_segm_n5_path = f"/home/buglakov/g_projects/platy_registration/data/sbem-6dpf-1-whole-segmented-nuclei.n5"
    nuclei_segm_n5_key = "setup0/timepoint0/s0"
    
    cell_segm_n5_path = f"/scratch/buglakova/data/platy5plus1/sbem-6dpf-1-whole-segmented-cells.n5"
    cell_segm_n5_key = "setup0/timepoint0/s2"

    muscle_segm_n5_path = f"/scratch/buglakova/data/platy5plus1/sbem-6dpf-1-whole-segmented-muscle.n5"
    muscle_segm_n5_key = "setup0/timepoint0/s0"
    muscle_res = [0.1, 0.1, 0.1]

    resolution = [0.129, 0.129, 0.129]

    # # Read raw EM

    # f = z5py.File(raw_n5_path, "r")
    # ds = f[raw_n5_key]
    # roi = np.s_[:]
    # raw = ds[roi]
    # print(raw.shape)
    # plt.imshow(raw[raw.shape[0] // 2, :, :])
    # plt.savefig("em.png", dpi=300)

    # raw = prealign(raw, order=2)
    # bbox = bbox_3D(raw > 0, margin=10)
    # zmin, zmax, ymin, ymax, xmin, xmax = [int(coord) for coord in bbox]
    # raw = raw[zmin:zmax, ymin:ymax, xmin:xmax]
    
    # plt.imshow(raw[raw.shape[0] // 2, :, :])
    # plt.savefig("em_transformed.png", dpi=300)

    # # Read muscle segmentation
    # f = z5py.File(muscle_segm_n5_path, "r")
    # ds = f[muscle_segm_n5_key]
    # roi = np.s_[:]
    # muscle = ds[roi]
    # print(muscle.shape)
    # plt.imshow(muscle[muscle.shape[0] // 2, :, :])
    # plt.savefig("em_muscle.png", dpi=300)

    # muscle = prealign(muscle, order=0, cur_res=muscle_res)
    # muscle = muscle[zmin:zmax, ymin:ymax, xmin:xmax] / 255 # segmentation goes up to 255

    # plt.imshow(muscle[muscle.shape[0] // 2, :, :])
    # plt.savefig("em_muscle_transformed.png", dpi=300)


    # # Write raw EM
    # raw = np.stack([raw, muscle])

    # axes = [
    # dict(name = 'c', type = 'channel'),
    # dict(name = 'z', type = 'space', unit = 'micrometer'),
    # dict(name = 'y', type = 'space', unit = 'micrometer'),
    # dict(name = 'x', type = 'space', unit = 'micrometer'),
    # ]

    # # Write raw EM and muscle segmentation
    # coordinate_transforms = [
    # [{'scale': [1, *resolution], 'type': 'scale'}],
    # ]

    # storage_options = dict(
    # chunks=(1, 256, 256, 256), # Output chunk size
    # compressors=BloscCodec(cname="blosclz"), # Compressor to be used, choose one: ['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']
    # overwrite=True  # Overwrite existing data if any
    # )   

    # store = parse_url(output_path, mode = 'w').store 
    # root = zarr.open_group(store)

    # writer.write_image(image = raw, # In this case, a numpy array
    #             group = root,
    #             axes = axes, # Dimensionality order
    #             scale_factors = [],
    #             coordinate_transformations = coordinate_transforms,
    #             storage_options = storage_options, zarr_format=3
    #             )

    # root.attrs["omero"] = {
    #     "channels": [{
    #         "color": 'gray',
    #         "label": "raw EM",
    #         "active": True,
    #     },
    #     {
    #         "color": 'yellow',
    #         "label": "muscle",
    #         "active": True,
    #     }]
    # }




    # # Read nuclei segmentation            
    # f = z5py.File(nuclei_segm_n5_path, "r")
    # ds = f[nuclei_segm_n5_key]
    # roi = np.s_[:]
    # nuclei_segm = ds[roi]
    # print(nuclei_segm.shape)
    # plt.imshow(nuclei_segm[nuclei_segm.shape[0] // 2, :, :])
    # plt.savefig("em_nuclei_segm.png", dpi=300)

    # nuclei_segm = prealign(nuclei_segm, order=0)
    # nuclei_segm = nuclei_segm[zmin:zmax, ymin:ymax, xmin:xmax]

    # plt.imshow(nuclei_segm[nuclei_segm.shape[0] // 2, :, :])
    # plt.savefig("em_nuclei_segm_transformed.png", dpi=300)

    # # Write nuclei segmentation  

    # labels_grp = root.create_group("labels")

    # axes = [
    # dict(name = 'z', type = 'space', unit = 'micrometer'),
    # dict(name = 'y', type = 'space', unit = 'micrometer'),
    # dict(name = 'x', type = 'space', unit = 'micrometer'),
    # ]
    # coordinate_transforms = [
    # [{'scale': resolution, 'type': 'scale'}],
    # ]
    # storage_options = dict(
    # chunks=(256, 256, 256), # Output chunk size
    # compressors=BloscCodec(cname="blosclz"), # Compressor to be used, choose one: ['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']
    # overwrite=True  # Overwrite existing data if any
    # )   

    # label_name = "nuclei_segmentation"
    # label_grp = labels_grp.create_group(label_name)
    # writer.write_image(nuclei_segm, label_grp, axes=axes, scale_factors=[], coordinate_transformations = coordinate_transforms, name=label_name,  storage_options = storage_options, zarr_format=3)
    # writer.write_label_metadata(labels_grp, "nuclei_segmentation")


    # # Read cell segmentation 
    # f = z5py.File(cell_segm_n5_path, "r")
    # ds = f[cell_segm_n5_key]
    # roi = np.s_[:]
    # cell_segm = ds[roi]
    # print(cell_segm.shape)
    # plt.imshow(cell_segm[cell_segm.shape[0] // 2, :, :])
    # plt.savefig("em_cell_segm.png", dpi=300)

    # cell_segm = prealign(cell_segm, order=0)
    # cell_segm = cell_segm[zmin:zmax, ymin:ymax, xmin:xmax]

    # plt.imshow(cell_segm[cell_segm.shape[0] // 2, :, :])
    # plt.savefig("em_cell_segm_transformed.png", dpi=300)


    # # Write cell segmentation   

    # label_name = "cell_segmentation"
    # label_grp = labels_grp.create_group(label_name)
    # writer.write_image(cell_segm , label_grp, axes=axes, scale_factors=[], coordinate_transformations = coordinate_transforms, name=label_name,  storage_options = storage_options, zarr_format=3)
    # writer.write_label_metadata(labels_grp, "cell_segmentation")

    # reader = Reader(parse_url(output_path))
    # nodes = list(reader())
    # for idx, node in enumerate(nodes):
    #     print(f"The node at the level {idx} is {node}")

    sample_images, resolution = read_sample_images(output_dir, "em")

    nuclei_segm = sample_images["nuclei_segm"]

    # Create registration ground truth volume
    reg_gt = pd.read_csv("data/match_gt.csv")
    gt_column = f"em"
    gt_dict = pd.Series(reg_gt.idx.values, index=reg_gt[gt_column]).to_dict()

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

    writer.write_label_metadata(labels_grp, "reg_gt")


if __name__ == "__main__":
    main()


