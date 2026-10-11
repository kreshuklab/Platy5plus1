from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from pathlib import Path

def read_sample_images(data_dir, sample_name):
    data_dir = Path(data_dir)
    reader = Reader(parse_url(data_dir / f"{sample_name}.ome.zarr"))
    nodes = list(reader())
#     for idx, node in enumerate(nodes):
#             print(f"The node at the level {idx} is {node.metadata}")
    
    image_node = nodes[0]

    raw = image_node.data[0][0, ...].compute()
    muscle = image_node.data[0][1, ...].compute()

    nuclei_segm_node = nodes[2]
    nuclei_segm = nuclei_segm_node.data[0].compute()

    cell_segm_node = nodes[3]
    cell_segm = cell_segm_node.data[0].compute()

    # reg_gt_node = nodes[4]
    # reg_gt = reg_gt_node.data[0].compute()

    sample_images = {"raw": raw,
                     "muscle": muscle,
                     "nuclei_segm": nuclei_segm,
                     "cell_segm": cell_segm,
                    #  "reg_gt": reg_gt
                     }

    resolution = nuclei_segm_node.metadata["coordinateTransformations"][0][0]["scale"]

    return sample_images, resolution