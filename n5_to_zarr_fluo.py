import z5py
import numpy as np
from pathlib import Path
from ome_zarr import writer, scale
from zarr.codecs import BloscCodec
from ome_zarr.io import parse_url
import zarr
from ome_zarr.reader import Reader
from skimage.transform import resize


def main():
    for i in range(1, 6):
        sample_name = f"Pl{i}"
        print(sample_name)
        raw_n5_path = f"/home/buglakov/g_projects/platy_registration/data/pipeline_output/sample_processing/St-Mhc--20250113Hyb--ALMF_Stellaris-40X-Sfg_{sample_name}/multiview_registration_St-Mhc--20250113Hyb--ALMF_Stellaris-40X-Sfg_{sample_name}.n5"
        segm_n5_path = f"/home/buglakov/g_projects/platy_registration/data/pipeline_output/sample_processing/St-Mhc--20250113Hyb--ALMF_Stellaris-40X-Sfg_{sample_name}/preprocessing_St-Mhc--20250113Hyb--ALMF_Stellaris-40X-Sfg_{sample_name}.n5"

        
        raw_n5_key = "raw/aligned/weighted_average_plantseg_res"
        f = z5py.File(raw_n5_path, "r")
        ds = f[raw_n5_key]
        roi = np.s_[:]
        vol = ds[roi]
        resolution = ds.attrs["resolution"]
        print(resolution)
        print(vol.shape)

        gene_chan_n5_key = "raw/aligned/weighted_average"
        ds = f[gene_chan_n5_key]
        roi = np.s_[1, ...]
        gene_vol = ds[roi]

        print("Upsample muscle staining")
        gene_vol = resize(gene_vol,  vol.shape, order=2, anti_aliasing=False)
        
        vol = np.array([vol, gene_vol])
        print(vol.shape)

        output_dir = Path("/g/kreshuk/buglakova/data/platy_reg_benchmark_zarr2/")
        output_path = output_dir / f"stmhc_{sample_name}.ome.zarr"

        axes = [
        dict(name = 'c', type = 'channel'),
        dict(name = 'z', type = 'space', unit = 'micrometer'),
        dict(name = 'y', type = 'space', unit = 'micrometer'),
        dict(name = 'x', type = 'space', unit = 'micrometer'),
        ]
        
        coordinate_transforms = [
        [{'scale': [1, *resolution], 'type': 'scale'}],
        ]

        storage_options = dict(
        chunks=(1, 256, 256, 256), # Output chunk size
        compressors=BloscCodec(cname="blosclz"), # Compressor to be used, choose one: ['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']
        overwrite=True  # Overwrite existing data if any
        )   

        store = parse_url(output_path, mode = 'w').store 
        root = zarr.open_group(store)

        writer.write_image(image = vol, # In this case, a numpy array
                    group = root,
                    axes = axes, # Dimensionality order
                    scale_factors = [],
                    coordinate_transformations = coordinate_transforms,
                    storage_options = storage_options, zarr_format=3
                    )
                    
        
        segm_n5_key = "segmentation/merged_plantseg_res"
        f = z5py.File(segm_n5_path, "r")
        ds = f[segm_n5_key]
        roi = np.s_[:]
        nuclei_segm = ds[roi]

        labels_grp = root.create_group("labels")

        axes = [
        dict(name = 'z', type = 'space', unit = 'micrometer'),
        dict(name = 'y', type = 'space', unit = 'micrometer'),
        dict(name = 'x', type = 'space', unit = 'micrometer'),
        ]
        coordinate_transforms = [
        [{'scale': resolution, 'type': 'scale'}],
        ]
        storage_options = dict(
        chunks=(256, 256, 256), # Output chunk size
        compressors=BloscCodec(cname="blosclz"), # Compressor to be used, choose one: ['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']
        overwrite=True  # Overwrite existing data if any
        )   

        label_name = "nuclei_segmentation"
        label_grp = labels_grp.create_group(label_name)
        writer.write_image(nuclei_segm, label_grp, axes=axes, scale_factors=[], coordinate_transformations = coordinate_transforms, name=label_name,  storage_options = storage_options, zarr_format=2)
        # labels_list = ["foreground"]
        # labels_grp.attrs["labels"] = labels_list 
        writer.write_label_metadata(labels_grp, "nuclei_segmentation")   


        segm_n5_key = "segmentation/cell_regions"
        f = z5py.File(segm_n5_path, "r")
        ds = f[segm_n5_key]
        roi = np.s_[:]
        cell_segm = ds[roi]
        cell_segm = resize(cell_segm,  nuclei_segm.shape, order=0, anti_aliasing=False)

        label_name = "cell_segmentation"
        label_grp = labels_grp.create_group(label_name)
        writer.write_image(cell_segm , label_grp, axes=axes, scale_factors=[], coordinate_transformations = coordinate_transforms, name=label_name,  storage_options = storage_options, zarr_format=2)
        writer.write_label_metadata(labels_grp, "cell_segmentation")

        reader = Reader(parse_url(output_path))
        nodes = list(reader())
        for idx, node in enumerate(nodes):
            print(f"The node at the level {idx} is {node}")



if __name__ == "__main__":
    main()


