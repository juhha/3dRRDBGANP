# 3D Volumetric Super-Resolution in Radiology Using 3D RRDB-GAN
<a href="https://arxiv.org/abs/2402.04171"><img src="https://img.shields.io/badge/ArXiv-2402.04171-brightgreen"></a>
<a href="https://juhha.github.io/isbi24_3dsr_page/"><img src="https://img.shields.io/badge/Page-Project_Page-blue"></a>

Official code for **3D Volumetric Super-Resolution in Radiology Using 3D RRDB-GAN** (ISBI 2024). Our 3D RRDG-GAN + 2.5D Perception (3D-RRDBGANp) achieves super-resolving finer image quality across four different experiments including 4 modalities (T1/ T2 MRI, MRH, CT), 2 species (human, mouse), and 2 body regions (brain, abdomen).

Example output of our model:
![thumbnail](./src/thumbnail.png)

## Usage
### Training:
#### GAN based method
    python train_srgan.py --exp_name rrdbgan_msd6_allviews
#### Non-GAN based method
    python train_sr.py --exp_name rrdb_msd6
You can change hyperparameters in the option scripts (store option script in `./options`) in yaml file format. For faster data I/O, I recommend using `--persistent_cache` option if there is enough storage for disk caching. Note that `exp_name` should be the filename of option script.

### Inference:
Not yet published. Inference and evaluation usage example will be updated soon.

## Dataset used
We used 4 different datasets including [Mice Brain](https://pubmed.ncbi.nlm.nih.gov/30524114/) (private), [MSD-Task6](http://medicaldecathlon.com/) (public), [OASIS-3](https://sites.wustl.edu/oasisbrains/home/oasis-3/) (public upon approval), and [HCP1200](https://humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release) (public upon approval).

### Data Structure:
    ./data
        - {data_source_name}
        - meta
            - {meta_json_for_data}.json
Note that `./data/meta/{meta_json_for_data}.json` should contain list of file locations stored in `./data/{data_source_name}` (see `./data/meta/msd_task06.json` as a reference). In option script, `data_opt-source` should have the same name as `{meta_json_for_data}`.

### Cite
```
@misc{ha20243d,
      title={3D Volumetric Super-Resolution in Radiology Using 3D RRDB-GAN}, 
      author={Juhyung Ha and Nian Wang and Surendra Maharjan and Xuhong Zhang},
      year={2024},
      eprint={2402.04171},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
This is ArXiv citation. Once ISBI paper is published, I will upload the citation for ISBI.