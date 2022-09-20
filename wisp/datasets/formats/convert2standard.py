
'''
Convert EG3D format into the format for single-scene-overfitting (SSO).

'''

import sys
if '/viscam/u/jrshue/neural-field-diffusion' not in sys.path:
    sys.path.append('/viscam/u/jrshue/neural-field-diffusion')
    
import os
import glob
import argparse
import json
import numpy as np



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                    help='directory where original NeRF data is', required=True)
    parser.add_argument('--outdir', type=str,
                    help='directory where to save formatted NeRF data to', required=True)
    parser.add_argument('--train_percent', type=int, default=70,
                    help='percent of training set to allocate to training.', required=False)
    parser.add_argument('--val_percent', type=int, default=10,
                    help='percent of training set to allocate to validation. The rest is allocated to testing.', required=False)
    parser.add_argument('--sso', default=False, required=False, action='store_true',
                    help='If we only want to export one scene for the overfitting experiment.')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    json_files = glob.glob(f'{args.data_dir}/*.json')

    # Extract JSON
    print(f'Loading json file from {json_files[0]}...')
    with open(f'{json_files[0]}', 'r') as json_file:
        transforms = json.load(json_file)

    print(f'transforms.keys(): {transforms.keys()}')

    if args.sso:
        # Make a list of the file names for SSO
        shape_idx = 0  # Index of the shape to use for SSO
        def filter_fn(x, idx=shape_idx):
            if (f'{str(idx).zfill(5)}/' in x[0]):
                return True
            else:
                return False
        file_transforms = list(filter(filter_fn, [x for x in transforms['labels']]))[:250]  # !Hard-coded placeholder!

        # Split into train/val/test
        transforms_dict = dict()
        transforms_dict['train'] = file_transforms[:len(file_transforms)*args.train_percent//100]
        transforms_dict['val'] = file_transforms[len(file_transforms)*args.train_percent//100:len(file_transforms)*(args.train_percent+args.val_percent)//100]
        transforms_dict['test'] = file_transforms[len(file_transforms)*(args.train_percent+args.val_percent)//100:]

        for key in transforms_dict:
            split_dict = dict()
            split_dict['fl_x'] = 1.025390625 * 64  # !Temporary hard-code!
            split_dict['frames'] = []
            
            for frame in transforms_dict[key]:
                split_dict['frames'].append(dict(
                    file_path=f'./{frame[0]}',
                    transform_matrix=np.array(frame[1][:16]).reshape(4, 4).tolist()
                    ))

            with open(f'{args.outdir}/transforms_{key}.json', "w") as outfile:
                json.dump(split_dict, outfile, indent=4)
        
        # Copy shape files into outdir
        os.system(f'cp -r {args.data_dir}/{str(shape_idx).zfill(5)} {args.outdir}/')

    # print(transforms['labels'][0])
    # with open(f'{args.data_dir}/')




if __name__ == "__main__":
    main()

